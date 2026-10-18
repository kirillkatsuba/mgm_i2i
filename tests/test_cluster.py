"""CPU regression checks for cluster models, planning, and training helpers."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.models import Generator, Field, TimeEmbedding, project, get_architecture_version
from mgm_i2i.cluster.data import CropBank, SampleRange
from mgm_i2i.losses import observation, variational_loss


@pytest.mark.parametrize('failure', [FileNotFoundError('git'), PermissionError('git'),
                                     subprocess.TimeoutExpired(['git'], 5), None])
def test_provenance_without_working_git(monkeypatch, failure):
    from mgm_i2i.cluster import engine

    def unavailable(*args, **kwargs):
        assert kwargs['timeout'] == 5
        if failure is not None:
            raise failure
        return SimpleNamespace(returncode=128, stdout='', stderr='not a repository')

    monkeypatch.setattr(engine.subprocess, 'run', unavailable)
    monkeypatch.setattr(engine.torch.cuda, 'get_device_name', lambda index: 'test GPU')
    result = engine.provenance()
    assert result['git_commit'] == result['git_status'] == 'unavailable'
    assert result['gpu'] == 'test GPU'
    assert result['torch'] == torch.__version__


def test_provenance_preserves_available_git_metadata(monkeypatch):
    from mgm_i2i.cluster import engine

    def available(command, **kwargs):
        output = 'abc123\n' if command[1] == 'rev-parse' else ''
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(engine.subprocess, 'run', available)
    monkeypatch.setattr(engine.torch.cuda, 'get_device_name', lambda index: 'test GPU')
    result = engine.provenance()
    assert result['git_commit'] == 'abc123'
    assert result['git_status'] == ''  # clean tree is different from unavailable


def test_projection_and_generator_enforce_same_lr():
    c = torch.randn(2,3,8,8)
    y = torch.randn(2,3,32,32)
    corrected = project(y,c,4)
    assert torch.allclose(F.avg_pool2d(corrected,4),c,atol=1e-6)
    gen = Generator(width=8)
    # Nonzero output head exercises the residual and projection together.
    torch.nn.init.normal_(gen.net.out.weight, std=.01)
    fake = gen(c, torch.randn_like(y))
    assert torch.allclose(F.avg_pool2d(fake,4),c,atol=1e-6)


def test_new_generator_can_predict_residuals_larger_than_one():
    gen = Generator(width=8, consistency=False)
    with torch.no_grad():
        gen.net.out.bias.fill_(3.)
    c = torch.zeros(1, 3, 8, 8)
    result = gen(c, torch.zeros(1, 3, 32, 32))
    assert torch.allclose(result, torch.full_like(result, 3.))


def test_time_embedding_conditions_field_and_receives_gradients():
    torch.manual_seed(12)
    field = Field(width=8)
    torch.nn.init.normal_(field.net.out.weight, std=.05)
    c, x = torch.randn(1, 3, 8, 8), torch.randn(1, 3, 32, 32)
    t = torch.tensor([.1, .4]).reshape(2, 1, 1, 1)
    result = field(c.expand(2, -1, -1, -1), t, x.expand(2, -1, -1, -1))
    assert result.shape == (2, 3, 32, 32)
    assert not torch.allclose(result[0], result[1])
    result.square().mean().backward()
    assert field.time_embedding.mlp[0].weight.grad.abs().sum() > 0
    assert field.net.mid[0].time_projection.weight.grad.abs().sum() > 0
    embedding = TimeEmbedding(8)
    assert torch.equal(embedding(t), embedding(t.reshape(2, 1)))
    with pytest.raises(ValueError, match='one MGM time'):
        embedding(torch.zeros(2, 2))


def test_unversioned_checkpoints_preserve_original_model_behavior():
    assert get_architecture_version({}) == 1
    assert get_architecture_version({'architecture_version': 2}) == 2
    with pytest.raises(ValueError, match='Unsupported architecture'):
        get_architecture_version({'architecture_version': 3})
    gen = Generator(width=8, consistency=False, architecture_version=1)
    field = Field(width=8, architecture_version=1)
    # Original checkpoints had six generator channels, ten field channels,
    # and no embedding parameters. Strict restore must keep that contract.
    assert gen.state_dict()['net.stem.weight'].shape[1] == 6
    assert field.state_dict()['net.stem.weight'].shape[1] == 10
    assert not any('time_' in key for key in field.state_dict())
    restored = Field(width=8, architecture_version=get_architecture_version({}))
    restored.load_state_dict(field.state_dict(), strict=True)
    with torch.no_grad():
        gen.net.out.bias.fill_(3.)
    result = gen(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 32, 32))
    assert torch.allclose(result, torch.full_like(result, torch.tanh(torch.tensor(3.)).item()))


def test_field_freeze_keeps_generator_gradient_paths():
    c = torch.randn(2,3,8,8)
    real = torch.randn(2,3,32,32)
    gen, field = Generator(width=8), Field(width=8)
    torch.nn.init.normal_(field.net.out.weight, std=.01)
    field.requires_grad_(False)
    fake = gen(c,torch.randn_like(real))
    t = torch.full((2,1,1,1),.3)
    mix = observation(fake,real,t,torch.ones_like(t,dtype=torch.bool),torch.zeros_like(real))
    variational_loss(field(c,t,mix),real-fake).backward()
    assert gen.net.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in field.parameters())


def test_stateless_crop_sequence_survives_prefetch_resume(tmp_path):
    a = np.random.default_rng(17).integers(0,256,(4,40,40,3),dtype=np.uint8)
    path = tmp_path/'train.npy'
    np.save(path,a)
    first, resumed = CropBank(path,32,seed=9), CropBank(path,32,seed=9)
    for i in range(30):  # emulate discarded worker prefetch
        first[i]
    for i in SampleRange(12,24):
        assert torch.equal(first[i],resumed[i])


def test_plan_budget_and_common_initialization(tmp_path):
    spec = importlib.util.spec_from_file_location('cluster_submit', ROOT/'scripts/cluster_submit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(data_root=tmp_path/'data', weights_root=tmp_path/'weights',
                           run_root=tmp_path/'runs', width=120,batch_size=16,cpus=4,fp32=False,
                           partition='rocky',max_parallel=1,constraint='',account='',
                           seeds=[0,1,2],variants=list(module.DEFAULT_VARIANTS),
                           warmup_minutes=module.DEFAULT_WARMUP_MINUTES,
                           train_minutes=module.DEFAULT_TRAIN_MINUTES)
    plan = module.make_plan(args)
    assert len(plan['warmup']) == 6 and len(plan['train']) == 15 and len(plan['eval']) == 24
    assert all(item['config']['seconds'] == (14400 if item['config']['method'] == 'i2sb' else 7200)
               for item in plan['warmup'] + plan['train'])
    assert all(item['config']['architecture_version'] == 2 for item in plan['warmup'] + plan['train'])
    assert all(item['config']['width'] == 120 for item in plan['warmup'] + plan['train'])
    assert all(item['config']['data_protocol'] == 'random128-v3' and
               item['config']['crop'] == item['config']['eval_crop'] == 128
               for item in plan['warmup'] + plan['train'])
    assert plan['budget']['training_gpu_hours'] == pytest.approx(48.)
    assert plan['budget']['allocation_gpu_hours_upper_bound'] == pytest.approx(87.5)
    assert plan['budget']['warmup_limit_minutes'] == 130
    assert plan['budget']['train_limit_minutes'] == 250
    # The earliest warning is time_limit - 120s - 60s. It must leave startup
    # headroom as well as the full requested training interval.
    for phase, key in [('warmup', 'warmup_limit_minutes'), ('train', 'train_limit_minutes')]:
        earliest_warning = plan['budget'][key]*60 - 120 - 60
        assert earliest_warning - plan[phase][0]['config']['seconds'] >= 120
    for seed in args.seeds:
        unet = [x for x in plan['train'] if x['config']['seed'] == seed and
                x['config']['method'] in ('mse', 'mse_perceptual', 'mgm')]
        assert len({x['init'] for x in unet}) == 1
        esrgan = next(x for x in plan['train'] if x['config']['seed'] == seed and x['config']['method'] == 'esrgan')
        assert esrgan['init'] != unet[0]['init']
    cmds = module.commands(plan, tmp_path/'plan.json')
    assert any('afterok:' in word for word in cmds[1])
    assert any('afterok:' in word for word in cmds[2])
    assert any('afterany:' in word for word in cmds[3])


def test_critic_only_noise_changes_generator_observation_but_keeps_field_noise(monkeypatch):
    from mgm_i2i.cluster import engine

    fake = torch.ones(1, 3, 4, 4, requires_grad=True)
    real = -torch.ones_like(fake)
    t = torch.full((1, 1, 1, 1), .25)
    monkeypatch.setattr(engine, 'random_bridge', lambda *args, **kwargs:
                        (t, torch.zeros_like(t, dtype=torch.bool), torch.ones_like(fake)))
    observations = []

    def field(c, time, mix):
        observations.append(mix.detach().clone())
        return .1 * mix

    cfg = dict(midpoint=False, noise_g=0., noise_f=.01, antithetic=False, amp=False)
    field_loss = engine.mgm_objective(field, None, fake.detach(), real, cfg, False)
    generator_loss = engine.mgm_objective(field, None, fake, real, cfg, True)
    clean = .75 * fake.detach() + .25 * real
    torch.testing.assert_close(observations[0], clean + (.01 * .25 * .75)**.5)
    torch.testing.assert_close(observations[1], clean)
    generator_loss.backward()
    assert torch.isfinite(field_loss)
    assert torch.isfinite(fake.grad).all() and fake.grad.abs().sum() > 0


def test_reuse_warmup_plan_preserves_initialization_and_isolates_noise(tmp_path):
    spec = importlib.util.spec_from_file_location('cluster_submit', ROOT/'scripts/cluster_submit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(data_root=tmp_path/'data', weights_root=tmp_path/'weights',
                           run_root=tmp_path/'source', width=120, batch_size=16, cpus=4, fp32=False,
                           partition='rocky', max_parallel=3, constraint='', account='',
                           seeds=[0, 1, 2], variants=['mgm'], warmup_minutes=120., train_minutes=120.)
    source = module.make_plan(args)
    args.run_root.mkdir()
    raw = json.dumps(source).encode()
    (args.run_root/'plan.json').write_bytes(raw)
    signature = args.run_root/'plan.sha256'
    signature.write_text(hashlib.sha256(raw).hexdigest())
    args.reuse_warmup_from, args.run_root = args.run_root, tmp_path/'critic'
    args.variants = ['mgm_critic_noise']
    plan = module.make_plan(args)
    assert len(plan['warmup']) == 0
    assert len(plan['train']) == len(plan['eval']) == len(plan['reused_warmup']) == 3
    assert plan['budget']['training_gpu_hours'] == 6.
    assert plan['budget']['reused_warmup_training_gpu_hours'] == 6.
    for reference, candidate in zip(source['train'], plan['train']):
        assert reference['init'] == candidate['init']
        assert candidate['config']['architecture_version'] == 2
        assert candidate['config']['noise_f'] == .01 and candidate['config']['noise_g'] == 0.
        differences = {key for key in reference['config']
                       if reference['config'][key] != candidate['config'][key]}
        assert differences == {'variant', 'noise_g'}
    commands = module.commands(plan, args.run_root/'plan.json')
    assert len(commands) == 3
    assert '--job-name=mgm_train' in commands[0]
    assert not any('--dependency=' in word for word in commands[0])
    assert '--dependency=afterok:<train_job_id>' in commands[1]
    assert '--dependency=afterany:<eval_job_id>' in commands[2]
    args.width = 48
    with pytest.raises(ValueError, match='Source warmup config differs: width'):
        module.make_plan(args)
    args.width = 120
    signature.write_text('bad signature')
    with pytest.raises(ValueError, match='checksum mismatch'):
        module.make_plan(args)


def test_progress_resumes_clock_and_writes_log_lines():
    from io import StringIO
    from mgm_i2i.cluster.progress import BudgetProgress

    output = StringIO()
    progress = BudgetProgress(7200, 'mgm/seed0', elapsed=3600, step=100, stream=output)
    assert progress.bar.n == 3600
    progress.update(5400, 200, {'g': .01})
    progress.close()
    log = output.getvalue()
    assert '\r' not in log
    assert '50.00%' in log and '75.00%' in log
    assert 'elapsed=01:30:00/02:00:00' in log
    assert 'remaining=00:30:00' in log
    assert 'step=200' in log and 'g=0.01' in log
    assert '100.00%' not in log  # interrupted close must not claim completion


def test_progress_clamps_deadline_overshoot():
    from io import StringIO
    from mgm_i2i.cluster.progress import BudgetProgress

    output = StringIO()
    progress = BudgetProgress(7200, 'mgm/seed0', stream=output)
    progress.update(7200.25, 300)
    progress.close()
    assert progress.bar.n == 7200
    assert '100.00%' in output.getvalue()
    assert 'remaining=00:00:00' in output.getvalue()


def test_central_bank_preserves_pixels_without_random_spatial_transforms(tmp_path):
    pixels = np.random.default_rng(31).integers(0, 256, (1, 128, 128, 3), dtype=np.uint8)
    path = tmp_path/'train.npy'
    np.save(path, pixels)
    expected = torch.from_numpy(pixels[0].transpose(2, 0, 1).copy())
    train = CropBank(path, seed=11, mode='center')
    validation = CropBank(path, training=False, mode='center')
    for index in [0, 1, 15, 10000]:
        assert torch.equal(train[index], expected)
    assert torch.equal(validation[0], expected)
    # A central crop of an old randomly selected 256 bank is not the center
    # of the original photograph and must not silently become the new data.
    np.save(path, np.zeros((1, 256, 256, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='original central'):
        CropBank(path, mode='center')


def test_central_protocol_rejects_old_manifest_but_legacy_runs_still_load(tmp_path):
    import json
    from mgm_i2i.cluster.data import validate_data_protocol
    manifest = tmp_path/'manifest.json'
    cfg = dict(data_protocol='center128-v2', crop=128, eval_crop=128)
    manifest.write_text(json.dumps({'preparation': {'bank_size': 256, 'patches_per_image': 8}}))
    with pytest.raises(ValueError, match='prepare fresh data'):
        validate_data_protocol(tmp_path, cfg)
    validate_data_protocol(tmp_path, {})
    manifest.write_text(json.dumps({'preparation': {
        'protocol': 'center128-v2', 'bank_size': 128, 'patches_per_image': 1}}))
    validate_data_protocol(tmp_path, cfg)
    with pytest.raises(ValueError, match='prepare fresh data'):
        validate_data_protocol(tmp_path, dict(cfg, eval_crop=256))


@pytest.mark.parametrize('crop_mode', ['center', 'random', 'multicrop'])
def test_preparation_extracts_original_centers_and_image_level_split(tmp_path, monkeypatch, crop_mode):
    import io
    import json
    import zipfile
    from PIL import Image
    spec = importlib.util.spec_from_file_location('cluster_prepare', ROOT/'scripts/cluster_prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archives, weights = tmp_path/'archives', tmp_path/'weights'
    archives.mkdir()
    # Odd dimensions exercise exact central coordinates; no resizing allowed.
    pixels = np.random.default_rng(19).integers(0, 256, (135, 143, 3), dtype=np.uint8)
    png = io.BytesIO()
    Image.fromarray(pixels).save(png, format='PNG')
    for split, ids in [('train', range(1, 801)), ('valid', range(801, 901))]:
        with zipfile.ZipFile(archives/f'DIV2K_{split}_HR.zip', 'w') as z:
            for image_id in ids:
                z.writestr(f'DIV2K/{image_id:04d}.png', png.getvalue())
    weight = weights/'hub/checkpoints/alexnet-owt-7be5be79.pth'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'unit-test placeholder, never loaded')
    original_digest = module.digest
    monkeypatch.setattr(module, 'digest', lambda path: '7be5be79' if Path(path) == weight
                        else original_digest(path))

    def offline_download(url, path):
        assert path.exists(), f'Unexpected download: {url}'

    monkeypatch.setattr(module, 'download', offline_download)
    args = SimpleNamespace(data_root=tmp_path/'center', weights_root=weights, archive_root=archives,
                           crop_mode=crop_mode, bank_size=128, patches_per_image=1, seed=123, skip_vgg=True)
    module.prepare(args)
    manifest = json.loads((args.data_root/'manifest.json').read_text())
    sets = []
    for split, count in [('train', 800), ('val', 50), ('test', 50)]:
        if split == 'train' and crop_mode in ('random', 'multicrop'):
            entries = manifest['train_images']
            assert len(entries) == 800
            assert not (args.data_root/'train.npy').exists()
            np.testing.assert_array_equal(np.load(args.data_root/entries[0]['path']), pixels)
            assert entries[0]['shape'] == list(pixels.shape)
            assert original_digest(args.data_root/entries[0]['path']) == entries[0]['sha256']
            sets.append({entry['image'] for entry in entries})
            continue
        bank = np.load(args.data_root/f'{split}.npy', mmap_mode='r')
        rows = manifest['banks'][f'{split}.npy']['crops']
        if crop_mode == 'multicrop':
            assert bank.shape == (count * 5, 128, 128, 3) and len(rows) == count * 5
            for image_id in {row['image'] for row in rows}:
                members = [r for r in rows if r['image'] == image_id]
                assert len({(r['top'], r['left']) for r in members}) == 5
                assert {r['crop_index'] for r in members} == set(range(5))
            for patch, row in zip(bank, rows):
                top, left = row['top'], row['left']
                np.testing.assert_array_equal(patch, pixels[top:top+128, left:left+128])
        else:
            assert bank.shape == (count, 128, 128, 3)
            np.testing.assert_array_equal(bank[0], pixels[3:131, 7:135])
            assert len(rows) == count
            assert all(row['top'] == 3 and row['left'] == 7 for row in rows)
        ids = {row['image'] for row in rows}
        assert len(ids) == count
        sets.append(ids)
    assert sets[0].isdisjoint(sets[1] | sets[2]) and sets[1].isdisjoint(sets[2])
    # Reusing the exact prepared dataset is idempotent and verifies hashes.
    module.prepare(args)
    if crop_mode == 'multicrop':
        from mgm_i2i.cluster.data import load_multicrop_evaluation
        from mgm_i2i.cluster.metrics import evaluation_bank
        manifest_hash = original_digest(args.data_root/'manifest.json')
        _, actual_hash = load_multicrop_evaluation(args.data_root, 'val', manifest_hash)
        assert actual_hash == manifest_hash
        with pytest.raises(ValueError, match='manifest changed'):
            load_multicrop_evaluation(args.data_root, 'test', 'wrong hash')
        # Frozen legacy models can use the same new evaluation bank without
        # modifying checkpoint config, training data hash or original data root.
        old_cfg = dict(data_root=str(tmp_path/'absent_old_data'), eval_crop=256)
        dataset, rows, _, dataset_name, size = evaluation_bank(old_cfg, 'old-training-hash', 'val',
                                                               args.data_root, manifest_hash)
        assert len(dataset) == len(rows) == 250 and dataset_name == 'DIV2K' and size == 128
        submit_spec = importlib.util.spec_from_file_location('multicrop_submit', ROOT/'scripts/cluster_submit.py')
        submit = importlib.util.module_from_spec(submit_spec)
        submit_spec.loader.exec_module(submit)
        source_args = SimpleNamespace(data_root=tmp_path/'old-data', weights_root=weights,
            run_root=tmp_path/'old-models', width=120, batch_size=16, cpus=4, fp32=False,
            partition='rocky', max_parallel=1, constraint='', account='', seeds=[0], variants=['mgm'],
            warmup_minutes=120, train_minutes=120)
        source_plan = submit.make_plan(source_args)
        submit.execute_plan(source_plan, source_args.run_root)
        eval_spec = importlib.util.spec_from_file_location('multicrop_eval_submit', ROOT/'scripts/cluster_eval_submit.py')
        evaluate = importlib.util.module_from_spec(eval_spec)
        eval_spec.loader.exec_module(evaluate)
        plan = evaluate.make_plan(SimpleNamespace(data_root=args.data_root, split='val',
            source_runs=[source_args.run_root], run_root=tmp_path/'shared-eval', partition='rocky',
            max_parallel=1, constraint='', account='', gpu_type='', memory_gb=32, eval_minutes=60))
        assert plan['evaluation_split'] == 'val' and plan['evaluation_dataset'] == 'DIV2K'
        assert len(plan['eval']) == 1 and not plan['train']
        assert plan['eval'][0]['evaluation_data_hash'] == manifest_hash
        assert plan['eval'][0]['config']['data_protocol'] == 'random128-v3'
    with pytest.raises(SystemExit, match='fresh --data-root'):
        module.prepare(SimpleNamespace(**dict(vars(args), seed=124)))
    if crop_mode in ('random', 'multicrop'):
        from mgm_i2i.cluster.data import training_dataset, RandomImageCrops, validate_data_protocol
        cfg = dict(data_protocol='random128-multicrop-v4' if crop_mode == 'multicrop' else 'random128-v3',
                   crop=128, eval_crop=128, seed=9)
        validate_data_protocol(args.data_root, cfg)
        assert isinstance(training_dataset(args.data_root, cfg), RandomImageCrops)
        submit_spec = importlib.util.spec_from_file_location('submit_check', ROOT/'scripts/cluster_submit.py')
        submit = importlib.util.module_from_spec(submit_spec)
        submit_spec.loader.exec_module(submit)
        monkeypatch.setattr(submit, 'digest', module.digest)
        submit.verify_data(args.data_root, weights)
        # Full-image corruption must be detected before a job can be submitted.
        (args.data_root/entries[0]['path']).write_bytes(b'corrupted')
        with pytest.raises(ValueError, match='Checksum mismatch'):
            submit.verify_data(args.data_root, weights)


def test_random_full_image_crops_are_stateless_unaugmented_and_cover_both_images(tmp_path):
    import json
    from mgm_i2i.cluster.data import RandomImageCrops, validate_data_protocol
    pixels = np.zeros((139, 147, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(139)[:, None]
    pixels[..., 1] = np.arange(147)[None, :]
    images, entries = [], []
    for i in range(2):
        image = pixels.copy()
        image[..., 2] = i
        np.save(tmp_path/f'{i}.npy', image)
        images.append(image)
        entries.append(dict(path=f'{i}.npy', image=f'{i}.png'))
    manifest = {'preparation': {'protocol': 'random128-v3', 'bank_size': 128,
                                'patches_per_image': 1}, 'train_images': entries}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    first, resumed = RandomImageCrops(tmp_path, seed=17), RandomImageCrops(tmp_path, seed=17)
    positions, selected = set(), set()
    for index in range(120):
        patch = first[index]
        assert patch.shape == (3, 128, 128) and patch.dtype == torch.uint8
        top, left, image_id = patch[:, 0, 0].tolist()
        expected = images[image_id][top:top+128, left:left+128].transpose(2, 0, 1)
        np.testing.assert_array_equal(patch.numpy(), expected)
        selected.add(image_id)
        positions.add((top, left))
    assert selected == {0, 1} and len(positions) > 50
    # Discarded prefetch and arbitrary access order cannot change future samples.
    for index in [130, 125, 122, 119, 300]:
        assert torch.equal(first[index], resumed[index])
    different_seed = RandomImageCrops(tmp_path, seed=18)
    assert any(not torch.equal(first[i], different_seed[i]) for i in range(5))
    with pytest.raises(ValueError, match='prepare fresh data'):
        validate_data_protocol(tmp_path, dict(data_protocol='center128-v2', crop=128, eval_crop=128))
    manifest['train_images'] = []
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='full train images'):
        validate_data_protocol(tmp_path, dict(data_protocol='random128-v3', crop=128, eval_crop=128))
