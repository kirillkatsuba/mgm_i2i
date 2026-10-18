"""Offline checks for external evaluation identity, model selection and saved outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.external_data import PROTOCOL, digest, load_bank, verify_source
from mgm_i2i.cluster.metrics import evaluation_bank


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bank(tmp_path):
    root = tmp_path/'bsd100'
    root.mkdir()
    # Constant but distinct intensities let us check reconstruction and image ordering.
    pixels = np.broadcast_to(np.arange(100, dtype=np.uint8)[:, None, None, None], (100,128,128,3)).copy()
    np.save(root/'external.npy', pixels)
    manifest = dict(preparation=dict(protocol=PROTOCOL, crop=128, scale=4),
                    banks={'external.npy': dict(sha256=digest(root/'external.npy'),
                        crops=[dict(image=f'img_{i:03d}.png', top=0, left=0) for i in range(100)])})
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root


def signed_plan(root, plan):
    root.mkdir(parents=True)
    raw = json.dumps(plan).encode()
    (root/'plan.json').write_bytes(raw)
    (root/'plan.sha256').write_text(hashlib.sha256(raw).hexdigest())


def test_external_bank_is_common_to_legacy_and_v2_and_detects_changes(bank):
    _, checksum = load_bank(bank)
    for cfg in [dict(eval_crop=256), dict(eval_crop=128, architecture_version=2)]:
        dataset, rows, actual, name, crop = evaluation_bank(cfg, 'different training hash', 'external', bank, checksum)
        assert len(dataset) == len(rows) == 100 and actual == checksum
        assert name == 'BSD100' and crop == 128
        assert dataset[17].shape == (3,128,128) and dataset[17].unique().item() == 17
    with pytest.raises(ValueError, match='manifest changed'):
        load_bank(bank, 'wrong')
    with pytest.raises(ValueError, match='DIV2K multicrop'):
        evaluation_bank({}, 'training', 'test', bank, checksum)
    data = np.load(bank/'external.npy')
    data[0,0,0,0] += 1
    np.save(bank/'external.npy', data)
    with pytest.raises(ValueError, match='bank checksum'):
        load_bank(bank)


def test_pinned_download_verifies_git_blob_bytes(tmp_path):
    path = tmp_path/'sample.png'
    raw = b'fixture bytes'
    path.write_bytes(raw)
    entry = dict(size=len(raw), git_blob_sha1=hashlib.sha1(b'blob 13\0'+raw).hexdigest())
    verify_source(path, entry)
    path.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='source checksum'):
        verify_source(path, entry)


def test_v1_wide_plan_uses_v1_warmup_and_rejects_v2_reuse(tmp_path, monkeypatch):
    submit = script('cluster_submit')
    source, target = tmp_path/'v2', tmp_path/'v1'
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-root', str(source), '--variants', 'mgm'])
    submit.main()
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-root', str(target), '--variants', 'mgm',
                                    '--architecture-version', '1', '--width', '120'])
    submit.main()
    plan = json.loads((target/'plan.json').read_text())
    assert [len(plan[k]) for k in ('warmup','train','eval')] == [3,3,3]
    assert plan['budget']['training_gpu_hours'] == 12.
    assert all(x['config']['architecture_version'] == 1 and x['config']['width'] == 120
               for phase in ('warmup','train','eval') for x in plan[phase])
    assert all(x['config']['noise_g'] == x['config']['noise_f'] == .01 for x in plan['train'])
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-root', str(tmp_path/'invalid'), '--variants', 'mgm',
                                    '--architecture-version', '1', '--reuse-warmup-from', str(source)])
    with pytest.raises(ValueError, match='architecture_version'):
        submit.main()


def test_external_plan_keeps_all_seeds_nfes_and_legacy_models(bank, tmp_path):
    planner, submit = script('cluster_eval_submit'), script('cluster_submit')
    legacy, current = tmp_path/'legacy', tmp_path/'current'
    cfg = dict(method='mgm', variant='mgm', seed=0, eval_crop=256, eval_batch_size=4, cpus=4)
    signed_plan(legacy, dict(train=[dict(config=cfg, output=str(legacy/'mgm/seed0'))]))
    bridge = dict(cfg, method='i2sb', variant='i2sb', eval_crop=128, architecture_version=2)
    entries = [dict(config=dict(bridge, seed=seed), nfe=nfe, checkpoint=str(current/f'seed{seed}/latest.pt'),
                    label=f'i2sb_nfe{nfe}', output='unused') for seed in (0,1,2) for nfe in (1,4,16,64)]
    signed_plan(current, dict(train=[], eval=entries))
    args = SimpleNamespace(data_root=bank, source_runs=[legacy,current], run_root=tmp_path/'eval',
                           partition='rocky', max_parallel=4, constraint='', account='', gpu_type='',
                           memory_gb=32, eval_minutes=30)
    plan = planner.make_plan(args)
    assert not plan['train'] and not plan['warmup'] and len(plan['eval']) == 13
    assert plan['eval'][0]['config']['eval_crop'] == 256  # Training metadata is untouched.
    assert all(x['eval_batch_size'] == 1 for x in plan['eval'])
    assert len({x['evaluation_data_hash'] for x in plan['eval']}) == 1
    commands = submit.commands(plan, args.run_root/'plan.json')
    assert len(commands) == 2 and '--array=0-12%4' in commands[0]
    assert not any('--dependency' in arg for arg in commands[0])
    assert '--dependency=afterany:<eval_job_id>' in commands[1]
    with pytest.raises(FileNotFoundError, match='Trained checkpoint missing'):
        planner.verify_inputs(plan)


def test_external_runner_selects_external_split_and_common_batch(bank, tmp_path, monkeypatch):
    from mgm_i2i.cluster import metrics
    cfg = dict(method='mgm', variant='mgm', seed=0, eval_batch_size=4)
    entry = dict(config=cfg, checkpoint='trained.pt', output='new_output', nfe=1, eval_batch_size=1,
                 evaluation_data_root=str(bank), evaluation_data_hash=digest(bank/'manifest.json'))
    root = tmp_path/'run'
    signed_plan(root, dict(evaluation_split='external', eval=[entry], train=[], warmup=[]))
    calls = []
    monkeypatch.setattr(metrics, 'evaluate', lambda *args, **kwargs: calls.append((args,kwargs)))
    monkeypatch.setattr(sys, 'argv', ['run', '--plan', str(root/'plan.json'), '--phase', 'eval', '--index', '0'])
    script('cluster_run').main()
    args, kwargs = calls[0]
    assert args[2:5] == ('external',1,1)
    assert kwargs == dict(evaluation_data_root=str(bank), evaluation_data_hash=entry['evaluation_data_hash'],
                          expected_config=cfg)


def test_external_evaluator_outputs_all_images_and_collector_separates_training_data(bank, tmp_path, monkeypatch):
    from mgm_i2i.cluster import metrics

    class Nearest(nn.Module):
        def forward(self, lr, z):
            return torch.nn.functional.interpolate(lr, scale_factor=4, mode='nearest')

    class FakeLPIPS(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, a, b):
            return (a-b).square().mean((1,2,3), keepdim=True)

    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kwargs: self)
    monkeypatch.setattr(nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(metrics, 'seed_all', lambda seed: torch.manual_seed(seed))
    monkeypatch.setattr(metrics, 'provenance', lambda: {'gpu': 'CPU test fixture'})
    monkeypatch.setattr(metrics, 'build_generator', lambda cfg: Nearest())
    monkeypatch.setitem(sys.modules, 'lpips', SimpleNamespace(LPIPS=FakeLPIPS))
    monkeypatch.setattr(torch.hub, 'set_dir', lambda path: None)
    weights = tmp_path/'weights/hub/checkpoints'
    weights.mkdir(parents=True)
    (weights/'alexnet-owt-7be5be79.pth').touch()
    cfg = dict(method='mse', variant='mse', seed=0, cpus=1, workers=0, scale=4,
               weights_root=str(tmp_path/'weights'), eval_crop=256, eval_batch_size=4)
    checkpoint = tmp_path/'trained.pt'
    torch.save(dict(complete=True, config=cfg, ema={}, data_hash='legacy-training',
                    step=5, successful_g=5, consumed=80, elapsed_training_seconds=60.), checkpoint)
    checksum = digest(bank/'manifest.json')
    result = metrics.evaluate(checkpoint, tmp_path/'output', 'external', 25,
                              evaluation_data_root=bank, evaluation_data_hash=checksum, expected_config=cfg)
    assert result['images'] == result['crops'] == 100 and result['evaluation_crop'] == 128
    assert result['training_data_hash'] == 'legacy-training'
    assert result['evaluation_data_hash'] == checksum
    assert result['metrics']['model_psnr'] == pytest.approx(120.)
    assert result['metrics']['model_lpips'] == pytest.approx(0.)
    assert len(list((tmp_path/'output/samples').glob('*.png'))) == 100
    with Image.open(next((tmp_path/'output/samples').glob('*.png'))) as image:
        assert image.size == (128*7, 128)
    plan = dict(evaluation_split='external', evaluation_dataset='BSD100', train=[], eval=[])
    for seed, train_hash in [(0,'legacy-training'), (1,'new-training')]:
        out = tmp_path/f'seed{seed}'
        out.mkdir()
        (out/'external_metrics.json').write_text(json.dumps(dict(result, seed=seed, data_hash=train_hash)))
        plan['eval'].append(dict(config=dict(cfg, seed=seed), output=str(out), evaluation_data_hash=checksum))
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    collector = script('cluster_collect')
    collector.collect(path)
    summary = (tmp_path/'external_summary.md').read_text()
    assert 'BSD100' in summary and '2/2' in summary and 'Training datasets/protocols differ' in summary
    plan['eval'][0]['evaluation_data_hash'] = 'wrong dataset'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='External dataset mismatch'):
        collector.collect(path)
