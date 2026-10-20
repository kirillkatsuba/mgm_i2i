"""Inference boundary/sampler parity, fixed draws and independent offline Slurm chains."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['XFORMERS_DISABLED'] = '1'
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster import sr_reference as ref


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources():
    for name in ('omegaconf', 'timm', 'einops', 'diffusers', 'fairscale'):
        pytest.importorskip(name)
    roots = {name: ROOT/'third_party'/pin['directory'] for name, pin in ref.PINS.items()}
    if not all(root.exists() for root in roots.values()):
        pytest.skip('Pinned author sources required for real upstream tests')
    ref.verify_sources(roots, 'sinsr')
    ref.verify_sources(roots, 'vosr')
    return roots


def test_reflection_supports_lr32_and_preserves_upstream_for_larger_inputs():
    image = torch.arange(3*32*32).reshape(1, 3, 32, 32).float()
    with pytest.raises(RuntimeError):
        F.pad(image, (0, 32, 0, 32), mode='reflect')
    padded = ref.reflect_to_multiple(image)
    assert padded.shape == (1, 3, 64, 64)
    torch.testing.assert_close(padded[..., :32, :32], image, rtol=0, atol=0)
    torch.testing.assert_close(padded[..., :32, 32:63], image[..., :, :31].flip(-1), rtol=0, atol=0)
    large = torch.randn(1, 3, 70, 80)
    torch.testing.assert_close(ref.reflect_to_multiple(large), F.pad(large, (0, 48, 0, 58), mode='reflect'), rtol=0, atol=0)
    full = torch.randn(1, 3, 64, 64)
    assert ref.reflect_to_multiple(full) is full


def test_fixed_draws_restore_rng_and_do_not_depend_on_batch_size():
    class RandomModel:
        def restore(self, image):
            return F.interpolate(image, scale_factor=4)+torch.randn(1, 3, 128, 128)*.01
    c = torch.zeros(2, 3, 32, 32)
    before = torch.get_rng_state().clone()
    both = ref.predict(RandomModel(), c, offset=7)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    single = ref.predict(RandomModel(), c[1:], offset=8)
    torch.testing.assert_close(both[1:], single, rtol=0, atol=0)
    assert not torch.equal(both[:1], ref.predict(RandomModel(), c[:1], offset=7, draw=1))


def test_sinsr_wrapper_matches_upstream_one_step_sampler(sources):
    from omegaconf import OmegaConf
    ref._package('_mgm_sinsr_models', sources['sinsr']/'models')
    ref._package('ldm', sources['sinsr']/'ldm')
    cfg = OmegaConf.to_container(OmegaConf.load(sources['sinsr']/'configs/SinSR.yaml'))
    params = dict(cfg['diffusion']['params'], timestep_respacing=15)
    sampler = importlib.import_module('_mgm_sinsr_models.script_util').create_gaussian_diffusion(**params)
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.tensor(.2)); self.calls = []
        def forward(self, x, t, lq):
            self.calls.append(t.item())
            return self.weight*x+.1*lq
    class AE(nn.Module):
        def __init__(self):super().__init__();self.weight = nn.Parameter(torch.tensor(1.))
        def encode(self, x):return F.avg_pool2d(x, 4)*self.weight
        def decode(self, x):return F.interpolate(x, scale_factor=4)*self.weight
    model = ref.SinSR.__new__(ref.SinSR)
    nn.Module.__init__(model)
    model.net, model.ae, model.diffusion = Net(), AE(), sampler
    c = torch.randn(1, 3, 32, 32)*.1
    actual = ref.predict(model, c, offset=3)
    with ref.draw_rng('cpu', 20260101+3*2):
        padded = ref.reflect_to_multiple(c)
        expected = sampler.p_sample_loop(y=padded, model=model.net, first_stage_model=model.ae,
                                        clip_denoised=False, model_kwargs={'lq': padded},
                                        one_step=True, progress=False, apply_decoder=True)
    torch.testing.assert_close(actual, expected[..., :128, :128].clamp(-1, 1), rtol=0, atol=0)
    assert model.net.calls == [14, 14]  # One actual network evaluation each, original final timestep.


def test_original_backbones_accept_required_shapes_and_have_author_parameter_counts(sources):
    from omegaconf import OmegaConf
    ref._package('_mgm_sinsr_models', sources['sinsr']/'models')
    cls = importlib.import_module('_mgm_sinsr_models.unet').UNetModelSwin
    cfg = OmegaConf.to_container(OmegaConf.load(sources['sinsr']/'configs/SinSR.yaml'))
    with torch.device('meta'):
        net = cls(**cfg['model']['params'])
        assert sum(p.numel() for p in net.parameters()) == 118593583
        assert net(torch.empty(1, 3, 64, 64), torch.zeros(1, dtype=torch.long),
                   lq=torch.empty(1, 3, 64, 64)).shape == (1, 3, 64, 64)
    dit, _, core, _, _ = ref.vosr_code(sources['vosr'])
    # Actual upstream flexible-resolution DiT and original one-step solver, CPU numerical forward.
    tiny = dit(input_size=64, patch_size=2, in_channels=8, out_channels=4, hidden_size=64,
               depth=1, num_heads=4, z_dims=32, encdim_ratio=1, use_rope=True,
               use_rmsnorm=True, auxiliary_time_cond=True).eval()
    tiny.forward = tiny.forward_flexible
    solver = core(accelerator=SimpleNamespace(device='cpu'))
    latent, features = torch.randn(1, 4, 16, 16), [torch.randn(1, 1024, 32)]
    with torch.no_grad(), ref.draw_rng('cpu', 42):
        result = solver.sample_onestep(tiny, latent, features, n_steps=1)
    assert result.shape == latent.shape and torch.isfinite(result).all()


def test_bicubic_preserves_float_pooling_input():
    c = torch.full((1, 3, 32, 32), .123456)
    up = ref.tensor_bicubic(c)
    assert up.shape == (1, 3, 128, 128)
    torch.testing.assert_close(up, torch.full_like(up, .123456), atol=1e-6, rtol=0)
    assert not torch.allclose((up+1)*127.5, ((up+1)*127.5).round())


def test_prepare_bundle_reuses_verified_assets_and_rejects_changes(tmp_path, monkeypatch):
    prepare = script('prepare_sr_references')
    names = ['SinSR_v1.pth', 'autoencoder_vq_f4.pth']
    folder = tmp_path/'weights/sinsr_v1'
    folder.mkdir(parents=True)
    for name in names:(folder/name).write_bytes(b'valid')
    specs = {name:dict(url='unused', size=5, sha256=None) for name in names}
    monkeypatch.setitem(ref.ASSETS, 'sinsr', specs)
    code = tmp_path/'code'/ref.PINS['sinsr']['directory'];code.mkdir(parents=True)
    monkeypatch.setattr(prepare.subprocess, 'check_output', lambda *a, **k:ref.PINS['sinsr']['commit'])
    monkeypatch.setattr(prepare, 'verify_sources', lambda *a:None)
    prepare.prepare('sinsr', tmp_path/'code', tmp_path/'weights')
    manifest = ref.load_bundle(folder/'manifest.json', 'sinsr')
    assert set(manifest['files']) == set(names)
    prepare.prepare('sinsr', tmp_path/'code', tmp_path/'weights')  # No network needed on repeat.
    (folder/names[0]).write_bytes(b'other')
    with pytest.raises(ValueError, match='changed reference asset'):
        ref.load_bundle(folder/'manifest.json', 'sinsr')


@pytest.mark.parametrize('response_code', [200, 206])
def test_download_resumes_or_restarts_when_server_ignores_range(tmp_path, monkeypatch, response_code):
    import hashlib
    prepare = script('prepare_sr_references')
    path = tmp_path/'weight.pth'
    path.with_suffix('.pth.part').write_bytes(b'abc')
    payload = b'def' if response_code == 206 else b'abcdef'
    class Response:
        status_code = response_code
        headers = {'Content-Range': 'bytes 3-5/6'}
        def __enter__(self):return self
        def __exit__(self, *a):pass
        def raise_for_status(self):pass
        def iter_content(self, n):yield payload
    def get(url, headers, **kwargs):
        assert headers == {'Range': 'bytes=3-'}
        return Response()
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=get))
    spec = dict(url='unused', size=6, sha256=hashlib.sha256(b'abcdef').hexdigest())
    prepare.download_file(path, spec, True)
    assert path.read_bytes() == b'abcdef'
    prepare.download_file(path, spec, False)


@pytest.mark.parametrize('method', ['sinsr', 'vosr'])
def test_separate_slurm_chains_have_probe_dependencies_and_no_ram_requests(tmp_path, monkeypatch, method):
    submit = script('cluster_sr_reference_submit')
    args = SimpleNamespace(data_root=tmp_path/'data', weights_root=tmp_path/'weights',
                           third_party=ROOT/'third_party', run_root=tmp_path/'runs', split='val',
                           cpus=4, partition='rocky', account='', constraint='', gpu_type='',
                           probe_minutes=20, eval_minutes=90)
    monkeypatch.setattr(submit, 'load_multicrop_evaluation', lambda *a:({'preparation': {'eval_crops_per_image': 5}}, 'common'))
    monkeypatch.setattr(submit, 'load_bundle', lambda *a:dict(checkpoint='weight'))
    monkeypatch.setattr(submit, 'verify_sources', lambda *a:None)
    monkeypatch.setattr(submit, 'digest', lambda *a:'7be5be79abcdef')
    plan, root = submit.make_plan(args, method)
    assert root == tmp_path/'runs'/method
    assert plan['eval'][0]['config']['reference_method'] == method
    assert len(plan['eval']) == len(plan['probe']) == 1
    common = script('cluster_submit')
    commands = []
    def sbatch(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout=f'{800+len(commands)}\n')
    monkeypatch.setattr(common.subprocess, 'run', sbatch)
    common.execute_plan(plan, root, True)
    assert '--gpus=1' in commands[0] and '--gpus=1' in commands[1]
    assert '--dependency=afterok:801' in commands[1]
    assert '--dependency=afterany:802' in commands[2]
    assert all(not arg.startswith('--mem') for cmd in commands for arg in cmd)
    with pytest.raises(FileExistsError):common.execute_plan(plan, root, True)


@pytest.mark.parametrize('seconds,complete', [(1., True), (100., False)])
def test_probe_gates_slow_evaluation_and_records_hardware(tmp_path, monkeypatch, seconds, complete):
    worker = script('cluster_sr_reference_run')
    from mgm_i2i.cluster import engine, metrics
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:True)
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda:None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda:100)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda:200)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **k:self)
    monkeypatch.setattr(engine, 'provenance', lambda:dict(gpu='CPU fixture'))
    monkeypatch.setattr(ref, 'load_for_evaluation', lambda *a:({}, None, None, {}))
    bank = [torch.zeros(3, 128, 128, dtype=torch.uint8)]*250
    monkeypatch.setattr(metrics, 'evaluation_bank', lambda *a:(bank, None, 'common', None, 128))
    monkeypatch.setattr(metrics, 'predict_sr', lambda model, c, *a, **k:F.interpolate(c, scale_factor=4))
    ticks = iter([v for _ in range(5) for v in (0., seconds)])
    monkeypatch.setattr(worker.time, 'perf_counter', lambda:next(ticks))
    item = dict(config=dict(cpus=1, reference_bundle_sha256='bundle'), checkpoint='unused',
                evaluation_data_root='unused', evaluation_data_hash='common')
    if complete:worker.probe(item, tmp_path, 'val', 90)
    else:
        with pytest.raises(RuntimeError, match='allocation'):worker.probe(item, tmp_path, 'val', 90)
    report = json.loads((tmp_path/'probe.json').read_text())
    assert report['complete'] == complete and report['peak_allocated_bytes'] == 100


def test_reference_evaluator_uses_shared_metrics_and_unknown_training_cost(tmp_path, monkeypatch):
    from mgm_i2i.cluster import metrics
    class FakeLPIPS(nn.Module):
        def __init__(self, **k):super().__init__()
        def forward(self, a, b):return (a-b).square().mean((1, 2, 3), keepdim=True)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **k:self)
    monkeypatch.setattr(nn.Module, 'cuda', lambda self:self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda:123)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda:456)
    monkeypatch.setattr(metrics, 'seed_all', lambda s:None)
    monkeypatch.setattr(metrics, 'provenance', lambda:dict(gpu='CPU fixture'))
    monkeypatch.setattr(torch.hub, 'set_dir', lambda p:None)
    monkeypatch.setitem(sys.modules, 'lpips', SimpleNamespace(LPIPS=FakeLPIPS))
    cfg = dict(method='sr_reference', reference_method='sinsr', variant='sinsr_official_v1',
               cpus=1, workers=0, seed=0, scale=4, weights_root=str(tmp_path/'weights'))
    alex = tmp_path/'weights/hub/checkpoints/alexnet-owt-7be5be79.pth'
    alex.parent.mkdir(parents=True);alex.write_bytes(b'fixture')
    state = dict(complete=True, config=cfg, data_hash=None, step=None, successful_g=None,
                 consumed=None, elapsed_training_seconds=None, training_stop='external_pretrained')
    class Model:
        def restore(self, c):return F.interpolate(c, scale_factor=4)
    monkeypatch.setattr(ref, 'load_for_evaluation', lambda *a:(state, Model(), None, {}))
    bank = [torch.full((3, 128, 128), 100, dtype=torch.uint8)]*2
    monkeypatch.setattr(metrics, 'evaluation_bank', lambda *a:(bank, [dict(image=f'{i}.png') for i in range(2)], 'common', 'DIV2K', 128))
    checkpoint = tmp_path/'weight';checkpoint.write_bytes(b'fixture')
    result = metrics.evaluate(checkpoint, tmp_path/'eval', 'val', 1, expected_config=cfg,
                              evaluation_data_root=tmp_path, evaluation_data_hash='common')
    assert result['metrics']['model_lpips'] == pytest.approx(0, abs=1e-12)
    assert result['step'] is None and result['training_seconds'] is None
    assert result['evaluation_data_hash'] == 'common'
    plan = dict(run_kind='sr-reference-evaluation', external_pretrained=True, train=[],
                eval=[dict(config=cfg, output=str(tmp_path/'eval'), nfe=1)])
    path = tmp_path/'plan.json';path.write_text(json.dumps(plan))
    script('cluster_collect').collect(path)
    text = (tmp_path/'val_summary.md').read_text()
    assert 'SinSR/VOSR' in text and 'Official I2SB uses' not in text
