"""Offline checks of baseline math, gradients, and immutable Slurm orchestration."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.baselines import (BridgeDenoiser, I2SBBridge, RRDBNet, VGGDiscriminator,
                                     VGGPerceptual, relativistic_loss)
from mgm_i2i.cluster.engine import generator_objective
from mgm_i2i.cluster.metrics import predict_sr
from mgm_i2i.cluster.models import Field, Generator


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_sizes_match_selected_specification():
    with torch.device('meta'):
        for model, expected in [(RRDBNet(), 16697987), (VGGDiscriminator(), 14499401),
                                (BridgeDenoiser(), 17289963)]:
            assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize('method', ['mse', 'mse_perceptual', 'rrdb_l1', 'esrgan', 'mgm', 'i2sb'])
def test_method_objective_backpropagates_and_optimizer_updates(method):
    torch.manual_seed(10)
    config = dict(method=method, amp=False, scale=4, perceptual_weight=.01, pixel_weight=.01,
                  gan_weight=.005, reconstruction_weight=0., noise_g=.01, noise_f=.01,
                  midpoint=False, antithetic=False)
    real = torch.rand(2, 3, 128, 128) * 2 - 1
    c = F.avg_pool2d(real, 4)
    model = (RRDBNet(features=8, blocks=1, growth=4) if method in ('esrgan', 'rrdb_l1') else
             BridgeDenoiser(8) if method == 'i2sb' else Generator(8))
    aux = VGGDiscriminator(features=8) if method == 'esrgan' else Field(8) if method == 'mgm' else None
    if method == 'mgm':
        # Engine updates the field before G. A still-zero field head carries
        # no generator signal, so exercise a nonzero learned-field fixture.
        nn.init.normal_(aux.net.out.weight, std=.01)
    if aux is not None:
        # G must receive gradients through the auxiliary input while its weights freeze.
        aux.requires_grad_(False)
    bridge = I2SBBridge(20) if method == 'i2sb' else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.out.weight.detach().clone() if method == 'esrgan' else next(model.parameters()).detach().clone()
    loss, mse, output = generator_objective(model, aux, c, real, config,
                                             perceptual=lambda a, b: F.l1_loss(a, b), bridge=bridge)
    assert torch.isfinite(loss) and torch.isfinite(mse) and output.shape == real.shape
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    optimizer.step()
    if aux is not None:
        assert all(p.grad is None for p in aux.parameters())
    if method == 'esrgan':
        assert not torch.equal(model.out.weight, before)
        aux.requires_grad_(True)
        loss_d = relativistic_loss(aux((real+1)/2), aux((output.detach()+1)/2), generator=False)
        loss_d.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in aux.parameters())


def test_bridge_target_inverse_sampler_endpoints_nfe_and_fixed_draws():
    bridge = I2SBBridge(20, .3)
    clean, degraded = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    indices = torch.tensor([0, 19])
    state, label = bridge.training_pair(clean, degraded, indices, torch.zeros_like(clean))
    torch.testing.assert_close(state - bridge.std(indices) * label, clean)
    assert (bridge.std(indices) > 0).all()
    assert not torch.equal(state, clean) and not torch.equal(state, degraded)

    class Oracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x, t):
            self.calls += 1
            step = ((t - 1e-4) / (1-1e-4) * 19).round().long()
            return (x - clean) / bridge.std(step)

    model = Oracle()
    single = bridge.sample(model, degraded, 1)
    ratio = bridge.variance_fwd[0] / bridge.variance_fwd[-1]
    torch.testing.assert_close(single, (1-ratio)*clean + ratio*degraded)
    assert model.calls == 1
    model.calls = 0
    first = bridge.sample(model, degraded, 4, generator=torch.Generator().manual_seed(7))
    assert model.calls == 4
    second = bridge.sample(model, degraded, 4, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match='NFE'):
        bridge.sample(model, degraded, 20)
    with pytest.raises(ValueError, match='even'):
        I2SBBridge(19)

    denoiser = BridgeDenoiser(8).eval()
    lr = torch.randn(2, 3, 8, 8)
    cfg = dict(method='i2sb', scale=4)
    batch = predict_sr(denoiser, lr, None, cfg, bridge=bridge, nfe=4)
    pieces = torch.cat([predict_sr(denoiser, lr[i:i+1], None, cfg, bridge=bridge, nfe=4, offset=i)
                        for i in range(2)])
    torch.testing.assert_close(batch, pieces)


def test_vgg_local_loading_pre_relu_and_frozen_target_branch(tmp_path):
    from torchvision.models.vgg import cfgs, make_layers
    source = make_layers(cfgs['E'])
    path = tmp_path/'vgg.pth'
    torch.save({'features.'+key: value for key, value in source.state_dict().items()}, path)
    loss_module = VGGPerceptual(path)
    assert sum(p.numel() for p in loss_module.parameters()) == 20024384
    assert isinstance(loss_module.features[-1], nn.Conv2d)
    assert not any(p.requires_grad for p in loss_module.parameters())
    prediction, target = torch.rand(1, 3, 32, 32, requires_grad=True), torch.rand(1, 3, 32, 32)
    loss_module(prediction, target).backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert all(p.grad is None for p in loss_module.parameters())


def test_slurm_submission_dependencies_and_duplicate_guard(tmp_path, monkeypatch):
    submit = load_script('cluster_submit')
    commands = []
    def fake_sbatch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=f'{700+len(commands)};cluster\n')
    monkeypatch.setattr(submit.subprocess, 'run', fake_sbatch)
    monkeypatch.setattr(submit, 'verify_data', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['cluster_submit', '--run-root', str(tmp_path), '--pilot', '--submit'])
    submit.main()
    plan = json.loads((tmp_path/'plan.json').read_text())
    assert [len(plan[k]) for k in ('warmup', 'train', 'eval')] == [2, 5, 6]
    assert len(commands) == 4
    assert '--array=0-1%4' in commands[0]
    assert '--dependency=afterok:701' in commands[1]
    assert '--dependency=afterok:702' in commands[2]
    assert '--dependency=afterany:703' in commands[3]
    assert all('--gpus=1' in cmd for cmd in commands[:3])
    with pytest.raises(FileExistsError, match='Submission already'):
        submit.main()
    assert len(commands) == 4


def test_i2sb_only_plan_needs_no_warmup_and_collect_keeps_nfe_separate(tmp_path, monkeypatch, capsys):
    submit = load_script('cluster_submit')
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-root', str(tmp_path), '--variants', 'i2sb',
                                    '--seeds', '0', '--i2sb-nfe', '1', '4'])
    submit.main()
    plan = json.loads((tmp_path/'plan.json').read_text())
    assert not plan['warmup'] and len(plan['train']) == 1 and len(plan['eval']) == 2
    assert 'init' not in plan['train'][0]
    assert plan['train'][0]['config']['seconds'] == 14400
    commands = submit.commands(plan, tmp_path/'plan.json')
    assert not any('--dependency=' in arg for arg in commands[0])
    for entry in plan['eval']:
        folder = Path(entry['output'])
        folder.mkdir(parents=True)
        metrics = {key: 1. for key in ('model_psnr', 'model_ssim', 'model_lpips', 'diversity_lpips',
                                      'model_lr_mse_raw', 'model_projected_psnr', 'model_projected_lpips')}
        metrics.update({name+'_'+metric: 1. for name in ('bicubic', 'projected_bicubic')
                        for metric in ('psnr', 'ssim', 'lpips')})
        data = dict(variant='i2sb', seed=0, split='val', nfe=entry['nfe'], metrics=metrics, step=4,
                    successful_g=4, training_seconds=14400, evaluation_seconds=10,
                    inference_ms_per_image=entry['nfe']*10, images=50, crops=50, data_hash='same')
        (folder/'val_metrics.json').write_text(json.dumps(data))
    collect = load_script('cluster_collect')
    collect.collect(tmp_path/'plan.json')
    summary = json.loads((tmp_path/'val_summary.json').read_text())
    assert [item['variant'] for item in summary] == ['i2sb_nfe1', 'i2sb_nfe4']


def test_runner_separates_training_from_nfe_evaluation_and_can_continue(tmp_path, monkeypatch):
    submit = load_script('cluster_submit')
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-root', str(tmp_path), '--pilot'])
    submit.main()
    plan_path = tmp_path/'plan.json'
    plan = json.loads(plan_path.read_text())
    from mgm_i2i.cluster import engine, metrics
    calls = []
    def train(config, output, init, resume):
        calls.append(('train', config['method'], resume))
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/'latest.pt').touch()
    def evaluate(checkpoint, output, split, batch, nfe=1):
        calls.append(('eval', str(checkpoint), split, nfe))
    monkeypatch.setattr(engine, 'train', train)
    monkeypatch.setattr(metrics, 'evaluate', evaluate)
    runner = load_script('cluster_run')
    monkeypatch.setattr(sys, 'argv', ['run', '--plan', str(plan_path), '--phase', 'train', '--index', '0'])
    runner.main()
    assert calls == [('train', 'mse', False)]
    monkeypatch.setattr(sys, 'argv', ['run', '--plan', str(plan_path), '--phase', 'train', '--index', '0', '--resume-if-exists'])
    runner.main()
    assert calls[-1] == ('train', 'mse', True)
    index = next(i for i, entry in enumerate(plan['eval']) if entry['config']['method'] == 'i2sb' and entry['nfe'] == 4)
    monkeypatch.setattr(sys, 'argv', ['run', '--plan', str(plan_path), '--phase', 'eval', '--index', str(index)])
    runner.main()
    assert calls[-1] == ('eval', plan['eval'][index]['checkpoint'], 'val', 4)
