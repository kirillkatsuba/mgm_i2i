"""Authored for execution on the cluster; not executed during code preparation."""
import importlib.util
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
from mgm_i2i.cluster.models import Generator, Field, project
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
                           run_root=tmp_path/'runs', width=48,batch_size=16,cpus=4,fp32=False,
                           partition='rocky',max_parallel=1,constraint='',account='',
                           seeds=[0,1,2],variants=list(module.VARIANTS),warmup_minutes=12,train_minutes=18)
    plan = module.make_plan(args)
    assert len(plan['warmup']) == 3 and len(plan['train']) == 18
    assert plan['budget']['training_gpu_hours'] == 6
    assert plan['budget']['allocation_gpu_hours_upper_bound'] == 7.75
    # The earliest warning is time_limit - 120s - 60s. It must leave startup
    # headroom as well as the full requested training interval.
    for phase, key in [('warmup', 'warmup_limit_minutes'), ('train', 'train_limit_minutes')]:
        earliest_warning = plan['budget'][key]*60 - 120 - 60
        assert earliest_warning - plan[phase][0]['config']['seconds'] >= 120
    for seed in args.seeds:
        assert len({x['init'] for x in plan['train'] if x['config']['seed'] == seed}) == 1
    cmds = module.commands(plan, tmp_path/'plan.json')
    assert any('afterok:' in word for word in cmds[1])
    assert any('afterany:' in word for word in cmds[2])
