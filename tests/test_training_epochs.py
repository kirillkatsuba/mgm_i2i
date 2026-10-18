"""Epoch targets, checkpoint continuation and Slurm planning, without CUDA training."""
from io import StringIO
import json
import signal
import sys

import pytest
import torch

from test_mgm_training import script
from mgm_i2i.cluster.budget import TrainingBudget
from mgm_i2i.cluster.progress import BudgetProgress


def test_epochs_override_wall_time_and_resume_from_consumed_patches():
    cfg = dict(seconds=1., epochs=320, samples_per_epoch=800, batch_size=32)
    budget = TrainingBudget(cfg, 800)
    assert budget.target_samples == 256000
    assert not budget.reached(255968, 1000000.)
    assert budget.reached(256000, 0.)
    assert len(range(128000, budget.target_samples, 32)) == 4000
    assert budget.metadata(128000)['epochs_completed'] == 160
    legacy = TrainingBudget(dict(seconds=7200., batch_size=16), 800)
    assert not legacy.reached(1000000, 7199.)
    assert legacy.reached(0, 7200.)
    with pytest.raises(ValueError, match='image count'):
        TrainingBudget(cfg, 799)
    with pytest.raises(ValueError, match='divisible'):
        TrainingBudget(dict(cfg, epochs=1, batch_size=48), 800)


def test_epoch_progress_does_not_claim_completion_when_wall_time_exceeds_hint():
    stream = StringIO()
    progress = BudgetProgress(256000, 'mgm', elapsed=20000, step=4000,
                              samples_per_epoch=800, consumed=128000, stream=stream)
    assert progress.bar.n == 128000
    progress.update(21000, 4001, consumed=128032)
    progress.close()
    assert 'epoch=160.040/320' in stream.getvalue()
    assert '100.00%' not in stream.getvalue()


def test_epoch_plan_keeps_reused_warmup_and_separates_field_target_and_slurm_limits(tmp_path, monkeypatch):
    submit = script('cluster_submit')
    source, dest = tmp_path/'source', tmp_path/'epoch'
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-root', str(source), '--variants', 'mgm',
                                    '--data-protocol', 'random128-v3'])
    submit.main()
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-root', str(dest), '--variants', 'mgm_critic_noise',
        '--data-protocol', 'random128-v3',
        '--reuse-warmup-from', str(source), '--mgm-optimizer', 'authors', '--mgm-beta1', '.8',
        '--mgm-microbatch-size', '32', '--grad-accumulation', '1',
        '--field-initialization', 'warmup_features', '--field-warmup-epochs', '100',
        '--train-epochs', '320', '--train-time-limit-minutes', '240',
        '--field-warmup-time-limit-minutes', '60'])
    submit.main()
    plan = json.loads((dest/'plan.json').read_text())
    assert not plan['warmup'] and len(plan['reused_warmup']) == 3
    assert all('epochs' not in job['config'] for job in plan['reused_warmup'])
    assert all(job['config']['epochs'] == 100 for job in plan['field_warmup'])
    assert all(job['config']['epochs'] == 320 for job in plan['train'])
    for job in plan['field_warmup'] + plan['train']:
        assert job['config']['samples_per_epoch'] == 800 and job['config']['batch_size'] == 32
    assert plan['budget']['training_gpu_hours'] is None
    cmds = submit.commands(plan, dest/'plan.json')
    assert '--time=60' in cmds[0] and '--time=240' in cmds[1]


def test_real_training_loop_resumes_interrupted_epoch_target_on_cpu(tmp_path, monkeypatch):
    """Run actual optimizer/save/restore/loop logic; emulate CUDA allocation and signal only."""
    from mgm_i2i.cluster import engine
    from mgm_i2i.cluster.models import Generator
    handlers = {}
    sync_calls = 0
    interrupt = True

    def sync_cuda():
        nonlocal sync_calls
        sync_calls += 1
        if interrupt and sync_calls == 2:  # Initial checkpoint, then first completed update.
            handlers[signal.SIGUSR1](signal.SIGUSR1, None)

    class CPUCuda:
        is_available = staticmethod(lambda: True)
        synchronize = staticmethod(sync_cuda)
        reset_peak_memory_stats = staticmethod(lambda: None)
        max_memory_allocated = max_memory_reserved = staticmethod(lambda: 0)
        manual_seed_all = staticmethod(lambda seed: None)
        get_rng_state_all = staticmethod(lambda: [])
        set_rng_state_all = staticmethod(lambda state: None)

    class CPUTorch:
        cuda = CPUCuda()
        def __getattr__(self, name):
            return getattr(torch, name)
        @staticmethod
        def device(name):
            return torch.device('cpu')

    class Patches:
        def __len__(self):
            return 8
        def __getitem__(self, index):
            return torch.full((3, 32, 32), (index * 17) % 256, dtype=torch.uint8)

    def register(signum, handler):
        previous = handlers.get(signum, 0)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(engine, 'torch', CPUTorch())
    monkeypatch.setattr(engine, 'provenance', lambda: {})
    monkeypatch.setattr(engine, 'validate_data_protocol', lambda *a: None)
    monkeypatch.setattr(engine, 'training_dataset', lambda *a: Patches())
    monkeypatch.setattr(engine, 'build_generator', lambda cfg: Generator(8))
    monkeypatch.setattr(engine.signal, 'signal', register)
    (tmp_path/'manifest.json').write_text('{}')
    cfg = dict(seed=0, width=8, architecture_version=2, scale=4, projection=True,
               method='mse', variant='mse', data_root=str(tmp_path), cpus=1, workers=0,
               amp=False, lr_g=1e-4, ema_decay=.995, grad_clip=10., checkpoint_seconds=10000.,
               log_every=1, batch_size=4, epochs=2, samples_per_epoch=8, seconds=1e-9)
    with pytest.raises(SystemExit) as error:
        engine.train(cfg, tmp_path/'resumed')
    assert error.value.code == 75
    status = json.loads((tmp_path/'resumed/status.json').read_text())
    assert status['consumed'] == 4 and status['epochs_completed'] == .5 and not status['complete']
    interrupt = False
    resumed = engine.train(cfg, tmp_path/'resumed', resume=True)
    assert resumed['complete'] and resumed['consumed'] == 16 and resumed['step'] == 4
    assert resumed['epochs_completed'] == resumed['target_epochs'] == 2
    reference = engine.train(cfg, tmp_path/'reference')
    for key, weight in reference['generator'].items():
        torch.testing.assert_close(resumed['generator'][key], weight, rtol=0, atol=0)
    for actual, expected in zip(resumed['optimizer_g']['state'].values(), reference['optimizer_g']['state'].values()):
        assert actual['step'] == expected['step'] == 4
        torch.testing.assert_close(actual['exp_avg'], expected['exp_avg'], rtol=0, atol=0)
