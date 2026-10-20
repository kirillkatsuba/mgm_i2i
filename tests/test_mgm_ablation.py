"""Noise controls: observation math, shared initializers and sweep scheduling."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from cluster_mgm_ablate import build_plan, verify_sources
from cluster_submit import VARIANTS, commands
from mgm_i2i.cluster import engine
from mgm_i2i.cluster.mgm_training import validate_field_checkpoint


@pytest.mark.parametrize('name,noise_f,noise_g,midpoint', [
    ('mgm_no_noise', 0., 0., False),
    ('mgm', .01, .01, False),
    ('mgm_critic_noise', .01, 0., False),
    ('mgm_matched_noise_005', .05, .05, False),
    ('mgm_midpoint_critic_noise', .01, 0., True),
])
def test_noise_regimes_feed_the_correct_observation_to_each_player(monkeypatch, name, noise_f, noise_g, midpoint):
    cfg = dict(method='mgm', noise_f=.01, noise_g=.01, midpoint=False, antithetic=False, amp=False)
    cfg.update(VARIANTS[name])
    assert cfg['midpoint'] == midpoint
    fake = torch.full((2, 3, 4, 4), .2, requires_grad=True)
    real = torch.full_like(fake, -.3)
    seen = []

    def bridge(fake, midpoint):
        t = torch.full((len(fake), 1, 1, 1), .5 if midpoint else .25)
        b = torch.tensor([False, True]).reshape(2, 1, 1, 1)
        return t, b, torch.ones_like(fake)

    def field(c, t, mix):
        seen.append((t.clone(), mix.detach().clone()))
        return .1*mix

    monkeypatch.setattr(engine, 'random_bridge', bridge)
    engine.mgm_objective(field, None, fake.detach(), real, cfg, False)
    loss = engine.mgm_objective(field, None, fake, real, cfg, True)
    t, _ = seen[0]
    weight = torch.where(torch.tensor([False, True]).reshape(2, 1, 1, 1), 1-t, t)
    clean = (1-weight)*fake.detach() + weight*real
    for (_, mix), strength in zip(seen, [noise_f, noise_g]):
        torch.testing.assert_close(mix, clean + (strength*t*(1-t)).sqrt())
    loss.backward()
    assert torch.isfinite(fake.grad).all() and fake.grad.abs().sum() > 0


def prepare_args(tmp_path, monkeypatch, suite='core', retrain=False):
    import cluster_submit
    for version, variant in [(1, 'mgm'), (2, 'mgm_critic_noise')]:
        source = tmp_path/f'source_v{version}'
        monkeypatch.setattr(sys, 'argv', [
            'submit', '--run-root', str(source), '--data-root', str(tmp_path/'data'),
            '--weights-root', str(tmp_path/'weights'), '--architecture-version', str(version),
            '--variants', variant, '--warmup-epochs', '50', '--field-warmup-epochs', '50',
            '--train-epochs', '80', '--field-initialization', 'warmup_features',
            '--mgm-optimizer', 'authors', '--mgm-beta1', '.8', '--mgm-microbatch-size', '32'])
        cluster_submit.main()
    from types import SimpleNamespace
    return SimpleNamespace(run_root=tmp_path/'ablation', source_v1=tmp_path/'source_v1',
                           source_v2=tmp_path/'source_v2', data_root=tmp_path/'data',
                           weights_root=tmp_path/'weights', seeds=[0, 1, 2], suite=suite,
                           retrain_controls=retrain, max_parallel=2, cpus=4, field_minutes=90,
                           train_minutes=240, eval_minutes=30, partition='rocky', constraint='',
                           account='', gpu_type='')


@pytest.mark.parametrize('suite,retrain,training,evaluations', [
    ('core', False, 12, 18), ('paper', False, 24, 30), ('core', True, 18, 18),
])
def test_sweep_shares_warmups_keeps_architectures_separate_and_caps_all_jobs(
        tmp_path, monkeypatch, suite, retrain, training, evaluations):
    args = prepare_args(tmp_path, monkeypatch, suite, retrain)
    plan = build_plan(args)
    assert not plan['warmup']
    assert len(plan['reused_warmup']) == 6
    assert len(plan['train']) == len(plan['field_warmup']) == training
    assert len(plan['eval']) == evaluations
    assert len(plan['ablation']['reference_checkpoints']) == (0 if retrain else 6)
    assert len({item['output'] for item in plan['eval']}) == evaluations
    assert len({item['label'] for item in plan['eval']}) == evaluations//3
    for item in plan['train']:
        cfg = item['config']
        version, seed = cfg['architecture_version'], cfg['seed']
        assert item['init'] == str(tmp_path/f'source_v{version}/warmup/seed{seed}/latest.pt')
        assert cfg['epochs'] == 80 and cfg['samples_per_epoch'] == 3200
        assert cfg['batch_size'] == 32 and cfg.get('grad_accumulation', 1) == 1
        assert cfg['lr_g'] == cfg['lr_aux'] == 1e-5 and cfg['adam_betas'] == [.8, .999]
        assert cfg['projection'] and cfg['reconstruction_weight'] == 0.
        field = next(x for x in plan['field_warmup'] if x['output'] == str(Path(item['field_init']).parent))
        assert field['init'] == item['init'] and field['config']['epochs'] == 50
        for key in ('noise_f', 'noise_g', 'midpoint'):
            assert field['config'][key] == cfg[key]
    jobs = commands(plan, args.run_root/'plan.json')
    assert len(jobs) == 4
    assert f'--array=0-{training-1}%2' in jobs[0]
    assert '--dependency=afterok:<field_warmup_job_id>' in jobs[1]
    assert '--dependency=afterok:<train_job_id>' in jobs[2]
    assert all(not arg.startswith('--mem') for cmd in jobs for arg in cmd)
    assert plan['budget']['training_gpu_hours'] is None
    assert plan['budget']['estimated_training_gpu_hours_v100'] == pytest.approx(training*150.6/60)
    if not retrain:
        with pytest.raises(FileNotFoundError, match='control checkpoint missing'):
            verify_sources(plan)
    # Noisy and midpoint fields cannot silently share incompatible checkpoints.
    cfg = plan['train'][0]['config']
    state = dict(config=dict(cfg, field_only=True), complete=True, data_hash='data',
                 init_hash='init', step=20, skipped_aux=0)
    for key in ('noise_f', 'midpoint', 'antithetic'):
        changed = dict(cfg, **{key: not cfg[key] if isinstance(cfg[key], bool) else .2})
        with pytest.raises(ValueError, match=key):
            validate_field_checkpoint(state, changed, 'data', 'init')


def test_source_signature_and_old_control_policy_are_checked_before_submission(tmp_path, monkeypatch):
    args = prepare_args(tmp_path, monkeypatch)
    plan = build_plan(args)
    path = args.source_v1/'plan.json'
    source = json.loads(path.read_text())
    source['train'][0]['config']['noise_g'] = 0.
    raw = json.dumps(source).encode()
    path.write_bytes(raw)
    import hashlib
    path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match='does not match the current recipe'):
        build_plan(args)
    with pytest.raises(ValueError, match='Source plan changed'):
        verify_sources(plan)


def test_collector_does_not_average_different_architectures_together(tmp_path, monkeypatch, capsys):
    from cluster_collect import collect
    args = prepare_args(tmp_path, monkeypatch)
    plan = build_plan(args)
    args.run_root.mkdir()
    plan_path = args.run_root/'plan.json'
    plan_path.write_text(json.dumps(plan))
    for item in plan['eval']:
        cfg = item['config']
        value = float(cfg['architecture_version'])
        metrics = {f'{base}_{metric}': value
                   for base in ('model', 'bicubic', 'projected_bicubic')
                   for metric in ('psnr', 'ssim', 'lpips')}
        metrics.update(diversity_lpips=.01, model_lr_mse_raw=0.)
        result = dict(variant=cfg['variant'], seed=cfg['seed'], split='val', nfe=1,
                      metrics=metrics, step=8000, successful_g=8000, training_seconds=7100.,
                      evaluation_seconds=10., inference_ms_per_image=9., images=50, crops=250,
                      data_hash='shared')
        folder = Path(item['output'])
        folder.mkdir(parents=True)
        (folder/'val_metrics.json').write_text(json.dumps(result))
    collect(plan_path)
    rows = json.loads((args.run_root/'val_summary.json').read_text())
    assert len(rows) == 6
    for row in rows:
        assert row['completed'] == row['expected'] == 3
        expected = float(row['variant'][1])
        assert row['metrics']['model_psnr']['mean'] == expected
        assert row['metrics']['model_psnr']['std'] == 0.


def test_test_phase_uses_frozen_checkpoints_and_separate_samples_after_validation(tmp_path, monkeypatch):
    import hashlib
    import cluster_run
    args = prepare_args(tmp_path, monkeypatch, suite='paper')
    args.with_test = True
    plan = build_plan(args)
    assert len(plan['train']) == 24 and len(plan['eval']) == len(plan['test']) == 30
    assert plan['collect_splits'] == ['val', 'test']
    for validation, test in zip(plan['eval'], plan['test']):
        assert validation['config'] == test['config']
        assert validation['checkpoint'] == test['checkpoint']
        assert test['output'] == str(Path(validation['output'])/'test')
    jobs = commands(plan, args.run_root/'plan.json')
    assert len(jobs) == 5
    assert '--dependency=afterok:<eval_job_id>' in jobs[3]
    assert '--array=0-29%2' in jobs[3] and jobs[3][-1] == 'test'
    assert '--dependency=afterany:<test_job_id>' in jobs[4]
    assert plan['budget']['allocation_gpu_hours_upper_bound'] == 162.
    args.run_root.mkdir()
    path = args.run_root/'plan.json'
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest())
    calls = []
    from mgm_i2i.cluster import metrics
    monkeypatch.setattr(engine, 'train', lambda *a, **k: pytest.fail('Test phase attempted training'))
    monkeypatch.setattr(metrics, 'evaluate', lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(sys, 'argv', ['run', '--plan', str(path), '--phase', 'test', '--index', '0'])
    cluster_run.main()
    assert len(calls) == 1
    positional, kwargs = calls[0]
    assert positional[2] == 'test'
    assert positional[0] == Path(plan['test'][0]['checkpoint'])
    assert positional[1] == plan['test'][0]['output']
    assert kwargs['expected_config'] == plan['test'][0]['config']


def test_collector_writes_both_split_summaries_without_mixing_rows(tmp_path, monkeypatch):
    import cluster_collect
    args = prepare_args(tmp_path, monkeypatch)
    args.with_test = True
    plan = build_plan(args)
    args.run_root.mkdir()
    path = args.run_root/'plan.json'
    path.write_text(json.dumps(plan))
    for phase, split, value in [('eval', 'val', 1.), ('test', 'test', 2.)]:
        for item in plan[phase]:
            cfg = item['config']
            metrics = {f'{base}_{metric}': value
                       for base in ('model', 'bicubic', 'projected_bicubic')
                       for metric in ('psnr', 'ssim', 'lpips')}
            metrics.update(diversity_lpips=.01, model_lr_mse_raw=0.)
            result = dict(variant=cfg['variant'], seed=cfg['seed'], split=split, nfe=1,
                          metrics=metrics, step=8000, successful_g=8000, training_seconds=7100.,
                          evaluation_seconds=10., inference_ms_per_image=9., images=50, crops=250,
                          data_hash='shared')
            folder = Path(item['output'])
            folder.mkdir(parents=True)
            (folder/f'{split}_metrics.json').write_text(json.dumps(result))
    monkeypatch.setattr(sys, 'argv', ['collect', '--plan', str(path)])
    cluster_collect.main()
    for split, value in [('val', 1.), ('test', 2.)]:
        rows = json.loads((args.run_root/f'{split}_summary.json').read_text())
        assert len(rows) == 6
        assert all(row['completed'] == 3 and row['metrics']['model_psnr']['mean'] == value for row in rows)
