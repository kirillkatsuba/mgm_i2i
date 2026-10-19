"""Collect completed evaluations, show missing jobs, and average independent seeds."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def collect(plan_path, split=None):
    plan = json.loads(plan_path.read_text())
    split = split or plan.get('evaluation_split', 'val')
    results, missing = [], []
    items = plan.get('eval', plan['train'])
    for item in items:
        cfg = item['config']
        metric_path = Path(item['output'])/f'{split}_metrics.json'
        if not metric_path.exists():
            missing.append(f"{item.get('label', cfg['variant'])}/seed{cfg['seed']}: no {split} metrics")
            continue
        result = json.loads(metric_path.read_text())
        if result['variant'] != cfg['variant'] or result['seed'] != cfg['seed'] or result['split'] != split:
            raise ValueError(f'Metadata mismatch: {metric_path}')
        if result.get('nfe', 1) != item.get('nfe', 1):
            raise ValueError(f'NFE mismatch: {metric_path}')
        if item.get('evaluation_data_hash') is not None and result.get('evaluation_data_hash') != item['evaluation_data_hash']:
            raise ValueError(f'External dataset mismatch: {metric_path}')
        results.append(dict(variant=item.get('label', cfg['variant']), seed=cfg['seed'], **result['metrics'],
                            steps=result['step'], successful_g=result['successful_g'],
                            consumed=result.get('consumed'), training_stop=result.get('training_stop', 'time'),
                            target_epochs=result.get('target_epochs'), epochs_completed=result.get('epochs_completed'),
                            training_seconds=result['training_seconds'],
                            warmup_training_seconds=result.get('warmup_training_seconds', 0.),
                            field_warmup_seconds=result.get('field_warmup_seconds', 0.),
                            evaluation_seconds=result['evaluation_seconds'],
                            inference_ms_per_image=result['inference_ms_per_image'],
                            gpu=result.get('environment', {}).get('gpu', 'unknown'),
                            eval_batch_size=result.get('eval_batch_size'),
                            images=result['images'], crops=result['crops'], data_hash=result['data_hash'],
                            evaluation_data_hash=result.get('evaluation_data_hash', result['data_hash'])))
    if len({r['evaluation_data_hash'] for r in results}) > 1:
        raise ValueError('Cannot compare runs evaluated on different dataset manifests')
    destination = plan_path.parent
    if results:
        with (destination/f'{split}_runs.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    names = list(dict.fromkeys(item.get('label', item['config']['variant']) for item in items))
    columns = ['model_psnr', 'model_ssim', 'model_lpips', 'diversity_lpips', 'model_lr_mse_raw', 'steps']
    if results and all('model_projected_lpips' in row for row in results):
        columns += ['model_projected_psnr', 'model_projected_lpips', 'inference_ms_per_image']
    coverage_note = ('One official externally pretrained checkpoint; no independent training-seed SD. '
                     'Training steps and GPU cost are unknown, not zero.' if plan.get('external_pretrained') else
                     'Mean ± sample SD across seeds; n=1 has no SD.')
    report = [f'# {plan.get("evaluation_dataset", "DIV2K")} x4 — {split}', '',
              f'Completed evaluations: {len(results)}/{len(items)}. '+coverage_note,
              'Crops are averaged within each source image before calculating run means.', '',
              '| Variant | Seeds | '+ ' | '.join(columns)+' |',
              '|---|---|'+ '|'.join(['---']*len(columns))+'|']
    summary = []
    for name in names:
        members = [r for r in results if r['variant'] == name]
        expected = sum(item.get('label', item['config']['variant']) == name for item in items)
        values = []
        row = dict(variant=name, completed=len(members), expected=expected, metrics={})
        for key in columns:
            data = [r[key] for r in members if r[key] is not None]
            mean = statistics.mean(data) if data else None
            sd = statistics.stdev(data) if len(data) > 1 else None
            row['metrics'][key] = dict(mean=mean, std=sd)
            values.append('—' if mean is None else f'{mean:.5g}' + (f' ± {sd:.3g}' if sd is not None else ''))
        report.append(f'| {name} | {len(members)}/{expected} | '+' | '.join(values)+' |')
        summary.append(row)
    if results:
        first = results[0]
        report += ['', 'Fixed baselines (identical data and no training):', '',
                   '| Baseline | PSNR | SSIM | LPIPS |', '|---|---|---|---|']
        for baseline in ('bicubic', 'projected_bicubic'):
            report.append('| '+baseline+' | '+' | '.join(f'{first[baseline+"_"+key]:.5g}' for key in ('psnr','ssim','lpips'))+' |')
    report += ['', '## Coverage', ''] + (['- '+s for s in missing] or ['All planned evaluations are present.'])
    if len({r['gpu'] for r in results}) > 1 or len({r['eval_batch_size'] for r in results}) > 1:
        report += ['', 'Inference timings use different GPU models or batch sizes and must not be directly ranked.']
    if len({r['data_hash'] for r in results}) > 1:
        report += ['', 'Training datasets/protocols differ; common evaluation does not isolate architecture or objective effects.']
    training_note = ('Official I2SB uses ImageNet256/ADM pretraining and the original 552.81M-parameter network. '
                     'Evaluation is on the shared DIV2K crop128 bank without resize; this is not a full ImageNet256 reproduction.'
                     if plan.get('external_pretrained') else
                     'Training, warmup and evaluation times are in the per-run CSV. MSE warmup is shared by U-Net methods; ESRGAN has its own RRDB L1 warmup. I2SB trains from scratch.')
    report += ['', training_note,
               'Training stop policy, sampled patches and epoch counts are in the per-run CSV. Equal sampled patches and equal GPU time are different comparisons. Compare convergence curves as well as final scores.',
               'Larger diversity is not automatically better; inspect quality and LR consistency together.',
               'This is synthetic average-pooling SR, not the standard full-frame bicubic benchmark.', '']
    (destination/f'{split}_summary.md').write_text('\n'.join(report))
    (destination/f'{split}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('\n'.join(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--split', choices=['val','test','external'], default=None)
    args = p.parse_args()
    collect(args.plan, args.split)
