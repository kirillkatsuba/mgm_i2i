"""Collect completed evaluations, show missing jobs, and average independent seeds."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def collect(plan_path, split='val'):
    plan = json.loads(plan_path.read_text())
    results, missing = [], []
    for item in plan['train']:
        cfg = item['config']
        metric_path = Path(item['output'])/f'{split}_metrics.json'
        if not metric_path.exists():
            missing.append(f"{cfg['variant']}/seed{cfg['seed']}: no {split} metrics")
            continue
        result = json.loads(metric_path.read_text())
        if result['variant'] != cfg['variant'] or result['seed'] != cfg['seed'] or result['split'] != split:
            raise ValueError(f'Metadata mismatch: {metric_path}')
        results.append(dict(variant=cfg['variant'], seed=cfg['seed'], **result['metrics'],
                            steps=result['step'], successful_g=result['successful_g'],
                            training_seconds=result['training_seconds'],
                            evaluation_seconds=result['evaluation_seconds'],
                            inference_ms_per_image=result['inference_ms_per_image'],
                            images=result['images'], crops=result['crops'], data_hash=result['data_hash']))
    if len({r['data_hash'] for r in results}) > 1:
        raise ValueError('Cannot compare runs from different dataset manifests')
    destination = plan_path.parent
    if results:
        with (destination/f'{split}_runs.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    names = list(dict.fromkeys(item['config']['variant'] for item in plan['train']))
    columns = ['model_psnr', 'model_ssim', 'model_lpips', 'diversity_lpips', 'model_lr_mse_raw', 'steps']
    report = [f'# DIV2K x4 — {split}', '',
              f'Completed evaluations: {len(results)}/{len(plan["train"])}. Mean ± sample SD across seeds; n=1 has no SD.',
              'Crops are averaged within each source image before calculating run means.', '',
              '| Variant | Seeds | '+ ' | '.join(columns)+' |',
              '|---|---|'+ '|'.join(['---']*len(columns))+'|']
    summary = []
    for name in names:
        members = [r for r in results if r['variant'] == name]
        expected = sum(item['config']['variant'] == name for item in plan['train'])
        values = []
        row = dict(variant=name, completed=len(members), expected=expected, metrics={})
        for key in columns:
            data = [r[key] for r in members]
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
    report += ['', 'Training and evaluation times are in the per-run CSV. Warmup is shared across methods.',
               'Wall-time budgets yield different update counts. Compare convergence curves as well as final scores.',
               'Larger diversity is not automatically better; inspect quality and LR consistency together.',
               'This is synthetic average-pooling SR, not the official DIV2K bicubic benchmark.', '']
    (destination/f'{split}_summary.md').write_text('\n'.join(report))
    (destination/f'{split}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('\n'.join(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--split', choices=['val','test'], default='val')
    args = p.parse_args()
    collect(args.plan, args.split)
