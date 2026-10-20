"""Offline GPU probe and shared evaluation for pretrained SR references."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def probe(item, output, split, eval_minutes):
    import torch
    from torch.nn import functional as F
    from mgm_i2i.cluster.engine import atomic_json, provenance
    from mgm_i2i.cluster.metrics import evaluation_bank, predict_sr
    from mgm_i2i.cluster.sr_reference import load_for_evaluation
    if not torch.cuda.is_available():
        raise RuntimeError('Probe must run on a CUDA GPU node')
    cfg = item['config']
    torch.set_num_threads(cfg['cpus'])
    torch.cuda.reset_peak_memory_stats()
    _, model, _, metadata = load_for_evaluation(cfg, item['checkpoint'])
    bank, _, data_hash, _, _ = evaluation_bank(cfg, None, split,
                                              item['evaluation_data_root'], item['evaluation_data_hash'])
    timings = []
    for index in range(5):
        # Warm two forwards, then measure three real crops, including complete preprocessing.
        real = bank[index % len(bank)].unsqueeze(0).cuda().float()/127.5-1
        c = F.avg_pool2d(real, 4)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = predict_sr(model, c, None, cfg, offset=index)
        torch.cuda.synchronize()
        if result.shape != real.shape or not torch.isfinite(result).all():
            raise ValueError('Invalid reference output')
        if index >= 2:
            timings.append(1000*(time.perf_counter()-start))
    estimate = max(timings)*2*len(bank)/60000
    # Loader provenance also contains bundle IDs. Measured/planned fields take
    # precedence; passing both as dict keyword arguments raises TypeError.
    report = dict(metadata)
    report.update(complete=estimate*2+10 <= eval_minutes, reference_bundle_sha256=cfg['reference_bundle_sha256'],
                  evaluation_data_hash=data_hash, evaluation_crop=128, lr_crop=32,
                  batch_size=1, measured_ms=timings, estimated_two_draw_eval_minutes_without_metrics=estimate,
                  eval_allocation_limit_minutes=eval_minutes,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), environment=provenance())
    if not report['complete']:
        report['error'] = 'Probe suggests allocation is too short; use a fresh run root and larger --eval-minutes'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'probe.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['complete']:
        raise RuntimeError(report['error'])


def main():
    from mgm_i2i.cluster.sr_reference import digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--phase', choices=['probe', 'eval'], required=True)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    if digest(args.plan) != args.plan.with_suffix('.sha256').read_text().strip():
        raise ValueError('Evaluation plan changed after creation')
    plan = json.loads(args.plan.read_text())
    if plan.get('run_kind') != 'sr-reference-evaluation' or not 0 <= args.index < len(plan[args.phase]):
        raise ValueError('Wrong plan or array index')
    item = plan[args.phase][args.index]
    split = plan['evaluation_split']
    if args.phase == 'probe':
        probe(item, Path(item['output']), split, plan['budget']['eval_limit_minutes'])
        return
    report = json.loads((Path(plan['probe'][args.index]['output'])/'probe.json').read_text())
    if (not report['complete'] or report['reference_bundle_sha256'] != item['config']['reference_bundle_sha256'] or
            report['evaluation_data_hash'] != item['evaluation_data_hash']):
        raise ValueError('Matching successful probe required')
    from mgm_i2i.cluster.metrics import evaluate
    evaluate(item['checkpoint'], item['output'], split, 1, 1,
             evaluation_data_root=item['evaluation_data_root'], evaluation_data_hash=item['evaluation_data_hash'],
             expected_config=item['config'])


if __name__ == '__main__':
    main()
