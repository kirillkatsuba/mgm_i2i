"""Execute one immutable plan entry; never called automatically on the Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--phase', choices=['warmup', 'field_warmup', 'train', 'eval', 'test'], required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume-if-exists', action='store_true', help='Continue existing checkpoints, start missing jobs')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--split', choices=['val', 'test', 'external'], default=None)
    args = p.parse_args()
    raw = args.plan.read_bytes()
    expected = args.plan.with_suffix('.sha256').read_text().strip()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Plan was edited after creation. Create a new plan instead.')
    plan = json.loads(raw)
    if args.phase == 'test':
        if args.split is not None and args.split != 'test':
            p.error('The test phase requires --split test or no explicit split')
        args.split = 'test'
    else:
        args.split = args.split or plan.get('evaluation_split', 'val')
    jobs = plan[args.phase]
    if not 0 <= args.index < len(jobs):
        p.error('array index outside plan')
    item = jobs[args.index]
    from mgm_i2i.cluster.engine import train
    from mgm_i2i.cluster.metrics import evaluate
    if args.phase in ('eval', 'test'):
        if args.resume or args.resume_if_exists:
            p.error('--resume applies to training, not evaluation')
        evaluate(Path(item['checkpoint']), item['output'], args.split,
                 item.get('eval_batch_size', item['config']['eval_batch_size']), item.get('nfe', 1),
                 evaluation_data_root=item.get('evaluation_data_root'),
                 evaluation_data_hash=item.get('evaluation_data_hash'), expected_config=item['config'])
        return
    if not args.evaluate_only:
        if args.split == 'test':
            p.error('Held-out test requires --evaluate-only after fixing the protocol')
        resume = args.resume or (args.resume_if_exists and (Path(item['output'])/'latest.pt').exists())
        extra = {'field_init': item['field_init']} if item.get('field_init') else {}
        train(item['config'], item['output'], item.get('init'), resume, **extra)
    if args.evaluate_only or (args.phase == 'train' and not plan.get('separate_evaluation')):
        evaluate(Path(item['output'])/'latest.pt', item['output'], args.split,
                 item['config']['eval_batch_size'])


if __name__ == '__main__':
    main()
