"""Infer full DIV2K photos with completed EMA checkpoints; never train or overwrite crop metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--index', type=int, help='Train-array index from the existing immutable plan')
    parser.add_argument('--output-root', type=Path, help='Default: next to plan, full_<split>')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--limit', type=int, default=8, help='First N full photos in split; 0 means all (default: 8)')
    parser.add_argument('--image-ids', nargs='+', help='Explicit DIV2K IDs in this split, overrides --limit')
    parser.add_argument('--archive', type=Path, help='Original DIV2K_valid_HR.zip, if moved since preparation')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--list', action='store_true', dest='list_jobs', help='Print plan indices without loading models')
    parser.add_argument('--gallery-only', action='store_true', help='CPU-only comparison of completed full-frame runs')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    raw = args.plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.plan.with_suffix('.sha256').read_text().strip():
        parser.error('Plan checksum mismatch; use the unchanged training plan')
    plan = json.loads(raw)
    if args.list_jobs:
        for index, item in enumerate(plan['train']):
            cfg = item['config']
            print(f'{index}: {cfg["variant"]}/seed{cfg["seed"]} — {Path(item["output"])/"latest.pt"}')
        return
    output_root = args.output_root.resolve() if args.output_root else args.plan.resolve().parent/f'full_{args.split}'
    # Keep weights/metrics/training status outside the output tree even with custom paths.
    for item in plan['warmup'] + plan['train']:
        training_output = Path(item['output']).resolve()
        if output_root == training_output or training_output in output_root.parents or output_root in training_output.parents:
            parser.error('--output-root must be separate from all training/warmup output directories')
    from mgm_i2i.cluster.full_images import comparison_gallery, infer_full
    if args.gallery_only:
        comparison_gallery(plan, output_root)
        return
    if args.index is None or not 0 <= args.index < len(plan['train']):
        parser.error('--index must identify one train entry in the plan')
    infer_full(plan['train'][args.index], output_root, args.split, args.limit, args.image_ids,
               args.archive, args.precision)


if __name__ == '__main__':
    main()
