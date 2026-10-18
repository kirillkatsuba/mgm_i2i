"""Download 100 pinned BSD100 HR PNGs and prepare the shared external evaluation bank."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.external_data import prepare, load_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/bsd100_center128'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    prepare(args.data_root, args.workers)
    _, checksum = load_bank(args.data_root)
    print(f'BSD100: 100 center128 images; manifest SHA256 {checksum}; root {args.data_root.resolve()}')


if __name__ == '__main__':
    main()
