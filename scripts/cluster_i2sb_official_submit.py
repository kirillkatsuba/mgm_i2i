"""Plan official pretrained I2SB: one CUDA probe, four NFE evaluations, collect."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from cluster_submit import execute_plan
from mgm_i2i.cluster.data import load_multicrop_evaluation
from mgm_i2i.cluster.official_i2sb import digest, load_bundle, verify_upstream


def make_plan(args):
    data, data_hash = load_multicrop_evaluation(args.data_root, args.split)
    if data['preparation']['eval_crops_per_image'] != 5:
        raise ValueError('This run requires the same five fixed evaluation crops per photograph as MGM')
    manifest, checkpoint = load_bundle(args.bundle)
    verify_upstream(args.upstream)
    if not digest(args.weights_root/'hub/checkpoints/alexnet-owt-7be5be79.pth').startswith('7be5be79'):
        raise ValueError('Local LPIPS AlexNet weights are missing/corrupt')
    config = dict(method='i2sb_official', variant='i2sb_official_pretrained', seed=0,
                  architecture_version='official_ADM256', width=256, scale=4,
                  crop=128, eval_crop=128, data_protocol='random128-multicrop-v4',
                  eval_crops_per_image=5, data_root=str(args.data_root.resolve()),
                  weights_root=str(args.weights_root.resolve()), cpus=args.cpus,
                  workers=max(0,args.cpus-2), eval_batch_size=1, projection=False,
                  official_bundle=str(args.bundle.resolve()), official_bundle_sha256=digest(args.bundle),
                  official_upstream=str(args.upstream.resolve()))
    root = args.run_root.resolve()
    def entry(nfe):
        return dict(config=dict(config), checkpoint=str(checkpoint.resolve()),
                    output=str(root/f'eval_nfe{nfe}'), nfe=nfe,
                    label=f'i2sb_official_nfe{nfe}', eval_batch_size=1,
                    evaluation_data_root=str(args.data_root.resolve()), evaluation_data_hash=data_hash)
    probe = entry(4)
    probe['output'] = str(root/'probe')
    return dict(version=1, run_kind='i2sb-official-evaluation', external_pretrained=True,
                separate_evaluation=True, project_dir=str(ROOT), evaluation_split=args.split,
                evaluation_dataset='DIV2K', phase_order=['probe','eval'],
                worker_script='slurm/i2sb_official.sbatch', warmup=[], train=[],
                probe=[probe], eval=[entry(nfe) for nfe in args.nfe],
                protocol='Original 552.81M ADM / official sr4x-pool EMA; common crop128 without resize; '
                         'native and final-projected outputs, FP32 batch1, fixed draws; no training',
                scheduler=dict(partition=args.partition, max_parallel=args.max_parallel,
                               account=args.account, constraint=args.constraint, gpu_type=args.gpu_type,
                               job_name_prefix='i2sb_ref'),
                budget=dict(training_gpu_hours=0, external_training_gpu_hours=None,
                            probe_limit_minutes=args.probe_minutes, eval_limit_minutes=args.eval_minutes,
                            allocation_gpu_hours_upper_bound=(args.probe_minutes+len(args.nfe)*args.eval_minutes)/60,
                            note='Evaluation only. One pretrained checkpoint, not three training seeds. '
                                 'Probe records actual GPU memory/time; eval limits are allocation caps.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1'))
    p.add_argument('--data-root',type=Path,default=Path('data/div2k_multicrop128'))
    p.add_argument('--weights-root',type=Path,default=Path('weights'))
    p.add_argument('--bundle',type=Path,default=Path('weights/i2sb_official_sr4x_pool/manifest.json'))
    p.add_argument('--upstream',type=Path,default=Path('third_party/I2SB-official'))
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--nfe',type=int,nargs='+',default=[1,4,16,64])
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--max-parallel',type=int,default=2)
    p.add_argument('--probe-minutes',type=int,default=20)
    p.add_argument('--eval-minutes',type=int,default=180)
    p.add_argument('--partition',default='rocky')
    p.add_argument('--account',default='')
    p.add_argument('--constraint',default='')
    p.add_argument('--gpu-type',default='')
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    if min(args.cpus,args.max_parallel,args.probe_minutes,args.eval_minutes)<1:
        p.error('CPUs, parallelism and time limits must be positive')
    if len(set(args.nfe))!=len(args.nfe) or any(not 0<n<1000 for n in args.nfe):
        p.error('NFE must be unique and in [1,999]')
    if args.split=='test' and args.run_root.name=='i2sb_official_sr4x_pool_multi5_val_r1':
        p.error('Use a dedicated --run-root for the explicit final test')
    plan=make_plan(args)
    execute_plan(plan,args.run_root,args.submit)


if __name__=='__main__':
    main()
