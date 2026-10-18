"""Measure physical MGM batch memory in fresh CUDA subprocesses; never changes trained checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def probe(checkpoint, batch_size, steps, noise_g, beta1=.8):
    import copy
    import torch
    from mgm_i2i.cluster.baselines import build_generator
    from mgm_i2i.cluster.models import Field, get_architecture_version
    from mgm_i2i.cluster.engine import mgm_objective, seed_all
    from mgm_i2i.cluster.mgm_training import initialize_field, mgm_step

    if not torch.cuda.is_available():
        raise RuntimeError('The memory probe requires a CUDA allocation')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not state['complete'] or state['config']['method'] != 'mse':
        raise ValueError('Expected a completed generator MSE warmup')
    cfg = dict(state['config'], method='mgm', batch_size=batch_size, grad_accumulation=1,
               lr_g=1e-5, lr_aux=1e-5, adam_betas=[beta1,.999], noise_g=noise_g,
               reconstruction_weight=0., field_initialization='warmup_features')
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg['cpus'])
    gen = build_generator(cfg).cuda().train()
    gen.load_state_dict(state['generator'])
    field = Field(cfg['width'],cfg['conditioned'],get_architecture_version(cfg)).cuda()
    initialize_field(field,gen)
    ema = copy.deepcopy(gen).eval().requires_grad_(False)
    opt_g = torch.optim.Adam(gen.parameters(),lr=cfg['lr_g'],betas=tuple(cfg['adam_betas']))
    opt_f = torch.optim.Adam(field.parameters(),lr=cfg['lr_aux'],betas=tuple(cfg['adam_betas']))
    scaler = torch.amp.GradScaler('cuda',enabled=cfg['amp'])
    torch.cuda.reset_peak_memory_stats()
    attempts, successful = 0, 0
    while successful < steps and attempts < steps+40:
        # Shapes/dtypes/graphs/optimizers match MGM training. Random targets are
        # only for profiling memory; this is not a quality/throughput benchmark.
        real = torch.rand(batch_size,3,cfg['crop'],cfg['crop'],device='cuda')*2-1
        result = mgm_step(gen,field,real,cfg,opt_g,opt_f,scaler,mgm_objective)
        attempts += 1
        successful += bool(result['generator']['success'] and result['field']['success'])
        if result['generator']['success']:
            with torch.no_grad():
                for target,current in zip(ema.parameters(),gen.parameters()):
                    target.lerp_(current,1-cfg['ema_decay'])
        torch.cuda.synchronize()
    if successful < steps:
        raise RuntimeError('Probe did not populate both optimizers with successful steps; inspect AMP/gradients')
    total = torch.cuda.get_device_properties(0).total_memory
    return dict(status='ok', batch_size=batch_size, successful_steps=successful, attempts=attempts,
                gpu=torch.cuda.get_device_name(0),
                amp=cfg['amp'], architecture_version=get_architecture_version(cfg), width=cfg['width'],
                adam_betas=cfg['adam_betas'], lr_g=cfg['lr_g'], lr_aux=cfg['lr_aux'], grad_accumulation=1,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, total_gib=total/2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[16,32,48,64])
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--noise-g',type=float,default=0.)
    parser.add_argument('--mgm-beta1',type=float,default=.8)
    parser.add_argument('--worker-batch',type=int)
    args = parser.parse_args()
    if args.steps < 2 or any(n < 1 for n in args.batch_sizes) or args.noise_g < 0:
        parser.error('At least two steps, positive batches and nonnegative noise required')
    if not 0 <= args.mgm_beta1 < 1:
        parser.error('MGM Adam beta1 must be in [0, 1)')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.worker_batch is not None:
        import torch
        try:
            result = probe(args.checkpoint,args.worker_batch,args.steps,args.noise_g,args.mgm_beta1)
        except torch.cuda.OutOfMemoryError:
            result = dict(status='oom',batch_size=args.worker_batch)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        return
    results = []
    for batch in sorted(set(args.batch_sizes)):
        path = args.output.with_name(args.output.stem+f'_batch{batch}.json')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(args.checkpoint),
                        '--output',str(path),'--worker-batch',str(batch),'--steps',str(args.steps),
                        '--noise-g',str(args.noise_g),'--mgm-beta1',str(args.mgm_beta1)],check=True)
        result = json.loads(path.read_text())
        results.append(result)
        print(json.dumps(result),flush=True)
        if result['status']=='oom':
            break
    candidates = [r['batch_size'] for r in results if r['status']=='ok' and
                  r['peak_reserved_gib'] <= .85*r['total_gib']]
    report = dict(results=results,recommended_physical_batch=max(candidates) if candidates else None,
                  note='Recommendation keeps >=15% reserved-memory headroom; measured synthetic training steps, '
                       'not a guarantee for every future configuration. No gradient accumulation is used.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
