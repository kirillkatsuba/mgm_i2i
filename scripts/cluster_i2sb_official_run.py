"""Execute an immutable official-I2SB probe or validation entry; never download."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def probe(item, output, split, max_nfe=64, eval_minutes=180):
    import torch
    from torch.nn import functional as F
    from mgm_i2i.cluster.engine import atomic_json, provenance, seed_all
    from mgm_i2i.cluster.metrics import evaluation_bank, predict_sr
    from mgm_i2i.cluster.official_i2sb import load_for_evaluation
    cfg=item['config']
    if not torch.cuda.is_available():
        raise RuntimeError('The official GPU probe must run on a CUDA node')
    seed_all(cfg['seed']);torch.set_num_threads(cfg['cpus'])
    torch.cuda.reset_peak_memory_stats()
    _,gen,bridge,metadata=load_for_evaluation(cfg,item['checkpoint'])
    bank,_,data_hash,_,crop=evaluation_bank(cfg,None,split,item['evaluation_data_root'],item['evaluation_data_hash'])
    real=bank[0].unsqueeze(0).cuda().float()/127.5-1
    c=F.avg_pool2d(real,4)
    # Warmup and two measured samplers on an actual fixed validation crop.
    predict_sr(gen,c,None,cfg,bridge=bridge,nfe=1)
    timings={}
    for nfe in (1,4):
        torch.cuda.synchronize();start=time.perf_counter()
        result=predict_sr(gen,c,None,cfg,bridge=bridge,nfe=nfe)
        torch.cuda.synchronize();timings[str(nfe)]=1000*(time.perf_counter()-start)
        if result.shape!=real.shape or not torch.isfinite(result).all():
            raise RuntimeError('Official inference produced an invalid output')
    peak=torch.cuda.max_memory_allocated()
    report=dict(complete=True,official_bundle_sha256=cfg['official_bundle_sha256'],
                evaluation_data_hash=data_hash,evaluation_crop=crop,batch_size=1,precision='fp32',
                inference_ms=timings,peak_allocated_bytes=peak,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),environment=provenance(),**metadata)
    # Extrapolation only; it excludes metrics, disk, model loading and warmup.
    report['estimated_nfe64_two_draws_250_crops_minutes']=timings['4']/4*64*2*250/60000
    estimate=timings['4']/4*max_nfe*2*len(bank)/60000
    report['estimated_longest_eval_minutes_without_metrics']=estimate
    report['eval_allocation_limit_minutes']=eval_minutes
    if estimate*1.5+5 > eval_minutes:
        report['complete']=False
        report['error']='Measured probe suggests the eval allocation is too short; increase --eval-minutes in a fresh plan'
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'probe.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not report['complete']:
        raise RuntimeError(report['error'])


def main():
    from mgm_i2i.cluster.official_i2sb import digest
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--phase',choices=['probe','eval'],required=True)
    p.add_argument('--index',type=int,required=True)
    args=p.parse_args()
    if digest(args.plan)!=args.plan.with_suffix('.sha256').read_text().strip():
        raise ValueError('Official evaluation plan was edited after creation')
    plan=json.loads(args.plan.read_text())
    if plan.get('run_kind')!='i2sb-official-evaluation':
        raise ValueError('Not an official I2SB plan')
    if not 0<=args.index<len(plan[args.phase]):
        p.error('Array index outside plan')
    item=plan[args.phase][args.index]
    split=plan['evaluation_split']
    if args.phase=='probe':
        probe(item,Path(item['output']),split,max(x['nfe'] for x in plan['eval']),plan['budget']['eval_limit_minutes'])
        return
    report=json.loads((Path(plan['probe'][0]['output'])/'probe.json').read_text())
    if (not report['complete'] or report['official_bundle_sha256']!=item['config']['official_bundle_sha256'] or
            report['evaluation_data_hash']!=item['evaluation_data_hash']):
        raise ValueError('A matching successful GPU probe is required')
    from mgm_i2i.cluster.metrics import evaluate
    print(f'Official I2SB: {split}, NFE={item["nfe"]}, FP32 batch1, crop128',flush=True)
    evaluate(item['checkpoint'],item['output'],split,1,item['nfe'],
             evaluation_data_root=item['evaluation_data_root'],evaluation_data_hash=item['evaluation_data_hash'],
             expected_config=item['config'])


if __name__=='__main__':
    main()
