"""Evaluate a saved local checkpoint, including an untouched test split."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['TORCH_HOME']=str(ROOT/'offline/torch-cache')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import torch
from train import device_for,models
from mgm_i2i.evaluate import evaluate_toy,evaluate_images

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint')
p.add_argument('--split',choices=['val','test'],default='val')
p.add_argument('--limit',type=int,default=256)
p.add_argument('--device',default='auto')
p.add_argument('--lpips',action='store_true')
p.add_argument('--output')
args=p.parse_args()
torch.set_num_threads(2)
torch.manual_seed(12345)
device=device_for(args.device)
checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
cfg=checkpoint['config'].copy()
cfg.update(eval_limit=args.limit,data_root=str(ROOT/'data/cifar10'))
g,_,_=models(cfg,device)
g.load_state_dict(checkpoint['ema'])
output=Path(args.output) if args.output else Path(args.checkpoint).parent/('evaluation_'+args.split)
output.mkdir(parents=True,exist_ok=True)
metrics=evaluate_toy(g,device,cfg['method']=='mse',output) if cfg['dataset']=='toy' else evaluate_images(g,cfg,device,output,args.lpips,args.split)
(output/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
