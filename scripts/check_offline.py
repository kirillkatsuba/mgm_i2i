"""Verify local files, dependencies, models, and LPIPS with network disabled."""
from pathlib import Path
import importlib.metadata
import json
import os
import socket
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['TORCH_HOME']=str(ROOT/'offline/torch-cache')
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib')

def denied(*args,**kwargs):
    raise RuntimeError('Network disabled by offline check')

socket.create_connection=denied
socket.socket.connect=denied
socket.socket.connect_ex=denied
import torch
torch.set_num_threads(1)
from mgm_i2i.models import ImageGenerator,ImageField
from mgm_i2i.data import CIFARData,latent
sys.path.insert(0,str(ROOT/'scripts'))
from train import bridge_objective, DEFAULTS
import lpips

data=CIFARData('cpu',root=ROOT/'data/cifar10',split='val',limit=4)
c,y=data.batch(2)
g=ImageGenerator(width=8,blocks=1,project=True)
f=ImageField(width=8,blocks=1)
pred=g(c,latent(y))
assert f(c,torch.full((2,1,1,1),.25),pred).shape==y.shape
assert torch.allclose(torch.nn.functional.avg_pool2d(pred,2),c,atol=1e-6)
for p in f.parameters(): p.requires_grad_(False)
loss=bridge_objective(g,f,c,y,DEFAULTS,'generator')
loss.backward()
assert any(p.grad is not None and p.grad.abs().max()>0 for p in g.parameters())
assert all(p.grad is None for p in f.parameters())
metric=lpips.LPIPS(net='alex',verbose=False).eval()
with torch.no_grad():
    yy=torch.nn.functional.interpolate(y,size=(64,64),mode='nearest')
    distance=metric(yy,yy)
assert distance.abs().max()<1e-5
missing=[str(p) for p in [ROOT/'materials/midpoint-generative-models-v2.pdf',ROOT/'offline/guide.html'] if not p.exists()]
if missing: raise SystemExit(f'Missing: {missing}')
report={'status':'passed','network':'blocked by socket hooks',
        'torch':torch.__version__,'python':sys.version.split()[0],
        'platform':sys.platform,'mps_available':torch.backends.mps.is_available(),
        'checks':['CIFAR load','model forward','MGM generator backward through frozen field','data consistency','LPIPS with cached weights','offline guide'],
        'dependencies':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow','matplotlib','lpips','pytest']}}
(ROOT/'offline/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
