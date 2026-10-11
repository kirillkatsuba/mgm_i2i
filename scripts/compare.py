"""Build a local results table and training plots; no network calls."""
import csv
import json
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

paths=[Path(p) for p in sys.argv[1:]]
if not paths:
    raise SystemExit('Usage: .venv/bin/python scripts/compare.py runs/toy_mse runs/toy_mgm ...')
rows=[]
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for path in paths:
    metrics=json.loads((path/'metrics.json').read_text())
    cfg=json.loads((path/'config.json').read_text())
    row={'run':str(path),'dataset':cfg['dataset'],'method':cfg['method'],'seed':cfg['seed'],**metrics}
    rows.append(row)
    with (path/'train.csv').open() as f:
        log=list(csv.DictReader(f))
    axes[0].plot([int(a['step']) for a in log],[float(a['loss_g']) for a in log],label=path.name)
    if cfg['dataset']=='toy':
        axes[1].bar(path.name,metrics['conditional_w1'])
    else:
        axes[1].bar(path.name,metrics['psnr'])
keys=list(dict.fromkeys(k for r in rows for k in r))
output=ROOT/'runs/comparison';output.mkdir(parents=True,exist_ok=True)
with (output/'results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
axes[0].set(title='Training objectives (not comparable across methods)',xlabel='step')
axes[0].legend(fontsize=8)
axes[1].set_title('Conditional W1 (toy) or PSNR (images); do not mix datasets')
axes[1].tick_params(axis='x',rotation=25)
fig.savefig(output/'comparison.png',dpi=150)
print(output/'results.csv')
