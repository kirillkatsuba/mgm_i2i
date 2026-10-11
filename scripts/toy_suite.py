"""Small multi-seed experiment matrix. Output dirs are never overwritten."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--steps',type=int,default=4000)
p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
p.add_argument('--output',default='runs/toy_suite')
args=p.parse_args()
variants={'mse':['--method','mse','--warmup-steps','0'],
          'mgm':[], 'midpoint':['--midpoint'],
          'unconditioned':['--no-conditioned'],
          'critic_noise':['--critic-only-noise'],
          'antithetic':['--antithetic']}
for seed in args.seeds:
    for name,extra in variants.items():
        output=Path(args.output)/f'{name}_seed{seed}'
        if (output/'metrics.json').exists():
            print('Already complete:',output);continue
        if (output/'config.json').exists():
            print('Incomplete run; resume manually:',output);continue
        subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--config',str(ROOT/'configs/toy_mgm.json'),
                        '--steps',str(args.steps),'--seed',str(seed),'--output',str(output),
                        '--threads','1','--log-every',str(args.steps),*extra],cwd=ROOT,check=True)
