"""Offline training of paired MSE, conditional GAN, and conditional MGM."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ['TORCH_HOME'] = str(ROOT/'offline/torch-cache')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))
import numpy as np
import torch
from torch.nn import functional as F
from mgm_i2i.data import ToyData, CIFARData, latent
from mgm_i2i.models import ToyGenerator, ToyField, ToyDiscriminator, ImageGenerator, ImageField, ImageDiscriminator
from mgm_i2i.losses import observation, random_bridge, variational_loss, field_loss
from mgm_i2i.evaluate import evaluate_toy, evaluate_images

DEFAULTS = dict(dataset='toy', method='mgm', steps=3000, warmup_steps=200, batch_size=128,
                width=64, blocks=3, lr_g=.001, lr_f=.001, critic_steps=2,
                noise_strength=.01, critic_only_noise=False, midpoint=False, flip=True,
                conditioned=True, project=True, antithetic=False, supervised_weight=0.,
                seed=0, data_seed=123, device='auto', threads=2, ema=.99,
                train_limit=2048, eval_limit=256, data_root=str(ROOT/'data/cifar10'),
                log_every=100, save_every=500, output=str(ROOT/'runs/toy_mgm'),
                init=None, resume=None, lpips=False, eval_split='val')


def config():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    for key, value in DEFAULTS.items():
        name = '--'+key.replace('_','-')
        if isinstance(value, bool):
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=None)
        else:
            kind = type(value) if value is not None else str
            p.add_argument(name, type=kind, default=None)
    args = vars(p.parse_args())
    cfg = DEFAULTS.copy()
    if args['config']:
        updates = json.loads(Path(args['config']).read_text())
        unknown = set(updates)-set(cfg)
        if unknown:
            p.error(f'Unknown config keys: {unknown}')
        cfg.update(updates)
    cfg.update({k:v for k,v in args.items() if k != 'config' and v is not None})
    if cfg['method'] not in ['mse','mgm','gan'] or cfg['dataset'] not in ['toy','cifar10']:
        p.error('method must be mse/mgm/gan; dataset must be toy/cifar10')
    if cfg['steps'] < 1 or cfg['critic_steps'] < 1 or cfg['batch_size'] < 1:
        p.error('steps, critic-steps and batch-size must be positive')
    if cfg['noise_strength'] < 0 or not 0 <= cfg['ema'] < 1 or cfg['supervised_weight'] < 0:
        p.error('invalid noise, EMA, or supervised weight')
    if cfg['antithetic'] and not cfg['flip']:
        p.error('antithetic averaging requires flip')
    if cfg['resume']:
        cfg['init'] = None  # A resumed checkpoint already contains initialization.
    return cfg


def device_for(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def models(cfg, device):
    toy = cfg['dataset']=='toy'
    g = (ToyGenerator if toy else ImageGenerator)(width=cfg['width'], blocks=cfg['blocks'], project=cfg['project']).to(device)
    f = (ToyField if toy else ImageField)(width=cfg['width'], blocks=cfg['blocks'], conditioned=cfg['conditioned']).to(device)
    d = (ToyDiscriminator if toy else ImageDiscriminator)(width=cfg['width']).to(device)
    return g,f,d


def bridge_objective(g, f, c, y, cfg, player):
    fake = g(c, latent(y))
    if player == 'field':
        fake = fake.detach()
    t,b,noise = random_bridge(fake, cfg['midpoint'], cfg['flip'])
    strength = 0. if player == 'generator' and cfg['critic_only_noise'] else cfg['noise_strength']
    delta = y-fake
    def term(branch):
        x = observation(fake, y, t, branch, noise, strength)
        pred = f(c,t,x)
        return field_loss(pred,delta) if player=='field' else variational_loss(pred,delta)
    if cfg['antithetic']:
        return .5*(term(torch.zeros_like(b))+term(torch.ones_like(b)))
    return term(b)


def update(opt, loss, params):
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss; latest checkpoint is saved periodically. Lower learning rates.')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
    opt.step()


def main():
    cfg = config()
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = device_for(cfg['device'])
    output = Path(cfg['output'])
    if (output/'config.json').exists() and not cfg['resume']:
        raise SystemExit(f'{output} already contains a run; choose another --output or use --resume')
    output.mkdir(parents=True, exist_ok=True)
    start_time=time.perf_counter()
    data = ToyData(device,seed=cfg['seed']) if cfg['dataset']=='toy' else CIFARData(device,root=cfg['data_root'],split='train',limit=cfg['train_limit'],seed=cfg['data_seed'])
    g,f,d=models(cfg,device)
    og=torch.optim.Adam(g.parameters(),lr=cfg['lr_g'],betas=(0.,.999))
    auxiliary=d if cfg['method']=='gan' else f
    of=torch.optim.Adam(auxiliary.parameters(),lr=cfg['lr_f'],betas=(0.,.999))
    start_step=0
    elapsed_before=0.
    if cfg['init']:
        initial=torch.load(cfg['init'],map_location=device,weights_only=False)
        g.load_state_dict(initial['generator'])
        print('Initialized generator from',cfg['init'],flush=True)
    elif cfg['resume']:
        state=torch.load(cfg['resume'],map_location=device,weights_only=False)
        mutable={'steps','device','threads','output','resume','init','lpips','eval_split','eval_limit','log_every','save_every'}
        changed={k for k in cfg if k not in mutable and cfg[k] != state['config'].get(k)}
        if changed:
            raise SystemExit(f'Resume configuration differs: {changed}. Use the original --config and overrides only for runtime/total steps.')
        g.load_state_dict(state['generator']); auxiliary.load_state_dict(state['auxiliary'])
        og.load_state_dict(state['optimizer_g']); of.load_state_dict(state['optimizer_f'])
        start_step=state['step']; elapsed_before=state['elapsed_seconds']
        torch.set_rng_state(state['torch_rng'].cpu()); data.rng.set_state(state['data_rng'].cpu())
        if device.type=='cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        if device.type=='mps' and state.get('mps_rng') is not None:
            torch.mps.set_rng_state(state['mps_rng'].cpu())
    elif cfg['method']!='mse' and cfg['warmup_steps']:
        for step in range(cfg['warmup_steps']):
            c,y=data.batch(cfg['batch_size'])
            loss=(g(c,latent(y,True))-y).square().mean()
            update(og,loss,g.parameters())
        print('Paired MSE warmup finished:',cfg['warmup_steps'],'steps',flush=True)
    ema=copy.deepcopy(g).eval()
    if cfg['resume']:
        ema.load_state_dict(state['ema'])
    for p in ema.parameters():
        p.requires_grad_(False)
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    info={'torch':torch.__version__,'python':sys.version,'device':str(device),
          'generator_parameters':sum(p.numel() for p in g.parameters()),
          'auxiliary_parameters':sum(p.numel() for p in auxiliary.parameters()),
          'degradation':'average-pool x2' if cfg['dataset']=='cifar10' else 'two-pixel average',
          'status':'research prototype; smoke tests do not establish quality'}
    (output/'environment.json').write_text(json.dumps(info,indent=2))
    print(info,flush=True)
    completed=start_step
    def save():
        state={'config':cfg,'step':completed,'generator':g.state_dict(),'ema':ema.state_dict(),
               'auxiliary':auxiliary.state_dict(),'optimizer_g':og.state_dict(),'optimizer_f':of.state_dict(),
               'torch_rng':torch.get_rng_state(),'data_rng':data.rng.get_state(),
               'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None,
               'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None,
               'elapsed_seconds':elapsed_before+time.perf_counter()-start_time}
        temporary=output/'latest.tmp'
        torch.save(state,temporary); temporary.replace(output/'latest.pt')
    log_path=output/'train.csv'
    with log_path.open('a' if cfg['resume'] else 'w',newline='') as logfile:
        writer=csv.DictWriter(logfile,fieldnames=['step','loss_g','loss_f','elapsed_seconds'])
        if not cfg['resume'] or log_path.stat().st_size==0:
            writer.writeheader()
        try:
            for step in range(start_step+1,cfg['steps']+1):
                loss_f=torch.zeros((),device=device)
                if cfg['method']!='mse':
                    for p in auxiliary.parameters(): p.requires_grad_(True)
                    for _ in range(cfg['critic_steps']):
                        c,y=data.batch(cfg['batch_size'])
                        if cfg['method']=='mgm':
                            loss_f=bridge_objective(g,f,c,y,cfg,'field')
                        else:
                            with torch.no_grad(): fake=g(c,latent(y))
                            loss_f=F.softplus(-d(c,y)).mean()+F.softplus(d(c,fake)).mean()
                        update(of,loss_f,auxiliary.parameters())
                    # Parameters fixed; autograd through the input stays enabled.
                    for p in auxiliary.parameters(): p.requires_grad_(False)
                c,y=data.batch(cfg['batch_size'])
                if cfg['method']=='mse':
                    loss_g=(g(c,latent(y,True))-y).square().mean()
                elif cfg['method']=='mgm':
                    loss_g=bridge_objective(g,f,c,y,cfg,'generator')
                else:
                    loss_g=F.softplus(-d(c,g(c,latent(y)))).mean()
                if cfg['method']!='mse' and cfg['supervised_weight']:
                    loss_g=loss_g+cfg['supervised_weight']*(g(c,latent(y))-y).square().mean()
                update(og,loss_g,g.parameters())
                with torch.no_grad():
                    for pe,p in zip(ema.parameters(),g.parameters()):
                        pe.lerp_(p,1-cfg['ema'])
                completed=step
                if step==1 or step%cfg['log_every']==0 or step==cfg['steps']:
                    row=dict(step=step,loss_g=loss_g.item(),loss_f=loss_f.item(),elapsed_seconds=elapsed_before+time.perf_counter()-start_time)
                    writer.writerow(row); logfile.flush(); print(row,flush=True)
                if step%cfg['save_every']==0: save()
        except KeyboardInterrupt:
            save(); print('Interrupted; checkpoint saved. Resume with --resume',output/'latest.pt',flush=True)
            return
        except Exception:
            save()
            raise
    save()
    training_seconds=elapsed_before+time.perf_counter()-start_time
    evaluation_start=time.perf_counter()
    torch.manual_seed(12345)
    metrics=evaluate_toy(ema,device,cfg['method']=='mse',output) if cfg['dataset']=='toy' else evaluate_images(ema,cfg,device,output,cfg['lpips'],cfg['eval_split'])
    metrics['elapsed_training_seconds']=training_seconds
    metrics['elapsed_evaluation_seconds']=time.perf_counter()-evaluation_start
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
