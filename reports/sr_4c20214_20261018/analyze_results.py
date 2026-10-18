"""Analyze archived JSON/logs/PNG only. Does not import or execute model code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

P = argparse.ArgumentParser(description=__doc__)
P.add_argument('--root', type=Path, default=Path('tmp/research/sr_4c20214_results/cluster_runs/sr_4c20214'))
P.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = P.parse_args()
ROOT, OUT = args.root.resolve(), args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
load = lambda path: json.loads(path.read_text())
plan = load(ROOT/'plan.json')
issues, warnings = [], []

def check(condition, description):
    if not condition:
        issues.append(description)


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csv_write(name, rows):
    if rows:
        with (OUT/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


check(hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest() == (ROOT/'plan.sha256').read_text().strip(), 'Plan checksum')
manifest_path = ROOT.parents[1]/'data/div2k_random128/manifest.json'
manifest = load(manifest_path)
data_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
sets = {'train': {row['image'] for row in manifest['train_images']}}
for split in ('val', 'test'):
    rows = manifest['banks'][split+'.npy']['crops']
    sets[split] = {row['image'] for row in rows}
    check(len(rows) == len(sets[split]) == 50, split+' has 50 distinct images/crops')
check(len(sets['train']) == 800, '800 train images')
check(not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']), 'Disjoint source splits')
check(manifest['preparation']['protocol'] == 'random128-v3', 'Data preparation protocol')

runs, training_csv, window_csv = [], [], []
METHODS = ['warmup', 'rrdb_warmup', 'mse', 'mse_perceptual', 'esrgan', 'mgm', 'i2sb']
COLORS = dict(zip(METHODS, ['#64748b', '#8d5a35', '#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed']))
LABELS = {'warmup':'MSE warmup', 'rrdb_warmup':'RRDB L1 warmup', 'mse':'MSE',
          'mse_perceptual':'MSE + perceptual', 'esrgan':'ESRGAN', 'mgm':'MGM', 'i2sb':'I2SB'}
for phase in ('warmup', 'train'):
    for item in plan[phase]:
        cfg = item['config']
        path = ROOT/cfg['variant']/f'seed{cfg["seed"]}'
        status, env = load(path/'status.json'), load(path/'environment.json')
        entries = [json.loads(line) for line in (path/'train.jsonl').read_text().splitlines() if line.strip()]
        arrays = {key: np.array([row[key] if row[key] is not None else np.nan for row in entries]) for key in entries[0]}
        label = f'{cfg["variant"]}/seed{cfg["seed"]}'
        check(load(path/'config.json') == cfg, label+' config matches plan')
        check(status['complete'] and not status['received_signals'], label+' completed uninterrupted')
        check(status['data_hash'] == data_hash, label+' data hash')
        check(status['step']//cfg['log_every'] == len(entries), label+' complete log coverage')
        check(np.all(np.diff(arrays['step']) == cfg['log_every']) and np.all(np.diff(arrays['seconds']) > 0), label+' monotone log')
        check(status['successful_g'] + status['skipped_g'] == status['step'], label+' update counters')
        check(status['consumed'] == status['step']*cfg['batch_size'], label+' consumed count')
        check(cfg['seconds'] <= status['elapsed_training_seconds'] < cfg['seconds']+5, label+' training time')
        for key in ('seconds','g','mse'):
            check(np.isfinite(arrays[key]).all(), label+' finite '+key)
        if cfg['method'] in ('mgm','esrgan'):
            check(np.isfinite(arrays['aux']).all(), label+' finite aux')
        check(len(env) == 1, label+' one training session')
        windows = {}
        duration = cfg['seconds']
        for name, lo, hi in [('first10',0,600),('second10',600,1200),
                             ('previous10',duration-1200,duration-600),('last10',duration-600,duration+5),
                             ('last30',duration-1800,duration+5)]:
            mask = (arrays['seconds'] >= lo) & (arrays['seconds'] < hi)
            record = dict(method=cfg['variant'], seed=cfg['seed'], window=name, n=int(mask.sum()))
            for key in ('g','mse','aux'):
                values = arrays[key][mask]
                values = values[np.isfinite(values)]
                record[key] = float(values.mean()) if len(values) else None
                record[key+'_median'] = float(np.median(values)) if len(values) else None
            windows[name] = record
            window_csv.append(record)
        bins=[]
        for lo in range(0, int(duration), 300):
            mask=(arrays['seconds']>=lo)&(arrays['seconds']<lo+300)
            bins.append(dict(minutes=(lo+150)/60, **{k:float(arrays[k][mask].mean()) if mask.any() and np.isfinite(arrays[k][mask]).all() else None for k in ('g','mse','aux')}))
        summary = dict(method=cfg['variant'], seed=cfg['seed'], phase=phase,
                       seconds=status['elapsed_training_seconds'], steps=status['step'],
                       consumed=status['consumed'], image_equivalent_passes=status['consumed']/800,
                       skipped_g=status['skipped_g'], skipped_aux=status['skipped_aux'],
                       skipped_g_pct=100*status['skipped_g']/status['step'],
                       skipped_aux_pct=100*status['skipped_aux']/status['step'],
                       peak_vram_gib=status['peak_allocated_bytes_this_session']/2**30,
                       first_log_seconds=float(arrays['seconds'][0]),
                       g_min=float(arrays['g'].min()), g_max=float(arrays['g'].max()),
                       g_median=float(np.median(arrays['g'])),
                       g_below_minus01=int((arrays['g'] < -.1).sum()),
                       g_below_minus1=int((arrays['g'] < -1).sum()),
                       mse_max=float(arrays['mse'].max()), mse_above01=int((arrays['mse'] > .1).sum()),
                       git_commit=env[0]['git_commit'], git_status=env[0]['git_status'],
                       gpu=env[0]['gpu'])
        training_csv.append(summary)
        runs.append(dict(summary=summary, config=cfg, status=status, env=env, windows=windows, bins=bins, arrays=arrays))

for seed in range(3):
    same = [r['status']['init_hash'] for r in runs if r['config']['seed']==seed and r['config']['variant'] in ('mse','mse_perceptual','mgm')]
    check(len(same)==3 and len(set(same))==1 and same[0] is not None, f'Seed{seed} shared MSE initialization')

# Native sacct JobID selects arrays; skip .batch/.extern/.0 to avoid double counting.
accounting=[]
submission=load(ROOT/'submission.json')
with (ROOT/'slurm_accounting.txt').open() as stream:
    for row in csv.DictReader(stream, delimiter='|'):
        job=row['JobID']
        phase=next((phase for phase,base in submission['jobs'].items() if job==base or job.startswith(base+'_')),None)
        if phase is None or '.' in job:
            continue
        elapsed=row['Elapsed']
        days, clock=(elapsed.split('-') if '-' in elapsed else ('0',elapsed))
        h,m,s=map(int,clock.split(':'))
        seconds=int(days)*86400+h*3600+m*60+s
        check(row['State']=='COMPLETED' and row['ExitCode']=='0:0', 'Slurm '+job)
        accounting.append(dict(phase=phase,job=job,seconds=seconds,state=row['State'],resources=row['AllocTRES']))
check(len(accounting)==46, 'Slurm has all 46 parent jobs')

metrics_rows, evals, arrays_by_method = [], [], {}
reference_baselines=None
reference_grid=None
for item in plan['eval']:
    cfg=item['config']; label=item.get('label',cfg['variant'])
    folder=ROOT/cfg['variant']/f'seed{cfg["seed"]}'/f'eval_nfe{item["nfe"]}'
    result=load(folder/'val_metrics.json'); images=load(folder/'val_images.json'); patches=load(folder/'val_patches.json')
    run=next(r for r in runs if r['config']['variant']==cfg['variant'] and r['config']['seed']==cfg['seed'])
    check(result['data_hash']==data_hash and result['step']==run['status']['step'], label+' eval/checkpoint metadata')
    check(result['nfe']==item['nfe'] and result['variant']==cfg['variant'] and result['seed']==cfg['seed'], label+' NFE/seed')
    check(result['precision']=='fp32' and result['eval_batch_size']==1, label+' FP32 batch1')
    check(len(images)==len(patches)==50 and result['images']==result['crops']==50, label+' eval counts')
    check({r['image'] for r in images}==sets['val'], label+' correct val image IDs')
    check([r['image'] for r in patches]==[r['image'] for r in manifest['banks']['val.npy']['crops']], label+' patch order')
    images=sorted(images,key=lambda r:r['image'])
    for key,value in result['metrics'].items():
        recomputed=float(np.mean([r[key] for r in images]))
        check(np.isfinite(value) and abs(recomputed-value)<1e-6, label+' aggregate '+key)
    baseline={r['image']:{k:v for k,v in r.items() if k.startswith(('bicubic_','projected_bicubic_'))} for r in images}
    if reference_baselines is None:
        reference_baselines=baseline
    else:
        for image in baseline:
            for key in baseline[image]:
                check(abs(reference_baselines[image][key]-baseline[image][key])<1e-6, label+' fixed baseline '+key)
    grid=np.array(Image.open(folder/'val_samples.png'))
    check(grid.shape==(1024,768,3),label+' grid shape')
    if reference_grid is None:
        reference_grid=grid[:,:512]
    else:
        check(np.array_equal(grid[:,:512],reference_grid),label+' same LR/baselines/HR in visual grid')
    entry=dict(label=label,seed=cfg['seed'],nfe=item['nfe'],gpu=result['environment']['gpu'],
               training_seconds=result['training_seconds'],warmup_seconds=result['warmup_training_seconds'],
               eval_seconds=result['evaluation_seconds'],ms=result['inference_ms_per_image'],
               **result['metrics'])
    metrics_rows.append(entry)
    evals.append(dict(item=item,result=result,images=images,grid=grid))
    arrays_by_method.setdefault(label,{})[cfg['seed']]={k:np.array([r[k] for r in images]) for k in result['metrics']}
for name in ('mse','mse_perceptual','esrgan','mgm'):
    for r in runs:
        if r['config']['variant']==name:
            check(r['status']['init_hash'] is not None,name+' warmup hash exists')
for seed in range(3):
    hashes={e['result']['checkpoint_sha256'] for e in evals if e['item']['config']['variant']=='i2sb' and e['item']['config']['seed']==seed}
    check(len(hashes)==1,f'I2SB seed{seed} same checkpoint across NFE')

method_stats={}
for name in METHODS:
    members=[r for r in runs if r['config']['variant']==name]
    g0=np.mean([r['windows']['first10']['g'] for r in members]);g1=np.mean([r['windows']['last10']['g'] for r in members])
    m0=np.mean([r['windows']['first10']['mse'] for r in members]);m1=np.mean([r['windows']['last10']['mse'] for r in members])
    a=[r['windows']['first10']['aux'] for r in members]
    stat=dict(g_first10=float(g0),g_last10=float(g1),g_relative_change_pct=float(100*(g1-g0)/abs(g0)),
              mse_first10=float(m0),mse_last10=float(m1),mse_relative_change_pct=float(100*(m1-m0)/m0),
              g_previous10=float(np.mean([r['windows']['previous10']['g'] for r in members])),
              mse_previous10=float(np.mean([r['windows']['previous10']['mse'] for r in members])),
              steps_mean=float(np.mean([r['status']['step'] for r in members])),
              vram_gib_max=max(r['summary']['peak_vram_gib'] for r in members),
              skipped_g=sum(r['status']['skipped_g'] for r in members),
              skipped_aux=sum(r['status']['skipped_aux'] for r in members))
    if a[0] is not None:
        stat.update(aux_first10=float(np.mean(a)),aux_last10=float(np.mean([r['windows']['last10']['aux'] for r in members])))
    method_stats[name]=stat

metric_summary={}
for label in arrays_by_method:
    rows=[r for r in metrics_rows if r['label']==label]
    keys=['model_psnr','model_ssim','model_lpips','model_projected_psnr','model_projected_ssim','model_projected_lpips',
          'model_lr_mse_raw','model_lr_mse_clipped','model_out_of_range','model_projected_lr_mse_raw',
          'model_projected_lr_mse_clipped','model_projected_out_of_range','diversity_lpips','ms']
    metric_summary[label]={key:dict(mean=float(np.mean([r[key] for r in rows])),sd=float(np.std([r[key] for r in rows],ddof=1))) for key in keys}

# Paired bootstrap over 50 image IDs, conditional on the three trained seeds.
comparisons=[]
rng=np.random.default_rng(20261018)
bootstrap=rng.integers(0,50,size=(20000,50))
for name,reference,projection in [('mgm','mse',False),('mgm','mse_perceptual',False),('mgm','esrgan',False),
                                  ('mse_perceptual','mse',False),('esrgan','mse_perceptual',False),
                                  ('i2sb_nfe64','mse_perceptual',True),('mgm','i2sb_nfe1',False),
                                  ('mgm','i2sb_nfe64',False),('mgm','i2sb_nfe64',True),('mgm','esrgan',True)]:
    prefix='model_projected_' if projection else 'model_'
    for metric in ('lpips','psnr'):
        a=np.stack([arrays_by_method[name][s][prefix+metric] for s in range(3)])
        b=np.stack([arrays_by_method[reference][s][prefix+metric] for s in range(3)])
        differences=(a-b).mean(0)
        distribution=differences[bootstrap].mean(1)
        wins=differences<0 if metric=='lpips' else differences>0
        comparisons.append(dict(method=name,reference=reference,projected=projection,metric=metric,
                                mean_difference=float(differences.mean()),median_difference=float(np.median(differences)),
                                wins=int(wins.sum()),images=50,ci95=np.quantile(distribution,[.025,.975]).tolist()))

warnings += [f'{sum(r["env"][0]["git_commit"]=="unavailable" for r in runs)}/21 training runs lack recorded Git commit.',
             'Available Git provenance reports modified scheduler scripts; modified files are absent from archive.',
             'No model checkpoints or original HR/LR banks in archive: cannot rerun inference or recompute float metrics.',
             'Only 8 fixed images have saved PNG grids; 50 have per-image metric values.',
             'All supplied evaluations are validation; held-out test not evaluated.',
             'No intermediate validation checkpoints, gradient norms, clip counts, AMP scale or separate ESRGAN loss terms.',
             'Latency is a synchronized mean over 49 images, no per-image times or repeated timing trials retained.',
             'Epoch-equivalent passes count sampled crops/800; this is not an exhaustive pass over image areas.']
summary=dict(issues=issues,warnings=warnings,plan_sha256=hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest(),
             data_sha256=data_hash,total_log_rows=sum(len(r['arrays']['step']) for r in runs),
             training_gpu_hours=sum(r['status']['elapsed_training_seconds'] for r in runs)/3600,
             slurm_training_gpu_hours=sum(r['seconds'] for r in accounting if r['phase'] in ('warmup','train'))/3600,
             slurm_eval_gpu_hours=sum(r['seconds'] for r in accounting if r['phase']=='eval')/3600,
             summed_eval_function_seconds=sum(e['result']['evaluation_seconds'] for e in evals),
             accounting=accounting,methods=method_stats,metrics=metric_summary,comparisons=comparisons,
             visual_image_ids=[r['image'] for r in manifest['banks']['val.npy']['crops'][:8]])
dump('analysis.json',summary)
csv_write('training_runs.csv',training_csv)
csv_write('training_windows.csv',window_csv)
csv_write('metrics_runs.csv',metrics_rows)

plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
for ax,name in zip(axes.flat,METHODS):
    for run in [r for r in runs if r['config']['variant']==name]:
        bins=run['bins'];x=[r['minutes'] for r in bins]
        ax.plot(x,[r['g'] for r in bins],alpha=.75,label=f'seed {run["config"]["seed"]}')
    ax.set(title=LABELS[name],xlabel='Main-stage minutes',ylabel='Own training objective (5-min mean)')
    if name=='mgm':ax.axhline(0,color='black',lw=.7)
    ax.legend(fontsize=8)
for ax in axes.flat[len(METHODS):]:ax.set_visible(False)
fig.suptitle('Different objectives: scales are NOT comparable between methods',fontsize=14)
fig.savefig(OUT/'training_objectives.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for name in ('warmup','mse','mse_perceptual','esrgan','mgm'):
    members=[r for r in runs if r['config']['variant']==name]
    values=np.array([[b['mse'] for b in r['bins']] for r in members]);x=[b['minutes'] for b in members[0]['bins']]
    mean=values.mean(0);sd=values.std(0,ddof=1)
    axes[0].plot(x,mean,color=COLORS[name],label=LABELS[name]);axes[0].fill_between(x,mean-sd,mean+sd,color=COLORS[name],alpha=.12)
axes[0].set(title='Actual SR output vs paired HR',xlabel='Stage minutes',ylabel='Train MSE on [-1,1]');axes[0].legend(fontsize=8)
for run in [r for r in runs if r['config']['variant']=='mgm']:
    x=[b['minutes'] for b in run['bins']]
    axes[1].plot(x,[b['aux'] for b in run['bins']],label=f'Field loss, seed {run["config"]["seed"]}')
axes[1].set(title='MGM field regression',xlabel='Main-stage minutes',ylabel='Field MSE');axes[1].legend(fontsize=8)
for run in [r for r in runs if r['config']['variant']=='i2sb']:
    axes[2].plot([b['minutes'] for b in run['bins']],[b['mse'] for b in run['bins']],label=f'seed {run["config"]["seed"]}')
axes[2].set(title='I2SB: intermediate-state clean estimate',xlabel='Training minutes',ylabel='Diagnostic MSE, NOT final SR inference');axes[2].legend(fontsize=8)
fig.savefig(OUT/'training_diagnostics.png',dpi=160);plt.close(fig)

labels=list(metric_summary)
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for label in labels:
    s=metric_summary[label]
    color=COLORS['i2sb' if label.startswith('i2sb') else label]
    axes[0].errorbar(s['model_psnr']['mean'],s['model_lpips']['mean'],xerr=s['model_psnr']['sd'],yerr=s['model_lpips']['sd'],fmt='o',color=color)
    axes[0].annotate(label,(s['model_psnr']['mean'],s['model_lpips']['mean']),xytext=(5,5),textcoords='offset points',fontsize=8)
    axes[1].plot(s['ms']['mean'],s['model_lpips']['mean'],'o',color=color)
    axes[1].plot(s['ms']['mean'],s['model_projected_lpips']['mean'],'s',color=color,mfc='white')
    axes[1].plot([s['ms']['mean']]*2,[s['model_lpips']['mean'],s['model_projected_lpips']['mean']],color=color,alpha=.5)
    axes[1].annotate(label,(s['ms']['mean'],s['model_lpips']['mean']),xytext=(5,5),textcoords='offset points',fontsize=8)
axes[0].set(xlabel='PSNR (higher is better)',ylabel='LPIPS (lower is better)',title='Native output: mean ± SD across 3 seeds')
axes[1].set(xscale='log',xlabel='FP32 batch1 native inference, ms/image (log)',ylabel='LPIPS (lower is better)',title='Circles: native; open squares: final projection')
fig.savefig(OUT/'quality_latency.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for projected,marker in [(False,'o'),(True,'s')]:
    prefix='model_projected_' if projected else 'model_'
    nfes=[1,4,16,64]
    for ax,key in zip(axes,['psnr','lpips']):
        ax.errorbar(nfes,[metric_summary[f'i2sb_nfe{n}'][prefix+key]['mean'] for n in nfes],
                    yerr=[metric_summary[f'i2sb_nfe{n}'][prefix+key]['sd'] for n in nfes],marker=marker,label='Final projection' if projected else 'Native')
        ax.set(xscale='log',xlabel='NFE',ylabel=key.upper(),title='I2SB quality vs NFE');ax.set_xticks(nfes,labels=nfes);ax.legend()
fig.savefig(OUT/'i2sb_nfe.png',dpi=160);plt.close(fig)

# Scientific contact sheets use the saved PNG pixels; no inference/enhancement.
# Only native outputs were saved. Do not label these as final-projected outputs.
visual_variants=['mse','mse_perceptual','esrgan','mgm','i2sb_nfe1','i2sb_nfe64']
for row in (0,1,2,3,4,5,6,7):
    image_id=summary['visual_image_ids'][row]
    fig,axes=plt.subplots(1,len(visual_variants)+1,figsize=(14,2.35),layout='constrained')
    gt=reference_grid[row*128:(row+1)*128,384:512]
    axes[0].imshow(gt,interpolation='nearest');axes[0].set_title('HR target')
    for ax,label in zip(axes[1:],visual_variants):
        e=next(e for e in evals if e['item']['label']==label and e['result']['seed']==0)
        tile=e['grid'][row*128:(row+1)*128,512:640]
        ax.imshow(tile,interpolation='nearest')
        score=next(r for r in e['images'] if r['image']==image_id)
        ax.set_title(f'{label}\nPSNR {score["model_psnr"]:.2f}; LPIPS {score["model_lpips"]:.3f}',fontsize=9)
    for ax in axes:ax.axis('off')
    fig.suptitle(f'{image_id} | seed 0 | native saved PNG outputs',fontsize=11)
    fig.savefig(OUT/f'sample_{Path(image_id).stem}.png',dpi=160)
    plt.close(fig)

# Across-seed view: fixed first image for reproducibility, not selected for appearance.
fig,axes=plt.subplots(3,4,figsize=(9,7),layout='constrained')
for seed in range(3):
    for col,label in enumerate(['mse_perceptual','esrgan','mgm','i2sb_nfe64']):
        e=next(e for e in evals if e['item']['label']==label and e['result']['seed']==seed)
        axes[seed,col].imshow(e['grid'][:128,512:640],interpolation='nearest')
        axes[seed,col].set_title(f'{label}, seed {seed}',fontsize=9);axes[seed,col].axis('off')
fig.suptitle(f'{summary["visual_image_ids"][0]} | native outputs across training seeds')
fig.savefig(OUT/'across_seeds.png',dpi=150);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for r in [r for r in runs if r['config']['variant']=='mgm']:
    a=r['arrays']; name=f'seed {r["config"]["seed"]}'
    axes[0,0].scatter(a['seconds']/60,a['g'],s=3,alpha=.5,label=name)
    axes[0,1].scatter(a['seconds']/60,a['mse'],s=3,alpha=.4,label=name)
    x=[];means=[];medians=[]
    for lo in range(0,7200,300):
        mask=(a['seconds']>=lo)&(a['seconds']<lo+300)
        x.append((lo+150)/60);means.append(float(a['g'][mask].mean()));medians.append(float(np.median(a['g'][mask])))
    axes[1,0].plot(x,means,label=name)
    axes[1,1].plot(x,medians,label=name)
axes[0,0].set_yscale('symlog',linthresh=1e-4)
axes[0,0].set(title='MGM G objective: individual logged batches',xlabel='Minutes',ylabel='G loss (symlog)')
axes[0,1].set(title='MGM paired reconstruction: individual batches',xlabel='Minutes',ylabel='Train MSE [-1,1]')
axes[1,0].set(title='MGM G objective: 5-minute mean',xlabel='Minutes',ylabel='G loss')
axes[1,1].set(title='MGM G objective: 5-minute median',xlabel='Minutes',ylabel='G loss')
for ax in axes.flat:ax.legend(fontsize=8)
fig.savefig(OUT/'mgm_instability.png',dpi=160);plt.close(fig)
print('Issues:',len(issues),issues)
print('Time:',{k:summary[k] for k in ('training_gpu_hours','slurm_training_gpu_hours','slurm_eval_gpu_hours')})
print('Training:',json.dumps(method_stats,indent=2))
print('Paired:',json.dumps(comparisons,indent=2))
print('Visual IDs:',summary['visual_image_ids'])
