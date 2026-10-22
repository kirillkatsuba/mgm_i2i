"""Build scientific figures from frozen exports; never run or modify models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DOCS=Path(__file__).resolve().parent
ROOT=DOCS.parent
ASSETS=DOCS/'assets'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality_plot():
    with (DOCS/'results/test_metrics.csv').open()as stream:
        rows={(r['model'],r['output']):r for r in csv.DictReader(stream)}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10.8,4.5),layout='constrained')
    points=[('mse','MSE','#7b8794',(7,7)),
            ('mse_perceptual','MSE + perceptual','#bd7d23',(7,10)),
            ('mgm_v2_no_noise','MGM v2, 1 NFE','#b7353d',(-2,-19)),
            ('esrgan','ESRGAN + P','#7953a5',(7,-7)),
            ('sinsr','SinSR + P, 1 NFE','#168b85',(9,-17))]
    for name,label,color,offset in points:
        row=rows[name,'projected'];x=float(row['inference_ms']);y=float(row['lpips'])
        ax.scatter(x,y,s=80 if name.startswith('mgm')else 55,c=color,zorder=3)
        ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',color=color,
                    weight='bold'if name.startswith('mgm')else'normal')
    xs=[];ys=[]
    for n in (1,4,16,64):
        row=rows[f'i2sb_nfe{n}','projected'];x=float(row['inference_ms']);y=float(row['lpips'])
        xs.append(x);ys.append(y)
        ax.annotate(f'{n} NFE',(x,y),xytext=(5,8),textcoords='offset points',fontsize=9,color='#2873a4')
    ax.plot(xs,ys,'o-',color='#2873a4',lw=1.5,ms=5,label='I²SB-small + P')
    ax.set(xscale='log',xlim=(7.5,1100),ylim=(.126,.26),
           xlabel='Native inference latency, ms/patch (log scale, lower is better)',
           ylabel='Test LPIPS (lower is better)',title='Common test pairs and final projection')
    ax.grid(alpha=.2);ax.legend(loc='upper right',frameon=False)
    fig.savefig(ASSETS/'quality_latency.png',dpi=170);plt.close(fig)


def gallery():
    mgm=ROOT/'tmp/mgm_noise_20261021/cluster_runs/mgm_noise_paper_val_test_p8_20261021/v2/mgm_no_noise/seed0/eval_nfe1/test'
    baselines=ROOT/'tmp/baselines_test_20261021/cluster_runs/baselines_test_multi5_20261021'
    classic=baselines/'baselines_multi4_eval5_w50_g80_b32_r2'
    folders={'mgm':mgm,'perc':classic/'mse_perceptual/seed0/eval_nfe1',
             'esr':classic/'esrgan/seed0/eval_nfe1',
             'bridge':baselines/'i2sb_multi4_eval5_e130_b32_r2/i2sb/seed0/eval_nfe16',
             'sin':ROOT/'tmp/sinsr_test_20261021/cluster_runs/sr_refs_multi5_test_sinsr_20261021/sinsr/eval'}
    patches=json.loads((mgm/'test_patches.json').read_text())
    photos=sorted({p['image']for p in patches})[:4]
    columns=[('HR','mgm',3),('MSE + perceptual','perc',4),('MGM v2','mgm',4),
             ('ESRGAN + P','esr',6),('I²SB16 + P','bridge',6),('SinSR + P','sin',6)]
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),12)
    canvas=Image.new('RGB',(128*len(columns),30+150*len(photos)),'white');draw=ImageDraw.Draw(canvas)
    for col,(label,_,_)in enumerate(columns):draw.text((col*128+4,5),label,font=font,fill='black')
    provenance=[]
    for row,photo in enumerate(photos):
        index=next(i for i,p in enumerate(patches)if p['image']==photo and p['crop_index']==0)
        filename=f'{index:03d}_{Path(photo).stem}.png'
        draw.text((4,30+row*150),f'{photo} · crop0 · draw1',font=font,fill='black')
        inputs=None
        for col,(label,name,tile)in enumerate(columns):
            path=folders[name]/'samples'/filename
            with Image.open(path)as im:
                assert im.size==(896,128)
                current=np.asarray(im)[:,:512]
                if inputs is None:inputs=current.copy()
                else:np.testing.assert_array_equal(current,inputs)
                panel=im.crop((tile*128,0,(tile+1)*128,128))
                canvas.paste(panel,(col*128,50+row*150))
                provenance.append(dict(photo=photo,crop_index=0,draw=1,label=label,
                    source=str(path.relative_to(ROOT)),source_sha256=sha(path),tile=tile,
                    panel_sha256=hashlib.sha256(panel.tobytes()).hexdigest()))
    canvas.save(ASSETS/'test_samples.png')
    (ASSETS/'test_samples.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery',action='store_true',help='Requires the original extracted result archives.')
    args=parser.parse_args();ASSETS.mkdir(parents=True,exist_ok=True)
    quality_plot()
    if args.gallery:gallery()
    print('Built figures from frozen evidence; no model inference or training.')
