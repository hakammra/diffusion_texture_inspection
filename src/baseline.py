"""CPU baseline and deterministic synthetic pipeline check, not a paper reproduction."""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def font(size=20):
    for name in ('DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def reference_label():
    im = Image.new('L', (360, 224), 238)
    d = ImageDraw.Draw(im)
    d.rectangle((12, 12, 347, 211), outline=30, width=2)
    d.text((28, 28), 'QUALITY CONTROL', font=font(25), fill=20)
    d.text((28, 75), 'ITEM A-104', font=font(24), fill=20)
    d.text((28, 112), 'BATCH 2026-01', font=font(20), fill=25)
    rng = np.random.default_rng(100)
    x = 30
    while x < 270:
        width = int(rng.integers(1, 4))
        d.rectangle((x, 156, x + width, 193), fill=25)
        x += width + int(rng.integers(2, 5))
    return im


def make_sample(base, kind, rng):
    altered = base.copy()
    d = ImageDraw.Draw(altered)
    original = np.asarray(base)
    if kind == 'missing_print':
        ys, xs = np.where((original[75:105, 28:200] < 70))
        k = int(rng.integers(len(xs)))
        x, y = int(xs[k] + 28), int(ys[k] + 75)
        d.rectangle((x - 2, y - 4, x + 5, y + 6), fill=238)
    elif kind == 'smudge':
        x, y = int(rng.integers(190, 290)), int(rng.integers(70, 125))
        d.ellipse((x, y, x + 22, y + 13), fill=int(rng.integers(30, 100)))
    elif kind == 'tear':
        y = int(rng.integers(55, 165))
        d.polygon([(340,y-10),(359,y-20),(359,y+22),(325,y+7)], fill=100)
    elif kind != 'good':
        raise ValueError(f'Unknown sample type: {kind}')
    mask = np.asarray(altered) != original
    signal = np.asarray(altered, dtype=np.float32)
    # Nuisance variation is not counted as a defect. Layout remains aligned.
    signal = signal * rng.uniform(.9, 1.05) + rng.normal(0, 1.5, signal.shape)
    return Image.fromarray(np.clip(signal,0,255).astype('uint8')), Image.fromarray(mask.astype('uint8')*255)


def generate_dataset(root, seed=230224):
    root = Path(root)
    base = reference_label()
    rows = []
    specs = [('train','good',48),('val','good',16)] + [('test',k,12) for k in ['good','missing_print','smudge','tear']]
    for group,(split,kind,count) in enumerate(specs):
        rng = np.random.default_rng(np.random.SeedSequence([seed,group]))
        for i in range(count):
            image, mask = make_sample(base,kind,rng)
            relative = Path(split)/kind/f'{i:03d}.png'
            target = root/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            image.save(target)
            mask_path = ''
            if split == 'test':
                relmask = Path('masks')/kind/f'{i:03d}.png'
                (root/relmask).parent.mkdir(parents=True,exist_ok=True)
                mask.save(root/relmask)
                mask_path=relmask.as_posix()
            rows.append({'split':split,'kind':kind,'image':relative.as_posix(),'mask':mask_path})
    (root/'manifest.json').write_text(json.dumps({'synthetic':True,'seed':seed,'generator_version':1,'samples':rows},indent=2))
    return rows


def read_normalized(path):
    with Image.open(path) as im:
        x=np.asarray(im.convert('L'),dtype=np.float32)/255.
    # White-background label assumption; real photographs need registration first.
    return np.clip(x / max(float(np.median(x)),1e-6),0,1)


def score_image(image,template):
    if image.shape != template.shape:
        raise ValueError(f'Shape mismatch: {image.shape} != {template.shape}')
    squared = (image-template)**2
    # This baseline explicitly quantizes residuals to 8 bits before median filtering.
    # Saved scores retain all resulting levels, rather than only a binary mask.
    quantized = Image.fromarray(np.clip(squared*255,0,255).astype('uint8'))
    return np.asarray(quantized.filter(ImageFilter.MedianFilter(3)),dtype=np.float32)/255.


def fit_template(paths):
    if not paths:
        raise ValueError('Normal training images are required')
    return np.median(np.stack([read_normalized(p) for p in paths]),axis=0)


def calibrate_threshold(normal_scores,target_fpr=.005):
    if not 0 < target_fpr < 1:
        raise ValueError('target_fpr must be strictly between 0 and 1')
    if not normal_scores:
        raise ValueError('Normal validation scores are required')
    return float(np.quantile(np.concatenate([s.ravel() for s in normal_scores]),1-target_fpr,method='higher'))


def mask_metrics(pred,truth):
    pred,truth=np.asarray(pred,dtype=bool),np.asarray(truth,dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError('Prediction and reference shapes must match')
    tp=int(np.sum(pred & truth));fp=int(np.sum(pred & ~truth));fn=int(np.sum(~pred & truth));tn=int(np.sum(~pred & ~truth))
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}


def preview(rows,root,template,threshold,output):
    selected = [next(r for r in rows if r['split']=='test' and r['kind']==k) for k in ['good','missing_print','smudge','tear']]
    sheet=Image.new('RGB',(1180,1110),'#edf2f4');d=ImageDraw.Draw(sheet)
    d.text((30,20),'LabelInspect | CPU baseline pipeline check',fill='#172c38',font=font(27))
    d.text((30,60),'SYNTHETIC DATA ONLY - not diffusion output or real-world validation',fill='#9d351c',font=font(18))
    for c,title in enumerate(['Input','Known synthetic mask','Template baseline mask']):
        d.text((30+c*380,104),title,fill='#172c38',font=font(19))
    for idx,row in enumerate(selected):
        y=150+idx*235
        image=Image.open(root/row['image']).convert('RGB')
        truth=Image.open(root/row['mask']).convert('RGB')
        pred=score_image(read_normalized(root/row['image']),template)>threshold
        pred=Image.fromarray(pred.astype('uint8')*255).convert('RGB')
        for c,im in enumerate([image,truth,pred]):sheet.paste(im.resize((342,213)),(30+c*380,y))
        d.text((30,y-20),row['kind'],fill='#172c38',font=font(15))
    sheet.save(output)


def run_demo(output,seed=230224):
    output=Path(output);root=output/'data';output.mkdir(parents=True,exist_ok=True)
    rows=generate_dataset(root,seed)
    template=fit_template([root/r['image'] for r in rows if r['split']=='train'])
    threshold=calibrate_threshold([score_image(read_normalized(root/r['image']),template) for r in rows if r['split']=='val'])
    np.save(output/'template.npy',template)
    results=[];elapsed=[]
    for row in rows:
        if row['split']!='test':continue
        start=time.perf_counter()
        scores=score_image(read_normalized(root/row['image']),template)
        pred=scores>threshold;elapsed.append((time.perf_counter()-start)*1000)
        truth=np.asarray(Image.open(root/row['mask']).convert('L'))>0
        results.append({'image':row['image'],'kind':row['kind'],**mask_metrics(pred,truth)})
        dest=output/'predictions'/row['kind'];dest.mkdir(parents=True,exist_ok=True)
        name=Path(row['image']).name
        Image.fromarray(pred.astype('uint8')*255).save(dest/name)
        np.save(dest/(Path(name).stem+'_scores.npy'),scores)
    defects=[r for r in results if r['kind']!='good'];normal=[r for r in results if r['kind']=='good']
    report={'status':'synthetic_pipeline_check_only','method':'illumination_normalized_median_template_squared_residual_3x3_median','seed':seed,'train_count':48,'normal_validation_count':16,'test_count':48,'test_defective_count':36,'test_normal_count':12,'threshold_source':'normal_validation_only','target_normal_pixel_fpr':.005,'threshold':threshold,'mean_dice_defective_images':float(np.mean([r['dice'] for r in defects])),'mean_iou_defective_images':float(np.mean([r['iou'] for r in defects])),'normal_pixel_fpr':sum(r['fp'] for r in normal)/sum(r['fp']+r['tn'] for r in normal),'median_ms_including_image_load':float(np.median(elapsed)),'limitations':['Synthetic, perfectly aligned fixed layout','No real-world accuracy claim','Not DTU-Net or Tsimplex','Synthetic threshold cannot be assumed valid for real data']}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    with (output/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    preview(rows,root,template,threshold,output/'preview.png')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['demo'])
    parser.add_argument('--output',default='artifacts/synthetic_check')
    parser.add_argument('--seed',type=int,default=230224)
    args=parser.parse_args();run_demo(args.output,args.seed)
