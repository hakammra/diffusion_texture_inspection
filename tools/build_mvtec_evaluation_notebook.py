from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r'C:\Users\abdul\Documents\LabelInspect')
source_nb = json.loads((ROOT/'notebooks'/'05_mvtec_leather_train.ipynb').read_text(encoding='utf-8'))
author_writer = ''.join(source_nb['cells'][3]['source'])


def lines(text): return text.splitlines(keepends=True) or ['']
def md(text): return {'cell_type':'markdown','metadata':{},'source':lines(text.strip()+'\n')}
def code(text): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':lines(text.strip()+'\n')}

cells=[
md('''# Stage 3D — MVTec leather validation and locked test evaluation

This notebook loads the completed 2,000-step checkpoint and performs the first reduced evaluation.

Protocol decisions are fixed before reading test masks:
- `t_distance = 250`, following the fixed-time author evaluation routine and the paper's partial-diffusion analysis.
- The binary pixel threshold is the 99.5th percentile of reconstruction residuals across the 49 held-out **normal validation** images.
- The official test set is evaluated once after the threshold is frozen.
- Training and validation use no anomalous images or masks.

Reported outputs include pixel AUROC, image AUROC, mean Dice/IoU on defective images, normal pixel false-positive rate, per-defect results, latency, and qualitative examples. This is a one-category, 2,000-step reduced experiment rather than a reproduction of the paper's 3,000-epoch result.
'''),
code(r'''from pathlib import Path
import csv, importlib.util, json, os, random, shutil, subprocess, sys, time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

assert torch.cuda.is_available(), 'Select a T4 GPU runtime first.'
from google.colab import drive
drive.mount('/content/drive')

DATA_ROOT = Path('/content/drive/MyDrive/mvtec_anomaly_detection')
CHECKPOINT = Path('/content/drive/MyDrive/LabelInspect/checkpoints/mvtec_leather/latest.pt')
WORK = Path('/content/labelinspect')
OUTPUT = WORK/'mvtec_leather_evaluation'
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 230224
IMAGE_SIZE = 224
BATCH_SIZE = 4
T_DISTANCE = 250
TARGET_NORMAL_PIXEL_FPR = 0.005
print('GPU:', torch.cuda.get_device_name(0))
print('Checkpoint:', CHECKPOINT)
'''),
code(r'''missing=[]
for package,module in [('timm','timm'),('einops','einops'),('numba','numba'),('scikit-learn','sklearn')]:
    if importlib.util.find_spec(module) is None: missing.append(package)
if missing:
    subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)
print('Dependencies ready.')
'''),
code(author_writer),
code(r'''spec=importlib.util.spec_from_file_location('labelinspect_author_smoke',WORK/'author_smoke.py')
author=importlib.util.module_from_spec(spec);sys.modules[spec.name]=author;spec.loader.exec_module(author)
import numba
numba.set_num_threads(min(2,numba.get_num_threads()))
source_cache=WORK/'upstream'/author.COMMIT
hashes=author.fetch_sources(source_cache)
model_module,diffusion_ns,Adapter,compatibility=author.load_author_components(source_cache,OUTPUT)
print('Pinned author commit:',author.COMMIT)
'''),
code(r'''def locate_category(root:Path,category='leather'):
    candidates=[root/category,root/'mvtec_anomaly_detection'/category]
    if root.name==category:candidates.insert(0,root)
    for candidate in candidates:
        if (candidate/'train'/'good').is_dir() and (candidate/'test').is_dir():return candidate
    raise FileNotFoundError(f'Could not find leather category below {root}.')

category_root=locate_category(DATA_ROOT)
all_normal=sorted((category_root/'train'/'good').glob('*.png'))
assert len(all_normal)==245,f'Expected 245 train/good images, found {len(all_normal)}'
split_rng=random.Random(SEED);shuffled=all_normal.copy();split_rng.shuffle(shuffled)
val_count=max(1,round(len(shuffled)*.20));val_paths=sorted(shuffled[:val_count]);train_paths=sorted(shuffled[val_count:])
test_paths=sorted((category_root/'test').glob('*/*.png'))
assert len(train_paths)==196 and len(val_paths)==49 and len(test_paths)==124

RESAMPLE=getattr(Image,'Resampling',Image)
def load_image(path):
    with Image.open(path) as image:
        image=image.convert('RGB').resize((IMAGE_SIZE,IMAGE_SIZE),RESAMPLE.BILINEAR)
        array=np.asarray(image,dtype=np.float32).copy()/127.5-1.
    return torch.from_numpy(array).permute(2,0,1)
def load_mask(path):
    with Image.open(path) as image:
        array=np.asarray(image.convert('L').resize((IMAGE_SIZE,IMAGE_SIZE),RESAMPLE.NEAREST))>0
    return array

def mask_for_test(path):
    kind=path.parent.name
    return None if kind=='good' else category_root/'ground_truth'/kind/f'{path.stem}_mask.png'

for path in test_paths:
    mask=mask_for_test(path)
    if mask is not None:assert mask.is_file(),f'Missing mask {mask}'
print('Protocol:',len(train_paths),'train normal,',len(val_paths),'validation normal,',len(test_paths),'official test')
'''),
code(r'''assert CHECKPOINT.is_file(),'The 2,000-step checkpoint is missing from Google Drive.'
device=torch.device('cuda:0')
checkpoint=torch.load(CHECKPOINT,map_location=device,weights_only=False)
assert checkpoint['step']==2000,f"Expected step 2000, found {checkpoint['step']}"
assert checkpoint['author_commit']==author.COMMIT
model_config=checkpoint['model_config']
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
backbone=model_module.UDHVT(**model_config).to(device);model=Adapter(backbone)
model.load_state_dict(checkpoint['model']);model.eval()
diffusion=diffusion_ns['GaussianDiffusionModel'](
    [224,224],diffusion_ns['get_beta_schedule'](1000,'cosine'),img_channels=3,
    loss_type='l2',noise='4dsimplex',octave=6,frequency=64,persistence=.9,train=False)
# Compile Tsimplex before timing evaluation.
diffusion.noise_fn(torch.zeros(1,1,4,4,device=device),torch.tensor([5],device=device))
torch.cuda.reset_peak_memory_stats()
print('Loaded trained model with',sum(p.numel() for p in model.parameters()),'parameters.')
'''),
code(r'''def reconstruct_paths(paths,batch_size=BATCH_SIZE):
    reconstructions=[];inputs=[];seconds=[]
    for start in range(0,len(paths),batch_size):
        batch_paths=paths[start:start+batch_size]
        x=torch.stack([load_image(p) for p in batch_paths]).to(device)
        tick=time.perf_counter()
        with torch.inference_mode():
            recon=diffusion.forward_backward(model,x,None,see_whole_sequence=None,
                                              t_distance=T_DISTANCE,denoise_fn='noise_fn')
        torch.cuda.synchronize()
        seconds.append(time.perf_counter()-tick)
        inputs.append(x.cpu());reconstructions.append(recon.cpu())
        print(f'Reconstructed {min(start+len(batch_paths),len(paths))}/{len(paths)}',flush=True)
    return torch.cat(inputs),torch.cat(reconstructions),seconds

def residual_maps(inputs,recons):
    return (inputs-recons).square().mean(dim=1).numpy().astype(np.float32)
'''),
code(r'''# Threshold calibration uses only held-out normal images.
print('Reconstructing held-out normal validation images...')
val_inputs,val_recons,val_batch_seconds=reconstruct_paths(val_paths)
val_scores=residual_maps(val_inputs,val_recons)
threshold=float(np.quantile(val_scores.reshape(-1),1-TARGET_NORMAL_PIXEL_FPR,method='higher'))
observed_val_fpr=float(np.mean(val_scores>threshold))
calibration={
    'source':'49 held-out MVTec leather train/good images only',
    'target_normal_pixel_fpr':TARGET_NORMAL_PIXEL_FPR,
    'threshold':threshold,
    'observed_validation_pixel_fpr':observed_val_fpr,
    't_distance':T_DISTANCE,
}
(OUTPUT/'calibration.json').write_text(json.dumps(calibration,indent=2))
print(json.dumps(calibration,indent=2))
'''),
code(r'''# The threshold is now frozen. Test masks are loaded only for evaluation.
print('Running the locked official test evaluation...')
test_inputs,test_recons,test_batch_seconds=reconstruct_paths(test_paths)
test_scores=residual_maps(test_inputs,test_recons)
test_predictions=test_scores>threshold

ground_truth=[];image_labels=[];kinds=[]
for path in test_paths:
    kind=path.parent.name;kinds.append(kind);image_labels.append(kind!='good')
    mask_path=mask_for_test(path)
    ground_truth.append(np.zeros((IMAGE_SIZE,IMAGE_SIZE),bool) if mask_path is None else load_mask(mask_path))
ground_truth=np.stack(ground_truth)
image_labels=np.asarray(image_labels,bool)

from sklearn.metrics import roc_auc_score
pixel_auc=float(roc_auc_score(ground_truth.reshape(-1),test_scores.reshape(-1)))
# Robust high-percentile image score is declared in advance here and retained in the report.
image_scores=np.quantile(test_scores.reshape(len(test_paths),-1),.995,axis=1)
image_auc=float(roc_auc_score(image_labels,image_scores))

def confusion_metrics(pred,truth):
    tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth));fn=int(np.sum(~pred&truth));tn=int(np.sum(~pred&~truth))
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None}

rows=[]
for i,path in enumerate(test_paths):
    metrics=confusion_metrics(test_predictions[i],ground_truth[i])
    rows.append({'image':path.relative_to(category_root).as_posix(),'kind':kinds[i],
                 'image_label':int(image_labels[i]),'image_score_p995':float(image_scores[i]),**metrics})
with (OUTPUT/'per_image.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

defective=[row for row in rows if row['kind']!='good'];normal=[row for row in rows if row['kind']=='good']
normal_fpr=sum(row['fp'] for row in normal)/sum(row['fp']+row['tn'] for row in normal)
per_kind={}
for kind in sorted(set(kinds)):
    indices=np.asarray([value==kind for value in kinds])
    group=[rows[i] for i in np.flatnonzero(indices)]
    entry={'count':len(group)}
    if kind=='good':
        entry['pixel_false_positive_rate']=sum(x['fp'] for x in group)/sum(x['fp']+x['tn'] for x in group)
    else:
        entry.update({'mean_dice':float(np.mean([x['dice'] for x in group])),
                      'mean_iou':float(np.mean([x['iou'] for x in group])),
                      'pixel_auc':float(roc_auc_score(ground_truth[indices].reshape(-1),test_scores[indices].reshape(-1)))})
    per_kind[kind]=entry
'''),
code(r'''# Qualitative preview: one normal and one example per anomaly kind.
selected=[]
for kind in ['good','color','cut','fold','glue','poke']:
    selected.append(next(i for i,value in enumerate(kinds) if value==kind))
fig,axes=plt.subplots(len(selected),5,figsize=(14,3*len(selected)))
for row,index in enumerate(selected):
    original=((test_inputs[index].permute(1,2,0).numpy()+1)/2).clip(0,1)
    recon=((test_recons[index].permute(1,2,0).numpy()+1)/2).clip(0,1)
    panels=[original,recon,test_scores[index],ground_truth[index],test_predictions[index]]
    titles=[kinds[index],'Reconstruction','Squared residual','Ground truth','Prediction']
    for col,(panel,title) in enumerate(zip(panels,titles)):
        axes[row,col].imshow(panel,cmap=None if col<2 else 'magma' if col==2 else 'gray')
        axes[row,col].set_title(title);axes[row,col].axis('off')
fig.tight_layout();fig.savefig(OUTPUT/'evaluation_preview.png',dpi=150,bbox_inches='tight');plt.show()

report={
    'status':'passed','scope':'one_category_reduced_evaluation','paper_result_reproduced':False,
    'dataset':'MVTec AD','category':'leather','checkpoint_step':int(checkpoint['step']),
    'author_commit':author.COMMIT,'model_configuration':model_config,
    'train_normal_count':len(train_paths),'validation_normal_count':len(val_paths),
    'test_count':len(test_paths),'test_defective_count':int(image_labels.sum()),'test_normal_count':int((~image_labels).sum()),
    't_distance':T_DISTANCE,'t_distance_source':'Author fixed-time evaluation routine and paper partial-diffusion analysis; not selected on this test set.',
    'threshold':threshold,'threshold_source':'99.5th percentile of held-out normal validation residual pixels',
    'pixel_auroc':pixel_auc,'image_auroc_p995_score':image_auc,
    'mean_dice_defective_images':float(np.mean([x['dice'] for x in defective])),
    'mean_iou_defective_images':float(np.mean([x['iou'] for x in defective])),
    'mean_precision_defective_images':float(np.mean([x['precision'] or 0. for x in defective])),
    'mean_recall_defective_images':float(np.mean([x['recall'] or 0. for x in defective])),
    'normal_test_pixel_fpr':float(normal_fpr),'per_kind':per_kind,
    'median_validation_batch_seconds':float(np.median(val_batch_seconds)),
    'median_test_batch_seconds':float(np.median(test_batch_seconds)),
    'approx_test_seconds_per_image':float(sum(test_batch_seconds)/len(test_paths)),
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'limitations':['One MVTec category','2,000 optimizer steps rather than paper 3,000 epochs','One reconstruction per image','Normal-only threshold protocol differs from test-optimized Dice selection','Architecture is the inspected SPE+DMHA+HFF+refinement configuration'],
}
(OUTPUT/'evaluation_report.json').write_text(json.dumps(report,indent=2))
# Compact half-precision maps are for qualitative audit; metrics above use float32 maps.
np.savez_compressed(OUTPUT/'test_maps_float16.npz',scores=test_scores.astype(np.float16),predictions=test_predictions,ground_truth=ground_truth)
archive=shutil.make_archive(str(WORK/'mvtec_leather_evaluation_results'),'zip',OUTPUT)
print(json.dumps(report,indent=2))
print('\nLOCKED TEST EVALUATION PASSED')
print('Download:',archive)
''')]

notebook={'cells':cells,'metadata':{'accelerator':'GPU','colab':{'name':'06_mvtec_leather_evaluate.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.x'}},'nbformat':4,'nbformat_minor':5}
out=ROOT/'notebooks'/'06_mvtec_leather_evaluate.ipynb';out.write_text(json.dumps(notebook,indent=1),encoding='utf-8');print(out)
