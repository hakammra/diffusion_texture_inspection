from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "06_mvtec_leather_evaluate_kaggle.ipynb"
TRAINING = ROOT / "notebooks" / "08_fabric_stain_train_kaggle.ipynb"
OUTPUT = ROOT / "notebooks" / "09_fabric_stain_evaluate_kaggle.ipynb"


def lines(text: str):
    return [line + "\n" for line in text.strip("\n").splitlines()]


nb = json.loads(SOURCE.read_text(encoding="utf-8"))
training_nb = json.loads(TRAINING.read_text(encoding="utf-8"))

nb["cells"][0]["source"] = lines(
    """
# Fabric stain locked image-level evaluation

This notebook loads the completed 2,000-step DTU-Net/Tsimplex checkpoint and
performs the first locked fabric-stain evaluation.

Attach the same `fabric_stain_pilot` dataset and the newest checkpoint input.
The protocol was fixed before evaluation:

- 10 validation-normal images calibrate the image-score cutoff.
- 10 untouched normal images and a seed-fixed subset of 100 stain images form
  the locked test set. The smaller stain subset limits free-GPU use.
- Each complete photograph is resized to 224 × 224 so defects are not cropped
  away. This creates a scale mismatch with 512-pixel training crops and is
  recorded as a limitation.
- `t_distance = 250`, matching the earlier author-path evaluation decision.
- The image score is the 99.5th percentile of squared reconstruction residuals.

The stain cohort has no pixel masks. This run reports image AUROC, average
precision, thresholded sensitivity/specificity, latency, and qualitative maps.
It does not report Dice or IoU.
"""
)

nb["cells"][1]["source"] = lines(
    """
from pathlib import Path
import csv, importlib.util, json, random, shutil, subprocess, sys, time, zipfile

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT=Path('/kaggle/input')
WORK=Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True,exist_ok=True)
OUTPUT=WORK/'fabric_stain_evaluation'
OUTPUT.mkdir(parents=True,exist_ok=True)

def find_dataset(search_root):
    found=[]
    for path in search_root.rglob('fabric_stain_pilot'):
        if path.is_dir() and (path/'val_normal').is_dir() and (path/'test'/'stain').is_dir():
            found.append(path)
    return sorted(set(found))

dataset_candidates=find_dataset(KAGGLE_INPUT)
if not dataset_candidates:
    matching=[]
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            names=['/'+item.filename.replace('\\\\','/').lstrip('/') for item in archive.infolist()]
            if any('/fabric_stain_pilot/val_normal/' in name for name in names): matching.append(archive_path)
    if len(matching)==1:
        extraction_root=WORK/'uploaded_data';extraction_root.mkdir(parents=True,exist_ok=True)
        resolved=extraction_root.resolve()
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                target=(extraction_root/item.filename).resolve()
                if target!=resolved and resolved not in target.parents: raise ValueError(f'Unsafe ZIP member: {item.filename}')
            archive.extractall(extraction_root)
        dataset_candidates=find_dataset(extraction_root)
if len(dataset_candidates)!=1:
    raise FileNotFoundError('Expected one fabric_stain_pilot dataset: '+repr([str(p) for p in dataset_candidates]))
DATA_ROOT=dataset_candidates[0]

all_pt_files=sorted(KAGGLE_INPUT.rglob('*.pt'))
if not all_pt_files:
    extracted_roots=[]
    for data_pickle in KAGGLE_INPUT.rglob('data.pkl'):
        candidate=data_pickle.parent
        if (candidate/'data').is_dir() and (candidate/'version').is_file(): extracted_roots.append(candidate)
    extracted_roots=sorted(set(extracted_roots))
    if len(extracted_roots)==1:
        archive_root=extracted_roots[0]
        rebuilt=WORK/'rebuilt_fabric_stain_checkpoint.pt'
        with zipfile.ZipFile(rebuilt,'w',compression=zipfile.ZIP_STORED) as archive:
            for source_file in sorted(archive_root.rglob('*')):
                if source_file.is_file():
                    archive.write(source_file,f'{archive_root.name}/{source_file.relative_to(archive_root).as_posix()}')
        all_pt_files=[rebuilt]
        print('Rebuilt Kaggle-extracted checkpoint:',rebuilt)
if len(all_pt_files)!=1:
    raise FileNotFoundError('Expected one checkpoint input; found: '+repr([str(p) for p in all_pt_files]))
CHECKPOINT=all_pt_files[0]

SEED=230224
IMAGE_SIZE=224
BATCH_SIZE=4
T_DISTANCE=250
IMAGE_SCORE_QUANTILE=.995
STAIN_TEST_COUNT=100
print('GPU:',torch.cuda.get_device_name(0))
print('Dataset:',DATA_ROOT)
print('Checkpoint:',CHECKPOINT)
"""
)

# Keep the verified dependency and embedded author-source cells.
nb["cells"][3]["source"] = training_nb["cells"][3]["source"]
nb["cells"][4]["source"] = lines(
    """
spec=importlib.util.spec_from_file_location('labelinspect_author_smoke',WORK/'author_smoke.py')
author=importlib.util.module_from_spec(spec);sys.modules[spec.name]=author;spec.loader.exec_module(author)
import numba
numba.set_num_threads(min(2,numba.get_num_threads()))
source_cache=WORK/'upstream'/author.COMMIT
hashes=author.fetch_sources(source_cache)
model_module,diffusion_ns,Adapter,compatibility=author.load_author_components(source_cache,OUTPUT)
print('Pinned author commit:',author.COMMIT)
"""
)

nb["cells"][5]["source"] = lines(
    """
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png'}
def image_paths(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

val_paths=image_paths(DATA_ROOT/'val_normal')
all_test_good=image_paths(DATA_ROOT/'test'/'good')
all_test_stain=image_paths(DATA_ROOT/'test'/'stain')
assert (len(val_paths),len(all_test_good),len(all_test_stain))==(10,10,398)
selection_rng=random.Random(SEED)
selected_stain=all_test_stain.copy();selection_rng.shuffle(selected_stain)
test_stain_paths=sorted(selected_stain[:STAIN_TEST_COUNT])
test_paths=all_test_good+test_stain_paths
image_labels=np.asarray([0]*len(all_test_good)+[1]*len(test_stain_paths),dtype=np.int64)
(OUTPUT/'locked_test_files.txt').write_text('\\n'.join(path.relative_to(DATA_ROOT).as_posix() for path in test_paths))

RESAMPLE=getattr(Image,'Resampling',Image).BILINEAR
def load_image(path):
    with Image.open(path) as image:
        image=image.convert('RGB').resize((IMAGE_SIZE,IMAGE_SIZE),RESAMPLE)
        array=np.asarray(image,dtype=np.float32).copy()/127.5-1.
    return torch.from_numpy(array).permute(2,0,1)

print('Protocol:',len(val_paths),'validation normal,',len(all_test_good),'test normal,',len(test_stain_paths),'test stain')
"""
)

nb["cells"][6]["source"] = lines(
    """
device=torch.device('cuda:0')
checkpoint=torch.load(CHECKPOINT,map_location=device,weights_only=False)
assert checkpoint['step']==2000,f"Expected step 2000, found {checkpoint['step']}"
assert checkpoint['author_commit']==author.COMMIT
assert checkpoint.get('dataset_category')=='fabric_stain_pilot'
model_config=checkpoint['model_config']
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
backbone=model_module.UDHVT(**model_config).to(device);model=Adapter(backbone)
model.load_state_dict(checkpoint['model']);model.eval()
diffusion=diffusion_ns['GaussianDiffusionModel'](
    [224,224],diffusion_ns['get_beta_schedule'](1000,'cosine'),img_channels=3,
    loss_type='l2',noise='4dsimplex',octave=6,frequency=64,persistence=.9,train=False)
diffusion.noise_fn(torch.zeros(1,1,4,4,device=device),torch.tensor([5],device=device))
torch.cuda.reset_peak_memory_stats()
print('Loaded step',checkpoint['step'],'model with',sum(p.numel() for p in model.parameters()),'parameters.')
"""
)

# The generic batch reconstruction and residual helpers are unchanged.
nb["cells"][7]["source"] = nb["cells"][7]["source"]

nb["cells"][8]["source"] = lines(
    """
print('Reconstructing held-out normal validation images...')
val_inputs,val_recons,val_batch_seconds=reconstruct_paths(val_paths)
val_maps=residual_maps(val_inputs,val_recons)
val_image_scores=np.quantile(val_maps.reshape(len(val_paths),-1),IMAGE_SCORE_QUANTILE,axis=1)
# With only ten validation normals, the maximum is a conservative normal-only cutoff.
threshold=float(np.max(val_image_scores))
calibration={
    'source':'10 held-out normal images only',
    'image_score_quantile':IMAGE_SCORE_QUANTILE,
    'threshold_rule':'maximum validation-normal image score',
    'threshold':threshold,
    'observed_validation_image_fpr':float(np.mean(val_image_scores>threshold)),
    't_distance':T_DISTANCE,
}
(OUTPUT/'calibration.json').write_text(json.dumps(calibration,indent=2))
print(json.dumps(calibration,indent=2))
"""
)

nb["cells"][9]["source"] = lines(
    """
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

print('Running locked normal-versus-stain evaluation...')
test_inputs,test_recons,test_batch_seconds=reconstruct_paths(test_paths)
test_maps=residual_maps(test_inputs,test_recons)
image_scores=np.quantile(test_maps.reshape(len(test_paths),-1),IMAGE_SCORE_QUANTILE,axis=1)
predictions=image_scores>threshold

image_auc=float(roc_auc_score(image_labels,image_scores))
average_precision=float(average_precision_score(image_labels,image_scores))
tn,fp,fn,tp=confusion_matrix(image_labels,predictions,labels=[0,1]).ravel()
sensitivity=float(tp/(tp+fn)) if tp+fn else None
specificity=float(tn/(tn+fp)) if tn+fp else None
balanced_accuracy=float((sensitivity+specificity)/2)

rows=[]
for path,label,score,prediction in zip(test_paths,image_labels,image_scores,predictions):
    rows.append({'image':path.relative_to(DATA_ROOT).as_posix(),'label':int(label),
                 'kind':'stain' if label else 'good','image_score_p995':float(score),'prediction':int(prediction)})
with (OUTPUT/'per_image.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

fpr,tpr,_=roc_curve(image_labels,image_scores)
fig,ax=plt.subplots(figsize=(5,5));ax.plot(fpr,tpr,label=f'AUROC = {image_auc:.3f}')
ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='Fabric stain image-level ROC')
ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUTPUT/'roc_curve.png',dpi=160);plt.show()
"""
)

nb["cells"][10]["source"] = lines(
    """
# Show a fixed normal plus low/median/high-scoring stain examples for failure analysis.
stain_indices=np.flatnonzero(image_labels==1)
ordered=stain_indices[np.argsort(image_scores[stain_indices])]
selected=[0,int(ordered[0]),int(ordered[len(ordered)//2]),int(ordered[-1])]
fig,axes=plt.subplots(len(selected),3,figsize=(10,3*len(selected)))
for row,index in enumerate(selected):
    original=((test_inputs[index].permute(1,2,0).numpy()+1)/2).clip(0,1)
    recon=((test_recons[index].permute(1,2,0).numpy()+1)/2).clip(0,1)
    panels=[original,recon,test_maps[index]];titles=[f"{rows[index]['kind']} | score {image_scores[index]:.4f}",'Reconstruction','Squared residual']
    for col,(panel,title) in enumerate(zip(panels,titles)):
        axes[row,col].imshow(panel,cmap=None if col<2 else 'magma');axes[row,col].set_title(title);axes[row,col].axis('off')
fig.tight_layout();fig.savefig(OUTPUT/'evaluation_preview.png',dpi=150,bbox_inches='tight');plt.show()

report={
    'status':'passed','scope':'locked_fabric_stain_image_evaluation','paper_result_reproduced':False,
    'dataset':'Fabric Defects Dataset - matched grayscale stain cohort','checkpoint_step':int(checkpoint['step']),
    'author_commit':author.COMMIT,'model_configuration':model_config,
    'train_normal_count':48,'validation_normal_count':len(val_paths),
    'test_normal_count':int(np.sum(image_labels==0)),'test_stain_count':int(np.sum(image_labels==1)),
    'stain_subset_selection':'100 images selected with Python random seed 230224 before inference',
    'preprocessing':'Complete image resized from 1984x1488 or 1488x1984 to 224x224 RGB',
    't_distance':T_DISTANCE,'image_score':'99.5th percentile of squared residual pixels',
    'threshold':threshold,'threshold_source':'Maximum image score over 10 validation normals',
    'image_auroc':image_auc,'average_precision':average_precision,
    'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
    'sensitivity':sensitivity,'specificity':specificity,'balanced_accuracy':balanced_accuracy,
    'normal_score_mean':float(np.mean(image_scores[image_labels==0])),
    'stain_score_mean':float(np.mean(image_scores[image_labels==1])),
    'median_batch_seconds':float(np.median(test_batch_seconds)),
    'approx_test_seconds_per_image':float(sum(test_batch_seconds)/len(test_paths)),
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'limitations':['Only ten independent normal test images','Seed-fixed 100-image subset of 398 stains',
                   'No pixel masks, so no segmentation Dice/IoU','Whole-image resize differs from 512-pixel training crop scale',
                   'One grayscale stain acquisition domain','One reconstruction per image'],
}
(OUTPUT/'evaluation_report.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(OUTPUT/'test_maps_float16.npz',scores=test_maps.astype(np.float16),image_scores=image_scores,
                    labels=image_labels,predictions=predictions)
archive=shutil.make_archive('/kaggle/working/fabric_stain_evaluation_results','zip',OUTPUT)
print(json.dumps(report,indent=2))
print('\\nLOCKED FABRIC STAIN EVALUATION PASSED')
print('Download:',archive)
"""
)

nb["metadata"]["colab"] = {"name": OUTPUT.name, "provenance": []}
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUTPUT)
