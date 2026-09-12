from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "05_mvtec_leather_train.ipynb"
OUTPUT = ROOT / "notebooks" / "07_fabric_stain_pilot_train_kaggle.ipynb"


def lines(text: str):
    return [line + "\n" for line in text.strip("\n").splitlines()]


nb = json.loads(SOURCE.read_text(encoding="utf-8"))
nb["cells"][0]["source"] = lines(
    """
# Fabric stain DTU-Net/Tsimplex training pilot

This notebook trains the paper authors' DTU-Net/Tsimplex path for 100 optimizer
steps using only normal fabric images. It is a pipeline and timing pilot, not a
completed detector and not a reproduced paper result.

Attach `fabric_stain_pilot.zip` as a Kaggle notebook input and enable a GPU and
Internet. The fixed data protocol is 48 training normals, 10 validation normals,
10 untouched test normals, and 398 test-only stain images. Defective images are
never loaded during training.

The selected stain cohort has no pixel masks. Later evaluation will report
image-level scores and qualitative anomaly maps; a labelled subset is required
for Dice and IoU.
"""
)

nb["cells"][1]["source"] = lines(
    """
from pathlib import Path
import importlib.util, json, os, random, shutil, subprocess, sys, time, zipfile

import numpy as np
import torch
from PIL import Image

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT = WORK/'fabric_stain_pilot_training'
OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = WORK/'checkpoints'/'fabric_stain'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def find_dataset(search_root):
    candidates=[]
    for path in search_root.rglob('fabric_stain_pilot'):
        if path.is_dir() and (path/'train_normal').is_dir() and (path/'val_normal').is_dir():
            candidates.append(path)
    return sorted(set(candidates))

dataset_candidates=find_dataset(KAGGLE_INPUT)
if not dataset_candidates:
    matching=[]
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            names=['/'+item.filename.replace('\\\\','/').lstrip('/') for item in archive.infolist()]
            if any('/fabric_stain_pilot/train_normal/' in name for name in names):
                matching.append(archive_path)
    if len(matching)==1:
        extraction_root=WORK/'uploaded_data'
        extraction_root.mkdir(parents=True,exist_ok=True)
        resolved=extraction_root.resolve()
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                target=(extraction_root/item.filename).resolve()
                if target != resolved and resolved not in target.parents:
                    raise ValueError(f'Unsafe ZIP member: {item.filename}')
            archive.extractall(extraction_root)
        dataset_candidates=find_dataset(extraction_root)
        print('Extracted:',matching[0])
if len(dataset_candidates)!=1:
    raise FileNotFoundError('Expected one fabric_stain_pilot dataset, found: '+repr([str(p) for p in dataset_candidates]))
DATA_ROOT=dataset_candidates[0]

SEED = 230224
TARGET_STEPS = 100
BATCH_SIZE = 2
SAVE_EVERY = 25
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
CROP_SIZE = 512
print('GPU:', torch.cuda.get_device_name(0))
print('Dataset:', DATA_ROOT)
print('Working output:', WORK)
"""
)

nb["cells"][5]["source"] = lines(
    """
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png'}
def image_paths(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

train_paths=image_paths(DATA_ROOT/'train_normal')
val_paths=image_paths(DATA_ROOT/'val_normal')
test_good_paths=image_paths(DATA_ROOT/'test'/'good')
test_stain_paths=image_paths(DATA_ROOT/'test'/'stain')
assert (len(train_paths),len(val_paths),len(test_good_paths),len(test_stain_paths)) == (48,10,10,398)
assert not (set(train_paths)&set(val_paths) or set(train_paths)&set(test_good_paths))
print('Training normals:',len(train_paths))
print('Held-out validation normals:',len(val_paths))
print('Untouched test:',len(test_good_paths),'normal +',len(test_stain_paths),'stain')

RESAMPLE=getattr(Image,'Resampling',Image).BILINEAR
FLIP=getattr(Image,'Transpose',Image).FLIP_LEFT_RIGHT

def load_training_crop(path,rng):
    with Image.open(path) as image:
        image=image.convert('RGB')
        width,height=image.size
        crop=min(CROP_SIZE,width,height)
        left=int(rng.integers(0,width-crop+1))
        top=int(rng.integers(0,height-crop+1))
        image=image.crop((left,top,left+crop,top+crop))
        if float(rng.random())<0.5:
            image=image.transpose(FLIP)
        image=image.resize((IMAGE_SIZE,IMAGE_SIZE),RESAMPLE)
        array=np.asarray(image,dtype=np.float32).copy()/127.5-1.0
    return torch.from_numpy(array).permute(2,0,1)

def deterministic_batch(step):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,step]))
    indices=rng.choice(len(train_paths),size=BATCH_SIZE,replace=False)
    return torch.stack([load_training_crop(train_paths[int(i)],rng) for i in indices])

def deterministic_times(step):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,step,1]))
    values=rng.integers(1,1000,size=BATCH_SIZE,dtype=np.int64)
    if len(set(values.tolist()))!=len(values):
        values[1]=(values[0]%999)+1
    return torch.from_numpy(values)

sample=deterministic_batch(0)
assert sample.shape==(BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE)
assert torch.isfinite(sample).all() and sample.min()>=-1 and sample.max()<=1
print('Training batch check:',tuple(sample.shape),float(sample.min()),float(sample.max()))
"""
)

# Keep the verified model construction cell and replace only the checkpoint logic.
cell6 = "".join(nb["cells"][6]["source"])
start = cell6.index("latest = CHECKPOINT_DIR/'latest.pt'")
prefix = cell6[:start]
checkpoint_logic = """
latest = CHECKPOINT_DIR/'fabric_stain_latest.pt'
all_pt_files = sorted(KAGGLE_INPUT.rglob('*.pt'))
if not all_pt_files:
    # Kaggle may unpack a PyTorch .pt file because torch.save uses a ZIP container.
    # Recreate that container when its extracted data.pkl/data/version structure is visible.
    extracted_roots=[]
    for data_pickle in KAGGLE_INPUT.rglob('data.pkl'):
        candidate=data_pickle.parent
        if (candidate/'data').is_dir() and (candidate/'version').is_file():
            extracted_roots.append(candidate)
    extracted_roots=sorted(set(extracted_roots))
    if len(extracted_roots)==1:
        archive_root=extracted_roots[0]
        rebuilt=WORK/'rebuilt_fabric_stain_checkpoint.pt'
        with zipfile.ZipFile(rebuilt,'w',compression=zipfile.ZIP_STORED) as archive:
            for source_file in sorted(archive_root.rglob('*')):
                if source_file.is_file():
                    relative=source_file.relative_to(archive_root).as_posix()
                    archive.write(source_file,f'{archive_root.name}/{relative}')
        all_pt_files=[rebuilt]
        print('Rebuilt Kaggle-extracted PyTorch checkpoint:',rebuilt)
    elif len(extracted_roots)>1:
        raise ValueError('More than one extracted PyTorch checkpoint is attached: '+repr([str(p) for p in extracted_roots]))
preferred = [path for path in all_pt_files if path.name == 'fabric_stain_latest.pt']
previous = preferred if preferred else (all_pt_files if len(all_pt_files) == 1 else [])
print('Checkpoint files visible under /kaggle/input:', [str(path) for path in all_pt_files])
start_step = 0
history = []
if len(previous)>1:
    raise ValueError('More than one possible checkpoint is attached. Keep only fabric_stain_latest.pt: '+repr([str(p) for p in previous]))
if len(previous)==1:
    shutil.copy2(previous[0],latest)
if latest.is_file():
    checkpoint=torch.load(latest,map_location=device,weights_only=False)
    if checkpoint.get('author_commit')!=author.COMMIT:
        raise ValueError('Checkpoint author commit does not match this notebook.')
    if checkpoint.get('model_config')!=model_config:
        raise ValueError('Checkpoint model configuration does not match this notebook.')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step=int(checkpoint['step'])
    history=list(checkpoint.get('history',[]))
    print('Resuming from step',start_step)
else:
    print('Starting a new 100-step fabric pilot.')
if start_step>TARGET_STEPS:
    raise ValueError(f'Checkpoint is at step {start_step}; set TARGET_STEPS to at least that value.')

probe=torch.zeros(1,1,4,4,device=device)
diffusion.noise_fn(probe,torch.tensor([5],device=device))
torch.cuda.reset_peak_memory_stats()
"""
nb["cells"][6]["source"] = lines(prefix + checkpoint_logic)

nb["cells"][7]["source"] = lines(
    """
def save_checkpoint(step):
    payload={
        'step':step,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'history':history,
        'author_commit':author.COMMIT,
        'source_sha256':hashes,
        'model_config':model_config,
        'noise_parameters':{'octave':6,'frequency':64,'persistence':0.9},
        'seed':SEED,
        'dataset_category':'fabric_stain_pilot',
        'protocol':'48 normal train; 10 normal validation; 10 normal + 398 stain locked test',
        'crop_size':CROP_SIZE,
    }
    temporary=CHECKPOINT_DIR/'fabric_stain_latest.tmp.pt'
    torch.save(payload,temporary)
    temporary.replace(latest)

model.train()
run_started=time.perf_counter()
step_times=[]
for step in range(start_step,TARGET_STEPS):
    step_started=time.perf_counter()
    x=deterministic_batch(step).to(device,non_blocking=True)
    t=deterministic_times(step).to(device)
    optimizer.zero_grad(set_to_none=True)
    losses,noisy,predicted=diffusion.calc_loss(model,x,None,t)
    loss=losses['loss'].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite loss at step {step+1}: {loss}')
    loss.backward()
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f'Non-finite gradient norm at step {step+1}')
    optimizer.step()
    elapsed=time.perf_counter()-step_started
    step_times.append(elapsed)
    history.append({'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad_norm),'seconds':elapsed})
    if (step+1)%10==0 or step==start_step:
        recent=np.mean([item['loss'] for item in history[-10:]])
        print(f'Step {step+1:4d}/{TARGET_STEPS} | loss {float(loss):.6f} | recent mean {recent:.6f} | {elapsed:.2f}s')
    if (step+1)%SAVE_EVERY==0 or step+1==TARGET_STEPS:
        save_checkpoint(step+1)

run_seconds=time.perf_counter()-run_started
print('Checkpoint:',latest)
print('Steps completed this run:',TARGET_STEPS-start_step)
"""
)

nb["cells"][8]["source"] = lines(
    """
import csv
import matplotlib.pyplot as plt

with (OUTPUT/'training_history.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['step','loss','grad_norm','seconds'])
    writer.writeheader(); writer.writerows(history)

loss_values=[item['loss'] for item in history]
window=min(20,len(loss_values))
smoothed=np.convolve(loss_values,np.ones(window)/window,mode='valid') if window else []
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(range(1,len(loss_values)+1),loss_values,alpha=.35,label='step loss')
if len(smoothed):
    ax.plot(range(window,len(loss_values)+1),smoothed,linewidth=2,label=f'{window}-step mean')
ax.set(xlabel='Optimizer step',ylabel='L2 noise-prediction loss',title='Fabric stain training pilot')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(OUTPUT/'loss_curve.png',dpi=160)
plt.show()

report={
    'status':'passed',
    'scope':'fabric_stain_training_pilot',
    'paper_result_reproduced':False,
    'dataset':'Fabric Defects Dataset - matched grayscale stain cohort',
    'train_normal_count':len(train_paths),
    'held_out_val_normal_count':len(val_paths),
    'locked_test_good_count':len(test_good_paths),
    'locked_test_stain_count':len(test_stain_paths),
    'test_images_used':0,
    'author_commit':author.COMMIT,
    'model_parameters':sum(p.numel() for p in model.parameters()),
    'target_steps':TARGET_STEPS,
    'start_step':start_step,
    'steps_completed_this_run':TARGET_STEPS-start_step,
    'initial_loss':history[0]['loss'] if history else None,
    'final_loss':history[-1]['loss'] if history else None,
    'last_20_mean_loss':float(np.mean(loss_values[-20:])) if loss_values else None,
    'median_step_seconds_this_run':float(np.median(step_times)) if step_times else None,
    'run_seconds':run_seconds,
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'gpu':torch.cuda.get_device_name(0),
    'checkpoint':str(latest),
    'next_decision':'Inspect pilot stability, then continue training before locked image-level evaluation.',
}
(OUTPUT/'pilot_report.json').write_text(json.dumps(report,indent=2))
shutil.copy2(latest,Path('/kaggle/working/fabric_stain_latest.pt'))
archive=shutil.make_archive('/kaggle/working/fabric_stain_pilot_results','zip',OUTPUT)
print(json.dumps(report,indent=2))
print('\\nFABRIC TRAINING PILOT PASSED')
print('Download checkpoint: /kaggle/working/fabric_stain_latest.pt')
print('Download result bundle:',archive)
"""
)

nb["metadata"].setdefault("kaggle", {})
nb["metadata"]["colab"] = {"name": OUTPUT.name, "provenance": []}
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUTPUT)
