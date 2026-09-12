from __future__ import annotations
import json
from pathlib import Path

root=Path(r'C:\Users\abdul\Documents\LabelInspect')
source=root/'notebooks'/'06_mvtec_leather_evaluate.ipynb'
nb=json.loads(source.read_text(encoding='utf-8'))
nb['cells'][0]['source']=[line+'\n' for line in '''# Stage 3D — MVTec leather validation and locked test evaluation (Kaggle)

This Kaggle version loads the completed 2,000-step checkpoint and performs the first reduced evaluation.

Before running, attach two notebook inputs:
1. The official MVTec AD data containing `leather/train/good`, `leather/test`, and `leather/ground_truth`.
2. A private dataset containing the Google Drive checkpoint file `latest.pt`.

Protocol decisions are fixed before reading test masks:
- `t_distance = 250`, following the fixed-time author evaluation routine and the paper's partial-diffusion analysis.
- The binary pixel threshold is the 99.5th percentile of reconstruction residuals across 49 held-out normal validation images.
- The official test set is evaluated once after the threshold is frozen.
- Training and validation use no anomalous images or masks.

This is a one-category, 2,000-step reduced experiment rather than a reproduction of the paper's 3,000-epoch result. Keep Kaggle Internet enabled so the notebook can obtain the five pinned, hash-verified author source files.'''.splitlines()]

setup='''from pathlib import Path
import csv, importlib.util, json, os, random, shutil, subprocess, sys, time, zipfile
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT = Path('/kaggle/input')
checkpoint_candidates = sorted(KAGGLE_INPUT.rglob('latest.pt'))
if len(checkpoint_candidates) != 1:
    raise FileNotFoundError(
        'Expected exactly one latest.pt below /kaggle/input, found: '
        + repr([str(path) for path in checkpoint_candidates])
    )
CHECKPOINT = checkpoint_candidates[0]

def find_leather_categories(search_root):
    found = []
    for good_dir in search_root.rglob('good'):
        if not good_dir.is_dir() or good_dir.parent.name != 'train':
            continue
        candidate = good_dir.parent.parent
        if candidate.name == 'leather' and (candidate/'test').is_dir() and (candidate/'ground_truth').is_dir():
            found.append(candidate)
    return sorted(set(found))

category_candidates = find_leather_categories(KAGGLE_INPUT)
if not category_candidates:
    matching_archives = []
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            normalized = ['/' + item.filename.replace('\\\\','/').lstrip('/') for item in archive.infolist()]
            if any('/leather/train/good/' in name for name in normalized):
                matching_archives.append(archive_path)
    if len(matching_archives) == 1:
        extraction_root = Path('/kaggle/working/uploaded_leather')
        extraction_root.mkdir(parents=True, exist_ok=True)
        resolved_root = extraction_root.resolve()
        with zipfile.ZipFile(matching_archives[0]) as archive:
            for item in archive.infolist():
                target = (extraction_root/item.filename).resolve()
                if target != resolved_root and resolved_root not in target.parents:
                    raise ValueError(f'Unsafe ZIP member: {item.filename}')
            archive.extractall(extraction_root)
        print('Extracted leather archive:', matching_archives[0])
        category_candidates = find_leather_categories(extraction_root)
    elif len(matching_archives) > 1:
        raise FileNotFoundError('More than one uploaded ZIP contains leather data: ' + repr([str(p) for p in matching_archives]))
if len(category_candidates) != 1:
    raise FileNotFoundError(
        'Expected exactly one leather category below /kaggle/input, found: '
        + repr([str(path) for path in category_candidates])
    )
CATEGORY_ROOT = category_candidates[0]
DATA_ROOT = CATEGORY_ROOT.parent

WORK = Path('/kaggle/working/labelinspect')
OUTPUT = WORK/'mvtec_leather_evaluation'
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 230224
IMAGE_SIZE = 224
BATCH_SIZE = 4
T_DISTANCE = 250
TARGET_NORMAL_PIXEL_FPR = 0.005
print('GPU:', torch.cuda.get_device_name(0))
print('Leather category:', CATEGORY_ROOT)
print('Checkpoint:', CHECKPOINT)
print('Output:', OUTPUT)
'''
nb['cells'][1]['source']=[line+'\n' for line in setup.splitlines()]
# Make the already located category explicit and update the download message.
for cell in nb['cells']:
    joined=''.join(cell.get('source',[]))
    joined=joined.replace("category_root=locate_category(DATA_ROOT)","category_root=CATEGORY_ROOT")
    joined=joined.replace("print('Download:',archive)","print('Download from Kaggle working files:',archive)")
    cell['source']=joined.splitlines(keepends=True)
nb['metadata']['colab']['name']='06_mvtec_leather_evaluate_kaggle.ipynb'
out=root/'notebooks'/'06_mvtec_leather_evaluate_kaggle.ipynb'
out.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print(out)
