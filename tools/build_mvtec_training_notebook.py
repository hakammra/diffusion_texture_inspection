from __future__ import annotations
import json
from pathlib import Path

root = Path(r'C:\Users\abdul\Documents\LabelInspect')
source = root/'notebooks'/'04_mvtec_leather_pilot_train.ipynb'
notebook = json.loads(source.read_text(encoding='utf-8'))
replacements = {
    'Stage 3B — MVTec leather DTU-Net/Tsimplex training pilot': 'Stage 3C — MVTec leather DTU-Net/Tsimplex training continuation',
    'This is a **reduced reproduction pilot**, based on Kumar et al., WACV 2025. It trains the authors’ DTU-Net/Tsimplex path on normal MVTec AD leather images for 100 optimizer steps and saves a resumable checkpoint.': 'This is a **reduced reproduction training run**, based on Kumar et al., WACV 2025. It resumes the verified 100-step pilot and trains the authors’ DTU-Net/Tsimplex path to 2,000 total optimizer steps.',
    'The purpose of this run is to measure stable loss, speed, and memory on the available Tesla T4 before selecting a defensible longer schedule. It is not a reproduced paper result and does not evaluate the official test set.': 'This run uses approximately 20 image-pass equivalents over the 196-image normal training partition. It remains a reduced experiment rather than a reproduced paper result, and it does not access the official test set.',
    "OUTPUT = WORK/'mvtec_leather_pilot'": "OUTPUT = WORK/'mvtec_leather_training'",
    'TARGET_STEPS = 100       # Increase later; this first run is a timing pilot.': 'TARGET_STEPS = 2000      # Total steps, including the completed 100-step pilot.',
    'SAVE_EVERY = 25': 'SAVE_EVERY = 250',
    "print('Starting a new pilot.')": "print('No earlier checkpoint found; starting the reduced training run from step 0.')",
    "title='Reduced MVTec leather pilot'": "title='Reduced MVTec leather training'",
    "'scope':'reduced_training_pilot_only'": "'scope':'reduced_training_run'",
    "'next_decision':'Choose longer target step count from measured throughput, then validate reconstruction on held-out normal images.'": "'next_decision':'Validate partial reconstructions on held-out normal images, calibrate without test masks, then evaluate once on the official test set.'",
    "str(WORK/'mvtec_leather_pilot_results')": "str(WORK/'mvtec_leather_training_results')",
    "print('\\nTRAINING PILOT PASSED')": "print('\\nTRAINING RUN PASSED')",
}
for cell in notebook['cells']:
    updated=[]
    for piece in cell.get('source',[]):
        for old,new in replacements.items():
            piece=piece.replace(old,new)
        updated.append(piece)
    cell['source']=updated
notebook['metadata']['colab']['name']='05_mvtec_leather_train.ipynb'
out=root/'notebooks'/'05_mvtec_leather_train.ipynb'
out.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
print(out)
