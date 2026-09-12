"""Run a bounded integration check of the author DTU-Net and Tsimplex code.

This is not training for anomaly detection, not a reproduced result, and not a
performance comparison. Downloads only five source files from a pinned commit.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import random
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

COMMIT = 'dc4a9bd2a2a5b1c31223daab4bdfea3f6a5b2990'
BASE_URL = f'https://raw.githubusercontent.com/MAXNORM8650/Annotsim/{COMMIT}/'
FILES = ['src/models/UModels/UDHVT.py','GaussianDiffusion.py',
         'utils/Simplex/constants.py','utils/Simplex/internals.py','utils/Simplex/noise.py']
EXPECTED = {
 'utils/Simplex/constants.py':'52bf6ba3e2c0d386fa420382de380093a8dd61f488765cb812b13e25d0be7294',
 'utils/Simplex/internals.py':'ef67562885dcfe3356acd97784fe10660bf21238be7bcc608e86053c529fd61a',
 'src/models/UModels/UDHVT.py':'f7303c4dd228a3f5e1ab98d16fe1db7c7abfecfa97f683e89449128c5a03a4c2',
 'GaussianDiffusion.py':'cdf7a2143a441d20a3250458c0683c53ac1484ef8f0d0927831e66a34b52ec9a',
 'utils/Simplex/noise.py':'d114b6898369a0299e48f95fe4165fb3d587dcb8d7e257b58aef02077b6249d3',
}


def fetch_sources(root):
    hashes={}
    for name in FILES:
        destination=root/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            print('Downloading',name,flush=True)
            last_error=None
            for attempt in range(1,4):
                try:
                    with urllib.request.urlopen(BASE_URL+name,timeout=45) as response:
                        payload=response.read()
                    break
                except urllib.error.URLError as exc:
                    last_error=exc
                    print(f'Network attempt {attempt}/3 failed: {exc}',flush=True)
                    if attempt < 3:time.sleep(2*attempt)
            else:
                raise RuntimeError(
                    'Could not download the pinned public author files. In Kaggle, '
                    'open Settings, turn Internet on, then rerun this cell.'
                ) from last_error
            if name in EXPECTED and hashlib.sha256(payload).hexdigest()!=EXPECTED[name]:
                raise ValueError(f'Inspected-source hash mismatch: {name}; stop and review this revision.')
            destination.write_bytes(payload)
        digest=hashlib.sha256(destination.read_bytes()).hexdigest()
        if name in EXPECTED and digest!=EXPECTED[name]:
            raise ValueError(f'Cached-source hash mismatch: {name}; use a fresh cache after review.')
        hashes[name]=digest
    return hashes


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_author_components(source_root,output):
    import numpy as np
    import torch
    import torch.nn as nn
    # Namespace isolation avoids importing the repository's unrelated experiments.
    package=types.ModuleType('labelinspect_author_simplex')
    package.__path__=[str(source_root/'utils/Simplex')]
    sys.modules[package.__name__]=package
    noise=load_module(package.__name__+'.noise',source_root/'utils/Simplex/noise.py')

    original=(source_root/'src/models/UModels/UDHVT.py').read_text(encoding='utf8')
    replacements={
      'from torchvision import models':'# Removed unused torchvision.models import.',
      'from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD':'# Removed unused timm image constants.',
      'from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv':'from timm.models._manipulate import named_apply',
      'from timm.models.layers import trunc_normal_, lecun_normal_, to_2tuple':'from timm.layers import trunc_normal_, lecun_normal_, to_2tuple',
      'from timm.models.registry import register_model':'# Removed unused timm registry import.',
    }
    for old,new in replacements.items():
        if original.count(old)!=1:raise ValueError('Compatibility patch no longer matches the inspected source: '+old)
        original=original.replace(old,new)
    patched=output/'author_UDHVT_compat.py'
    patched.write_text('import numpy as np\n'+original,encoding='utf8')
    model_module=load_module('labelinspect_author_model',patched)

    # Keep the author definitions, including its variance convention. Avoid the
    # top-level imports for unused image losses, plotting, datasets and backbones.
    tree=ast.parse((source_root/'GaussianDiffusion.py').read_text(encoding='utf8'))
    wanted={'get_beta_schedule','extract','mean_flat','generate_simplex_4noise','GaussianDiffusionModel'}
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in wanted]
    if {node.name for node in selected}!=wanted:raise ValueError('Required author diffusion definitions are missing')
    reduced=ast.Module(body=selected,type_ignores=[])
    namespace={'np':np,'torch':torch,'nn':nn,'OpenSimplex':noise.OpenSimplex}
    exec(compile(reduced,'author_diffusion_l2_subset.py','exec'),namespace)
    (output/'author_diffusion_l2_subset.py').write_text(ast.unparse(reduced),encoding='utf8')

    class NoisePredictionAdapter(nn.Module):
        def __init__(self,backbone):super().__init__();self.backbone=backbone
        def forward(self,x,t,y=None):
            if y is not None:raise ValueError('This initial adapter supports the normal-only, unconditioned path')
            result=self.backbone(x,t,y=None)
            prediction=result[0] if isinstance(result,tuple) else result
            if prediction.shape!=x.shape:raise ValueError('Noise prediction does not match input shape')
            return prediction

    return model_module,namespace,NoisePredictionAdapter,{
      'imports':replacements,'extra_import':'numpy for the author PositionalEmbedding helper',
      'adapter':'Select tuple element 0; preserve the backbone computation.',
      'diffusion_loading':'AST-load only the author definitions required for Gaussian/Tsimplex L2 and sampling; other losses are not supported.',
      'sampling':'Pass denoise_fn=noise_fn so reverse steps use configured O/mu/p instead of the author alternate branch defaults.',
      'calling_convention':'Set author diffusion train=False to select model(x,t,y=lab) during sampling. This is a dispatch flag; the model is explicitly switched with model.train()/eval().',
    }


def synthetic_batch(size,batch,device):
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont
    im=Image.new('L',(size,size),235);draw=ImageDraw.Draw(im)
    try:font=ImageFont.truetype('DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default(size=20)
    draw.rectangle((12,12,size-12,size-12),outline=20,width=2)
    draw.text((24,45),'LABEL A-104',font=font,fill=20)
    draw.text((24,90),'BATCH 2026',font=font,fill=20)
    x=torch.from_numpy(np.asarray(im).copy()).float()/127.5-1
    return x[None,None].repeat(batch,3,1,1).to(device)


def save_preview(x,reconstructed,destination):
    from PIL import Image,ImageDraw
    import numpy as np
    images=[]
    for tensor in [x,reconstructed]:
        array=((tensor[0].detach().float().cpu().permute(1,2,0).numpy()+1)/2*255).clip(0,255).astype(np.uint8)
        images.append(Image.fromarray(array))
    sheet=Image.new('RGB',(520,302),'white');draw=ImageDraw.Draw(sheet)
    draw.text((12,10),'INTEGRATION CHECK ONLY - TWO UPDATES',fill='darkred')
    draw.text((12,32),'Synthetic input',fill='black');draw.text((268,32),'8-step reconstruction',fill='black')
    for i,im in enumerate(images):sheet.paste(im.resize((224,224)),(12+i*256,54))
    draw.text((12,283),'No anomaly-removal or accuracy claim.',fill='darkred');sheet.save(destination)


def run(output,cache=None):
    import importlib.metadata
    import numpy as np
    import torch
    import numba
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('Select a GPU accelerator before running this notebook')
    cache=Path(cache) if cache else output/'upstream'/COMMIT
    hashes=fetch_sources(cache)
    random.seed(230224);np.random.seed(230224);torch.manual_seed(230224)
    numba.set_num_threads(min(2,numba.get_num_threads()))
    module,ns,adapter_type,patches=load_author_components(cache,output)
    config={'img_size':224,'patch_size':16,'in_chans':3,'embed_dim':384,'depth':12,
            'num_heads':6,'mlp_ratio':4.,'num_classes':None,'mlp_time_embed':True,
            'use_dec':['DAFF','DAFF','DAFF'],'PE_type':'SPE','refinement':True,'qkv_bias':False}
    report={'status':'running','scope':'integration_check_only','upstream_commit':COMMIT,
            'source_sha256':hashes,'compatibility_changes':patches,'model_configuration':config,
            'configuration_note':'Illustrated SPE/DMHA/HFF/refinement variant. Code depth=12 gives six encoder blocks, one middle, six decoder blocks. This is not asserted to match every paper table.',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
            'gpu_vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
            'packages':{n:importlib.metadata.version(n) for n in ['timm','einops','numba','numpy']},
            'noise_parameters':{'octave':6,'frequency':64,'persistence':.9},
            'training_steps':2,'batch_size':2,'total_diffusion_steps':1000,'reconstruction_steps':8}
    (output/'integration_report.json').write_text(json.dumps(report,indent=2))
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats()
    print('Building author DTU-Net, width 384, six attention heads...',flush=True)
    backbone=module.UDHVT(**config).to(device);model=adapter_type(backbone)
    report['parameter_count']=sum(p.numel() for p in model.parameters())
    x=synthetic_batch(224,2,device)
    t=torch.tensor([50,150],device=device,dtype=torch.long)
    diffusion=ns['GaussianDiffusionModel']([224,224],ns['get_beta_schedule'](1000,'cosine'),img_channels=3,
                 loss_type='l2',noise='4dsimplex',octave=6,frequency=64,persistence=.9,train=False)
    print('Compiling the author 4D noise function on CPU; first use may take a few minutes...',flush=True)
    start=time.perf_counter()
    probe=torch.zeros(1,1,4,4,device=device)
    diffusion.noise_fn(probe,torch.tensor([5],device=device))
    report['noise_first_compile_seconds']=time.perf_counter()-start
    noise=diffusion.noise_fn(x,t).float()
    assert noise.shape==x.shape and torch.isfinite(noise).all()
    assert not torch.allclose(noise[0],noise[1]),'Different time coordinates unexpectedly generated identical samples'
    report['noise_shape']=list(noise.shape)
    report['noise_mean']=float(noise.mean());report['noise_std']=float(noise.std())
    report['noise_normalization']='Author raw amplitude retained; no per-sample standardization.'
    report['batch_noise_note']='The author generator uses t as the fourth coordinate. Duplicate time coordinates with one seed can produce identical noise across batch entries; this remains to be assessed during training.'
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
    report['losses']=[];report['gradient_norms']=[]
    tracked=backbone.pos_embed.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses,noisy,predicted=diffusion.calc_loss(model,x,None,t)
        loss=losses['loss'].mean()
        assert predicted.shape==x.shape and torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        report['losses'].append(float(loss.detach()))
        report['gradient_norms'].append(float(norm))
        print(f'Update {step+1}/2 passed; L2 noise loss {float(loss.detach()):.6f}',flush=True)
    assert not torch.equal(tracked,backbone.pos_embed.detach()),'Optimizer did not change the tracked parameter'
    report['tracked_parameter_changed']=True
    report['parameters_without_grad']=[name for name,p in model.named_parameters() if p.grad is None]
    model.eval()
    print('Checking eight author reverse-diffusion steps. This model is not trained for detection.',flush=True)
    with torch.no_grad():
        result=diffusion.forward_backward(model,x[:1],None,see_whole_sequence=None,t_distance=8,denoise_fn='noise_fn')
    assert result.shape==x[:1].shape and torch.isfinite(result).all()
    residual=(x[:1]-result).square().mean(dim=1)
    assert residual.shape==(1,224,224) and torch.isfinite(residual).all()
    save_preview(x,result,output/'integration_preview.png')
    report['reconstruction_shape']=list(result.shape);report['residual_shape']=list(residual.shape)
    report['peak_gpu_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    report['peak_gpu_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    report['status']='passed'
    report['not_completed']=['Training a useful anomaly model','Real label dataset','Paper metrics reproduction','Quality comparison with CPU baseline']
    (output/'integration_report.json').write_text(json.dumps(report,indent=2))
    print('\nDTU-NET + TSIMPLEX INTEGRATION CHECK PASSED',flush=True)
    print(json.dumps({k:report[k] for k in ['parameter_count','noise_shape','losses','reconstruction_shape','peak_gpu_allocated_gib','peak_gpu_reserved_gib']},indent=2))
    print('Saved:',output/'integration_report.json',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/author_integration')
    parser.add_argument('--cache',default=None)
    args=parser.parse_args();run(args.output,args.cache)
