"""Two fresh processes, same seed and unchanged GPU training settings."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deterministic',action='store_true');args=parser.parse_args()
    assert torch.cuda.is_available(),'GPU required'
    out=Path('reports/training/gpu-deterministic-01' if args.deterministic else 'reports/training/gpu-reproducibility-01');out.mkdir(exist_ok=False)
    root=Path('data/prepared/helmet-separate-v2-expanded-04')
    lock=json.loads((root/'dataset.lock.json').read_text())
    for name,h in lock.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
    prefix='detector-gpu-deterministic1950' if args.deterministic else 'detector-gpu-repro1950'
    jobs=[{'run':f'{prefix}-s20261124-r{i}','dataset':str(root),'seed':20261124,'images':50,'repeat':i} for i in (1,2)]
    environment=os.environ.copy()
    if args.deterministic:environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='20261124')
    assert all(not (Path('runs')/j['run']).exists() for j in jobs)
    protocol={'jobs':jobs,'optimizer_updates':1950,'checkpoint':'last.pt','initialization':'random','pretrained':False,
              'sampler':'original; no reservation or changed quotas','test_used':False,'negative_exposure_logging':True,
              'trace':'Inputs, targets, CPU/CUDA/Python/NumPy RNG, losses and actual sampler masks each step; gradients before clipping and model state first10 steps then every25; first-step.pt holds updated model and clipped gradients.',
              'primary':'Bitwise initial, input, RNG, masks, gradient and state equality; first divergence, then final metrics.',
              'gpu_flags':'Observed current defaults, no deterministic algorithms enabled for this baseline experiment.',
              'environment':{k:environment.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED','CUDA_LAUNCH_BLOCKING')},
              'dataset_lock_sha256':{str(root):hashlib.sha256((root/'dataset.lock.json').read_bytes()).hexdigest()}}
    protocol['deterministic_gpu']=args.deterministic
    if args.deterministic:
        protocol['gpu_flags']='Strict deterministic algorithms, warn_only=False, cudnn deterministic=True/benchmark=False, TF32 disabled; eager torchvision reference RoIAlign. New condition, unchanged original sampler.'
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for job in jobs:
        command=[str(Path('.venv/Scripts/python.exe').resolve()),'training/train.py','--task','detector','--separate-classes',
                 '--dataset',str(root),'--epochs','150','--batch-size','2','--workers','0','--seed','20261124',
                 '--constant-lr','--max-updates','1950','--head-probe-every','10','--photometric','--feature-stride','8',
                 '--proposal-multiplier','4','--log-negative-exposure','--log-reproducibility','--run',job['run']]
        print('START '+job['run'],flush=True)
        if args.deterministic:command.append('--deterministic-gpu')
        with (out/(job['run']+'.log')).open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=environment)
        result=json.loads((Path('runs')/job['run']/'result.json').read_text());assert result['optimizer_updates']==1950
        print('DONE '+job['run'],flush=True)


if __name__=='__main__':main()
