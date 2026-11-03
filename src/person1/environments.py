"""Subprocess environments; native legacy kernels are never silently replaced."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request


def prepare(name, root, log, runner):
    root=Path(root);log=Path(log)
    if name in ('deepjscc','isec'):
        runner([sys.executable,'-m','pip','install','pytorch-msssim==1.0.0','lpips==0.1.4'],log,1200)
        return sys.executable,dict(os.environ)
    if sys.platform!='linux':
        raise RuntimeError('Native TensorFlow/NATTEN GPU environments are qualified on Kaggle Linux only')
    binary=root/'.runtime_deps/micromamba'
    if not binary.is_file():
        binary.parent.mkdir(parents=True,exist_ok=True)
        archive=binary.parent/'micromamba.tar.bz2'
        urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/linux-64/latest',archive)
        with tarfile.open(archive) as contents:
            member=contents.getmember('bin/micromamba')
            with contents.extractfile(member) as incoming,binary.open('wb') as target:target.write(incoming.read())
        binary.chmod(0o755)
    destination=root/'.venv-team'/('person1-'+name)
    python=destination/'bin/python'
    version='3.10' if name=='adjscc' else '3.8'
    if not python.is_file():
        runner([str(binary),'create','-y','-p',str(destination),'-c','conda-forge','python='+version,'pip'],log,1800)
    env=dict(os.environ);env['PYTHONPATH']=str(root);env['SEMANTIC_LOSSLESS_ROOT']=str(root)
    pip=[str(python),'-m','pip','install']
    if name=='adjscc':
        requirements=['tensorflow==2.14.1','tensorflow-compression==2.14.1','tensorflow-probability==0.22.1',
            'numpy==1.26.4','Pillow==10.4.0','nvidia-cuda-runtime-cu11==11.8.89',
            'nvidia-cudnn-cu11==8.7.0.84','nvidia-cublas-cu11==11.11.3.6','nvidia-cufft-cu11==10.9.0.58',
            'nvidia-curand-cu11==10.3.0.86','nvidia-cusolver-cu11==11.4.1.48','nvidia-cusparse-cu11==11.7.5.86',
            'nvidia-cuda-cupti-cu11==11.8.87']
        runner(pip+requirements,log,2400)
        libs=list(destination.glob('lib/python*/site-packages/nvidia/*/lib'))
        env['LD_LIBRARY_PATH']=':'.join(map(str,libs))+':'+env.get('LD_LIBRARY_PATH','')
        env['TF_DETERMINISTIC_OPS']='1'
    else:
        runner(pip+['numpy==1.23.5','Pillow==10.4.0','setuptools<70','wheel'],log,1200)
        runner(pip+['torch==1.13.0+cu117','torchvision==0.14.0+cu117','-f','https://download.pytorch.org/whl/torch_stable.html'],log,2400)
        runner(pip+['timm==0.4.12','compressai==1.2.0','configargparse==1.7','wandb==0.16.6',
                    'einops==0.4.1','PyYAML==6.0.2','pypng==0.20220715.0'],log,1800)
        # The original wheel index has an expired TLS certificate (measured
        # 2026-10-28). Use the exact official release asset with a verified hash.
        # Measured v12 hit Torch1.9's strict deterministic indexing assertion.
        # Retain NATTEN0.14.6 using its official Torch1.13 build.
        wheel='https://github.com/SHI-Labs/NATTEN/releases/download/v0.14.6/natten-0.14.6%2Btorch1130cu117-cp38-cp38-linux_x86_64.whl'
        runner(pip+['-v',wheel+'#sha256=ae0d892f5914d5a9b1d93a29e3fb13f3ea2c3aad0c0ecd4ea555eec46aa10577'],log,1200)
        env.update(MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
    return str(python),env
