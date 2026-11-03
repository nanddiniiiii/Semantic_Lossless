"""Read-only trained-weight CPU inference timing; run each framework separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import zipfile

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(function):
    for _ in range(3):function()
    values=[]
    for _ in range(10):
        start=time.perf_counter();function();values.append(time.perf_counter()-start)
    return dict(repeats_seconds=values,p50_seconds=float(np.median(values)),
                p95_seconds=float(np.quantile(values,.95)),mean_seconds=statistics.mean(values))


def hardware():
    return dict(platform=platform.platform(),processor=platform.processor(),machine=platform.machine(),
                logical_cpu_count=os.cpu_count(),cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').is_file() else None,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                python=sys.version,environment={k:os.environ.get(k) for k in
                    ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')})


def main():
    p=argparse.ArgumentParser();p.add_argument('framework',choices=['deepjscc','adjscc'])
    p.add_argument('--asset',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--expected-sha',required=True);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    before=sha(a.checkpoint)
    if before!=a.expected_sha:raise ValueError('Checkpoint hash mismatch')
    image=np.load(a.input)
    if image.shape!=(1,512,768,3) or image.dtype!=np.float32:raise ValueError('Expected shared float32 real Kodak image1x512x768x3')
    config=json.loads((a.asset/'config.json').read_text())
    result=dict(model=a.framework,hardware=hardware(),checkpoint_sha256=before,input_sha256=sha(a.input),
                input_shape=list(image.shape),precision='float32',device='CPU',warmup=3,repeats=10,
                intraop_threads=2,interop_threads=1,execution='native eager inference; no gradients; no training',
                endpoint='includes synchronous output numpy materialization/copy; excludes input decode, checkpoint load and model build',
                component_note='independent timed calls with fixed precomputed inputs; component medians need not sum to end-to-end',
                timings={})
    if a.framework=='deepjscc':
        import torch,importlib.util
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(22)
        source=a.asset/'paper_deepjscc.py'
        spec=importlib.util.spec_from_file_location('timing_frozen_deep',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if saved['progress']['completed_epochs']!={'codec':150}:raise ValueError('DeepJSCC training incomplete')
        if sha(source)!=saved['options']['runner_sha256']['paper_deepjscc.py']:raise ValueError('DeepJSCC source differs from trained runtime')
        model=module.DeepJSCC(config).cpu().eval();model.load_state_dict(saved['state_dict'],strict=True)
        if any(v.dtype!=torch.float32 for v in model.parameters()):raise ValueError('Unexpected model precision')
        x=torch.from_numpy(image).permute(0,3,1,2).contiguous()
        def channel(z):
            flat=z.flatten(1);k=flat.shape[1]//2
            signal=flat*(k/flat.square().sum(1,keepdim=True).clamp_min(1e-12)).sqrt()
            return (signal+torch.randn_like(signal)*(10**(-model.snr/20)/2**.5)).reshape_as(z)
        def materialize(t):return t.detach().numpy().copy()
        with torch.no_grad():
            torch.manual_seed(22);full=model(x)['reconstruction']
            torch.manual_seed(22);latent=model.encoder(x);received=channel(latent);decomposed=model.decoder(received)
            if not torch.equal(full,decomposed):raise ValueError('Deep channel decomposition differs from native forward')
            functions=dict(encoder=lambda:materialize(model.encoder(x)),channel=lambda:materialize(channel(latent)),
                           decoder=lambda:materialize(model.decoder(received)),end_to_end=lambda:materialize(model(x)['reconstruction']))
            for name,function in functions.items():result['timings'][name]=measure(function)
        result.update(framework_version=torch.__version__,strict_checkpoint_reload=True,trained_epochs=150,
                      model_source_sha256=sha(source),component_equivalence='torch.equal with identical channel RNG',
                      parameters=sum(p.numel() for p in model.parameters()))
    else:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU');tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.utils.set_random_seed(22)
        from src.person1.tf_worker import build
        model,optimizer=build(config,tf.distribute.OneDeviceStrategy('/cpu:0'))
        checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(a.checkpoint) as z:
                if z.testzip() is not None:raise ValueError('AD checkpoint ZIP CRC failure')
                for item in z.infolist():(Path(folder)/item.filename).resolve().relative_to(Path(folder).resolve())
                z.extractall(folder)
            progress=json.loads((Path(folder)/'progress.json').read_text())
            if progress['completed_epochs']!={'codec':150}:raise ValueError('ADJSCC training incomplete')
            checkpoint.read(str(Path(folder)/'state')).assert_consumed()
        x=tf.convert_to_tensor(image*255);snr=tf.constant([[10.]],tf.float32)
        encoder=tf.keras.Model(model.inputs,model.get_layer('en5_conv').output)
        channel=model.get_layer('channel')
        decoder=tf.keras.Model([channel.output,model.inputs[1]],model.output)
        tf.random.set_seed(22);full=model([x,snr],training=False).numpy()
        tf.random.set_seed(22);latent=encoder([x,snr],training=False);received=channel(latent,snr);decomposed=decoder([received,snr],training=False).numpy()
        if not np.array_equal(full,decomposed):raise ValueError('AD channel decomposition differs from native forward')
        functions=dict(encoder=lambda:encoder([x,snr],training=False).numpy(),channel=lambda:channel(latent,snr).numpy(),
                       decoder=lambda:decoder([received,snr],training=False).numpy(),end_to_end=lambda:model([x,snr],training=False).numpy())
        for name,function in functions.items():result['timings'][name]=measure(function)
        result.update(framework_version=tf.__version__,strict_checkpoint_reload=True,trained_epochs=150,
                      component_equivalence='numpy.array_equal with identical channel RNG',parameters=model.count_params(),
                      physical_devices=[str(d) for d in tf.config.list_physical_devices()])
    if sha(a.checkpoint)!=before:raise ValueError('Checkpoint changed during timing')
    result['checkpoint_unchanged']=True
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='hardware'},indent=2),flush=True)


if __name__=='__main__':main()
