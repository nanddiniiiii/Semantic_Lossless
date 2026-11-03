"""Read-only final ISEC evaluation on a reviewed portable input bundle.

No training, credential handling, Drive mount, or source downloads. The bundle
contains the exact frozen adapter, pinned author source and Kodak RGB inputs.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(root):
    manifest=json.loads((root/'bundle_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path=(root/relative).resolve();path.relative_to(root.resolve())
        if sha(path)!=expected:raise ValueError('Bundle hash differs: '+relative)
    return manifest

def main(root):
    root=Path(root).resolve();manifest=verify(root)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    sys.path.insert(0,str(root/'runtime'))
    import numpy as np
    import torch
    from src.person1.isec import build
    from src.person1.torch_worker import predict
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no silent CPU fallback')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    config=json.loads((root/'resolved_config.json').read_text())
    if config['iterations']!=50 or config['trials']!=10:raise ValueError('Native protocol differs')
    saved=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    if saved['progress']['completed_epochs']!={'jscc':150,'denoiser':150}:raise ValueError('Incomplete checkpoint')
    model=build(config,'cuda:0',root/'author')
    model.load_state_dict(saved['state_dict'],strict=True);model.stage='denoiser';model.eval()
    output=root.parent/'isec-colab-output';output.mkdir(exist_ok=True)
    (output/'predictions').mkdir(exist_ok=True)
    identity=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.version,
                  checkpoint_sha256=sha(root/'last.pt'),strict_reload=True,training_updates=0,
                  parent_manifest_identity=manifest['dataset_identity'],iterations=50,
                  backend_adaptation='Colab CUDA runtime; source and weights unchanged')
    (output/'strict_reload.json').write_text(json.dumps(identity,indent=2))
    completed={(r['sample'],r['trial']) for r in json.loads((root/'parent_predictions.json').read_text())['rows']}
    rows=[]
    # A resumed Colab session may already have completed some missing trials.
    if (output/'predictions.json').exists():
        old=json.loads((output/'predictions.json').read_text())
        if old['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Output checkpoint differs')
        for row in old['rows']:
            if sha(output/'predictions'/row['path'])!=row['sha256']:raise ValueError('Existing trial corrupt')
            if (row['sample'],row['trial']) in completed:raise ValueError('Duplicate parent trial')
            rows.append(row);completed.add((row['sample'],row['trial']))
    def publish():
        path=output/'predictions.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(rows=rows,checkpoint_sha256=identity['checkpoint_sha256'],
              inherited_trials=len(json.loads((root/'parent_predictions.json').read_text())['rows'])),indent=2))
        temp.replace(path)
    print(json.dumps(identity),flush=True)
    for sample in range(24):
        arr=np.load(root/'kodak'/('%02d.npy'%sample),allow_pickle=False)
        if arr.dtype!=np.uint8 or arr.ndim!=3 or arr.shape[2]!=3:raise ValueError('Invalid Kodak input')
        # Identical to frozen Images.array: RGB float32 / 255, original dimensions.
        x=torch.from_numpy(arr.astype(np.float32)/255).permute(2,0,1)[None].to('cuda:0')
        for trial in range(10):
            if (sample,trial) in completed:continue
            seed=config['seed']+10000*sample+trial;torch.manual_seed(seed)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():out=predict(model,x,config)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            y=out['reconstruction']
            if y.shape!=x.shape or not torch.isfinite(y).all():raise ValueError('Invalid reconstruction')
            name='%03d-%02d.npy'%(sample,trial);target=output/'predictions'/name
            np.save(target,y[0].detach().cpu().permute(1,2,0).numpy())
            row=dict(sample=sample,trial=trial,seed=seed,path=name,sha256=sha(target),seconds=seconds,
                 complex_channel_uses=float(out['complex_uses'].sum()),geometry=out['geometry'],
                 evaluation_device='cuda:0/'+identity['device'],evaluation_origin='Colab Amrita evaluation-only')
            rows.append(row);completed.add((sample,trial));publish();print(json.dumps(row),flush=True)
    if len(completed)!=240:raise ValueError('Missing trials')
    (output/'evaluation_completed.json').write_text(json.dumps(dict(passed=True,combined_trials=240,
       new_trials=len(rows),strict_reload=True,training_updates=0,checkpoint_sha256=identity['checkpoint_sha256']),indent=2))
    files={str(p.relative_to(output)).replace('\\','/'):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='artifacts.json'}
    (output/'artifacts.json').write_text(json.dumps(dict(files=files),indent=2))
    import shutil
    archive=shutil.make_archive(str(output),'zip',output)
    print('Completed output:',archive,'SHA256:',sha(archive),flush=True)

if __name__=='__main__':main(sys.argv[1])
