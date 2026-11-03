"""Resume only missing ISEC Kodak trials from a verified published run.

No training or model modification. Reuses the pinned native model/predict code.
Partial GPU trials keep their device labels; CPU fallback never supplies a GPU
latency claim. Completed source artifacts are copied and hash checked first.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(parent, run, root, device='cpu'):
    from src.person1.controller import command,score
    from src.person1.protocol import DATASET_IDENTITY
    from src.dataset import discover,freeze
    from src.sources import fetch_source,source_path
    from src.person1.isec import build
    from src.person1.torch_worker import predict
    from src.runtime import export
    parent,run,root=Path(parent),Path(run),Path(root)
    run.mkdir(parents=True,exist_ok=True);(run/'checkpoints').mkdir(exist_ok=True)
    source=json.loads((parent/'artifacts.json').read_text())
    entries={x['path']:x for x in source['files']}
    def copy_verified(relative,target):
        entry=entries[relative];path=(parent/relative).resolve();path.relative_to(parent.resolve())
        if digest(path)!=entry['sha256']:raise ValueError('Published source bytes differ: '+relative)
        target=Path(target);target.resolve().relative_to(run.resolve());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if digest(target)!=entry['sha256']:raise ValueError('Evaluation copy failed: '+relative)
    copy_verified('checkpoints/last.pt',run/'checkpoints/last.pt')
    copy_verified('training_summary.json',run/'training_summary.json')
    summary=json.loads((run/'training_summary.json').read_text())
    if summary['status']!='training_completed' or summary['completed_epochs']!={'jscc':150,'denoiser':150}:
        raise ValueError('Evaluation requires both full training stages')
    config=json.loads((parent/'resolved_config.json').read_text())
    manifest=run/'dataset.json';frozen=freeze(discover('/kaggle/input'),manifest,workers=2)
    if frozen['identity']!=DATASET_IDENTITY:raise ValueError('Frozen benchmark data differ')
    fetch_source('isec');config.update(manifest=str(manifest),source_dir=str(source_path('isec')))
    (run/'resolved_config.json').write_text(json.dumps(config,indent=2))
    command([__import__('sys').executable,'-m','pip','install','pytorch-msssim==1.0.0','lpips==0.1.4'],run/'dependencies.log',1200)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=build(config,device,config['source_dir'])
    saved=torch.load(run/'checkpoints/last.pt',map_location=device,weights_only=False)
    if saved['progress']['completed_epochs']!={'jscc':150,'denoiser':150}:raise ValueError('Checkpoint epochs differ')
    model.load_state_dict(saved['state_dict'],strict=True);model.stage='denoiser';model.eval()
    (run/'strict_reload.json').write_text(json.dumps(dict(passed=True,checkpoint_sha256=digest(run/'checkpoints/last.pt'),
         completed_epochs=saved['progress']['completed_epochs'],device=device,source_commit=model.metadata['source_commit']),indent=2))
    rows=[];seen=set();predictions=run/'predictions';predictions.mkdir(exist_ok=True)
    original=json.loads((parent/'predictions.json').read_text())['rows']
    for row in original:
        key=(row['sample'],row['trial'])
        if key in seen:raise ValueError('Duplicate parent trial')
        copy_verified('predictions/'+row['path'],predictions/row['path'])
        item=dict(row,evaluation_device=row.get('evaluation_device','cuda:0/TeslaT4'),
                  evaluation_origin=row.get('evaluation_origin','verified parent run'))
        rows.append(item);seen.add(key)
    from src.person1.protocol import Images
    dataset=Images(manifest,'test',config['crop'])
    if len(dataset)!=24 or config['trials']!=10:raise ValueError('Expected Kodak24 and10 channel trials')
    def save_rows():
        temporary=run/'predictions.json.tmp';temporary.write_text(json.dumps(dict(rows=rows,pilot=False),indent=2))
        temporary.replace(run/'predictions.json')
    try:
        for i in range(24):
            x=torch.from_numpy(dataset.array(i)).permute(2,0,1)[None].to(device)
            for trial in range(10):
                if (i,trial) in seen:continue
                seed=config['seed']+10000*i+trial;torch.manual_seed(seed)
                if device.startswith('cuda'):torch.cuda.synchronize()
                tick=time.perf_counter()
                with torch.no_grad():out=predict(model,x,config)
                if device.startswith('cuda'):torch.cuda.synchronize()
                seconds=time.perf_counter()-tick;y=out['reconstruction']
                if y.shape!=x.shape or not torch.isfinite(y).all():raise ValueError('Invalid native reconstruction')
                name='%03d-%02d.npy'%(i,trial)
                np.save(predictions/name,y[0].detach().cpu().permute(1,2,0).numpy())
                row=dict(sample=i,trial=trial,seed=seed,path=name,seconds=seconds,
                    complex_channel_uses=float(out['complex_uses'].sum()),geometry=out['geometry'],
                    evaluation_device=device,evaluation_origin='native evaluation-only continuation')
                rows.append(row);seen.add((i,trial));save_rows();print(json.dumps(row),flush=True)
        if len(seen)!=240:raise ValueError('Incomplete Kodak trials')
        rows.sort(key=lambda x:(x['sample'],x['trial']));save_rows()
        score(dict(run=str(run),config=config))
        metrics=run/'metrics/summary.json';value=json.loads(metrics.read_text())
        value.pop('one_pass_seconds',None)
        value.pop('one_pass_seconds_ci95',None)
        value.update(evaluation_device_scope=sorted({r['evaluation_device'] for r in rows}),
            mixed_device_quality_evaluation=len({r['evaluation_device'] for r in rows})>1,
            aggregate_latency_eligible=False,timing_scope='Do not aggregate CPU and GPU trial times; use separately profiled matched-device timing',
            rng_scope='native per-trial seed retained; CPU/CUDA random streams and floating backend may differ',
            inherited_trials=len(original),continued_trials=240-len(original))
        metrics.write_text(json.dumps(value,indent=2))
        devices={(r['sample'],r['trial']):(r['evaluation_device'],r['evaluation_origin']) for r in rows}
        per_image=run/'metrics/per_image.json';table=json.loads(per_image.read_text())
        for item in table['rows']:
            item['evaluation_device'],item['evaluation_origin']=devices[(item['sample'],item['trial'])]
        per_image.write_text(json.dumps(table,indent=2))
        with (run/'metrics/per_image.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table['rows'][0]));writer.writeheader();writer.writerows(table['rows'])
        (run/'evaluation_completed.json').write_text(json.dumps(dict(passed=True,trials=240,
              strict_reload=True,no_training_updates=True,checkpoint_sha256=digest(run/'checkpoints/last.pt')),indent=2))
    finally:
        save_rows();export(run)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--run',required=True)
    p.add_argument('--root',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    execute(a.parent,a.run,a.root,a.device)


if __name__=='__main__':main()
