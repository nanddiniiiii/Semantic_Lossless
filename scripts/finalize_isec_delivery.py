"""Verify Colab retrieval, merge immutable native trials and score Kodak240."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'Results/person1_final/isec'
PARENT=ROOT/'Results/team_downloads/1/hmmkk20__semantic-notebook-1-isec/05a60cc829cb47a986e620fbba637425/published_output/team_results/1/20261103T101702Z-e89966bf/isec'
TRANSFER=ROOT/'Results/person1_colab/isec-evaluation-input'
RUN=BASE/'combined_run'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False))
def copy(p,t,expected=None):
    if expected and sha(p)!=expected:raise ValueError('Source hash differs: '+str(p))
    t.parent.mkdir(parents=True,exist_ok=True)
    if t.exists():
        if sha(t)!=sha(p):raise ValueError('Existing destination differs')
    else:shutil.copy2(p,t)
def main():
    archive=Path(sys.argv[1]);expected='6f36c8520b9738421348a66f47b1839cc790acecafc978101128e79ee3701ec6'
    if sha(archive)!=expected:raise ValueError('Downloaded ZIP differs from Colab runtime hash')
    BASE.mkdir(parents=True,exist_ok=True);copy(archive,BASE/'retrieved/isec-colab-output.zip',expected)
    extracted=BASE/'colab_output';extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
        for name in z.namelist():(extracted/name).resolve().relative_to(extracted.resolve())
        z.extractall(extracted)
    files=json.loads((extracted/'artifacts.json').read_text())['files']
    for relative,digest in files.items():
        p=(extracted/relative).resolve();p.relative_to(extracted.resolve())
        if sha(p)!=digest:raise ValueError('Colab artifact hash mismatch: '+relative)
    completion=json.loads((extracted/'evaluation_completed.json').read_text())
    if not completion['passed'] or completion['combined_trials']!=240 or not completion['strict_reload']:raise ValueError('Evaluation gate failed')
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from src.person1.isec import build
    from src.evaluate import image_quality
    from PIL import Image
    config=json.loads((TRANSFER/'resolved_config.json').read_text())
    state=torch.load(TRANSFER/'last.pt',map_location='cpu',weights_only=False)
    if state['progress']['completed_epochs']!={'jscc':150,'denoiser':150}:raise ValueError('Training incomplete')
    model=build(config,'cpu',ROOT/'.upstream/isec');model.load_state_dict(state['state_dict'],strict=True);model.eval()
    check=sha(TRANSFER/'last.pt')
    if check!=completion['checkpoint_sha256']:raise ValueError('Evaluation checkpoint differs')
    RUN.mkdir(parents=True,exist_ok=True)
    copy(TRANSFER/'last.pt',RUN/'checkpoints/last.pt',check)
    for name in ['training_summary.json','model.json','source.json','environment.json','profile.json',
                 'training_profile-jscc.json','training_profile-denoiser.json','dataset_identity.json']:
        if (PARENT/name).is_file():copy(PARENT/name,RUN/name)
    write(RUN/'resolved_config.json',config)
    source_entries={x['path']:x for x in json.loads((PARENT/'artifacts.json').read_text())['files']}
    original=json.loads((PARENT/'predictions.json').read_text())['rows']
    continued=json.loads((extracted/'predictions.json').read_text())['rows']
    rows=[];keys=set()
    for origin,collection,folder in [('Kaggle CUDA12.8',original,PARENT),('Colab CUDA13.0',continued,extracted)]:
        for row in collection:
            key=(row['sample'],row['trial'])
            if key in keys or row['seed']!=22+10000*key[0]+key[1]:raise ValueError('Duplicate or invalid trial seed')
            relative='predictions/'+row['path']
            digest=source_entries[relative]['sha256'] if folder==PARENT else files[relative]
            copy(folder/relative,RUN/relative,digest)
            rows.append(dict(row,evaluation_origin=origin,evaluation_device='Tesla T4'));keys.add(key)
    if keys!={(i,t) for i in range(24) for t in range(10)}:raise ValueError('Kodak coverage incomplete')
    rows.sort(key=lambda r:(r['sample'],r['trial']));write(RUN/'predictions.json',dict(rows=rows))
    write(BASE/'retrieval_verified.json',dict(passed=True,archive_sha256=expected,colab_files=len(files),
          parent_trials=len(original),continued_trials=len(continued),trials=240,checkpoint_sha256=check,strict_reload=True))
    metrics=[];(RUN/'reconstructions').mkdir(exist_ok=True)
    for row in rows:
        a=np.load(TRANSFER/'kodak'/('%02d.npy'%row['sample'])).astype(np.float32)/255
        b=np.load(RUN/'predictions'/row['path'],allow_pickle=False)
        if a.shape!=b.shape or not np.isfinite(b).all():raise ValueError('Geometry/nonfinite reconstruction')
        value=image_quality(torch.from_numpy(a).permute(2,0,1)[None],torch.from_numpy(b).permute(2,0,1)[None],True)
        if any(value.get(k) is None or not np.isfinite(value[k]) for k in ['psnr_db','ssim','ms_ssim','lpips','mse']):raise ValueError('Required metric missing')
        h,w=a.shape[:2]
        metrics.append(dict(algorithm='isec',sample=row['sample'],trial=row['trial'],seed=row['seed'],height=h,width=w,
          channel='awgn',snr_db=10,complex_channel_uses=row['complex_channel_uses'],transmitted_side_information=0,
          cbr_complex_per_rgb_sample=row['complex_channel_uses']/(3*h*w),one_pass_seconds=row['seconds'],
          evaluation_origin=row['evaluation_origin'],evaluation_device=row['evaluation_device'],**value))
        Image.fromarray((np.clip(b,0,1)*255).round().astype(np.uint8)).save(RUN/'reconstructions'/row['path'].replace('.npy','.png'))
        if len(metrics)%10==0:print('Verified/scored',len(metrics),'/240',flush=True)
    summary=dict(requested_trials=240,successful_trials=240,failed_trials=0,completed_epochs={'jscc':150,'denoiser':150},
       split='Kodak24 test',iterations=50,complete_link_certified=True,pilot=False,
       ci_method='bootstrap24 image means;2000 resamples;seed22',
       latency_aggregation_eligible=False,latency_note='Two TeslaT4 runtime stacks; per-trial times descriptive only; no matched-host ranking',
       rate_scope='native CBR1/16; DeepJSCC/ADJSCC CBR1/6; not an equal-rate ranking')
    for key in ['psnr_db','ssim','ms_ssim','lpips','mse','cbr_complex_per_rgb_sample']:
        means=np.array([np.mean([r[key] for r in metrics if r['sample']==i]) for i in range(24)])
        boot=means[np.random.RandomState(22).randint(0,24,(2000,24))].mean(1)
        summary[key]=float(means.mean());summary[key+'_ci95']=np.quantile(boot,[.025,.975]).tolist()
    write(RUN/'metrics/summary.json',summary);write(RUN/'metrics/per_image.json',dict(rows=metrics,failures=[]))
    with (RUN/'metrics/per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    write(RUN/'evaluation_reload.json',dict(strict_reload=True,checkpoint_sha256=check,completed_epochs=state['progress']['completed_epochs']))
    for phase in ['train','evaluate']:write(RUN/(phase+'_status.json'),dict(status='completed',evidence='verified final150 checkpoint and combined240trial artifacts'))
    from src.sharing import package_run, verify_bundle
    target=ROOT/'Models/teammate_1_models/ISEC/final150-kodak240'
    if target.exists():verify_bundle(target)
    else:package_run(RUN,ROOT/'Models','1','final150-kodak240')
    write(BASE/'completion.json',dict(verified=True,model_bundle=str(target),metrics=summary,checkpoint_sha256=check,
         scope='training, quality, communication and artifact verification; see separate logical complexity report'))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
