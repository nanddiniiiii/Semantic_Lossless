"""Prepare and submit one private CPU-only matched timing job, with receipts."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.person1.auth import ensure_fresh_authorization
from src.person1.collection import owned_dataset_exists


def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def cli(args,log):
    result=subprocess.run(['kaggle']+args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=900)
    log.write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:raise RuntimeError('CLI failed; inspect '+str(log))
    return result.stdout


def main():
    os.chdir(ROOT);os.environ['PATH']=str(ROOT/'.venv-collector/Scripts')+os.pathsep+os.environ['PATH']
    ensure_fresh_authorization('hmmmkk')
    destination=ROOT/'Results/person1_timing';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'submission.json').exists():raise RuntimeError('Timing job already submitted; inspect existing receipt before any retry')
    run=ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1-deepjscc/7a5a49a0c063462fa6f3d63a86217fd6/published_output/team_results/1/20261031T003230Z-cc4c86d3/deepjscc'
    checkpoint=run/'checkpoints/last.pt';config=json.loads((run/'resolved_config.json').read_text())
    summary=json.loads((run/'training_summary.json').read_text())
    if summary.get('completed_epochs')!={'codec':150} or summary['status']!='training_completed':raise ValueError('Deep150 missing')
    source=ROOT/'src/person1/paper_deepjscc.py'
    if sha(source)!=config['runner_sha256']['paper_deepjscc.py']:raise ValueError('Deep frozen source hash mismatch')
    asset=destination/'deep-asset';asset.mkdir(exist_ok=True)
    expected=sha(checkpoint);shutil.copy2(checkpoint,asset/'deep-last.pt');shutil.copy2(source,asset/'paper_deepjscc.py')
    if sha(asset/'deep-last.pt')!=expected:raise ValueError('Checkpoint copy mismatch')
    write(asset/'config.json',config);write(asset/'training_summary.json',summary)
    dataset='hmmmkk/p1-deepjscc-final150-timing-'+expected[:8]
    write(asset/'dataset-metadata.json',dict(id=dataset,title='Person1 DeepJSCC trained150 timing asset',licenses=[{'name':'other'}]))
    if not owned_dataset_exists(dataset,dict(os.environ)):
        cli(['datasets','create','-p',str(asset),'-t'],destination/'deep-upload.log')
    status=cli(['datasets','status',dataset],destination/'deep-dataset-status.log')
    if 'ready' not in status.lower():raise RuntimeError('Dataset processing; rerun preparation after ready; no duplicate creation')
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest
    api=KaggleApi();api.authenticate();request=ApiGetDatasetMetadataRequest();request.owner_slug='hmmmkk';request.dataset_slug=dataset.split('/')[1]
    with api.build_kaggle_client() as client:view=client.datasets.dataset_api_client.get_dataset_metadata(request)
    if not view.info or view.info.is_private is not True:raise ValueError('Deep dataset privacy unverified')
    write(destination/'deep-asset-receipt.json',dict(dataset=dataset,sha256=expected,is_private=True,original_checkpoint=str(checkpoint),original_unchanged=sha(checkpoint)==expected))
    original=ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1/b20ed0d81fa44a48b7241b17baec8225/submitted-notebook.ipynb'
    book=json.loads(original.read_text());book['cells']=book['cells'][:3]
    worker=base64.b64encode((ROOT/'scripts/timing_person1_cpu.py').read_bytes()).decode()
    program='''
import os,json,base64,hashlib,shutil
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
from PIL import Image
import numpy as np
from src.person1.controller import command
from src.person1.environments import prepare
from src.sources import fetch_source,source_path
from src.runtime import export
run=Path('/kaggle/working/person1-timing');run.mkdir()
deep=next(Path('/kaggle/input').rglob('deep-last.pt'));ad=next(Path('/kaggle/input').rglob('checkpoint.bin'))
worker=run/'timing_worker.py';worker.write_bytes(base64.b64decode(WORKER))
images=sorted(p for p in Path('/kaggle/input').rglob('*') if p.is_file() and p.suffix.lower()=='.png' and 'kodak' in str(p).lower())
selected=None
for path in images:
 with Image.open(path) as handle:
  if handle.size==(768,512):
   value=np.asarray(handle.convert('RGB'),dtype=np.float32)[None].copy()/255.;selected=path;break
if selected is None:raise ValueError('No real Kodak512x768 input')
np.save(run/'input.npy',value)
(run/'input_identity.json').write_text(json.dumps(dict(source=str(selected),source_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),shape=list(value.shape),synthetic=False)))
adasset=run/'ad-asset';adasset.mkdir();config=json.loads((ad.parent/'config.json').read_text());fetch_source('adjscc_author');config['source_dir']=str(source_path('adjscc_author'));(adasset/'config.json').write_text(json.dumps(config))
python,env=prepare('adjscc',REPO,run/'dependencies.log',command)
env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
try:
 command([sys.executable,'-u',str(worker),'deepjscc','--asset',str(deep.parent),'--checkpoint',str(deep),'--expected-sha',DEEP_SHA,'--input',str(run/'input.npy'),'--output',str(run/'deepjscc.json')],run/'deepjscc.log',7200,env)
 command([python,'-u',str(worker),'adjscc','--asset',str(adasset),'--checkpoint',str(ad),'--expected-sha',AD_SHA,'--input',str(run/'input.npy'),'--output',str(run/'adjscc.json')],run/'adjscc.log',14400,env)
 results={name:json.loads((run/(name+'.json')).read_text()) for name in ('deepjscc','adjscc')}
 a,b=results.values()
 for key in ('platform','processor','machine','cpuinfo','logical_cpu_count','affinity'):
  if a['hardware'][key]!=b['hardware'][key]:raise ValueError('Hardware identities differ: '+key)
 if a['input_sha256']!=b['input_sha256']:raise ValueError('Input identities differ')
 (run/'comparison.json').write_text(json.dumps(dict(same_host=True,matched_input=True,device='CPU',precision='float32',results=results),indent=2))
finally:export(run)
'''.replace('WORKER',repr(worker)).replace('DEEP_SHA',repr(expected)).replace('AD_SHA',repr('385b3c1a85b764c66e8e5ae156cbcab21a944873fdbcea511830db5d56784df7'))
    compile(program,'timing-notebook','exec')
    book['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=program.splitlines(keepends=True)))
    write(destination/'notebook.ipynb',book)
    kernel='hmmmkk/person1-deep-ad-matched-cpu-timing'
    write(destination/'kernel-metadata.json',dict(id=kernel,title='Person1 Deep AD matched CPU timing',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=True,dataset_sources=['drxinchengzhu/kodak24','hmmmkk/p1-adjscc-final150-evaluation',dataset]))
    status=subprocess.run(['kaggle','kernels','status',kernel],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    (destination/'pre-submit-kernel-status.log').write_text(status.stdout+'\n'+status.stderr)
    if status.returncode==0:raise RuntimeError('Kernel identity already exists; inspect before retrying, no push attempted')
    if '404' not in status.stderr and 'not found' not in (status.stdout+status.stderr).lower():
        import csv,io
        listing=cli(['kernels','list','--mine','--csv','--page-size','100'],destination/'owned-kernel-inventory.log')
        lines=listing.splitlines();start=next((i for i,line in enumerate(lines) if line.lstrip('\ufeff').startswith('ref,')),None)
        if start is None:raise RuntimeError('Owned inventory unavailable; no push')
        records=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]).lstrip('\ufeff'))))
        if len(records)>=100 or any(row.get('ref')==kernel for row in records):raise RuntimeError('Absence unverified or job already exists; no push')
    acknowledgement=cli(['kernels','push','-p',str(destination)],destination/'push.log')
    write(destination/'submission.json',dict(kernel=kernel,acknowledgement=acknowledgement,enable_gpu=False,deep_checkpoint_sha256=expected,ad_checkpoint_sha256='385b3c1a85b764c66e8e5ae156cbcab21a944873fdbcea511830db5d56784df7',status='submitted'))
    print(acknowledgement,flush=True)


if __name__=='__main__':main()
