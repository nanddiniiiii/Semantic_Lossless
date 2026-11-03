"""Private evaluation-only continuation; pinned kernel outputs avoid weight uploads."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ['PATH']=str(ROOT/'.venv-collector/Scripts')+os.pathsep+os.environ['PATH']
from src.person1.auth import ensure_fresh_authorization
from src.persistence import file_lock


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--username',default='hmmkk20')
    parser.add_argument('--parent-version',type=int,default=17);parser.add_argument('--wait',action='store_true')
    args,_=parser.parse_known_args();owner=args.username
    folder=ROOT/'Results/person1_campaign'/owner/'isec-final-evaluation';folder.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'Results/NO_GITHUB_PUBLICATION').exists():raise RuntimeError('Publication block required')
    with file_lock(folder/'submit.lock') as owned:
        if not owned:raise RuntimeError('Evaluation preparation already active')
        if (folder/'submission_acknowledged.json').exists():raise RuntimeError('Evaluation already submitted; collect existing job')
        parent=owner+'/semantic-notebook-1-isec'
        while True:
            ensure_fresh_authorization(owner)
            result=subprocess.run(['kaggle','kernels','status',parent],capture_output=True,text=True,encoding='utf-8',timeout=90)
            if result.returncode==0 and any(s in result.stdout for s in ['COMPLETE','ERROR']):break
            if not args.wait:raise RuntimeError('Parent output not terminal; do not interrupt it')
            print('Waiting for parent evaluation to publish; no GPU job duplicated',flush=True);time.sleep(60)
        base=ROOT/'Results/team_downloads/1'/(owner+'__semantic-notebook-1-isec')
        receipts=[p.parent for p in base.glob('*/submitted_version.json') if json.loads(p.read_text())['version']==args.parent_version]
        if len(receipts)!=1:raise ValueError('Pinned source notebook receipt is missing/ambiguous')
        source=receipts[0];book=json.loads((source/'submitted-notebook.ipynb').read_text());book['cells']=book['cells'][:3]
        worker=ROOT/'scripts/isec_finish_evaluation_worker.py';data=worker.read_bytes();sha=hashlib.sha256(data).hexdigest()
        program=f'''
import base64,hashlib,json,os,subprocess
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from src.runtime import export
parents=[]
for candidate in Path('/kaggle/input').rglob('training_summary.json'):
 try:
  value=json.loads(candidate.read_text())
  if candidate.parent.name=='isec' and value.get('completed_epochs')==dict(jscc=150,denoiser=150):parents.append(candidate.parent)
 except (ValueError,OSError):pass
if len(parents)!=1:raise ValueError('Expected exactly one pinned completed ISEC run')
run=Path('/kaggle/working/isec-final-evaluation');run.mkdir()
payload=base64.b64decode({base64.b64encode(data).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
worker=run/'evaluation_worker.py';worker.write_bytes(payload)
(run/'parent_identity.json').write_text(json.dumps(dict(kernel={parent!r},version={args.parent_version},source=str(parents[0]),worker_sha256={sha!r},training_updates=0,evaluation_device='CPU'),indent=2))
from src.person1.controller import command
try:
 command([sys.executable,'-u',str(worker),'--parent',str(parents[0]),'--run',str(run),'--root',str(REPO),'--device','cpu'],run/'evaluation.log',36000,dict(os.environ))
finally:export(run)
'''
        book['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=program.splitlines(keepends=True)))
        for c in book['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'<isec-eval>','exec')
        (folder/'notebook.ipynb').write_text(json.dumps(book))
        slug=owner+'/person1-isec-final150-evaluation'
        metadata=dict(id=slug,title='Person1 ISEC final150 evaluation continuation',code_file='notebook.ipynb',
            language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=True,
            dataset_sources=['soumikrakshit/div2k-high-resolution-images','daehoyang/flickr2k','drxinchengzhu/kodak24'],
            kernel_sources=[parent+'/'+str(args.parent_version)])
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
        subprocess.run(['kaggle','kernels','push','-p',str(folder)],check=True,timeout=180)
        (folder/'submission_acknowledged.json').write_text(json.dumps(dict(kernel=slug,parent_kernel=parent,parent_version=args.parent_version,
             cpu_only=True,worker_sha256=sha,no_training=True),indent=2))
        print('Submitted private CPU evaluation continuation:',slug,flush=True)


if __name__=='__main__':main()
