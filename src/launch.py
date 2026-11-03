"""Submit one team notebook from the user's own CLI and collect saved outputs."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .monitor import collect, verify_download
from .runtime import write_json
from .sharing import import_bundles

ROOT=Path(__file__).resolve().parents[1]
DATASETS=['soumikrakshit/div2k-high-resolution-images','daehoyang/flickr2k','drxinchengzhu/kodak24']


def notebook_revision(notebook):
    revisions=[]
    for cell in notebook.get('cells',[]):
        if cell.get('cell_type','code')!='code':continue
        for node in ast.parse(''.join(cell.get('source',[]))).body:
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='REVISION' for target in node.targets):
                if not isinstance(node.value,ast.Constant) or not isinstance(node.value.value,str):
                    raise ValueError('Notebook REVISION must be a literal reviewed commit hash')
                revisions.append(node.value.value)
    if len(revisions)!=1 or len(revisions[0])!=40 or any(c not in '0123456789abcdef' for c in revisions[0]):
        raise RuntimeError('Notebook source revision is not published and pinned; refusing to submit')
    return revisions[0]


def bind_delivery(notebook, run_id, training_seconds=14400, model=None, mode=None, frozen_runtime=False, models=None):
    import copy
    if isinstance(model,(list,tuple)) and models is None:
        models=model;model=None
    notebook=copy.deepcopy(notebook)
    if models is not None:
        assigned=notebook.get('metadata',{}).get('experiment',{}).get('models',[])
        if not models or len(set(models))!=len(models) or set(models)-set(assigned):
            raise ValueError('Select a nonempty unique subset of this notebook roster')
    replacement='DELIVERY_RUN_ID = '+repr(run_id)+'\n'
    found=False
    for cell in notebook.get('cells',[]):
        if cell.get('cell_type','code')!='code':continue
        lines=''.join(cell.get('source',[])).splitlines(keepends=True)
        for index,line in enumerate(lines):
            if line.startswith('TRAINING_SECONDS = '):lines[index]='TRAINING_SECONDS = '+str(training_seconds)+'\n'
            if models is not None and line.startswith('MODELS_TO_RUN = '):
                lines[index]='MODELS_TO_RUN = '+repr(list(models))+'\n'
            if model is not None and line.startswith('PERSON1_MODEL = '):lines[index]='PERSON1_MODEL = '+repr(model)+'\n'
            if mode is not None and line.startswith('PERSON1_MODE = '):lines[index]='PERSON1_MODE = '+repr(mode)+'\n'
            if line.startswith('DELIVERY_RUN_ID = '):
                if found:raise ValueError('Duplicate notebook delivery identity')
                lines[index]=replacement;found=True
        cell['source']=lines
    if not found:
        cell=next(c for c in notebook['cells'] if c.get('cell_type','code')=='code')
        cell['source']=[replacement]+cell['source']
    if model is not None and (ROOT/'Results/NO_GITHUB_PUBLICATION').exists():
        if frozen_runtime:
            if not notebook.get('metadata',{}).get('private_runtime_overlay_sha256') or not any(
                    c.get('id')=='private-runtime-overlay' for c in notebook['cells']):
                raise ValueError('Continuation notebook lacks a frozen runtime overlay')
        else:
            notebook['cells']=[c for c in notebook['cells'] if c.get('id')!='private-runtime-overlay']
            notebook=private_runtime_overlay(notebook)
    return notebook


def private_runtime_overlay(notebook):
    """Carry local runtime adapters in the private notebook, without a Git push."""
    import base64,hashlib,zlib
    paths=list((ROOT/'src/person1').glob('*.py'))
    if (ROOT/'src/dataset.py').is_file():paths.append(ROOT/'src/dataset.py')
    files={p.relative_to(ROOT).as_posix():base64.b64encode(p.read_bytes()).decode('ascii')
           for p in sorted(paths)}
    raw=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
    sha=hashlib.sha256(raw).hexdigest();payload=base64.b64encode(zlib.compress(raw)).decode('ascii')
    code="""# Private, hash-checked local runtime adapters; no GitHub publication.
import base64,hashlib,json,zlib
_overlay=zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(_overlay).hexdigest()!=EXPECTED:raise RuntimeError('Private runtime bundle hash differs')
_files=json.loads(_overlay);_manifest={}
for _relative,_encoded in _files.items():
    _target=(REPO/_relative).resolve();_target.relative_to(REPO.resolve())
    _content=base64.b64decode(_encoded);_target.parent.mkdir(parents=True,exist_ok=True);_target.write_bytes(_content)
    _manifest[_relative]=hashlib.sha256(_content).hexdigest()
(REPO/'private_runtime_overlay.json').write_text(json.dumps(dict(sha256=EXPECTED,files=_manifest,base_revision=REVISION)))
print('Private runtime overlay:',EXPECTED,flush=True)
""".replace('PAYLOAD',repr(payload)).replace('EXPECTED',repr(sha))
    ast.parse(code)
    notebook['cells'].insert(2,dict(cell_type='code',id='private-runtime-overlay',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)))
    notebook.setdefault('metadata',{})['private_runtime_overlay_sha256']=sha
    return notebook


def current_version(username,slug):
    """Get actual current notebook version; only an HTTP 404 means absent."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    from requests import HTTPError
    api=KaggleApi();api.authenticate()
    request=ApiGetKernelRequest();request.user_name=username;request.kernel_slug=slug
    try:
        with api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.get_kernel(request)
    except HTTPError as error:
        if error.response is not None and error.response.status_code==404:return None
        raise
    metadata=response.metadata
    if metadata is None or not metadata.current_version_number:
        raise RuntimeError('Kaggle did not report a current notebook version')
    return int(metadata.current_version_number)


def finish_retrieval(destination, env, team, publish=True):
    from .publication import enqueue_bundles, flush
    saved=json.loads((destination/'submitted_version.json').read_text())
    ref=saved['kernel'];submitted=saved['version']
    raw=destination/'published_output'
    marker=destination/'retrieval_verified.json'
    if not marker.is_file():
        owner,slug=ref.split('/')
        if current_version(owner,slug)!=submitted:
            raise ValueError('Notebook version changed; refusing ambiguous retrieval')
        subprocess.run(['kaggle','kernels','output',ref,'-p',str(raw)],env=env,check=True,timeout=3600)
        if current_version(owner,slug)!=submitted:
            raise ValueError('Notebook version changed during retrieval')
        receipt=verify_download(raw)
        identities=[json.loads(path.read_text()) for path in raw.rglob('delivery_identity.json')]
        if not identities or any(item.get('delivery_run_id')!=saved['run_id'] or str(item.get('team'))!=str(team) for item in identities):
            raise ValueError('Downloaded outputs do not match this submission identity')
        write_json(marker,{**receipt,'kernel':ref,'version':submitted,
            'identity_scope':'latest output; current version checked before and after retrieval',
            'submission_identity_verified':True,'session_identity_verified':False,
            'identity_limitation':'A rerun of the same saved version may reuse the baked-in submission identity',
            'scientific_validity_certified':False})
    else:
        verify_download(raw)
    folders=import_bundles(raw,ROOT/'Models',team)
    if str(team)=='1':
        from .person1.collection import preserve_chunks
        preserve_chunks(ROOT,destination,env)
    enqueue_bundles(ROOT,folders)
    rows=collect(ROOT/'Results/team_downloads',ROOT/'Results/comparison.csv')
    if publish:flush(ROOT)
    print('Retrieved model folders:',folders,'; CSV rows:',rows,flush=True)


def watch(destination, env, team, interval=1800, once=False, publish=True, max_hours=24):
    """Resume local collection only; this function never submits a Kaggle job."""
    from .persistence import file_lock
    from .publication import flush
    destination=Path(destination).resolve()
    saved=json.loads((destination/'submitted_version.json').read_text())
    ref=saved['kernel'];started=time.monotonic();errors=0
    print('Recoverable collector:',destination,'poll interval:',interval,'seconds',flush=True)
    with file_lock(destination/'collector.lock') as owned:
        if not owned:raise RuntimeError('Another collector already owns this local run')
        while True:
            try:
                if str(team)=='1':
                    from .person1.auth import ensure_fresh_authorization
                    ensure_fresh_authorization(ref.split('/')[0])
                if publish:flush(ROOT)
                state_path=destination/'collector_state.json'
                already_done=state_path.is_file() and json.loads(state_path.read_text()).get('status')=='retrieved'
                if already_done:
                    pending=any(json.loads(p.read_text()).get('status')!='published' for p in (ROOT/'Results/publication_queue').glob('*.json')) if publish else False
                    if not pending:return
                    if once or time.monotonic()-started>=max_hours*3600:return
                    print('Retrieval complete; retrying pending GitHub events at next poll',flush=True)
                    time.sleep(interval)
                    continue
                if (destination/'retrieval_verified.json').is_file():
                    finish_retrieval(destination,env,team,publish)
                    write_json(destination/'collector_state.json',{'status':'retrieved','kernel':ref,'version':saved['version']})
                    if once:return
                    continue
                status=subprocess.run(['kaggle','kernels','status',ref],env=env,capture_output=True,text=True,timeout=90)
                message=(status.stdout+'\n'+status.stderr).strip()
                write_json(destination/'remote_status.json',{'kernel':ref,'version':saved['version'],
                    'status':message,'returncode':status.returncode,'scope':'Kaggle latest-run status'})
                if status.returncode:raise RuntimeError('Kaggle status unavailable; will retry without resubmitting')
                print('Kaggle status:',message,flush=True)
                upper=message.upper()
                terminal=any(word in upper for word in ('COMPLETE','ERROR','CANCELLED','CANCEL_ACKNOWLEDGED'))
                write_json(destination/'collector_state.json',{'status':'retrieving' if terminal else 'waiting',
                    'kernel':ref,'version':saved['version'],'poll_seconds':interval})
                if terminal:
                    finish_retrieval(destination,env,team,publish)
                    write_json(destination/'collector_state.json',{'status':'retrieved','kernel':ref,
                        'version':saved['version'],'remote_success':'COMPLETE' in upper})
                    if 'COMPLETE' not in upper:print('Notebook failed; partial evidence retained, completed model bundles processed',flush=True)
                    if once:return
                    continue
                errors=0
            except ValueError:
                # Identity/integrity violations require investigation, never a retry overwrite.
                raise
            except (OSError,RuntimeError,subprocess.SubprocessError) as error:
                errors+=1
                write_json(destination/'collector_state.json',{'status':'retry_pending','error':str(error),
                    'kernel':ref,'version':saved['version'],'consecutive_errors':errors})
                print('Collector retry pending:',str(error),flush=True)
                if errors>=6:raise RuntimeError('Six collector errors; state retained. Restart after fixing access/network.') from error
            if once or time.monotonic()-started>=max_hours*3600:return
            time.sleep(interval)


def notebook_slug(team,args):
    standard='semantic-notebook-'+str(team)
    override=getattr(args,'slug',None)
    if not override:return standard
    model=getattr(args,'model',None)
    expected=standard+'-'+(model or '').replace('_','-')
    if str(team)!='1' or not model or getattr(args,'mode',None)!='train' or override!=expected:
        raise ValueError('Parallel identity must be person1 training semantic-notebook-1-MODEL')
    return override


def main(team):
    parser=argparse.ArgumentParser(description='Submit this team notebook and retrieve its completed outputs.')
    parser.add_argument('--username',help='Your Kaggle username; prompted if omitted')
    parser.add_argument('--slug',help='Person1 independent training identity semantic-notebook-1-MODEL')
    parser.add_argument('--model',choices=['deepjscc','adjscc','isec','ntscc_plus'],help='Person1: one model per saved version')
    parser.add_argument('--mode',choices=['pilot','train'],default='pilot',help='Person1 only; default real-data pilot')
    parser.add_argument('--continue-from',help='Person1: private checkpoint Dataset owner/slug from verified earlier chunk')
    parser.add_argument('--runtime-notebook',help='Continuation only: exact saved submitted notebook preserving checkpoint runtime')
    parser.add_argument('--config-dir',help='Optional Kaggle CLI credential directory')
    parser.add_argument('--training-seconds',type=int,default=14400,help='Per-model training budget, at most14400 seconds')
    parser.add_argument('--models',nargs='+',help='Run only these assigned model IDs, in this order')
    parser.add_argument('--asset',action='append',default=[],help='Additional Kaggle dataset owner/slug containing model assets; repeat as needed')
    parser.add_argument('--interval',type=int,default=1800)
    parser.add_argument('--resume',nargs='?',const='latest',help='Resume a saved local run directory, or latest')
    parser.add_argument('--create-missing',action='store_true',help='Explicitly authorize first notebook creation when Kaggle masks an absent identity with HTTP403; verify account and mine-list first')
    parser.add_argument('--new-run',action='store_true',help='Explicitly submit another training run after earlier runs')
    parser.add_argument('--once',action='store_true',help='Check/retrieve/publish once without waiting')
    parser.add_argument('--no-publish',action='store_true',help='Retain results locally without GitHub publication')
    parser.add_argument('--publish-only',action='store_true',help='Retry local pending GitHub events; never submit training')
    parser.add_argument('--max-hours',type=float,default=24)
    args=parser.parse_args()
    if args.publish_only:
        from .publication import flush
        flush(ROOT);return
    username=(args.username or input('Your Kaggle username: ')).strip()
    if not username or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in username):
        raise ValueError('Invalid Kaggle username')
    from .persistence import file_lock
    with file_lock(ROOT/'Results/team_downloads'/team/(username+'__'+notebook_slug(team,args))/'launcher.lock') as owned:
        if not owned:raise RuntimeError('Another launcher/collector already owns this teammate notebook')
        return dispatch(team,args,username)


def dispatch(team,args,username):
    if not 0<args.training_seconds<=14400:raise ValueError('Training seconds must be in (0,14400]')
    if args.max_hours<=0:raise ValueError('max-hours must be positive')
    if args.interval<10:raise ValueError('Polling interval must be at least 10 seconds')
    if shutil.which('kaggle') is None:raise RuntimeError('Install and authenticate the Kaggle CLI in this Python environment first')
    env=os.environ.copy()
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    if args.config_dir:
        env['KAGGLE_CONFIG_DIR']=str(Path(args.config_dir).expanduser().resolve())
        os.environ['KAGGLE_CONFIG_DIR']=env['KAGGLE_CONFIG_DIR']
    slug=notebook_slug(team,args)
    ref=f'{username}/{slug}'
    base=ROOT/'Results/team_downloads'/team/(username+'__'+slug)
    receipts=sorted(base.glob('*/submission.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    prior=[p.parent for p in receipts]
    incomplete=[p for p in prior if not (p/'collector_state.json').is_file()
        or json.loads((p/'collector_state.json').read_text()).get('status')!='retrieved']
    target=None
    if args.resume:
        if args.resume=='latest':
            if not prior:raise FileNotFoundError('No local saved run to resume')
            target=prior[0]
        else:target=Path(args.resume).resolve()
    elif incomplete and not args.new_run:
        target=incomplete[0]
    elif prior and not args.new_run:
        from .publication import flush
        if not args.no_publish:flush(ROOT)
        print('Previous run retrieved. Use --new-run to explicitly start training again.',flush=True)
        return
    if target is not None:
        target.resolve().relative_to(base.resolve())
        if not (target/'submitted_version.json').is_file():
            raise RuntimeError('Submission receipt exists but version acknowledgement is missing. Verify Kaggle before any resubmission.')
        watch(target,env,team,args.interval,args.once,not args.no_publish,args.max_hours)
        return
    if str(team)=='1' and getattr(args,'mode',None)=='train':
        # Separate notebook identities must never train the same model twice.
        for path in (ROOT/'Results/team_downloads'/team).glob(username+'__*/*/submission.json'):
            saved=json.loads(path.read_text())
            if saved.get('person1_mode')!='train' or saved.get('person1_model')!=getattr(args,'model',None):continue
            state=path.parent/'collector_state.json'
            recovered=False
            if state.is_file():
                saved_state=json.loads(state.read_text())
                if saved_state.get('status')=='failed_checkpoint_recovered':
                    evidence_path=path.parent/'checkpoint_recovery.json'
                    if evidence_path.is_file():
                        evidence=json.loads(evidence_path.read_text())
                        checkpoint=(path.parent/evidence['checkpoint']).resolve()
                        checkpoint.relative_to(path.parent.resolve())
                        from .dataset import digest
                        recovered=(evidence.get('remote_terminal_status')=='ERROR' and
                                   evidence.get('training_complete') is False and
                                   checkpoint.is_file() and digest(checkpoint)==evidence.get('sha256'))
            if not recovered and (not state.is_file() or json.loads(state.read_text()).get('status')!='retrieved'):
                raise RuntimeError('Existing training receipt for this model must finish/retrieve first: '+str(path.parent))
    if str(team)=='1':
        from .person1.auth import ensure_fresh_authorization
        ensure_fresh_authorization(username)
        from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        with api.build_kaggle_client() as client:
            quota=client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
        gpu=quota.gpu_quota
        if gpu is None:raise RuntimeError('Cannot verify GPU quota before submission')
        remaining=(gpu.total_time_allowed-gpu.time_used-gpu.time_reserved).total_seconds()
        if remaining<1800:raise RuntimeError('Insufficient free GPU quota; checkpoint/receipts retained')
        print('Free GPU seconds available:',remaining,'weekly reset:',quota.quota_refresh_time,flush=True)
    notebook_path=ROOT/'Notebooks'/(team+'.ipynb')
    runtime=getattr(args,'runtime_notebook',None)
    if runtime:
        if not getattr(args,'continue_from',None):raise ValueError('Frozen runtime requires a continuation checkpoint')
        notebook_path=Path(runtime).resolve();notebook_path.relative_to((ROOT/'Results/team_downloads'/team).resolve())
    notebook=json.loads(notebook_path.read_text(encoding='utf-8'))
    revision=notebook_revision(notebook)
    try:
        existing=current_version(username,slug)
    except Exception as error:
        from requests import HTTPError
        if not (args.create_missing and isinstance(error,HTTPError) and error.response is not None and error.response.status_code==403):
            raise
        # Kaggle can mask absent private kernels with403. Creation still requires
        # explicit authorization plus an exact owner-filtered list check.
        import csv,io
        listing=subprocess.run(['kaggle','kernels','list','--mine','--csv','--page-size','100'],env=env,capture_output=True,text=True,check=True,timeout=90)
        lines=listing.stdout.splitlines()
        header=next((index for index,line in enumerate(lines) if line.lstrip('\ufeff').startswith('ref,title,')),None)
        if header is None:raise RuntimeError('Kaggle owned inventory did not contain the expected CSV header')
        rows=list(csv.DictReader(io.StringIO('\n'.join(lines[header:]).lstrip('\ufeff'))))
        if not rows or 'ref' not in rows[0]:raise RuntimeError('Cannot verify owned notebook inventory before creation')
        if len(rows)>=100:raise RuntimeError('Owned notebook inventory requires pagination; refusing to infer absence')
        if any(row.get('ref')==ref for row in rows):raise RuntimeError('Notebook exists but is inaccessible; refusing replacement')
        existing=None
        print('Explicit first-creation authorized; notebook absent from owned inventory',flush=True)
    if existing is not None:
        status=subprocess.run(['kaggle','kernels','status',ref],env=env,capture_output=True,text=True,timeout=90)
        message=(status.stdout+'\n'+status.stderr).strip()
        if status.returncode:raise RuntimeError('Cannot verify existing notebook status: '+message)
        if not any(word in message.upper() for word in ('COMPLETE','ERROR','CANCELLED','CANCEL_ACKNOWLEDGED')):
            raise RuntimeError('Existing notebook is active; refusing to replace it: '+message)
    run_id=uuid.uuid4().hex
    destination=ROOT/'Results'/'team_downloads'/team/(username+'__'+slug)/run_id
    destination.mkdir(parents=True,exist_ok=False)
    extra_inputs=list(args.asset)
    if getattr(args,'continue_from',None):
        if str(team)!='1':raise ValueError('Person1 continuation cannot change other teammate inputs')
        extra_inputs.append(args.continue_from)
    metadata={'id':ref,'title':slug,'code_file':'notebook.ipynb','language':'python',
        'kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
        'machine_shape':'NvidiaTeslaT4','dataset_sources':DATASETS+extra_inputs,'kernel_sources':[],
        'competition_sources':[],'model_sources':[]}
    write_json(destination/'submission.json',{'kernel':ref,'team':team,'datasets':DATASETS+extra_inputs,
        'source_revision':revision,'training_seconds_per_model':args.training_seconds,'selected_models':getattr(args,'models',None),
        'person1_model':getattr(args,'model',None) or ('deepjscc' if str(team)=='1' else None),
        'person1_mode':getattr(args,'mode',None) if str(team)=='1' else None,
        'previous_version':existing,'run_id':run_id,
        'note':'Stable kernel identity; do not create another version before retrieval'})
    print('Submitting private notebook:',ref,flush=True)
    print('The three common image datasets will be attached automatically. Additional private model assets may require setup.',flush=True)
    with tempfile.TemporaryDirectory(prefix='semantic-kaggle-') as tmp:
        folder=Path(tmp)
        bound=bind_delivery(notebook,run_id,args.training_seconds,
            (getattr(args,'model',None) or 'deepjscc') if str(team)=='1' else None,
            getattr(args,'mode','pilot') if str(team)=='1' else None,frozen_runtime=bool(runtime),models=getattr(args,'models',None))
        (destination/'submitted-notebook.ipynb').write_text(json.dumps(bound),encoding='utf-8')
        (folder/'notebook.ipynb').write_text(json.dumps(bound),encoding='utf-8')
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        subprocess.run(['kaggle','kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4'],env=env,check=True,timeout=180)
    submitted=current_version(username,slug)
    if submitted is None or submitted!=(existing or 0)+1:
        raise RuntimeError('Could not verify submitted notebook version; inspect Kaggle before retrying')
    write_json(destination/'submitted_version.json',{'kernel':ref,'version':submitted,'run_id':run_id})
    try:
        collect(ROOT/'Results'/'team_downloads',ROOT/'Results'/'comparison.csv')
    except TimeoutError:
        # Submission is already acknowledged. A concurrent table writer must
        # not abandon the saved job or trigger another submission.
        write_json(destination/'comparison_collection_pending.json',
                   {'reason':'comparison writer busy','submitted_version':submitted})
        print('Comparison table busy; continuing to monitor the acknowledged job.',flush=True)
    print('Notebook:',f'https://www.kaggle.com/code/{ref}',flush=True)
    print('Do not save a new version of this notebook while this monitor is running.',flush=True)
    watch(destination,env,team,args.interval,args.once,not args.no_publish,args.max_hours)
