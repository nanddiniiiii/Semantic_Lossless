"""Submit one team notebook from the user's own CLI and collect saved outputs."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
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


def bind_delivery(notebook, run_id, training_seconds=14400, models=None):
    import copy
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
            if line.startswith('DELIVERY_RUN_ID = '):
                if found:raise ValueError('Duplicate notebook delivery identity')
                lines[index]=replacement;found=True
        cell['source']=lines
    if not found:
        cell=next(c for c in notebook['cells'] if c.get('cell_type','code')=='code')
        cell['source']=[replacement]+cell['source']
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
                terminal=any(word in upper for word in ('COMPLETE','ERROR','CANCELLED'))
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


def main(team):
    parser=argparse.ArgumentParser(description='Submit this team notebook and retrieve its completed outputs.')
    parser.add_argument('--username',help='Your Kaggle username; prompted if omitted')
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
    with file_lock(ROOT/'Results/team_downloads'/team/(username+'__semantic-notebook-'+team)/'launcher.lock') as owned:
        if not owned:raise RuntimeError('Another launcher/collector already owns this teammate notebook')
        return dispatch(team,args,username)


def dispatch(team,args,username):
    if not 0<args.training_seconds<=14400:raise ValueError('Training seconds must be in (0,14400]')
    if args.max_hours<=0:raise ValueError('max-hours must be positive')
    if args.interval<10:raise ValueError('Polling interval must be at least 10 seconds')
    if shutil.which('kaggle') is None:raise RuntimeError('Install and authenticate the Kaggle CLI in this Python environment first')
    env=os.environ.copy()
    if args.config_dir:
        env['KAGGLE_CONFIG_DIR']=str(Path(args.config_dir).expanduser().resolve())
        os.environ['KAGGLE_CONFIG_DIR']=env['KAGGLE_CONFIG_DIR']
    slug=f'semantic-notebook-{team}'
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
    notebook_path=ROOT/'Notebooks'/(team+'.ipynb')
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
        if not any(word in message.upper() for word in ('COMPLETE','ERROR','CANCELLED')):
            raise RuntimeError('Existing notebook is active; refusing to replace it: '+message)
    run_id=uuid.uuid4().hex
    destination=ROOT/'Results'/'team_downloads'/team/(username+'__'+slug)/run_id
    destination.mkdir(parents=True,exist_ok=False)
    metadata={'id':ref,'title':slug,'code_file':'notebook.ipynb','language':'python',
        'kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
        'machine_shape':'NvidiaTeslaT4','dataset_sources':DATASETS+args.asset,'kernel_sources':[],
        'competition_sources':[],'model_sources':[]}
    write_json(destination/'submission.json',{'kernel':ref,'team':team,'datasets':DATASETS+args.asset,
        'source_revision':revision,'training_seconds_per_model':args.training_seconds,'selected_models':getattr(args,'models',None),
        'previous_version':existing,'run_id':run_id,
        'note':'Stable kernel identity; do not create another version before retrieval'})
    print('Submitting private notebook:',ref,flush=True)
    print('The three common image datasets will be attached automatically. Additional private model assets may require setup.',flush=True)
    with tempfile.TemporaryDirectory(prefix='semantic-kaggle-') as tmp:
        folder=Path(tmp)
        (folder/'notebook.ipynb').write_text(json.dumps(bind_delivery(notebook,run_id,args.training_seconds,getattr(args,'models',None))),encoding='utf-8')
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        subprocess.run(['kaggle','kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4'],env=env,check=True,timeout=180)
    submitted=current_version(username,slug)
    if submitted is None or submitted!=(existing or 0)+1:
        raise RuntimeError('Could not verify submitted notebook version; inspect Kaggle before retrying')
    write_json(destination/'submitted_version.json',{'kernel':ref,'version':submitted,'run_id':run_id})
    collect(ROOT/'Results'/'team_downloads',ROOT/'Results'/'comparison.csv')
    print('Notebook:',f'https://www.kaggle.com/code/{ref}',flush=True)
    print('Do not save a new version of this notebook while this monitor is running.',flush=True)
    watch(destination,env,team,args.interval,args.once,not args.no_publish,args.max_hours)
