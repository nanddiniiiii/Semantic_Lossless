"""Retrieve a saved Kaggle version without changing jobs or Git branches."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from functools import lru_cache

from .dataset import digest
from .runtime import write_json


def verify_download(directory):
    """Validate exported per-model files using their own relative roots."""
    manifests=list(Path(directory).rglob('artifacts.json'))
    verified=0
    for manifest in manifests:
        data=json.loads(manifest.read_text())
        for item in data['files']:
            root=manifest.parent.resolve();path=(root/item['path']).resolve()
            path.relative_to(root)
            if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                raise ValueError('Downloaded artifact failed verification: '+str(path))
            verified+=1
    if not manifests:raise FileNotFoundError('No exported artifact manifests found in downloaded outputs')
    return {'manifests':len(manifests),'verified_files':verified}


def collect(root, csv_path=None):
    """Combine verified training, evaluation, failure and status records once."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    output=Path(csv_path) if csv_path else root/'comparison.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.with_suffix(output.suffix+'.lock')
    from .persistence import file_lock
    with file_lock(lock) as owned:
        if not owned:raise TimeoutError('Another comparison collector owns '+str(lock))
        rows=[]
        paths={}
        # Submission/status records are local operational evidence, not verified
        # model metrics. Retain them even if remote setup produced no manifest.
        for submission in sorted(root.rglob('submission.json')):
            if 'published_output' in submission.parts:continue
            data=json.loads(submission.read_text(encoding='utf-8'))
            rows.append({**data,'record_type':'submission','artifact_source':submission.relative_to(root).as_posix()})
            status=submission.parent/'remote_status.json'
            if status.is_file():
                rows.append({**data,**json.loads(status.read_text(encoding='utf-8')),
                    'record_type':'remote_status','artifact_source':status.relative_to(root).as_posix()})
        # A concurrently downloading teammate's partial files are not evidence.
        for marker in sorted(root.rglob('retrieval_verified.json')):
            tree=marker.parent/'published_output'
            for manifest in tree.rglob('artifacts.json'):
                for item in json.loads(manifest.read_text())['files']:
                    path=(manifest.parent/item['path']).resolve()
                    path.relative_to(tree.resolve())
                    if path in paths and paths[path]!=item['sha256']:
                        raise ValueError('Conflicting artifact hashes: '+str(path))
                    paths[path]=item['sha256']

        @lru_cache(maxsize=None)
        def read_json(path):
            if path not in paths:return {}
            if digest(path)!=paths[path]:raise ValueError('Previously verified artifact changed: '+str(path))
            return json.loads(path.read_text(encoding='utf-8'))

        @lru_cache(maxsize=None)
        def identity(run):
            options=read_json(run/'resolved_config.json') or read_json(run/'requested_config.json')
            env=read_json(run/'environment.json')
            data=read_json(run/'dataset_identity.json')
            return {'algorithm':options.get('algorithm',run.name),'teammate':options.get('teammate'),
                    'train_coherence':options.get('train_coherence','symbol'),
                    'test_coherence':options.get('test_coherence','symbol'),
                    'channel_backend':options.get('channel_backend','native'),
                    'run_id':options.get('run_id',run.parent.name),'source_revision':env.get('revision'),
                    'working_tree_dirty':env.get('working_tree_dirty'),'dataset_identity':data.get('identity'),
                    **{key:options.get(key) for key in ('train_channel','train_snr_db','test_channel','test_snr_db','batch_size','crop','max_seconds')}}

        for path in sorted(paths):
            # Keep tiny CPU pilots out of the final comparison table.
            if any(part in path.parts for part in ('pilot','resume-check','resume-fixture')):continue
            source=path.relative_to(root.resolve()).as_posix()
            if path.name=='per_image.csv' and path.parent.name=='metrics':
                profile=read_json(path.parent/'profile.json') or read_json(path.parent.parent/'profile.json')
                summary=read_json(path.parent/'summary.json')
                fields={key:profile.get(key) for key in ('parameters','registered_kernel_flops',
                    'flops_coverage','latency_p50_seconds','latency_p95_seconds','peak_cuda_memory_bytes',
                    'incremental_cuda_memory_bytes','device','precision')}
                fields.update({('evaluation_'+key):summary.get(key) for key in
                    ('requested_trials','successful_trials','failed_trials','mean_over_successful_trials_only')})
                if digest(path)!=paths[path]:raise ValueError('Previously verified metrics changed: '+str(path))
                with path.open(newline='',encoding='utf-8') as stream:
                    for row in csv.DictReader(stream):
                        rows.append({**identity(path.parent.parent),**row,**fields,
                                     'record_type':'evaluation_trial','artifact_source':source})
            elif path.name=='per_image.json' and path.parent.name=='metrics':
                for failure in read_json(path).get('failures',[]):
                    rows.append({**identity(path.parent.parent),**failure,
                                 'record_type':'evaluation_failure','artifact_source':source})
            elif path.name=='training.jsonl':
                if digest(path)!=paths[path]:raise ValueError('Previously verified training log changed: '+str(path))
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        if line.strip():rows.append({**identity(path.parent),**json.loads(line),
                            'record_type':'training_step','artifact_source':source})
            elif path.name=='training_summary.json' or path.name.endswith('_status.json'):
                rows.append({**identity(path.parent),**read_json(path),
                    'record_type':'training_summary' if path.name=='training_summary.json' else 'stage_status',
                    'artifact_source':source})
            elif path.name.endswith('_initialization_failure.json'):
                rows.append({**read_json(path),'record_type':'initialization_failure','artifact_source':source})
            elif path.name=='session_summary.json':
                rows.append({**read_json(path),'record_type':'session_summary','artifact_source':source})
        keys=sorted(set().union(*(row.keys() for row in rows))) if rows else ['record_type','algorithm','teammate','run_id','artifact_source']
        temporary=output.with_suffix(output.suffix+'.tmp')
        with temporary.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
            for row in rows:
                writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value for key,value in row.items()})
        temporary.replace(output)
        return len(rows)


def main(team):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',required=True,help='owner/slug of the teammate notebook')
    parser.add_argument('--version',required=True,type=int,help='Expected saved notebook version; keep this kernel unchanged until retrieval')
    parser.add_argument('--output',type=Path,default=Path('Results/team_downloads'))
    parser.add_argument('--config-dir',type=Path,help='Optional isolated Kaggle credential directory; never print its contents')
    parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--max-hours',type=float,default=20)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval<10 or args.max_hours<=0 or args.version<=0:
        parser.error('interval>=10, max-hours>0 and a positive saved version are required')
    parts=args.kernel.split('/')
    if len(parts)!=2 or any(not p or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for ch in p) for p in parts):
        parser.error('kernel must be owner/slug')
    target=args.output.resolve()/team/(parts[0]+'__'+parts[1])/str(args.version)
    target.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ)
    if args.config_dir:environment['KAGGLE_CONFIG_DIR']=str(args.config_dir.resolve())
    ref=args.kernel+'/'+str(args.version)
    start=time.monotonic();previous=None
    while time.monotonic()-start<args.max_hours*3600:
        status=subprocess.run(['kaggle','kernels','status',ref],env=environment,capture_output=True,text=True,timeout=90)
        output=(status.stdout+'\n'+status.stderr).strip()
        # Kaggle CLI status resolves the latest run even when a version is
        # supplied. Never treat that response as proof of an exact version.
        record={'kernel':args.kernel,'expected_version':args.version,'returncode':status.returncode,'status':output,
                'status_scope':'latest kernel run; CLI does not establish version identity'}
        write_json(target/'remote_status.json',record)
        if output!=previous:print(output,flush=True);previous=output
        terminal=status.returncode==0 and any(word in output.upper() for word in ('COMPLETE','ERROR','CANCELLED'))
        if terminal:
            raise RuntimeError('Automatic retrieval disabled: Kaggle CLI status is latest-run scoped and cannot verify the requested version. Verify remote notebook identity and retrieve manually.')
        if args.once:return
        time.sleep(args.interval)
