"""Durable local results outbox. GitHub writes touch only curated result paths."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import re
import subprocess

from .dataset import digest
from .runtime import write_json
from .sharing import MODEL_NAMES, verify_bundle


def github_api(route, payload=None, method=None):
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        request=Request('https://api.github.com/'+route,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                     'Content-Type':'application/json','User-Agent':'SemanticLossless-results'},
            method=method or ('POST' if payload is not None else 'GET'))
        try:
            with urlopen(request,timeout=120) as response:
                raw=response.read().decode()
                return json.loads(raw) if raw.strip() else {}
        except (HTTPError,URLError) as error:
            raise RuntimeError('GitHub API request failed; results remain queued') from None
    if shutil.which('gh') is None:
        raise RuntimeError('GitHub authentication unavailable: install/login gh, or configure GH_TOKEN locally; results retained')
    command=['gh','api',route]
    if method:command+=['--method',method]
    if payload is not None:command+=['--input','-']
    result=subprocess.run(command,input=json.dumps(payload) if payload is not None else None,
                          capture_output=True,text=True,timeout=120)
    if result.returncode:
        # Authentication/server messages are not copied into public result artifacts.
        raise RuntimeError('GitHub API request failed; credentials/access/network may need attention')
    return json.loads(result.stdout) if result.stdout.strip() else {}


def repository_name(root):
    result=subprocess.run(['git','remote','get-url','origin'],cwd=root,capture_output=True,text=True,check=True)
    match=re.fullmatch(r'(?:https://github.com/|git@github.com:)([\w.-]+/[\w.-]+?)(?:\.git)?',result.stdout.strip())
    if not match:raise ValueError('Result publication requires a github.com origin')
    return match[1]


def enqueue_bundles(root, folders):
    root=Path(root)
    queued=[]
    for value in folders:
        folder=Path(value).resolve();metadata=verify_bundle(folder)
        key=f"{metadata['teammate']}-{metadata['run_id']}-{metadata['algorithm']}"
        # Identifiers are already validated by the importer; recheck external callers.
        if not re.fullmatch(r'[A-Za-z0-9_-]+',key):raise ValueError('Invalid event identifier')
        files={}
        for manifest_item in metadata['files']:
            rel=manifest_item['path']
            path=(folder/rel).resolve()
            path.relative_to(folder)
            allowed=(path.parent==folder/'metrics' and path.suffix in {'.csv','.json'}) or rel in {
                'training_summary.json','model.json','dataset_identity.json','native_data_identity.json','evaluation_reload.json'}
            if path.is_file() and allowed:
                if path.stat().st_size>4*1024*1024:raise ValueError('Result file exceeds publication size budget: '+rel)
                files[rel]={'local_path':str(path),'sha256':digest(path)}
        if not files:raise ValueError('No publishable model results: '+key)
        listed={item['path'] for item in metadata['files']}
        options=json.loads((folder/'resolved_config.json').read_text()) if 'resolved_config.json' in listed else {}
        protocol={k:options.get(k) for k in ('train_channel','test_channel','train_snr_db','test_snr_db',
            'train_coherence','test_coherence','channel_backend','crop','batch_size','seed','swin_variant','variant')}
        for stage in ('train','test'):
            protocol[stage+'_channel']=options.get(stage+'_channel','awgn') if metadata['algorithm']!='sqgan' else 'not_simulated'
            protocol[stage+'_snr_db']=options.get(stage+'_snr_db',options.get('snr_db',10)) if metadata['algorithm']!='sqgan' else None
            protocol[stage+'_coherence']=options.get(stage+'_coherence','symbol') if protocol[stage+'_channel']=='rayleigh_perfect_csi' else None
        protocol['channel_backend']=options.get('channel_backend','infrastructure' if metadata['algorithm'] in {'deepjscc','adjscc'} else 'native')
        if 'model.json' in listed:
            native=json.loads((folder/'model.json').read_text())
            protocol['native_channel']=native.get('evaluation_channel',native.get('channel'))
            protocol['rate_scope']=native.get('rate_scope')
        fingerprint=hashlib.sha256(json.dumps({'files':{k:v['sha256'] for k,v in files.items()},'protocol':protocol},sort_keys=True).encode()).hexdigest()
        target=root/'Results/publication_queue'/(key+'.json')
        if target.exists():
            prior=json.loads(target.read_text())
            if prior['fingerprint']!=fingerprint:raise ValueError('Publication event changed: '+key)
        else:
            write_json(target,{'event_id':key,'fingerprint':fingerprint,'algorithm':metadata['algorithm'],
                'teammate':metadata['teammate'],'run_id':metadata['run_id'],'protocol':protocol,'files':files,'status':'pending','attempts':0})
        queued.append(target)
    return queued


def merge_csv(previous, event, contents):
    rows=list(csv.DictReader(io.StringIO(previous))) if previous else []
    rows=[row for row in rows if row.get('event_id')!=event['event_id']]
    base={**event.get('protocol',{}),**{k:event[k] for k in ('event_id','algorithm','teammate','run_id')}}
    if 'metrics/per_image.csv' in contents:
        rows.extend({**row,**base,'record_type':'evaluation_trial'} for row in
                    csv.DictReader(io.StringIO(contents['metrics/per_image.csv'].decode('utf-8-sig'))))
    for name,kind in [('metrics/summary.json','evaluation_summary'),('training_summary.json','training_summary'),
                      ('metrics/profile.json','inference_profile')]:
        if name in contents:
            values=json.loads(contents[name])
            rows.append({**{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in values.items()},
                         **base,'record_type':kind})
    if not any(row.get('event_id')==event['event_id'] for row in rows):
        rows.append({**base,'record_type':'model_results_received'})
    fields=sorted(set().union(*(row.keys() for row in rows)))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()


def publish_event(root, event):
    repo=repository_name(root);route=f'repos/{repo}'
    info=github_api(route)
    if not info.get('permissions',{}).get('push'):
        return {'status':'pending','reason':'No GitHub write permission; retained locally'}
    branch=info['default_branch']
    head=github_api(route+'/git/ref/heads/'+branch)['object']['sha']
    commit=github_api(route+'/git/commits/'+head)
    tree=github_api(route+'/git/trees/'+commit['tree']['sha']+'?recursive=1')
    if tree.get('truncated'):raise RuntimeError('Remote tree truncated; cannot verify existing result event')
    prefix='Results/published/'+event['event_id']+'/'
    receipt=json.dumps({**{k:event[k] for k in ('event_id','algorithm','teammate','run_id','fingerprint')},'protocol':event.get('protocol',{})},sort_keys=True).encode()
    blob_sha=lambda data:hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    previous_receipt=next((item for item in tree['tree'] if item['path']==prefix+'receipt.json'),None)
    if previous_receipt:
        if previous_receipt['sha']!=blob_sha(receipt):raise ValueError('Conflicting published result event')
        return {'status':'published','repository':repo,'branch':branch,'already_present':True}
    contents={}
    for relative,item in event['files'].items():
        path=Path(item['local_path'])
        if digest(path)!=item['sha256']:raise ValueError('Queued result changed: '+relative)
        contents[relative]=path.read_bytes()
    old_csv=next((item for item in tree['tree'] if item['path']=='Results/comparison.csv'),None)
    previous=''
    if old_csv:
        blob=github_api(route+'/git/blobs/'+old_csv['sha'])
        previous=base64.b64decode(blob['content']).decode('utf-8-sig')
    output={prefix+name:data for name,data in contents.items()}
    output[prefix+'receipt.json']=receipt
    output['Results/comparison.csv']=merge_csv(previous,event,contents)
    if sum(map(len,output.values()))>16*1024*1024:raise ValueError('Result event exceeds Git publication budget')
    entries=[]
    for path,data in output.items():
        blob=github_api(route+'/git/blobs',{'content':base64.b64encode(data).decode(),'encoding':'base64'})
        entries.append({'path':path,'mode':'100644','type':'blob','sha':blob['sha']})
    new_tree=github_api(route+'/git/trees',{'base_tree':commit['tree']['sha'],'tree':entries})
    new_commit=github_api(route+'/git/commits',{'message':'Record model results: '+event['event_id'],
                     'tree':new_tree['sha'],'parents':[head]})
    # Non-fast-forward writes fail rather than overwriting another teammate's commit.
    github_api(route+'/git/refs/heads/'+branch,{'sha':new_commit['sha'],'force':False},method='PATCH')
    return {'status':'published','repository':repo,'branch':branch,'commit':new_commit['sha']}


def flush(root):
    """Attempt each pending event once; any failure remains recoverable on disk."""
    root=Path(root);outcomes=[]
    queue=root/'Results/publication_queue';queue.mkdir(parents=True,exist_ok=True)
    from .persistence import file_lock
    with file_lock(queue/'.writer.lock') as owned:
        if not owned:
            print('Another publisher owns the queue; pending results retained',flush=True)
            return []
        for path in sorted(queue.glob('*.json')):
            event=json.loads(path.read_text())
            if event['status']=='published':continue
            event['attempts']+=1
            try:event.update(publish_event(root,event))
            except (RuntimeError,ValueError,OSError,subprocess.SubprocessError) as error:
                event.update(status='pending',reason=str(error))
            write_json(path,event);outcomes.append(event['status'])
            print('Result publication:',event['event_id'],event['status'],event.get('reason',''),flush=True)
    return outcomes
