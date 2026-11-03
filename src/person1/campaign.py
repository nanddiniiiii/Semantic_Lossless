"""Serial person1 chunks using the existing stable notebook and receipts."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ORDER
from src.runtime import write_json
from src.persistence import file_lock


def require_pilot_qualification(folder):
    folder=Path(folder)
    for name in ('pilot_passed.json','resume_probe.json','pilot_only.json'):
        path=folder/name
        if not path.is_file():raise RuntimeError('Incomplete pilot qualification: missing '+name)
    if not json.loads((folder/'resume_probe.json').read_text()).get('passed'):
        raise RuntimeError('Pilot continuation check failed')
    if json.loads((folder/'pilot_only.json').read_text()).get('status')!='pilot_completed':
        raise RuntimeError('Pilot controller did not finish qualification')


def campaign_receipts(root,username):
    parent=Path(root)/'Results/team_downloads/1'
    return sorted(parent.glob(username+'__semantic-notebook-1*/*/submission.json'),key=lambda p:p.stat().st_mtime)


def external_training(submission,canonical):
    return (submission.get('kernel')!=canonical and submission.get('person1_mode')=='train'
            and submission.get('person1_model') in ORDER)


def next_training_model(models):
    return next((n for n in ORDER if models[n]['status'] not in ('training_completed','training_external')),None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--username',required=True);p.add_argument('--interval',type=int,default=1800)
    p.add_argument('--train-qualified-first',action='store_true',help='Train already-qualified models while other integrations await review')
    p.add_argument('--training-seconds',type=int,default=14400);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.username):raise ValueError('Invalid Kaggle username')
    a.username=a.username.lower()
    root=Path(__file__).resolve().parents[2];folder=root/'Results/person1_campaign'/a.username;folder.mkdir(parents=True,exist_ok=True)
    base=root/'Results/team_downloads/1'/(a.username+'__semantic-notebook-1')
    state_path=folder/'progress.json'
    state=json.loads(state_path.read_text()) if state_path.is_file() else dict(username=a.username,models={n:dict(status='pending') for n in ORDER},handled=[])
    if state.get('username')!=a.username:raise ValueError('Campaign state belongs to another Kaggle account')
    from src.launch import notebook_revision
    reviewed=notebook_revision(json.loads((root/'Notebooks/1.ipynb').read_text()))
    if state.get('reviewed_source') not in (None,reviewed) and any(
        entry['status'] in ('partial','training_completed') for entry in state['models'].values()):
        raise ValueError('Reviewed source changed after training progress; inspect contracts before continuation')
    state['reviewed_source']=reviewed
    for entry in state['models'].values():entry['pilot_passed']=entry.get('pilot_source')==reviewed
    with file_lock(folder/'campaign.lock') as owned:
        if not owned:raise RuntimeError('A person1 campaign is already running')
        while True:
            current=notebook_revision(json.loads((root/'Notebooks/1.ipynb').read_text()))
            if current!=reviewed:
                raise RuntimeError('Notebook source changed during campaign; stop and revalidate before submitting')
            receipts=campaign_receipts(root,a.username)
            active=[];external=[]
            for receipt in receipts:
                destination=receipt.parent
                collected=destination/'collector_state.json'
                if not collected.is_file() or json.loads(collected.read_text()).get('status')!='retrieved':
                    submission=json.loads(receipt.read_text())
                    if external_training(submission,a.username+'/semantic-notebook-1'):
                        external.append(destination)
                        state['models'][submission['person1_model']].update(status='training_external',receipt=str(destination))
                        continue
                    active.append(destination);continue
                if destination.name in state['handled']:continue
                submission=json.loads(receipt.read_text());name=submission.get('person1_model')
                if name not in ORDER:continue
                summaries=[path for path in (destination/'published_output').rglob('training_summary.json') if path.parent.name==name]
                if submission.get('person1_mode')=='pilot':
                    passed=list((destination/'published_output').rglob('pilot_passed.json'))
                    try:
                        if not passed:raise RuntimeError('Person1 pilot failed; inspect retrieved evidence before retry')
                        require_pilot_qualification(passed[0].parent)
                    except RuntimeError:
                        if not a.train_qualified_first:raise
                        state['models'][name].update(status='pilot_failed_awaiting_review',pilot_passed=False,pilot_source=None)
                        state.setdefault('pilot_reviews_required',[]).append(str(destination))
                        state['handled'].append(destination.name);write_json(state_path,state)
                        print('Pilot failed; evidence preserved for review; other qualified models remain eligible:',destination,flush=True)
                        continue
                    state['models'][name]['pilot_source']=submission['source_revision']
                    state['models'][name]['pilot_passed']=submission['source_revision']==reviewed
                elif summaries:
                    summary=json.loads(summaries[0].read_text());entry=state['models'][name]
                    if summary['status']=='training_completed':
                        packaged=summaries[0].parent/'package_result.json'
                        if not packaged.is_file() or not json.loads(packaged.read_text()).get('produced'):
                            raise RuntimeError('Training finished but evaluation/package failed; inspect '+str(destination))
                    entry.update(status=summary['status'],completed_epochs=summary['completed_epochs'],receipt=str(destination))
                    if summary['status']=='partial':
                        checkpoint=destination/('checkpoint_dataset-'+name+'.json')
                        if not checkpoint.is_file():raise RuntimeError('Partial chunk has no verified continuation input')
                        source=json.loads(checkpoint.read_text())
                        if source.get('status')!='ready':raise RuntimeError('Private continuation input not ready')
                        entry['dataset']=source['dataset']
                        book=destination/'submitted-notebook.ipynb'
                        if not book.is_file():raise RuntimeError('Continuation requires the exact submitted runtime notebook')
                        entry['runtime_notebook']=str(book)
                else:
                    raise RuntimeError('Training failed before valid progress; inspect '+str(destination))
                state['handled'].append(destination.name);write_json(state_path,state)
            if active:
                command=[sys.executable,str(root/'scripts/1.py'),'--username',a.username,'--resume',str(active[-1]),'--interval',str(a.interval),'--max-hours','24']
            else:
                selected=None;mode='pilot'
                if a.train_qualified_first:
                    selected=next((n for n in ORDER if state['models'][n].get('pilot_passed') and
                        state['models'][n]['status'] not in ('training_completed','training_external')),None)
                    if selected:mode='train'
                    else:selected=next((n for n in ORDER if not state['models'][n].get('pilot_passed') and
                        state['models'][n]['status']!='pilot_failed_awaiting_review'),None)
                else:selected=next((n for n in ORDER if not state['models'][n].get('pilot_passed')),None)
                if selected is None and not a.train_qualified_first:
                    selected=next_training_model(state['models'])
                    mode='train'
                if selected is None:
                    if external:
                        state['status']='waiting_for_external_training';write_json(state_path,state)
                        time.sleep(a.interval);continue
                    if any(entry['status']!='training_completed' for entry in state['models'].values()):
                        state['status']='pilot_review_required';write_json(state_path,state);return
                    state['status']='training_finished';write_json(state_path,state)
                    from .handoff import create
                    create(root);return
                command=[sys.executable,str(root/'scripts/1.py'),'--username',a.username,'--model',selected,'--mode',mode,
                         '--training-seconds',str(a.training_seconds),'--interval',str(a.interval),'--max-hours','24','--create-missing']
                if receipts:command.append('--new-run')
                if mode=='train' and state['models'][selected].get('dataset'):
                    book=state['models'][selected].get('runtime_notebook')
                    if not book:raise RuntimeError('Missing frozen continuation runtime; refusing changed code')
                    command+=['--continue-from',state['models'][selected]['dataset'],'--runtime-notebook',book]
            command.append('--no-publish')
            state.update(status='running',command=command,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write_json(state_path,state);print('Campaign command:',command,flush=True)
            result=subprocess.run(command,cwd=root)
            if result.returncode:
                state.update(status='needs_investigation',returncode=result.returncode)
                write_json(state_path,state);raise RuntimeError('Collector/submission failed; campaign stopped with receipts preserved')


if __name__=='__main__':main()
