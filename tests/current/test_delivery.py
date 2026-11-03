"""No external services: recovery, publication and merge behavior with local fixtures."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runtime import write_json
from src import launch,publication


def test_recovery_does_not_submit_training_and_retries_download(tmp_path,monkeypatch):
    destination=tmp_path/'Results/team_downloads/1/u__semantic-notebook-1/r'
    write_json(destination/'submitted_version.json',{'kernel':'u/semantic-notebook-1','version':2})
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    calls=[]
    monkeypatch.setattr(launch.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(returncode=0,stdout='COMPLETE',stderr='')))
    attempts=[]
    def finish(*args):
        attempts.append(1)
        if len(attempts)==1:raise OSError('temporary download interruption')
        write_json(destination/'retrieval_verified.json',{'verified':True})
    monkeypatch.setattr(launch,'finish_retrieval',finish)
    launch.watch(destination,{},'1',once=True,publish=False)
    assert json.loads((destination/'collector_state.json').read_text())['status']=='retry_pending'
    launch.watch(destination,{},'1',once=True,publish=False)
    assert json.loads((destination/'collector_state.json').read_text())['status']=='retrieved'
    assert len(attempts)==2 and all(cmd[2]=='status' for cmd in calls)


def test_os_lock_released_after_exception(tmp_path):
    from src.persistence import file_lock
    path=tmp_path/'lock'
    with pytest.raises(RuntimeError):
        with file_lock(path) as acquired:
            assert acquired
            with file_lock(path) as duplicate:assert not duplicate
            raise RuntimeError('simulate interrupted owner')
    with file_lock(path) as acquired:assert acquired


def test_publication_without_write_permission_stays_local(tmp_path,monkeypatch):
    monkeypatch.setattr(publication,'repository_name',lambda root:'owner/repo')
    calls=[]
    monkeypatch.setattr(publication,'github_api',lambda route,*a,**kw:(calls.append(route) or {'permissions':{'push':False}}))
    result=publication.publish_event(tmp_path,{})
    assert result['status']=='pending' and len(calls)==1


def test_persistent_publication_retry_and_no_duplicate_commit(tmp_path,monkeypatch):
    event=tmp_path/'Results/publication_queue/event.json'
    write_json(event,{'event_id':'event','status':'pending','attempts':0})
    attempts=[]
    def publish(root,record):
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('non-fast-forward')
        return {'status':'published','commit':'abc'}
    monkeypatch.setattr(publication,'publish_event',publish)
    assert publication.flush(tmp_path)==['pending']
    assert publication.flush(tmp_path)==['published']
    assert publication.flush(tmp_path)==[]
    assert len(attempts)==2 and json.loads(event.read_text())['attempts']==2


def test_atomic_github_commit_contains_only_results_and_preserves_other_models(tmp_path,monkeypatch):
    metrics=tmp_path/'per_image.csv';metrics.write_text('psnr_db\n30\n')
    from src.dataset import digest
    event={'event_id':'1-run-deepjscc','fingerprint':'fp','algorithm':'deepjscc','teammate':'1','run_id':'run',
           'files':{'metrics/per_image.csv':{'local_path':str(metrics),'sha256':digest(metrics)}}}
    monkeypatch.setattr(publication,'repository_name',lambda root:'owner/repo')
    captured=[];counter=iter(range(10))
    def api(route,payload=None,method=None):
        captured.append((route,payload,method))
        if route=='repos/owner/repo':return {'permissions':{'push':True},'default_branch':'main'}
        if route.endswith('/git/ref/heads/main'):return {'object':{'sha':'head'}}
        if route.endswith('/git/commits/head'):return {'tree':{'sha':'oldtree'}}
        if 'recursive=1' in route:return {'tree':[{'path':'Results/comparison.csv','sha':'oldcsv'}]}
        if route.endswith('/git/blobs/oldcsv'):
            return {'content':base64.b64encode(b'event_id,algorithm,psnr_db\nother,swinjscc,28\n').decode()}
        if route.endswith('/git/blobs'):return {'sha':'blob'+str(next(counter))}
        if route.endswith('/git/trees'):
            assert payload['base_tree']=='oldtree'
            assert all(item['path'].startswith('Results/') for item in payload['tree'])
            return {'sha':'newtree'}
        if route.endswith('/git/commits'):return {'sha':'newcommit'}
        if route.endswith('/git/refs/heads/main'):
            assert method=='PATCH' and payload=={'sha':'newcommit','force':False}
            return {}
        raise AssertionError(route)
    monkeypatch.setattr(publication,'github_api',api)
    result=publication.publish_event(tmp_path,event)
    assert result['status']=='published' and result['commit']=='newcommit'
    data=[base64.b64decode(payload['content']) for route,payload,_ in captured if route.endswith('/git/blobs')]
    assert any(b'other' in value and b'1-run-deepjscc' in value for value in data)
    assert not any(b'last.pt' in value for value in data)
