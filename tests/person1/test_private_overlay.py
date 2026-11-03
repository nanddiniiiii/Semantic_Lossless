import json
from src import launch


def test_private_runtime_overlay_roundtrip(tmp_path,monkeypatch):
    source=tmp_path/'local';folder=source/'src/person1';folder.mkdir(parents=True)
    content=b'VALUE = 17\n';(folder/'worker.py').write_bytes(content)
    monkeypatch.setattr(launch,'ROOT',source)
    notebook=launch.private_runtime_overlay({'cells':[{},{}]})
    destination=tmp_path/'remote';destination.mkdir()
    exec(''.join(notebook['cells'][2]['source']),{'REPO':destination,'REVISION':'a'*40})
    assert (destination/'src/person1/worker.py').read_bytes()==content
    receipt=json.loads((destination/'private_runtime_overlay.json').read_text())
    assert receipt['sha256']==notebook['metadata']['private_runtime_overlay_sha256']
    assert receipt['base_revision']=='a'*40


def test_continuation_keeps_existing_runtime_payload(tmp_path,monkeypatch):
    (tmp_path/'Results').mkdir();(tmp_path/'Results/NO_GITHUB_PUBLICATION').touch()
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    source={'cells':[{'cell_type':'code','source':['DELIVERY_RUN_ID = None\n']},
        {'id':'private-runtime-overlay','cell_type':'code','source':['IMMUTABLE_PAYLOAD = 17\n']}],
        'metadata':{'private_runtime_overlay_sha256':'f'*64}}
    bound=launch.bind_delivery(source,'next',model='adjscc',mode='train',frozen_runtime=True)
    assert bound['cells'][1]==source['cells'][1]
    assert bound['metadata']==source['metadata']
    assert len(bound['cells'])==2
