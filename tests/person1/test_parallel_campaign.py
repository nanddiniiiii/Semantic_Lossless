from src.person1.campaign import campaign_receipts,external_training,next_training_model
from src.person1.protocol import ORDER


def test_receipts_include_separate_training_identity_but_exclude_other_account(tmp_path):
    root=tmp_path/'Results/team_downloads/1'
    for folder in ('u__semantic-notebook-1','u__semantic-notebook-1-deepjscc','other__semantic-notebook-1'):
        p=root/folder/'run/submission.json';p.parent.mkdir(parents=True);p.write_text('{}')
    assert len(campaign_receipts(tmp_path,'u'))==2


def test_parallel_training_is_not_duplicated_or_treated_as_complete():
    job={'kernel':'u/semantic-notebook-1-deepjscc','person1_mode':'train','person1_model':'deepjscc'}
    assert external_training(job,'u/semantic-notebook-1')
    models={n:{'status':'pending'} for n in ORDER};models['deepjscc']['status']='training_external'
    assert next_training_model(models)=='adjscc'
    for n in ORDER[1:]:models[n]['status']='training_completed'
    assert next_training_model(models) is None
    assert models['deepjscc']['status']=='training_external'


def test_campaign_starts_qualified_adjscc_while_deepjscc_trains(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import pytest
    from src.person1 import campaign
    source='a'*40
    monkeypatch.setattr(campaign,'__file__',str(tmp_path/'src/person1/campaign.py'))
    notebook=tmp_path/'Notebooks/1.ipynb';notebook.parent.mkdir()
    notebook.write_text(json.dumps({'cells':[{'cell_type':'code','source':["REVISION = '"+source+"'\n"]}]}))
    folder=tmp_path/'Results/person1_campaign/u';folder.mkdir(parents=True)
    models={n:{'status':'pending','pilot_passed':n!='ntscc_plus',
               'pilot_source':source if n!='ntscc_plus' else None} for n in ORDER}
    (folder/'progress.json').write_text(json.dumps({'username':'u','models':models,'handled':[],'reviewed_source':source}))
    receipt=tmp_path/'Results/team_downloads/1/u__semantic-notebook-1-deepjscc/r/submission.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'kernel':'u/semantic-notebook-1-deepjscc','person1_mode':'train','person1_model':'deepjscc'}))
    monkeypatch.setattr('sys.argv',['campaign','--username','u','--train-qualified-first'])
    commands=[]
    monkeypatch.setattr(campaign.subprocess,'run',lambda command,**kwargs:(commands.append(command) or SimpleNamespace(returncode=1)))
    with pytest.raises(RuntimeError,match='campaign stopped'):campaign.main()
    assert commands[0][commands[0].index('--model')+1]=='adjscc'
    assert commands[0][commands[0].index('--mode')+1]=='train'
    assert '--no-publish' in commands[0]
