import json
import pytest
from src.launch import bind_delivery,notebook_revision
from src.person1.protocol import require_complete


def test_model_selection_is_bound_without_changing_source_identity():
    nb={'cells':[{'cell_type':'code','source':["REVISION = '"+'a'*40+"'\n",'TRAINING_SECONDS = 14400\n',
        'DELIVERY_RUN_ID = None\n',"PERSON1_MODEL = 'deepjscc'\n","PERSON1_MODE = 'pilot'\n"]}]}
    bound=bind_delivery(nb,'run',600,'isec','train')
    code=''.join(bound['cells'][0]['source']);scope={};exec(code,scope)
    assert scope['PERSON1_MODEL']=='isec' and scope['PERSON1_MODE']=='train'
    assert scope['TRAINING_SECONDS']==600 and scope['DELIVERY_RUN_ID']=='run'
    assert notebook_revision(bound)=='a'*40
    assert "PERSON1_MODEL = 'deepjscc'" in ''.join(nb['cells'][0]['source'])


def test_completed_means_every_required_epoch_and_stage():
    with pytest.raises(ValueError):require_complete({'status':'training_completed','completed_epochs':{'codec':149}},{'codec':150})
    with pytest.raises(ValueError):require_complete({'status':'training_completed','completed_epochs':{'jscc':150,'denoiser':149}},{'jscc':150,'denoiser':150})
    require_complete({'status':'training_completed','completed_epochs':{'jscc':150,'denoiser':150}},{'jscc':150,'denoiser':150})
