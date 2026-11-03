"""Keep the latest main's teammate selection alongside Person1 continuation."""
import ast
from pathlib import Path
import pytest

def binder():
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'src/launch.py').read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='bind_delivery')
    env={'ROOT':root}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'bind_delivery','exec'),env)
    return env['bind_delivery']

def notebook():
    return dict(metadata={'experiment':{'models':['x','y']}},cells=[dict(cell_type='code',source=[
        'DELIVERY_RUN_ID = None\n','MODELS_TO_RUN = []\n','PERSON1_MODEL = None\n','PERSON1_MODE = None\n'])])

def test_preserves_other_teammate_positional_subset():
    result=binder()(notebook(),'r',14400,['x'])
    assert "MODELS_TO_RUN = ['x']\n" in result['cells'][0]['source']

def test_person1_single_model_and_mode():
    result=binder()(notebook(),'r',14400,'isec','pilot')
    assert "PERSON1_MODEL = 'isec'\n" in result['cells'][0]['source']
    assert "PERSON1_MODE = 'pilot'\n" in result['cells'][0]['source']
