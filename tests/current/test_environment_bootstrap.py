import subprocess
import sys
import pytest
from src import notebook_support


def test_bootstrap_avoids_ensurepip_and_targets_model_environment(tmp_path,monkeypatch):
    calls=[]
    def create(self,path):
        calls.append(('create',self.with_pip,self.system_site_packages))
        path.mkdir(parents=True)
        (path/'pyvenv.cfg').write_text('test')
    monkeypatch.setattr(notebook_support.venv.EnvBuilder,'create',create)
    commands=[]
    python=notebook_support.setup_model(tmp_path,'ssc',tmp_path/'run',command_runner=lambda command,timeout:commands.append(command))
    assert calls==[('create',False,True)]
    assert commands[0][:5]==[sys.executable,'-m','pip','--python',str(python)]
    assert 'timm==1.0.29' in commands[0]
    assert 'from timm.layers import' in commands[1][-1]


def test_bootstrap_preserves_installer_stderr(tmp_path,monkeypatch):
    destination=tmp_path/'.venv-team/ssc';destination.mkdir(parents=True)
    (destination/'pyvenv.cfg').write_text('test')
    def fail(command,stdout,stderr,timeout):
        stdout.write('specific installer failure\n');stdout.flush()
        return subprocess.CompletedProcess(command,7)
    monkeypatch.setattr(notebook_support.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        notebook_support.setup_model(tmp_path,'ssc',tmp_path/'run')
    assert 'specific installer failure' in (tmp_path/'run/environment_setup.log').read_text()


def test_retry_selects_only_failed_model_without_changing_canonical_notebook():
    import json
    from pathlib import Path
    from src.launch import bind_delivery
    notebook=json.loads((Path(__file__).parents[2]/'Notebooks/2.ipynb').read_text())
    bound=bind_delivery(notebook,'retry',7200,['ssc'])
    text=lambda value:'\n'.join(''.join(cell['source']) for cell in value['cells'])
    assert "MODELS_TO_RUN = ['ssc']" in text(bound)
    assert "MODELS_TO_RUN = ASSIGNED[:]" in text(notebook)
    with pytest.raises(ValueError):bind_delivery(notebook,'retry',7200,['not_assigned'])


def test_single_gpu_mimo_binding_disables_parallelism_and_keeps_other_slots_unselected():
    import ast,json
    from pathlib import Path
    from src.launch import bind_delivery
    notebook=json.loads((Path(__file__).parents[2]/'Notebooks/3.ipynb').read_text())
    bound=bind_delivery(notebook,'mimo-single',14400,['deepjscc_mimo'],single_gpu=True)
    sources=[''.join(cell['source']) for cell in bound['cells'] if cell['cell_type']=='code']
    for source in sources:ast.parse(source)
    assert any("MODELS_TO_RUN = ['deepjscc_mimo']" in source for source in sources)
    assert any("OVERRIDES.setdefault(_model, {})['data_parallel']=False" in source for source in sources)
