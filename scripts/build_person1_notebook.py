"""Generate only person1's numbered notebook; other teammates are untouched."""
import argparse
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(revision):
    code=f'''TRAINING_SECONDS = 14400
DELIVERY_RUN_ID = None
PERSON1_MODEL = 'deepjscc'
PERSON1_MODE = 'pilot'
from pathlib import Path
import os, subprocess, sys
os.environ['GIT_LFS_SKIP_SMUDGE']='1'
URL = 'https://github.com/nanddiniiiii/Semantic_Lossless.git'
REVISION = {revision!r}
REPO = Path('/tmp/Semantic_Lossless')
if not REPO.exists():
    subprocess.run(['git','clone','--filter=blob:none',URL,str(REPO)],check=True,env={{**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'}})
subprocess.run(['git','-C',str(REPO),'fetch','origin',REVISION],check=True)
subprocess.run(['git','-C',str(REPO),'checkout','--detach',REVISION],check=True)
actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
if actual != REVISION:raise RuntimeError('Source revision mismatch')
os.environ['SEMANTIC_LOSSLESS_ROOT']=str(REPO)
os.environ['PYTHONPATH']=str(REPO)
os.chdir(REPO);sys.path.insert(0,str(REPO))
print('Reviewed source:',actual,'model:',PERSON1_MODEL,'mode:',PERSON1_MODE,flush=True)
'''
    sources=[('setup',code),('initialize',"from src.person1.controller import setup\ntry:\n    CONTEXT=setup(REPO,PERSON1_MODEL,PERSON1_MODE,DELIVERY_RUN_ID,TRAINING_SECONDS)\nexcept Exception:\n    from src.runtime import export\n    for folder in Path('/kaggle/working/team_results').glob('*'):\n        if folder.is_dir():export(folder)\n    raise\n"),
             ('pilot-train-evaluate-export',"from src.person1.controller import execute\nexecute(CONTEXT)\n")]
    cells=[dict(cell_type='markdown',id='overview',metadata={},source=[
        '# Person 1: DeepJSCC → ADJSCC → ISEC → NTSCC+\n',
        'One model per saved version, through scripts/1.py. Default is a real-data pilot only.\n',
        'Training chunks preserve progress and remain partial until150 complete epochs per required stage.\n',
        'Use T4×2, Internet, frozen DIV2K800 + Flickr2650 / DIV2K100 / Kodak24. Weights remain outside Git.\n'])]
    for identifier,source in sources:
        ast.parse(source);cells.append(dict(cell_type='code',id=identifier,metadata={},execution_count=None,outputs=[],source=source.splitlines(True)))
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),
                  person1_runtime=True),nbformat=4,nbformat_minor=5)
    (ROOT/'Notebooks/1.ipynb').write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);a=p.parse_args();build(a.revision)
