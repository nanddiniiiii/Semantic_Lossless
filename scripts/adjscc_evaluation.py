"""Prepare and submit private CPU evaluation of the completed ADJSCC weights."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['PATH'] = str(ROOT / '.venv-collector/Scripts') + os.pathsep + os.environ['PATH']
from src.person1.auth import ensure_fresh_authorization
from src.person1.collection import owned_dataset_exists

ensure_fresh_authorization('hmmmkk')
receipt = ROOT / 'Results/team_downloads/1/hmmmkk__semantic-notebook-1/b20ed0d81fa44a48b7241b17baec8225'
run = next((receipt / 'published_output').glob('team_results/1/*/adjscc'))
folder = ROOT / 'Results/person1_campaign/hmmmkk/adjscc-evaluation-only'
asset = folder / 'input'
asset.mkdir(parents=True, exist_ok=True)
for source, target in [('checkpoints/last-tf.zip', 'checkpoint.bin'), ('resolved_config.json', 'config.json'), ('training_summary.json', 'training_summary.json')]:
    shutil.copy2(run / source, asset / target)
SHA = '385b3c1a85b764c66e8e5ae156cbcab21a944873fdbcea511830db5d56784df7'
assert hashlib.sha256((asset / 'checkpoint.bin').read_bytes()).hexdigest() == SHA
assert json.loads((asset / 'training_summary.json').read_text())['completed_epochs']['codec'] == 150
ref = 'hmmmkk/p1-adjscc-final150-evaluation'
(asset / 'dataset-metadata.json').write_text(json.dumps(dict(id=ref, title='Private ADJSCC final150 evaluation', licenses=[{'name': 'other'}])))
if not owned_dataset_exists(ref, os.environ.copy()):
    subprocess.run(['kaggle', 'datasets', 'create', '-p', str(asset), '-t'], check=True)
for attempt in range(30):
    status = subprocess.run(['kaggle', 'datasets', 'status', ref], capture_output=True, text=True)
    if status.returncode == 0 and 'ready' in status.stdout.lower():
        break
    time.sleep(10)
else:
    raise RuntimeError('Private evaluation input is not ready')

notebook = json.loads((receipt / 'submitted-notebook.ipynb').read_text())
notebook['cells'] = notebook['cells'][:3]
evaluation = r'''
import json, shutil, hashlib, subprocess
from src.runtime import export
from src.person1.controller import command, score
from src.person1.environments import prepare
from src.dataset import discover, freeze
from src.sources import fetch_source, source_path
asset = next(Path('/kaggle/input').rglob('checkpoint.bin')).parent
assert hashlib.sha256((asset/'checkpoint.bin').read_bytes()).hexdigest() == '385b3c1a85b764c66e8e5ae156cbcab21a944873fdbcea511830db5d56784df7'
run = Path('/kaggle/working/adjscc-evaluation')
run.mkdir(); (run/'checkpoints').mkdir()
shutil.copy2(asset/'checkpoint.bin', run/'checkpoints/last-tf.zip')
shutil.copy2(asset/'training_summary.json', run/'training_summary.json')
config = json.loads((asset/'config.json').read_text())
manifest = run/'dataset.json'
freeze(discover('/kaggle/input'), manifest, workers=2)
config['manifest'] = str(manifest)
fetch_source('adjscc_author'); config['source_dir'] = str(source_path('adjscc_author'))
(run/'resolved_config.json').write_text(json.dumps(config))
python, env = prepare('adjscc', REPO, run/'dependencies.log', command)
try:
    # Seed before native channel graph construction; training weights stay intact.
    wrapper = 'from src.person1 import tf_worker;import tensorflow as tf;tf.keras.utils.set_random_seed(22);tf_worker.main()'
    command([python, '-u', '-c', wrapper, 'evaluate', '--config', str(run/'resolved_config.json'), '--run', str(run)], run/'evaluate.log', 36000, env)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytorch-msssim==1.0.0', 'lpips==0.1.4'], check=True)
    score(dict(run=str(run), config=config))
    summary = json.loads((run/'metrics/summary.json').read_text())
    summary.update(evaluation_device='CPU', gpu_latency_verified=False)
    (run/'metrics/summary.json').write_text(json.dumps(summary, indent=2))
    (run/'evaluation_completed.json').write_text(json.dumps(dict(passed=True, trials=240, device='CPU')))
finally:
    export(run)
'''
notebook['cells'].append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=evaluation.splitlines(True)))
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), 'evaluation-notebook', 'exec')
(folder / 'notebook.ipynb').write_text(json.dumps(notebook))
slug = 'hmmmkk/person1-adjscc-final150-evaluation'
(folder / 'kernel-metadata.json').write_text(json.dumps(dict(id=slug, title='Person1 ADJSCC final150 evaluation', code_file='notebook.ipynb', language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=True, dataset_sources=['soumikrakshit/div2k-high-resolution-images', 'daehoyang/flickr2k', 'drxinchengzhu/kodak24', ref])))
if (folder / 'submission_acknowledged.json').exists():
    raise RuntimeError('Evaluation was already submitted; inspect its status instead of duplicating')
subprocess.run(['kaggle', 'kernels', 'push', '-p', str(folder)], check=True)
(folder / 'submission_acknowledged.json').write_text(json.dumps(dict(kernel=slug, checkpoint_sha256=SHA, device='CPU')))
print('Submitted evaluation: https://www.kaggle.com/code/' + slug, flush=True)
