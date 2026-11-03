"""Fetch remaining reconstruction groups independently and verify before copying."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
BASE=ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only'
TARGET=BASE/'outputs/adjscc-evaluation'
manifest=json.loads((TARGET/'artifacts.json').read_text())
expected={r['path']:r for r in manifest['files']}
def group(indices):
    tag=str(indices[0]);folder=BASE/('parallel-'+tag)
    pattern='predictions/('+'|'.join('%03d'%i for i in indices)+')-[0-9]+[.]npy$'
    with (BASE/('parallel-'+tag+'.log')).open('w') as log:
        subprocess.run([str(ROOT/'.venv-collector/Scripts/kaggle.exe'),'kernels','output','hmmmkk/person1-adjscc-final150-evaluation','-p',str(folder),'--file-pattern',pattern],stdout=log,stderr=log,check=True)
    for p in folder.rglob('*.npy'):
        relative='predictions/'+p.name;spec=expected[relative]
        assert p.stat().st_size==spec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
        target=TARGET/relative
        if not target.exists():
            temporary=target.with_suffix('.verified-copy');shutil.copy2(p,temporary);temporary.replace(target)
    print('Verified download group',indices,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(group,[list(range(6,12)),list(range(12,18)),list(range(18,24))]))
