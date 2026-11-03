"""Collect and verify the separate ADJSCC evaluation; never submit a job."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ['PATH']=str(ROOT/'.venv-collector/Scripts')+os.pathsep+os.environ['PATH']
from src.person1.auth import ensure_fresh_authorization
from src.monitor import verify_download
from src.persistence import file_lock
folder=ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only'
with file_lock(folder/'collector.lock') as owned:
    if not owned:raise RuntimeError('Evaluation collector already active')
    for _ in range(1800):
        if not (folder/'submission_acknowledged.json').exists():
            time.sleep(15);continue
        ensure_fresh_authorization('hmmmkk')
        result=subprocess.run(['kaggle','kernels','status','hmmmkk/person1-adjscc-final150-evaluation'],capture_output=True,text=True)
        print(result.stdout.strip(),flush=True)
        if result.returncode:raise RuntimeError('Evaluation status unavailable; no submission attempted')
        if any(word in result.stdout for word in ['RUNNING','QUEUED']):
            time.sleep(60);continue
        output=folder/'outputs'
        subprocess.run(['kaggle','kernels','output','hmmmkk/person1-adjscc-final150-evaluation','-p',str(output)],check=True)
        verified=verify_download(output)
        (folder/'retrieval_verified.json').write_text(json.dumps(verified,indent=2))
        marker=next(output.rglob('evaluation_completed.json'),None)
        if marker is None:raise RuntimeError('Evaluation failed; downloaded logs preserved')
        summary=next(output.rglob('metrics/summary.json'))
        measured=json.loads(summary.read_text())
        assert measured['successful_trials']==240 and measured['failed_trials']==0
        (folder/'evaluation_verified.json').write_text(json.dumps(dict(passed=True,trials=240,device='CPU',verification=verified),indent=2))
        print('ADJSCC Kodak evaluation verified:240/240',flush=True)
        break
    else:raise TimeoutError('Evaluation collection window exceeded')
