"""Retrieve the already submitted ADJSCC arithmetic audit without resubmission."""
from pathlib import Path
import os,sys,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ['PATH']=str(ROOT/'.venv-collector/Scripts')+os.pathsep+os.environ['PATH']
from src.person1.auth import ensure_fresh_authorization
from src.monitor import verify_download
from src.persistence import file_lock
folder=ROOT/'Results/person1_flops/adjscc'
output=folder/(sys.argv[1] if len(sys.argv)>1 else 'outputs')
output.resolve().relative_to(folder.resolve())
with file_lock(folder/'collector.lock') as owned:
    if not owned:raise RuntimeError('Arithmetic audit collector already active')
    for _ in range(1440):
        ensure_fresh_authorization('hmmmkk')
        state=subprocess.run(['kaggle','kernels','status','hmmmkk/person1-adjscc-arithmetic-audit'],capture_output=True,text=True)
        print(state.stdout.strip(),flush=True)
        if state.returncode:raise RuntimeError('Cannot verify audit status')
        if 'RUNNING' in state.stdout or 'QUEUED' in state.stdout:
            time.sleep(60);continue
        subprocess.run(['kaggle','kernels','output','hmmmkk/person1-adjscc-arithmetic-audit','-p',str(output)],check=True)
        checked=verify_download(output)
        marker='retrieval_verified.json' if output.name=='outputs' else output.name+'-retrieval_verified.json'
        (folder/marker).write_text(json.dumps(checked,indent=2))
        summaries=list(output.rglob('summary.json'))
        if not summaries:raise RuntimeError('Audit failed; saved diagnostics require review')
        print('Arithmetic audit downloaded and hash-verified; logical coverage review still required',flush=True)
        break
