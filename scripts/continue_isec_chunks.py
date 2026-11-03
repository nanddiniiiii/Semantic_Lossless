"""Continue verified ISEC chunks on one stable account/notebook identity."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from src.person1.auth import ensure_fresh_authorization
from src.persistence import file_lock
from src.runtime import write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--username',required=True);a=parser.parse_args()
    owner=a.username.lower();slug='semantic-notebook-1-isec'
    folder=ROOT/'Results/person1_campaign'/owner;folder.mkdir(parents=True,exist_ok=True)
    progress=folder/'isec_chunk_coordinator.json'
    os.environ['PATH']=str(ROOT/'.venv-collector/Scripts')+os.pathsep+os.environ['PATH']
    if not (ROOT/'Results/NO_GITHUB_PUBLICATION').exists():raise RuntimeError('Publication block must remain present')
    with file_lock(folder/'isec_chunk_coordinator.lock') as owned:
        if not owned:raise RuntimeError('ISEC continuation coordinator already active')
        while True:
            ensure_fresh_authorization(owner)
            base=ROOT/'Results/team_downloads/1'/(owner+'__'+slug)
            files=sorted(base.glob('*/submission.json'),key=lambda p:p.stat().st_mtime)
            if not files:raise RuntimeError('No acknowledged ISEC job; coordinator never starts from scratch')
            receipt=files[-1].parent
            if not (receipt/'submitted_version.json').exists():time.sleep(60);continue
            state_file=receipt/'collector_state.json'
            state=json.loads(state_file.read_text()) if state_file.exists() else {}
            if state.get('status')!='retrieved':
                # The numbered launcher owns the identity lock for its entire
                # watch loop, even between short per-receipt collection locks.
                with file_lock(base/'launcher.lock') as launcher_available:pass
                if not launcher_available:
                    write_json(progress,dict(status='waiting_for_existing_collector',receipt=str(receipt)));time.sleep(60);continue
                with file_lock(receipt/'collector.lock') as available:pass
                if not available:
                    write_json(progress,dict(status='waiting_for_existing_collector',receipt=str(receipt)));time.sleep(60);continue
                write_json(progress,dict(status='recovering_existing_receipt',receipt=str(receipt)))
                command=[sys.executable,'scripts/1.py','--username',owner,'--slug',slug,'--model','isec','--mode','train','--resume',str(receipt),'--interval','60','--no-publish','--max-hours','24']
            else:
                if state.get('remote_success') is False:raise RuntimeError('ISEC run failed; preserve and investigate before another submission: '+str(receipt))
                verified=receipt/'retrieval_verified.json'
                if not verified.exists():raise RuntimeError('Downloaded ISEC outputs lack hash verification')
                summaries=[p for p in (receipt/'published_output').rglob('training_summary.json') if p.parent.name=='isec']
                if len(summaries)!=1:raise RuntimeError('Missing or ambiguous production training summary')
                summary=json.loads(summaries[0].read_text());epochs=summary['completed_epochs']
                # Performance reporting is local analysis of already verified
                # artifacts; missing diagnostics must not discard training.
                try:
                    from person1_speed_report import save_speed_report
                    report=save_speed_report(summaries[0].parent,receipt,folder)
                    print('Measured chunk performance:',json.dumps({k:report[k] for k in
                          ('completed_epochs','training_steps_per_second','tail_loading_seconds_median','peak_process_rss_bytes')}),flush=True)
                except (OSError,ValueError,KeyError,TypeError) as error:
                    write_json(receipt/'speed_report_unavailable.json',dict(reason=str(error),training_unchanged=True))
                if summary['status']=='training_completed':
                    if epochs!={'jscc':150,'denoiser':150}:raise RuntimeError('Required epochs are incomplete')
                    result=summaries[0].parent/'package_result.json'
                    if not result.exists() or not json.loads(result.read_text()).get('produced'):raise RuntimeError('Training completed but evaluation/package did not')
                    write_json(progress,dict(status='complete',completed_epochs=epochs,receipt=str(receipt)));return
                if summary['status']!='partial':raise RuntimeError('Unexpected training state')
                checkpoint=json.loads((receipt/'checkpoint_dataset-isec.json').read_text())
                if checkpoint.get('status')!='ready' or not checkpoint.get('is_private') or not checkpoint['dataset'].startswith(owner+'/'):raise RuntimeError('Private continuation input is not ready for this account')
                book=receipt/'submitted-notebook.ipynb'
                if not book.exists():raise RuntimeError('Frozen continuation notebook missing')
                chunk_seconds=14400
                cache_plan=folder/'decoded-cache-plan.json'
                if cache_plan.exists():
                    plan=json.loads(cache_plan.read_text())
                    if plan.get('enabled'):
                        if not plan.get('local_parity_tests_passed'):raise RuntimeError('Cache parity acceptance missing')
                        from person1_cache_notebook import prepare_cache_notebook
                        book=prepare_cache_notebook(book,ROOT/plan['module'],plan['sha256'])
                        chunk_seconds=int(plan.get('chunk_seconds',14400))
                        if not 1800<=chunk_seconds<=14400:raise ValueError('Chunk seconds outside reviewed bounds')
                        if plan.get('checkpoint_io'):
                            from person1_checkpoint_notebook import prepare_checkpoint_notebook
                            io=plan['checkpoint_io']
                            if plan.get('adaptive_bounded_chunks'):
                                from person1_chunk_policy import choose_chunk_seconds
                                chunk_seconds=choose_chunk_seconds(summaries[0].parent,io['sha256'],chunk_seconds,io.get('cudnn_plan_cache_limit'))
                            book=prepare_checkpoint_notebook(book,ROOT/io['module'],io['sha256'],chunk_seconds,
                                ROOT/io['resource_module'] if io.get('resource_module') else None,io.get('resource_sha256'),io.get('cudnn_plan_cache_limit'))
                command=[sys.executable,'scripts/1.py','--username',owner,'--slug',slug,'--model','isec','--mode','train','--continue-from',checkpoint['dataset'],'--runtime-notebook',str(book),'--new-run','--training-seconds',str(chunk_seconds),'--interval','60','--no-publish','--max-hours','24']
                write_json(progress,dict(status='continuing_verified_chunk',completed_epochs=epochs,receipt=str(receipt),dataset=checkpoint['dataset']))
            print('ISEC coordinator:',command,flush=True)
            result=subprocess.run(command,cwd=ROOT,env=os.environ.copy())
            if result.returncode:
                write_json(progress,dict(status='needs_investigation',receipt=str(receipt),returncode=result.returncode));raise RuntimeError('Continuation or collection failed; inspect preserved receipt')

if __name__=='__main__':main()
