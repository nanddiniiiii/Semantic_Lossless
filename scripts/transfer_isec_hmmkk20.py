"""Transfer the verified ISEC69 checkpoint privately and launch one continuation."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
OWNER='hmmkk20'
os.environ['PATH']=str(ROOT/'.venv-collector/Scripts')+os.pathsep+os.environ['PATH']
from src.person1.auth import ensure_fresh_authorization
from src.person1.collection import owned_dataset_exists
ensure_fresh_authorization(OWNER)
old=ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1-isec/45c1fef245084c1b8802f93da08bfc07'
proof=json.loads((old/'checkpoint_recovery_state.json').read_text())
assert proof['completed_epochs']=={'jscc':69,'denoiser':0} and proof['offset']==1884
run=next((old/'published_output').glob('team_results/1/*/isec'))
checkpoint=run/'checkpoints/last.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha==proof['sha256']=='a0d4a672788f42f34c281ab9a6cde6aaddd1abf4e64b9167907801bf8698cef0'
book=old/'submitted-notebook.ipynb';assert book.exists()
folder=ROOT/'Results/person1_account_transfer'/OWNER/'isec_resume69';folder.mkdir(parents=True,exist_ok=True)
if (folder/'continuation_launcher.json').exists():raise RuntimeError('Transfer continuation already launched; inspect receipt before retrying')
asset=folder/'input';asset.mkdir(exist_ok=True)
shutil.copy2(checkpoint,asset/'last.pt')
assert hashlib.sha256((asset/'last.pt').read_bytes()).hexdigest()==sha
identity=json.loads((run/'person1_run.json').read_text())
resume=dict(algorithm='isec',checkpoint='last.pt',sha256=sha,contract=proof['contract'],run_id=identity['run_id'])
(asset/'resume_receipt.json').write_text(json.dumps(resume,indent=2))
lineage=dict(source_account='hmmmkk',destination_account=OWNER,source_kernel='hmmmkk/semantic-notebook-1-isec',source_version=3,source_receipt=old.name,completed_epochs=proof['completed_epochs'],partial_epoch_offset=proof['offset'],step=proof['step'],checkpoint_sha256=sha,frozen_notebook_sha256=hashlib.sha256(book.read_bytes()).hexdigest(),contract=proof['contract'],remote_export_hash_verified_files=135)
(asset/'account_transfer.json').write_text(json.dumps(lineage,indent=2))
dataset=OWNER+'/p1-isec-resume69-'+sha[:12]
(asset/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,title='Person1 ISEC verified epoch69 continuation',licenses=[{'name':'other'}])))
if not owned_dataset_exists(dataset,os.environ.copy()):
    with (folder/'upload.log').open('w',encoding='utf-8') as log:
        subprocess.run(['kaggle','datasets','create','-p',str(asset),'-t'],stdout=log,stderr=log,check=True,timeout=1800)
for attempt in range(30):
    status=subprocess.run(['kaggle','datasets','status',dataset],capture_output=True,text=True,encoding='utf-8',timeout=90)
    if status.returncode==0 and 'ready' in status.stdout.lower():break
    print('Private continuation input processing',attempt+1,flush=True);time.sleep(10)
else:raise RuntimeError('Continuation input not ready; no training submitted')
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest
api=KaggleApi();api.authenticate();request=ApiGetDatasetMetadataRequest();request.owner_slug=OWNER;request.dataset_slug=dataset.split('/')[1]
with api.build_kaggle_client() as client:metadata=client.datasets.dataset_api_client.get_dataset_metadata(request)
if not metadata.info or metadata.info.is_private is not True:raise RuntimeError('Checkpoint input privacy not verified')
(folder/'input_verified.json').write_text(json.dumps(dict(dataset=dataset,is_private=True,local_sha256_verified=sha,remote_checkpoint_hash_verified=False,remote_hash_check='required by the GPU continuation loader'),indent=2))
args=[sys.executable,'scripts/1.py','--username',OWNER,'--slug','semantic-notebook-1-isec','--model','isec','--mode','train','--continue-from',dataset,'--runtime-notebook',str(book.relative_to(ROOT)),'--create-missing','--new-run','--training-seconds','14400','--interval','60','--no-publish','--max-hours','24']
stdout=ROOT/'Results/person1_campaign'/OWNER/'isec-continuation-v1.stdout.log';stdout.parent.mkdir(parents=True,exist_ok=True)
with stdout.open('w',encoding='utf-8') as out,stdout.with_name('isec-continuation-v1.stderr.log').open('w',encoding='utf-8') as err:
    process=subprocess.Popen(args,stdout=out,stderr=err,env=os.environ.copy())
(folder/'continuation_launcher.json').write_text(json.dumps(dict(pid=process.pid,args=args,dataset=dataset,checkpoint_lineage=lineage),indent=2))
print('ISEC continuation launcher started:',process.pid,'owner:',OWNER,flush=True)
