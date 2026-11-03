"""Archive the existing teammate workflow plus verified evidence, no Git weights."""
from pathlib import Path
import hashlib,json,re,zipfile,sys,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT.parent/'SharedModelHandoff';OUT.mkdir(exist_ok=True)
FINAL=OUT/'teammate_1_models_DeepJSCC_ADJSCC_ISEC_verified.zip'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    if FINAL.exists():raise RuntimeError('Preserve existing verified archive; inspect certificate')
    temp=FINAL.with_suffix('.in-progress.zip');entries=[];seen=set()
    def add(z,p,name):
        if name in seen:return
        seen.add(name);h=hashlib.sha256();size=0
        with p.open('rb') as source,z.open(name,'w',force_zip64=True) as target:
            for b in iter(lambda:source.read(1024*1024),b''):target.write(b);h.update(b);size+=len(b)
        entries.append(dict(path=name,bytes=size,sha256=h.hexdigest()))
    def tree(z,base,prefix,exclude=()):
        for p in sorted(base.rglob('*')):
            if p.is_file() and not any(x in exclude for x in p.relative_to(base).parts):add(z,p,prefix+'/'+p.relative_to(base).as_posix())
    deep=next((ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1-deepjscc/7a5a49a0c063462fa6f3d63a86217fd6/published_output').glob('team_results/1/*/deepjscc'))
    isec=ROOT/'Results/person1_final/isec/combined_run'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        tree(z,ROOT/'Models/teammate_1_models','Models/teammate_1_models')
        print('Model folders archived',flush=True)
        tree(z,deep,'Evidence/DeepJSCC',('checkpoints','pilot','resume-check','resume-fixture'))
        tree(z,ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only/outputs/adjscc-evaluation','Evidence/ADJSCC',('checkpoints',))
        tree(z,isec,'Evidence/ISEC',('checkpoints',))
        tree(z,ROOT/'Results/three_model_final','Comparison')
        for p in (ROOT/'Results/person1_flops').glob('*.json'):
            if 'provisional' not in p.name:add(z,p,'Complexity/'+p.name)
        tree(z,ROOT/'Results/person1_flops/final_counter_sources','Complexity/final_counter_sources')
        tree(z,ROOT/'Results/person1_timing/outputs','Timing/DeepAD_matchedCPU')
        for receipt in [ROOT/'Results/team_downloads/1/hmmkk20__semantic-notebook-1-isec/05a60cc829cb47a986e620fbba637425',
                        deep.parents[4]]:
            for filename in ['submitted-notebook.ipynb','retrieval_verified.json','submission.json','submitted_version.json']:
                if (receipt/filename).exists():add(z,receipt/filename,'Provenance/'+receipt.name+'/'+filename)
        for base in ['src/person1','scripts','Docs','Notebooks']:
            for p in (ROOT/base).glob('*'):
                if p.is_file() and p.suffix in ['.py','.md','.ipynb']:add(z,p,'Code/'+base+'/'+p.name)
        # Preserve readable failure evidence without leaking signed URL credentials.
        for p in (ROOT/'Results/person1_campaign').rglob('*.log'):
            text=p.read_text(encoding='utf-8',errors='replace')
            text=re.sub(r'https?://\S*\?\S+', '[URL query redacted]',text)
            text=re.sub(r'(?i)((?:access|refresh)_token[\s\"\x27:=]+)[^\s\"\x27,}]+',r'\1[REDACTED]',text)
            name='Logs/'+p.relative_to(ROOT/'Results/person1_campaign').as_posix()
            data=text.encode();z.writestr(name,data);entries.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),original_sha256=sha(p),sanitization='URL queries and OAuth token fields redacted; original local logs retained'))
        z.writestr('FILE_MANIFEST.json',json.dumps(dict(files=entries),indent=2))
        z.writestr('START_HERE.txt','Read Comparison/README.md. Final runs: DeepJSCC20261031T003230Z-cc4c86d3; ADJSCC/final150-kodak240; ISEC/final150-kodak240. Older partial folders remain historical, not additional completed models. Native CBR differs. See separate timing scopes; no equal-rate or unmatched-hardware ranking. Weights stay in this private OneDrive handoff, never Git.\n')
    print('Verifying every archived member',len(entries),flush=True)
    with zipfile.ZipFile(temp) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for item in entries:
            h=hashlib.sha256();n=0
            with z.open(item['path']) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
            if n!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('ZIP hash mismatch: '+item['path'])
        import torch
        from src.person1.isec import build
        from src.person1.paper_deepjscc import DeepJSCC
        checks={}
        with tempfile.TemporaryDirectory(prefix='person1-extracted-checkpoints-',dir=OUT) as extracted:
            extracted=Path(extracted)
            for model,run,filename in [('ISEC','final150-kodak240','last.pt'),('DeepJSCC','20261031T003230Z-cc4c86d3','last.pt'),('ADJSCC','final150-kodak240','last-tf.zip')]:
                prefix='Models/teammate_1_models/'+model+'/'+run+'/'
                path=extracted/(model+'-'+filename)
                with z.open(prefix+filename) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
                config=json.loads(z.read(prefix+'resolved_config.json'))
                if model=='ADJSCC':
                    with zipfile.ZipFile(path) as tf:
                        if tf.testzip():raise ValueError('Extracted TF checkpoint corrupt')
                    proof=json.loads((ROOT/'Results/person1_timing/outputs/person1-timing/adjscc.json').read_text())
                    if not proof['strict_checkpoint_reload'] or proof['checkpoint_sha256']!=sha(path):raise ValueError('Native TF strict reload evidence differs')
                    checks[model]=dict(archive_extracted=True,sha256=sha(path),strict_reload='original native TensorFlow runtime proof, same checkpoint bytes')
                else:
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    net=build(config,'cpu',ROOT/'.upstream/isec') if model=='ISEC' else DeepJSCC(config)
                    net.load_state_dict(state['state_dict'],strict=True)
                    checks[model]=dict(archive_extracted=True,sha256=sha(path),strict_reload=True,completed_epochs=state['progress']['completed_epochs'])
                    del net,state
    temp.replace(FINAL)
    report=dict(verified=True,archive=str(FINAL),bytes=FINAL.stat().st_size,sha256=sha(FINAL),manifest_files=len(entries),
        every_member_sha256_verified=True,extraction_checkpoint_checks=checks,storage='user-selected OneDrive image_semcom/SharedModelHandoff; sharing permissions unchanged')
    (OUT/'three_model_zip_verification.json').write_text(json.dumps(report,indent=2))
    gate=ROOT/'Results/three_model_final/acceptance_status.json';value=json.loads(gate.read_text());value.update(zip_verification_pending=False,archive_sha256=report['sha256'],archive_verified=True);gate.write_text(json.dumps(value,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
