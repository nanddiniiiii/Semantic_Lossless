"""Build a verified archive; final rename is gated on reviewed comparison files."""
from pathlib import Path
import hashlib,json,zipfile,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'SharedModelHandoff';OUT.mkdir(exist_ok=True)
TEMP=OUT/'DeepJSCC_ADJSCC_handoff.in-progress.zip'
FINAL=OUT/'DeepJSCC_ADJSCC_verified_handoff.zip'
if FINAL.exists():raise RuntimeError('Verified handoff already exists; preserve it')
manifest=[];seen=set()
def add(z,path,name):
    if name in seen:return
    seen.add(name)
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as source,z.open(name,'w',force_zip64=True) as target:
        for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk);target.write(chunk)
    after=path.stat()
    if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:raise RuntimeError('Source changed during packaging: '+str(path))
    manifest.append(dict(path=name,bytes=after.st_size,sha256=h.hexdigest()))
def tree(z,base,prefix,exclude=()):
    for p in sorted(base.rglob('*')):
        if p.is_file() and not any(part in exclude for part in p.relative_to(base).parts):
            add(z,p,prefix+'/'+p.relative_to(base).as_posix())
deep=ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1-deepjscc/7a5a49a0c063462fa6f3d63a86217fd6'
ad=ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1/b20ed0d81fa44a48b7241b17baec8225'
with zipfile.ZipFile(TEMP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    tree(z,next((deep/'published_output').glob('team_results/1/*/deepjscc')),'Models/DeepJSCC',('pilot','resume-check','resume-fixture'))
    tree(z,next((ad/'published_output').glob('team_results/1/*/adjscc')),'Models/ADJSCC/training',('pilot','resume-check','resume-fixture'))
    tree(z,ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only/outputs/adjscc-evaluation','Models/ADJSCC/evaluation',('checkpoints',))
    for label,receipt in [('DeepJSCC',deep),('ADJSCC',ad)]:
        for name in ['submitted-notebook.ipynb','submission.json','submitted_version.json','retrieval_verified.json']:
            add(z,receipt/name,'Provenance/'+label+'/'+name)
    for p in sorted((ROOT/'src/person1').glob('*.py')):add(z,p,'Code/src/person1/'+p.name)
    for p in sorted((ROOT/'scripts').glob('*.py')):add(z,p,'Code/scripts/'+p.name)
    tree(z,ROOT/'Results/person1_timing/outputs','Timing/raw')
    # Keep this preparatory archive distinguishable from the final deliverable.
    print('Base artifacts archived; waiting for final comparison acceptance',flush=True)
    gate=ROOT/'Results/two_model_final/acceptance_status.json'
    for _ in range(2880):
        state=json.loads(gate.read_text())
        if state.get('all_two_model_requirements_verified') is True:break
        time.sleep(30)
    else:raise TimeoutError('Final comparison gate not completed; archive remains in-progress')
    tree(z,ROOT/'Results/two_model_final','Comparison')
    tree(z,ROOT/'Results/person1_flops','Complexity',('outputs-v1-failed','outputs-v2-complete','reference_sources'))
    z.writestr('FILE_MANIFEST.json',json.dumps(dict(files=manifest),indent=2))
    z.writestr('README.txt','Person1 AWGN10 benchmark: DeepJSCC paper reproduction and native ADJSCC. Both150epochs,3450parents/epoch,Kodak24x10. CPU timing is explicitly labelled. Logical arithmetic follows the included convention; hardware instructions and RNG internals are excluded. Original logs are preserved. No Git publication is authorized.\n')
with zipfile.ZipFile(TEMP) as z:
    for item in manifest:
        h=hashlib.sha256()
        with z.open(item['path']) as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=item['sha256']:raise RuntimeError('Archive content hash mismatch: '+item['path'])
TEMP.replace(FINAL)
(OUT/'DeepJSCC_ADJSCC_zip_verification.json').write_text(json.dumps(dict(verified=True,files=len(manifest),archive=str(FINAL),method='every ZIP member streamed and SHA256 compared with source manifest'),indent=2))
print('Verified final archive:',FINAL,flush=True)
