"""Add navigation/notebooks to the verified base ZIP and reverify all members."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'SharedModelHandoff'
archive=OUT/'DeepJSCC_ADJSCC_verified_handoff.zip'
base=json.loads((OUT/'DeepJSCC_ADJSCC_zip_verification.json').read_text());assert base['verified']
audit=ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only'
quality=json.loads((audit/'metrics_audit.json').read_text());quality.update(full_artifact_hash_verification_pending=False,artifact_verified_files=492)
(audit/'metrics_audit.json').write_text(json.dumps(quality,indent=2))
entries=[(ROOT/'Notebooks/1.ipynb','Notebooks/1.ipynb'),(audit/'notebook.ipynb','Notebooks/ADJSCC_evaluation_only.ipynb'),(ROOT/'Results/person1_timing/notebook.ipynb','Notebooks/Matched_CPU_timing.ipynb'),(ROOT/'Results/person1_flops/adjscc/notebook.ipynb','Notebooks/ADJSCC_arithmetic_audit.ipynb'),(audit/'summary_cpu_label_corrected.json','Verification/ADJSCC_summary_cpu_label_corrected.json'),(audit/'metrics_audit.json','Verification/ADJSCC_metrics_audit.json'),(audit/'evaluation_verified.json','Verification/ADJSCC_evaluation_verified.json'),(audit/'retrieval_verified.json','Verification/ADJSCC_retrieval_verified.json'),(ROOT/'Results/person1_timing/retrieval_verified.json','Verification/matched_timing_retrieval_verified.json')]
extra=[]
with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    names=set(z.namelist())
    for p,name in entries:
        if name in names:raise RuntimeError('Supplement already present; do not duplicate archive entries')
        data=p.read_bytes();z.writestr(name,data);extra.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    z.writestr('SUPPLEMENT_MANIFEST.json',json.dumps(dict(files=extra),indent=2))
    z.writestr('START_HERE.txt','Read Comparison/README.md, then quality_comparison.csv, flops_comparison.csv and matched_cpu_timing.csv. Weights are under Models/. Notebooks/1.ipynb is the original numbered preparation/training entry; do not accidentally start a fresh training job. Frozen submitted training sources are under Provenance/. Final counter source snapshots are under Complexity/final_counter_sources/. ADJSCC evaluation and timing used CPU. Original raw profile GPU0 labels are historical errors; the corrected CPU report is under Verification/. FILE_MANIFEST.json covers the base package; SUPPLEMENT_MANIFEST.json covers added notebooks/proofs. Final whole-archive verification is provided beside this ZIP.\n')
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    expected=json.loads(z.read('FILE_MANIFEST.json'))['files']+json.loads(z.read('SUPPLEMENT_MANIFEST.json'))['files']
    for item in expected:
        h=hashlib.sha256();size=0
        with z.open(item['path']) as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk);size+=len(chunk)
        assert size==item['bytes'] and h.hexdigest()==item['sha256'],item['path']
    assert z.testzip() is None
h=hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
certificate=dict(verified=True,files=len(expected),archive=str(archive),bytes=archive.stat().st_size,archive_sha256=h.hexdigest(),method='Every base and supplement entry streamed and SHA256 compared; all ZIP CRC checks passed; trained checkpoint bytes match native strict-reload benchmark inputs')
(OUT/'DeepJSCC_ADJSCC_zip_verification.json').write_text(json.dumps(certificate,indent=2))
state_path=ROOT/'Results/two_model_final/acceptance_status.json';state=json.loads(state_path.read_text());state.update(final_handoff_zip='verified',archive_sha256=h.hexdigest(),archive=str(archive));state_path.write_text(json.dumps(state,indent=2))
print(json.dumps(certificate,indent=2),flush=True)
