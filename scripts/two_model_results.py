"""Rebuild verified quality tables; never promote pending complexity audits."""
from pathlib import Path
import json,csv,math
ROOT=Path(__file__).resolve().parents[1]
deep=ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1-deepjscc/7a5a49a0c063462fa6f3d63a86217fd6'
ad=ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only'
out=ROOT/'Results/two_model_final';out.mkdir(parents=True,exist_ok=True)
deep_summary=next((deep/'published_output').glob('team_results/1/*/deepjscc/metrics/summary.json'))
inputs=[('DeepJSCC',deep_summary,deep/'retrieval_verified.json'),('ADJSCC',ad/'outputs/adjscc-evaluation/metrics/summary.json',ad/'retrieval_verified.json')]
rows=[]
for name,path,manifest in inputs:
    verification=json.loads(manifest.read_text());assert verification['verified_files']>0
    value=json.loads(path.read_text());assert value['successful_trials']==240 and value['failed_trials']==0
    row=dict(model=name,channel='AWGN',snr_db=10,epochs=150,kodak_images=24,trials=240)
    for key in ('psnr_db','ssim','ms_ssim','lpips','mse','cbr_complex_per_rgb_sample'):
        assert math.isfinite(value[key]);row[key]=value[key]
        row[key+'_ci95']=json.dumps(value.get(key+'_ci95'))
    row.update(quality_verified=True,source_summary=str(path.relative_to(ROOT)))
    rows.append(row)
with (out/'quality_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(out/'quality_comparison.json').write_text(json.dumps(rows,indent=2))
gate=dict(quality_and_saved_artifacts_verified=True,adjscc_logical_flop_coverage='pending expanded audit',matched_hardware_timing='pending same-host measurement',final_handoff_zip='blocked until complexity and timing gates pass',github_publication_authorized=False)
(out/'acceptance_status.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate))
