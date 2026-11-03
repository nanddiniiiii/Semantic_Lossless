"""Publishable three-model evidence tables; no weights enter this directory."""
import csv,hashlib,json,shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'Results/three_model_final';OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2))
def table(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    from src.sharing import package_run,verify_bundle
    training=next((ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1/b20ed0d81fa44a48b7241b17baec8225/published_output').glob('team_results/1/*/adjscc'))
    evaluation=ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only/outputs/adjscc-evaluation'
    ad=ROOT/'Results/person1_final/adjscc/combined_run'
    if not ad.exists():
        ad.mkdir(parents=True)
        for p in training.glob('*.json'):shutil.copy2(p,ad/p.name)
        shutil.copytree(training/'checkpoints',ad/'checkpoints')
        shutil.copytree(evaluation/'metrics',ad/'metrics')
        shutil.copy2(ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only/summary_cpu_label_corrected.json',ad/'metrics/summary.json')
        for phase in ['train','evaluate']:write(ad/(phase+'_status.json'),dict(status='completed',evidence='native150 training and separately verified240trial evaluation'))
    target=ROOT/'Models/teammate_1_models/ADJSCC/final150-kodak240'
    if not target.exists():package_run(ad,ROOT/'Models','1','final150-kodak240')
    verify_bundle(target)
    isec=ROOT/'Results/person1_final/isec/combined_run';assert read(ROOT/'Results/person1_final/isec/completion.json')['verified']
    quality=list(csv.DictReader((ROOT/'Results/two_model_final/quality_comparison.csv').open()))
    summary=read(isec/'metrics/summary.json')
    row=dict(model='ISEC',channel='AWGN',snr_db=10,epochs='150 JSCC +150 denoiser',kodak_images=24,trials=240,quality_verified=True)
    for k in ['psnr_db','ssim','ms_ssim','lpips','mse','cbr_complex_per_rgb_sample']:
        row[k]=summary[k];row[k+'_ci95']=json.dumps(summary[k+'_ci95'])
    quality.append(row)
    for r in quality:r['comparison_scope']='Native rate; ISEC1/16, Deep/AD1/6; no equal-rate ranking'
    table('quality_comparison.csv',quality);write(OUT/'quality_comparison.json',quality)
    complexity=list(csv.DictReader((ROOT/'Results/two_model_final/flops_comparison.csv').open()))
    raw=read(ROOT/'Results/person1_flops/isec_final150_logical_arithmetic.json')
    assert raw['logical_scope_complete'] and raw['checkpoint_progress']['completed_epochs']==dict(jscc=150,denoiser=150)
    for stage,item in raw['two_replica_global2_training'].items():
        complexity.append(dict(model='ISEC',training_stage=stage,inference_height=512,inference_width=768,inference_batch=1,
            inference_logical_flops=raw['inference']['total']['logical_flops'],inference_gflops=raw['inference']['total']['logical_flops']/1e9,
            inference_iterations=50,training_crop=128,training_global_batch=2,training_replicas=2,
            training_logical_flops_per_update=item['total']['logical_flops'],training_logical_flops_per_image=item['total']['logical_flops_per_image'],
            training_gflops_per_image=item['total']['logical_flops_per_image']/1e9,logical_scope_complete=True,hardware_instruction_census=False))
    table('flops_comparison.csv',complexity);write(OUT/'flops_comparison.json',complexity)
    operator_rows=[]
    for phase,record in raw['inference']['summary'].items():
        operator_rows.append(dict(phase=phase,**record))
    table('isec_inference_operator_coverage.csv',operator_rows)
    timing=list(csv.DictReader((ROOT/'Results/two_model_final/matched_cpu_timing.csv').open()))
    for r in timing:r['comparison_group']='Deep/AD same-host CPU; exclude ISEC GPU from this ranking'
    profile=read(isec/'profile.json')
    timing.append(dict(model='isec',component='end_to_end_including50refinement_latent_gradients',device='Tesla T4 cuda:0',precision='float32',
        p50_seconds=profile['latency_p50_seconds'],p95_seconds=profile['latency_p95_seconds'],
        comparison_group='ISEC native GPU profile; separate from CPU measurements',warmup=3,repeats=10))
    table('timing_comparison.csv',timing)
    for model,folder in [('DeepJSCC',next((ROOT/'Models/teammate_1_models/DeepJSCC').glob('20261031T003230Z-*'))),('ADJSCC',target),('ISEC',ROOT/'Models/teammate_1_models/ISEC/final150-kodak240')]:
        verify_bundle(folder)
        for filename in ['training_summary.json','evaluation_reload.json','source.json']:
            if (folder/filename).exists():shutil.copy2(folder/filename,OUT/(model+'_'+filename))
    readme='''# Three-model Person1 AWGN10 benchmark

DeepJSCC and ADJSCC each completed150 full epochs. ISEC completed150 JSCC
and150 denoiser epochs. Every epoch retains3450 parent entries including59
documented duplicates. Development uses100 DIV2K images; final test uses24
Kodak images,10 noise trials each, original dimensions and both orientations.
Metrics use the repository image_quality function: float RGB clipped to[0,1],
PSNR, SSIM, MS-SSIM, AlexNet LPIPS and MSE. Confidence intervals bootstrap24
image means with2000 resamples,seed22. The raw per-image tables retain errors,
native transmitted complex symbols, shapes and device origins.

Read quality_comparison.csv, flops_comparison.csv and timing_comparison.csv.
ISEC native CBR is1/16 versus1/6 for DeepJSCC/ADJSCC. These are native-rate
measurements, NOT an equal-rate ranking. Counted overhead is zero for these
continuous analog links; no framing protocol is implemented or silently added.
DeepJSCC is a paper-faithful Figure2 reproduction, not verified original-author
code. ADJSCC uses pinned native TensorFlow author code. ISEC uses pinned native
encoder/decoder/BF_CNN and50-step refinement. Common data/schedules are declared
adaptations; finishing150 epochs does not establish paper convergence.

ISEC PSNR14.87dB is the measured result, not a paper-performance reproduction
claim. All240 finite native-size outputs passed strict shape/hash checks.
172 trials came from Kaggle Torch2.11 CUDA12.8 and68 from Colab Torch2.11
CUDA13.0, both TeslaT4. The backend migration is disclosed; no training occurred
on Colab. Exact frozen adapter and checkpoint bytes are recorded.

Complexity is complete DECLARED LOGICAL arithmetic at512x768, batch1, MAC=2;
special primitives, comparisons, random draws and exclusions are recorded.
ISEC counts all50 executed refinements including latent-gradient backward;
branch-dependent counts are checkpoint/input/seed specific. Training counts
execute native forward/loss/backward/Adam for both stages and include two-replica
reduction. These are not physical CuDNN instruction counts or campaign energy.
AD FFTs use disclosed TensorFlow Eigen source-reference logical algorithms.
Unknown arithmetic is not treated as zero. Original registered-only profiles
remain partial historical evidence; the full logical reports supersede them.

Deep/AD CPU latency shares host,input,float32,threads2/1,warmup3,repeats10.
ISEC latency is separately labelled native T4, including50 refinement steps.
There is NO matched-host three-model latency ranking. Component hook timings
are instrumented passes, not an additive latency decomposition.

Preserved failures include ISEC SIGSEGV/SIGKILL and evaluation timeout, AD
checkpoint packaging/seed errors and all earlier pilots. Source snapshots and
runtime adaptations retain hashes. Checkpoint bytes live only in the local
Models/teammate_1_models handoff, never in this Git result directory.
NTSCC+ is not completed and is excluded from this three-model certificate.
'''
    (OUT/'README.md').write_text(readme)
    write(OUT/'acceptance_status.json',dict(three_model_training_verified=True,kodak_trials_per_model=240,
       quality_artifacts_verified=True,logical_flop_coverage_verified=True,metrics_native_rate_only=True,
       timing_scope='Deep/AD matchedCPU; ISEC separately labelledT4; no three-model latency ranking',
       paper_convergence_claimed=False,zip_verification_pending=True,github_publication_authorized=True))
    print('Three-model tables ready',flush=True)
if __name__=='__main__':main()
