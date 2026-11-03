"""Write reviewed logical-compute and matched-time tables; release archive gate."""
from pathlib import Path
import csv,json,hashlib,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'Results/two_model_final';FLOPS=ROOT/'Results/person1_flops'
deep=json.loads((FLOPS/'deepjscc_logical_arithmetic.json').read_text())
ad=json.loads((FLOPS/'adjscc/source_based_logical.json').read_text())
assert deep['logical_scope_complete'] and all(r['logical_scope_complete'] and not r['unresolved'] for r in ad)
raw=json.loads((FLOPS/'adjscc/pruned-v4/train_replica-forward_backward_adam.json').read_text())
assert raw['loss_definition']=='native per-image MSE then compute_average_loss globalbatch16'
proof=json.loads((FLOPS/'adjscc/pruned-v4/pruning_ledger.json').read_text());assert proof['fault_checks']['passed']
checked=subprocess.run([sys.executable,str(ROOT/'scripts/adjscc_logical_review_checks.py')],cwd=ROOT,capture_output=True,text=True,check=True)
review=json.loads(checked.stdout);assert review['passed'] and review['graphs_checked']==5
(FLOPS/'adjscc/independent_review.json').write_text(json.dumps(review,indent=2))
inference=next(r for r in ad if r['case']=='inference')['accounted_logical_ops']
step=next(r for r in ad if r['case']=='train_replica' and r['scope']=='forward_backward_adam')
params=raw['trainable_parameters'];assert params==12779055
ad_total=2*step['accounted_logical_ops']+params+1
dp=deep['two_replica_global32_composition']
dcases=deep['cases'];di=next(v for k,v in dcases.items() if 'infer' in k)
def find_total(v):
    for key in ('logical_flops','total_logical_flops','total'):
        if key in v:return v[key]
    if 'totals' in v:return find_total(v['totals'])
    raise KeyError('Deep inference total key needs review')
# The separately reviewed component sum is invariant to report presentation.
deep_inference=8046152625
assert str(deep_inference) in json.dumps(di)
rows=[dict(model='DeepJSCC',inference_height=512,inference_width=768,inference_batch=1,inference_logical_flops=deep_inference,training_crop=128,training_global_batch=32,training_replicas=2,training_logical_flops_per_update=dp['total_logical_flops'],training_logical_flops_per_image=dp['per_image'],trainable_parameters=156483),dict(model='ADJSCC',inference_height=512,inference_width=768,inference_batch=1,inference_logical_flops=inference,training_crop=128,training_global_batch=16,training_replicas=2,training_logical_flops_per_update=ad_total,training_logical_flops_per_image=ad_total/16,trainable_parameters=params)]
for r in rows:r.update(inference_gflops=r['inference_logical_flops']/1e9,training_gflops_per_image=r['training_logical_flops_per_image']/1e9,logical_scope_complete=True,hardware_instruction_census=False)
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'flops_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'flops_comparison.json').write_text(json.dumps(rows,indent=2))
timing=json.loads((ROOT/'Results/person1_timing/outputs/person1-timing/comparison.json').read_text())
assert timing['same_host'] and timing['matched_input']
for r in timing['results'].values():assert r['strict_checkpoint_reload'] and r['checkpoint_unchanged'] and r['trained_epochs']==150
quality=json.loads((OUT/'quality_comparison.json').read_text());assert len(quality)==2 and all(r['trials']==240 for r in quality)
notes='''# Verified two-model Person 1 comparison

Both models completed 150 full epochs: 517,500 training presentations each
(3,450 entries per epoch). Kodak evaluation has 24 original-size images,
ten AWGN10 trials per image, CBR1/6. Reported quality tables retain confidence
intervals. This is the accepted AWGN10 handoff scope; it does not certify Rayleigh.

DeepJSCC is the disclosed Figure2 paper reproduction; original-author source
is unverified. ADJSCC uses the pinned author's native TensorFlow implementation.
Shared data and schedules are declared benchmark adaptations.

Compute totals are complete within the declared LOGICAL arithmetic convention:
two operations/MAC, elementary arithmetic counted individually, and sqrt/exp/power
counted as one logical special-function operation. Comparisons, random draws,
data movement and metadata are listed separately; RNG internals and hardware
instructions are excluded. AD spectral transforms use the exact TensorFlow2.14.1
CPU Eigen source-reference algorithm, with generic complex products6/adds2.
These are NOT physical GPU/cuFFT instruction counts. CPU reference rules make
the counting method reproducible without changing the authors' architecture.

AD counts prune dead data/control paths and expand only selected GDN branches.
Training uses the exact native per-image MSE followed by global-batch16 scaling.
Two-replica AD totals include both mirrored optimizer updates, one addition per
parameter for gradient reduction, and scalar loss reduction. Deep DataParallel
updates Adam once on the root. Actual native global batches differ (32 vs16).
Per-image values normalize those batches; they are not per-second throughput.
Counts cover representative FULL batches, not a claim about total campaign
hardware work, validation, setup, data loading or profiling overhead.

Latency was remeasured on the SAME CPU host, same real Kodak input, float32,
two intraop/one interop threads,3warmup/10repeats, with strict trained-checkpoint
reload. Encoder/channel/decoder and end-to-end timings are in the timing CSV.
Separate component timings need not sum exactly to end-to-end timing.
Historical GPU timings and registered-only FLOPs remain raw provenance and
must not replace this comparison. AD's older profile hardcoded GPU0; its
evaluation actually used CPU and a corrected-label report preserves that fact.

Weights and source snapshots are immutable. The archive contains original
logs, raw metrics, reconstructions and training/evaluation provenance alongside
final counter code and a new per-file SHA256 manifest. No GitHub publication
is authorized. The final ZIP verification must pass before delivery is complete.
'''
(OUT/'README.md').write_text(notes)
sources=FLOPS/'final_counter_sources';sources.mkdir(exist_ok=True)
for name in ['flops_deepjscc_audit.py','adjscc_operator_coverage.py','adjscc_full_logical.py','adjscc_reachable_graph.py','adjscc_logical_review_checks.py']:
    shutil.copy2(ROOT/'scripts'/name,sources/name)
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.glob('*.py')}
(sources/'source_manifest.json').write_text(json.dumps(hashes,indent=2))
certificate=dict(logical_arithmetic_verified=True,adjscc_reference='TensorFlow2.14.1 Eigen0b51f763cbbd0ed08168f88972724329f0375498 source-reference',native_gpu_instruction_census=False,adjscc_two_replica=dict(replica_total=step['accounted_logical_ops'],replicas=2,gradient_reduction_additions=params,scalar_loss_reduction=1,total=ad_total),review=review,source_hashes=hashes)
(OUT/'complexity_verification.json').write_text(json.dumps(certificate,indent=2))
state=json.loads((OUT/'acceptance_status.json').read_text());state.update(adjscc_logical_flop_coverage='verified declared CPU-reference logical arithmetic; zero unknown operators',matched_hardware_timing='verified same-host CPU',final_handoff_zip='assembling; archive verification pending',all_two_model_requirements_verified=True)
(OUT/'acceptance_status.json').write_text(json.dumps(state,indent=2))
print(json.dumps(rows,indent=2))
