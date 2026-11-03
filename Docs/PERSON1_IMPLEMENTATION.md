# Person 1 corrected author-source workflow — 28 October 2026

Gradient qualification correction, 30 October: NTSCC v13's independent forward
loss and reconstructed pixels matched exactly, while its separate backward
graph differed by up to0.000244140625 at gradient scale2130.3447265625. This
does not by itself prove a reduction error. The new probe captures each actual
GPU replica's gradients before DataParallel reduction, requires the incoming
native-loss coefficient to be exactly0.5 on both replicas, and compares the root
gradient with their float32 sum using torch.equal. Independent forward equality
is still required; independent backward differences remain reported diagnostics.
No model operations or training objectives change. Missing shards, wrong weights
and incorrect reductions are covered by failure-injection tests. GPU verification
of this revised probe remains required before accepting NTSCC qualification.

Runtime correction, 30 October: NTSCC pilot v12 failed inside PyTorch1.9's
deterministic CUDA indexing implementation. The next pilot uses Python3.8,
Torch1.13.0+cu117 and the official NATTEN0.14.6+torch1130cu117 wheel. Its SHA256 is
`ae0d892f5914d5a9b1d93a29e3fb13f3ea2c3aad0c0ecd4ea555eec46aa10577`.
The downloaded wheel passed ZIP integrity checks. This is a declared runtime
adaptation, not an author architecture or attention-release change. GPU
qualification remains required. Earlier Torch1.9 run records below are historical.

Status: implementation and preflight work. No150-epoch result is certified by this document.
The earlier runners and artifacts remain historical. Person1 now uses
`src/person1/` through `Notebooks/1.ipynb` and `scripts/1.py`.

DeepJSCC's version1 Kaggle pilot (source88b63e7) passed on two TeslaT4s:
two real-data updates/64 image presentations, strict checkpoint reload,
landscape/portrait/odd-size unpadding, and a split/uninterrupted continuation
fixture with zero maximum tensor difference. Local collection verified109 file
entries across two manifests. This does not count as a completed training epoch
or Kodak quality evaluation. The evidence receipt is under
`Results/published/person1-deepjscc-pilot-v1/`.
That mechanics pilot used normalized RGB MSE; review subsequently restored the
paper's native255-denormalized MSE objective. A fresh pilot at the newly pinned
source is required before production training. Campaign qualification is bound
to the reviewed source SHA, so an older pilot cannot silently qualify new code.

## Source fidelity

| Slot | Selected implementation | Native details / declared changes |
|---|---|---|
| DeepJSCC | Paper reproduction of arXiv1809.01733v4 Figure2 and equations1/3/5 | Encoder16/32/32/32/C, decoder32/32/32/16/3, all kernels5. TensorFlow SAME alignment reproduced explicitly; complex unit-power AWGN. Per-channel PReLU and PyTorch initialization are disclosed choices because the paper does not specify them. C16 gives1/6 at dimensions divisible by4. Original-author source remains unverified. |
| ADJSCC | alexxu1988/ADJSCC at e5332e95faf592aab9f440992de96029162dc7dd | Original TensorFlow Attention_Encoder/Decoder, SignalConv/GDN and Channel called directly from immutable source. Native pixel255 MSE, Adam1e-4, crop128, batch16, C16. Linux TF2.14.1/TFC2.14.1 compatibility and MirroredStrategy require the actual GPU pilot; this is distinct from the old DiffCom port. |
| ISEC | changwoolee/isec-deep-jscc at43d1a27beaeb89b7057f02bd2fa2dd05c46093e1 | OpenImages encoder/decoder and BF_CNN, native width6, real-noise convention, Adam betas and denoiser learning rate. Native positive-denoiser refinement equation and SNR scaling; Kodak/OpenImages example refinement lr0.001, alpha4, delta1. Random crop128 and horizontal flip.150 JSCC +150 denoiser epochs;50 final refinement iterations. Global batch2 is an explicit resource adaptation. |
| NTSCC+ | wsxtyrdd/NTSCC_plus at765c6f6c11cf3bbe249a26b624dbaf47a4f0676d | Native NTSCC_plus/NatCheckerboard2, native RateDistortionLoss with shipped-YAML lambda.72, main/aux Adam, step-based600000 MultiStepLR. Python3.8/Torch1.9 CUDA11.1/NATTEN0.14.6 exact wheel environment. No newer attention-kernel substitution. |

The membership is the frozen DIV2K800 + Flickr2K2650 training entries,
DIV2K100 development, Kodak24 test. The59 duplicate training entries are retained
and recorded. Kodak never selects a checkpoint or rate. Common data and fixed
AWGN10 are benchmark adaptations. NTSCC+ uses256 crops; the other three use128.
No extra resizing or flips are applied except ISEC's documented native flip.
The deterministic crop/order adapter preserves continuation across worker/session
boundaries; its exact RNG sequence is our adapter, not the authors' original RNG.

The old shi-labs wheel index currently fails certificate validation. The exact
official GitHub NATTENv0.14.6 release asset for torch190cu111/cp38 was downloaded
and SHA256-verified as
`f58e66c03de21849ffa7cde85d1d28c4a0b82958155ec20f12520ad39113f2a3`.
The environment installer uses that asset with hash enforcement. This diagnoses
the current index access failure; it does not prove the cause of older failed builds.

NTSCC+ serializes rate indices as PNG and reports their bytes plus the author's
assumed2/3-LDPC/16QAM symbol budget separately. The released source does not
provide that noisy side-bit decoder. `complete_link_certified=false` therefore
remains explicit; these values must not be ranked as a complete protected-link
measurement. NTSCC++ online editing is not included.

## Launch and continuation

Use the local CLI route only, account `hmmmkk`, stable `semantic-notebook-1`.
Each saved version selects one model. Default is a real-data pilot, including
two updates per native stage, reload, both-GPU work, development reconstruction,
and an interrupted/uninterrupted actual-model continuation check on a tiny crop
fixture. The latter fixture is mechanics evidence, never a quality result.

```powershell
python scripts/1.py --username hmmmkk --model deepjscc --mode pilot --create-missing
python scripts/1.py --username hmmmkk --campaign
```

The campaign waits for any incomplete receipt before another version. It runs
DeepJSCC, ADJSCC, ISEC, NTSCC+ in order, in four-hour chunks, and continues only
from private hash-checked Dataset checkpoints. Collector `--resume` means local
collection recovery; `--continue-from owner/slug` means training continuation.
Only all150 complete epochs per required stage qualifies for evaluation and a
completed model bundle. Partial/failing outputs remain evidence. A changed
training contract rejects automatic resume. GPU quota is checked before each
submission; no paid capacity is enabled.

## Evidence and delivery

All24 original Kodak images and10 reproducible channel trials per image are
scored by one shared RGB metric implementation. Native alignment padding is
removed before scoring and charged to channel uses/computation. Native source
channels retain their stated conventions. Reconstruction PNGs, float predictions,
raw metrics, image-bootstrap confidence intervals, time and supported-operator
FLOP profiles are saved. Unsupported operations and optimizer arithmetic remain
explicitly missing; partial counts are not complete model FLOPs.

Atomic checkpoints include optimizer/scheduler/auxiliary state, epoch/sample
position, RNG state, source/data contract, and per-stage completion. Torch GPU
continuation probes compare weights and optimizer state at declared tolerances;
TensorFlow probes compare checkpoint variables after the same native updates.
CUDA kernel determinism and native legacy dependencies still require live checks.

Git receives reviewed source/notebooks and curated result files. Newly generated
weights, private continuation inputs, raw local receipts, credentials and datasets
are excluded. Git Credential Manager may supply a token in process memory for
the existing results API publisher; its value is never logged or serialized.

At completion the campaign creates `SharedModelHandoff/teammate_1_full_handoff.zip`
beside the clone. It includes person1 weights/bundles, raw run evidence, code,
numbered notebook/script and a hash manifest. Every ZIP member is read back and
hash-checked. Each model's strict reload evidence must exist in the downloaded
bundle; CUDA-only architectures cannot be inferred to load on a CPU laptop.
