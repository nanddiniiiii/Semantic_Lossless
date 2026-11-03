# Three-model Person1 AWGN10 benchmark

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
