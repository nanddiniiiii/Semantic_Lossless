# Person 1 measured results

Final consolidated export: DeepJSCC, ADJSCC and ISEC, from multiple saved
continuation versions. DeepJSCC/ADJSCC150 full epochs each; ISEC150 JSCC plus
150 denoiser epochs. Frozen training membership3450 parents, including59
duplicates;100 DIV2K development images; Kodak24 original dimensions and both
orientations,10 noise trials/image,240/model. All720 trial artifacts and metric
means were verified. Resuming evaluation on Colab did not retrain the model.

comparison.csv is the aggregate three-model table. Each model contains its
training_summary.json and metrics/per_image.csv, summary.json, profile.json,
matching the Person2 result layout. Metrics: PSNR, SSIM, MS-SSIM, AlexNet LPIPS,
MSE and measured complex channel uses/CBR. Summary confidence intervals use
24 image means,2000 bootstrap resamples,seed22. No test image selected weights.

ISEC uses native CBR1/16 versus DeepJSCC/ADJSCC1/6. These are NOT equal-rate
measurements. ISEC measured14.87dB PSNR; completion is not paper convergence.
DeepJSCC is a paper-faithful reproduction, not verified original-author code.
ADJSCC uses pinned TensorFlow author code; ISEC uses native author encoder,
decoder and BF_CNN,50 refinement iterations. Common data/schedule adaptations
and each source's real/complex AWGN convention remain disclosed.

Profiles preserve original registered-only counts as partial and add complete
DECLARED LOGICAL inference/training arithmetic:2operations/MAC, explicit
special primitives, native backward and Adam/reduction, with exclusions noted.
AD FFTs follow the TensorFlow2.14.1 Eigen source-reference logical algorithm.
ISEC counts all50 actually executed refinements and latent-gradient backward;
its exact operation count is checkpoint/input/seed/branch specific. These are
not physical GPU instruction counts. Full raw audits stay in the private ZIP.

Native timing is labelled per device. AD final evaluation used CPU; the old
helper's GPU0 label is corrected. ISEC used T4 and includes50 refinements.
CPU and GPU timings must not be ranked as matched hardware. ISEC's172 Kaggle
CUDA12.8 plus68 Colab CUDA13.0 trials used identical final checkpoint/adapter
bytes and per-trial seeds; backend migration is explicitly disclosed.

Weights, raw reconstructions, source/audit snapshots and preserved failure logs
remain in the verified private OneDrive handoff under image_semcom/SharedModelHandoff.
ZIP SHA256:5058e94ea2f3df3d0975ef655a166b6f218f56121dc1be305e4c036e8b3d85aa.
No weights, datasets or credentials are published. NTSCC+ remains pending and
is excluded. The previous expanded three_model_final publication was removed
at the user's request; all its local evidence remains intact.
