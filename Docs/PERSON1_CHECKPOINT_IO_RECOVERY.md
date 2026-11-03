# ISEC checkpoint I/O recovery after version 5

Version 5 completed 140 JSCC epochs and saved offset 152 of epoch 141. The
checkpoint contains model, optimizer and RNG state and was CPU-loaded after
137 exported artifact entries were hash checked. The unfinished `last.tmp`
file is zero bytes; it is not a checkpoint and is never used for continuation.

Before the timeout, compute rose from about 0.08 seconds per step to hundreds
of seconds; repeated stack dumps then showed native Torch tensor serialization
inside `torch.save`. The controller stopped the worker at its 18,000-second
bound. The precise host/library cause of this slowdown is unproven. The decoded
cache was functioning, with roughly 0.01–0.02-second image loading and matching
native-loader inputs. No nonfinite objective or model architecture fault was
identified in this failure.

The declared runtime recovery stages checkpoint tensors to CPU without changing
the live training tensors, serializes with native `torch.save`, and retains the
original temporary-file/atomic-rename workflow. A 120-second save watchdog exits
the worker if snapshot or serialization stops progressing, preserving the prior
atomic checkpoint. Local tests cover metadata/shared object preservation,
non-mutation, exact next Adam update after reload, and a deliberately blocked save
in a subprocess. Native two-GPU pilot, strict reload and interrupted/resumed
acceptance checks remain mandatory in the new Kaggle run.

Recovery chunks are 5,400 seconds (within the four-hour maximum), with a
6,300-second controller bound to allow normal final checkpointing and export.
These limits change session scheduling, not batch, epochs, data, crop/flip RNG,
model, objective, channel or optimizer semantics. Both stages still require
150 complete epochs. Every continuation records the reviewed adapter hash and
keeps the immutable original model/runtime source cells. The full final
checkpoint remains authoritative, rather than elapsed time or session status.

No recurring monitoring timer is recreated. Existing local collection and chunk
continuation are separate from the removed timer. No Git publication is allowed.

Version 6 verified the save adapter on GPU, completed JSCC150 and denoiser10, then
was killed externally during denoiser11. Its last complete checkpoint includes
both optimizer states and RNG state and was CPU-loaded after 151 exported entries
were hash verified. This later failure was not the checkpoint-save watchdog.
The following recovery uses 2,700-second chunks and a 3,600-second controller
bound, disk-cache page reclamation, and persisted resource diagnostics. It keeps
all required full epochs and native two-GPU acceptance checks. A SIGKILL cause
must not be labelled confirmed OOM without kernel/cgroup evidence.

Later successful chunks showed process RSS growing to about 19 GiB in 45 minutes,
mostly anonymous memory. The exact retained-object/allocator cause is unproven.
The next reviewed adapter runs Python cyclic garbage collection and glibc
`malloc_trim` at existing checkpoint boundaries, recording RSS before/after and
collected-object counts. Local tests preserve live tensor values, Python/Torch
RNG and the exact next Adam update. Actual GPU memory improvement is pending.

The coordinator may select 90-minute chunks only after a matching-adapter normal
chunk supplies at least 20 minutes of diagnostics and three checkpoints, process
RSS never exceeds 8 GiB, post-reclamation RSS stays within 6 GiB, and all observed
OOM counters are zero. Missing evidence retains 45-minute chunks. This policy is
fault-tested and changes restart frequency only, not any training semantics.

## cuDNN execution-plan cache investigation

The measured garbage-collection adapter did not materially reduce the rising
anonymous RSS. Inspection of the exact Torch2.11.0 source found an explicit
per-thread cuDNN execution-plan cache lifetime leak, retained to avoid teardown
segfaults with newer cuDNN. DataParallel creates replica threads repeatedly, so
this is a specific candidate for the observed growth rather than a confirmed
GPU-measured attribution yet.

Pinned evidence:
https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cudnn/Conv_v8.cpp
lines243–270 define the environment setting; lines330–343 explain the lifetime.

The next training runtime sets `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=-1` before
worker import. Negative means no plan caching; zero means unlimited and is not
the intended setting. This does not disable cuDNN, its convolution operations,
CUDA tensor allocation caching, or deterministic settings. It does not enable
AMP/TF32 or change model, batch, losses, data or optimizer. The scope is native
pilot/resume-check/train only; ordinary evaluation caching remains unchanged.
The setting and prior adapter lineage are recorded in the notebook and runtime
adaptation JSON. GPU pilot/gradient/strict reload/split-resume acceptance remains
mandatory before production. End-to-end speed and host RSS must be compared
after execution, because rebuilding plans can also add compute overhead.
