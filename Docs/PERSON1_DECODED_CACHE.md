# Person 1 decoded-image cache

This is an I/O adaptation for the next verified ISEC continuation. The active
chunk is preserved. No author model, loss, optimizer, batch, channel, epoch
membership or augmentation is changed.

`scripts/person1_decoded_cache.py` caches EXIF-corrected RGB uint8 parents as
memory-mapped NPY files under `/kaggle/temp/person1-decoded-rgb`. It reproduces the frozen
loader's crop coordinates, random flip, float32 conversion and normalization.
Keys use the source file SHA256; decoded data is checked against the frozen
dimension-prefixed RGB SHA256. Duplicate manifest entries remain separate image
exposures. The cache has a 40 GiB budget and reserves 3 GiB free space. Insufficient
space falls back to the original decoding path without dropping samples.

After version 6 ended with SIGKILL during denoiser epoch 11, the adapter was
extended to verify that scratch is not tmpfs/ramfs, explicitly close each mmap
after producing its independent crop array, and request clean page-cache
reclamation with `posix_fadvise(DONTNEED)`. This bounds dataset-cache residency
without changing pixels. The SIGKILL origin is not proven to be memory pressure;
no kernel OOM counters were available in that run. The next run persists cgroup
memory/kill counters, process memory and GPU status every 30 seconds. Temporary
disk/source data are outside exported output directories. Tests cover mapping
closure, the page advice request and fallback on a RAM-backed cache directory.

The private notebook derivative embeds the reviewed adapter with a SHA256 check.
It wraps worker startup and records `decoded_cache_adaptation.json`; original
source cells, the private source overlay and checkpoint contract stay unchanged.
This additional runtime adapter is explicitly declared, rather than represented
as an unchanged runtime. Native GPU pilot and split-resume checks still run.

Local validation includes 90 exact array comparisons across image modes, EXIF,
orientations, splits, epochs, seeds and duplicate membership. Corrupt cache pixels
are rejected; disk-budget fallback preserves data. Generated notebook routing,
syntax, idempotence and preservation of original cells were checked. A local
synthetic PNG microbenchmark demonstrates loading improvement only; it is not a
Kaggle throughput measurement. See the account's `decoded-cache-plan.json` for
reviewed source hash and measurements.

The first pass fills the cache; subsequent epochs should benefit. Every new
Kaggle session has its own scratch cache. Existing dataset hash verification is
retained. Actual end-to-end speedup must be measured from cache hit counters and
training data/step times after GPU execution. No new monitoring timer is enabled.
