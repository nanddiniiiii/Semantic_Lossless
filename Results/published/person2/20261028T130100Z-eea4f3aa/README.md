# Person 2 measured results

Run: 20261028T130100Z-eea4f3aa. Kaggle satyapaladuguplus/semantic-notebook-2 version 2; source35dad214f91f343b0acf7f68c9355a92fda32c29. Exported 3 November 2026 from existing downloaded bundles. Published individual files were checked against their bundle SHA256 ledger; this does not certify all checkpoint files or independent result replication.

Training: approximately two hours per model, common DIV2K/Flickr2K mixture with documented pseudo-label adaptations. Test: Kodak24, three trials per image,72 trials per model. SQ-GAN is pretrained adversarial fine-tuning and ideal source coding; MDJCM and Shared Codebook use AWGN10. Rates differ and source bpp is not channel-use CBR. These are not matched-rate paper reproductions. SSC failed its pilot with a timm import error and has no completed results in this run.

Profiles: FP32,512x768 input,cuda:0 Tesla T4,3 warmups/10 repeats. Registered-kernel FLOPs are partial profiler counts with2FLOPs/MAC, not complete arithmetic totals. Whole-image exact reconstruction was zero. Prediction agreement/feature cosine are classifier proxies, not ground-truth semantic accuracy.

Model weights, model assets and credentials are deliberately excluded. Full checkpoints remain in local/Kaggle output storage. Older local master comparison CSV includes operational and historical entries and is not substituted for this verified run-specific table.
