# Semantic Lossless

Research code for image semantic communication experiments. The project name does not imply that neural models provide lossless reconstruction.

## Three-team experiments

The active workflow uses one shared codebase and three Kaggle notebooks:

| Person | Notebook | Models in execution order |
|---|---|---|
| 1 | [1.ipynb](Notebooks/1.ipynb) | DeepJSCC, ADJSCC, ISEC, NTSCC+ |
| 2 | [2.ipynb](Notebooks/2.ipynb) | Masked SQ-GAN, MDJCM, SharedSemanticCodebook, SSC |
| 3 | [3.ipynb](Notebooks/3.ipynb) | HJSCC, OFDM-guided JSCC, DeepJSCC-MIMO, DeepJSCC-WZ |

These twelve slots contain twelve distinct models; SwinJSCC is retained for evaluation. SSC, OFDM and MIMO adapters are implemented; their two-T4 preflight remains mandatory. See Docs/NATIVE_SYSTEMS_IMPLEMENTATION.md. The [setup guide](configs/README.md) explains datasets, credentials, model assets, scripts, and retrieval. Scripts create Kaggle jobs only when explicitly run. The numbered notebooks are pinned to the published implementation revision; a fresh Kaggle GPU pilot remains required.

## Project layout

| Path | Purpose |
|---|---|
| `src/` | Shared model adapters, dataset, training, evaluation, and launch code. |
| `configs/` | Model lists and operating guide. |
| `Notebooks/` | Three numbered notebook launchers. |
| `scripts/1.py`, `2.py`, `3.py` | Three local Kaggle launch and collection entry points. |
| `Results/` | Experiment outputs, logs, metrics and comparison CSV. |
| `Models/` | Checkpoints and model assets; existing checkpoints are immutable. |
| `Docs/` | Research, provenance, validation and previous work. |
| `Infrastructure/` | Channel models and independent train/test condition definitions. |
| `Archive/EncDec/`, `SharedCode/` | Older retained codec code and historical research components. |

## Validation and limitations

The datasets and model paths have bounded checks, not a completed comparison of all twelve slots. Some paths are adaptations, and SQ-GAN currently has no wireless channel in its source-codec path. Read the [validation record](Docs/TEAM_SUITE_VALIDATION.md) before making claims about model fidelity, results, or readiness. Saved outputs are not evidence of convergence or lossless delivery by themselves.

The earlier root README is preserved in [README_HISTORICAL_20261027.md](Docs/README_HISTORICAL_20261027.md); older notebooks and experimental artifacts are retained pending a provenance review.

## Results and sharing

The local launcher stores retrieved runs under `Results/team_downloads/` and updates `Results/comparison.csv`. Completed model bundles go to:

```text
Models/
  teammate_1_models/<Model>/<run-id>/
  teammate_2_models/<Model>/<run-id>/
  teammate_3_models/<Model>/<run-id>/
```

Zip a teammate folder to share it. Each bundle includes checkpoints, configuration, required available constructor assets, metrics and integrity metadata. Partial checkpoints remain with the raw run evidence. Source and dependencies are still required to load a model. These folders are created now; empty folders do not indicate newly trained models.

Result-only GitHub commits are now queued automatically after verified retrieval; source changes are never committed automatically. See the [workflow audit](Docs/TRAINING_WORKFLOW_AUDIT.md) for checks, remaining blockers and the distinction between completed execution and validated results.

The [collection guide](Docs/RESULT_COLLECTION.md) explains 30-minute polling, restart recovery, GitHub permissions and the saved-notebook-output limitation.

Start here: [teammate handoff and draft messages](Docs/TEAMMATE_HANDOFF.md).
