# Convergence v1 protocol

Prepared 2026-10-10, before new validation results.

Purpose: determine whether the conditioned encoder-transfer advantage survives longer target training. This is a validation-only duration sensitivity study, preceding source-budget/seed sensitivity and the full experiment.

- Models B (scratch) and D (source encoder), budgets 25/100%, target seeds 0/1/2: 12 fresh fits. Reinitialize rather than resume pilot best checkpoints, which do not preserve the final sampler/RNG state.
- Preserve pilot subsets, train-only normalization, paired initialization, source seed0 checkpoint (1500 updates), 16 base channels, batch size4, AdamW, weight decay1e-4 and selected LR3e-4. No new LR search; equal zero additional tuning budget isolates duration.
- Maximum150 epochs, patience25, validation each epoch. Select minimum whole-raster normalized-image geometry-group macro MAE; report matching RMSE. No test dataset access.
- Preserve base-only image geometry proxy and scalar/pattern inputs. Geographic overlap, renderer fidelity and physical outdoor masks remain unresolved.
- GPUs1/2 only; GPU0 remains reserved. Preflight checks file availability, paired IDs/normalization, source hash and unchanged model/data code. Snapshot protocol, input hashes, splits, manifest and code before launch. Refuse to overwrite prior runs.
- Report each run's learning curve, selected epoch, final epoch, stopping reason, best MAE/RMSE, runtime and relative change from pilot. Report paired B−D differences and per-budget mean/seed SD. An epoch-cap hit or best checkpoint in the final25 epochs is a conservative warning; early stopping is not proof of mathematical convergence.
- If either model remains capped or still improving near the cap, resolve training duration before the full experiment. If the paired advantage shrinks or reverses, preserve that result. No significance, independent-test improvement or data-saving claims from this study.

Run preflight: `python research/run_convergence.py --dry-run`

Run experiment: `python research/run_convergence.py`

Analyze completed artifacts: `python research/analyze_convergence.py`

Next stage: separately preregister source-pretraining budget and seed sensitivity, using equally trained scratch controls. Freeze the four-budget/five-target-seed experiment and test analysis after these checks.
