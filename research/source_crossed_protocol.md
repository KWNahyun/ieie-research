# Optional source crossed grid: validation-only extension

Prepared2026-10-10 before main test-array access. Priority: initial source sensitivity → main B/D → spatial A/C → this extension. Execute only within the existing approximately10-hour autonomous window; no source/target settings are changed based on main or ablation test results.

Purpose: complete source update budgets500/1500/4500 × source seeds0/1/2. Source_sensitivity_v1 covers five of these nine conditions; add500/seed1,500/seed2,4500/seed1,4500/seed2. Four new source runs plus24 new D fits. Reuse the five original source conditions and30 target fits (including the six convergence source0/1500 D fits). A complete grid contains nine source conditions and54 D fits across target budgets25/100%, target seeds0/1/2. Shared six scratch B controls remain shared references, not independent replicates.

Identical unmodified train_source.py/train_baseline.py, source/target splits, source validation probes every250 updates, minimum source validation checkpoint, encoder-only transfer, fresh paired target initialization, LR3e-4, target batch4/channels16, max150 epochs/patience25, base-only image proxy and whole-raster geometry macro MAE/RMSE. This validation-only grid uses the original sensitivity150-epoch rule, not main capped-fit300 replacements. Preserve and flag any capped fits; do not imply complete convergence.

No test evaluation is added for these source budgets. Verify all source/target checkpoints, paired normalization/IDs and source hashes; report every condition, selected source update, per-target differences, source/target seed variation and runtime. Source0's original1500 run used GPU1; all new runs use GPU2 with float32/TF32. Same seed/data order is not a bitwise-deterministic trajectory.

Stop starting new stages after2026-10-10 10:46:34UTC. If the grid is incomplete, retain a completion ledger and raw partial validation artifacts, and do not report an incomplete grid as a balanced source-budget comparison. Main/ablation test checkpoints and analyses remain fixed. Geometry/units/rendering/geographic-overlap limitations remain; no significance or data-saving claim from this extension.
