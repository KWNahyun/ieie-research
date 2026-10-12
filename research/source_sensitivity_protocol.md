# Source sensitivity v1

Frozen 2026-10-10 before new source/target validation results.

Question: how sensitive is the conditioned transfer advantage to source update budget and source initialization/data-order seed?

Budget axis: source seed0 at500/1500/4500 updates. Seed axis: source seeds0/1/2 at1500 updates. This is two separate sensitivity axes, not a fully crossed budget-by-seed design. Source split, DPM supervision, source architecture, batch16, AdamW LR3e-4 and fixed validation probes remain as in pilot_v1. Select minimum source validation MAE at250-update intervals; report actual selected update as well as maximum budget. Nested seed0 training budgets use the same seed/order from initialization. No source-convergence claim.

Reuse the existing seed0/1500 checkpoint and six longer D fits; reuse six longer scratch B controls. Train four new source conditions (500/seed0,4500/seed0,1500/seed1,1500/seed2) and six target D fits per condition: target budgets25/100%, target seeds0/1/2. Total new work: four source fits plus24 target fits. Reused controls are shared references, not additional independent replicates.

Every target fit starts from paired fresh initialization with encoder-only transfer, using convergence_v1 inputs/subsets/normalization, LR3e-4, batch4, channels16, max150 epochs and patience25. Equal zero additional tuning budget. Whole-raster normalized-image geometry-group macro MAE/RMSE; test never accessed. No stored augmentations. Existing geometry/rendering/geographic-overlap limitations remain.

Record source checkpoint hashes, input/code snapshots, histories, selected update, source validation MAE, target selected epochs, epoch-cap flags, runtimes and paired B−D differences. Verify source and target strict checkpoint loading/finite synthetic prediction, exact configs, source hash, paired IDs and normalization. Summarize means and target-seed SD for each source condition/budget. Do not treat shared controls or source conditions as independent statistical replicates. Do not select a favorable source condition and then claim unbiased validation performance.

Use GPU2 while GPU1 is occupied; exclude GPU0. Refuse overwrites. Run metadata preflight with `python research/run_source_sensitivity.py --dry-run`. Launch with `python research/run_source_sensitivity.py --gpu 2`. Completed artifacts are analyzed automatically by `analyze_source_sensitivity.py`.

If target runs hit the epoch cap or benefits reverse, report those outcomes before freezing the main experiment. Next: freeze10/25/50/100% budgets, five target seeds, source replication and independent test analysis. This study alone establishes neither significance nor data-saving rates.
