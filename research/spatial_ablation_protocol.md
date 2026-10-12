# Spatial ablation v1: preregistered supporting A/C comparison

Prepared2026-10-10 before any target test evaluation. Execute only after main_v1 completes, within the same approximate10-hour autonomous window; primary B/D work takes precedence.

Models A (scratch spatial B/T) and C (spatial encoder transfer from source seed0/1500). Four target geometry-group budgets10/25/50/100%, target seeds0–4:40 fresh fits. Pilot A/C fits are not reused because their30-epoch cap is insufficient for a matched duration comparison. Source seed0 is fixed independently of any test outcomes. This supporting study does not replicate C across source seeds; do not imply source-averaged A/C interaction.

Same source checkpoint, target subsets/splits, image proxy, base-only full-raster loss/metrics, channels16, batch4, AdamW LR3e-4, weight decay1e-4, max150 epochs/patience25, and same capped-fit fallback to fresh300 epochs as main_v1. No hyperparameter search. A/C share fresh decoder initialization for each target seed. Scalar/pattern normalization is still fitted from the identical target subset by the existing trainer, though A/C do not use conditioning. No augmentation or test-based selection.

Before test, verify all40 configs/checkpoints and freeze a separate ablation manifest. Compare A/C errors and the interaction (B−D_source0)−(A−C), using the matching already-frozen main B and source0 D checkpoints. Average over five target seeds, then bootstrap the same14 held-out image geometry groups with10,000 paired draws, RNG20261011. All ablation/interaction intervals are descriptive. Do not use source-averaged D for this interaction, and do not claim conditioning-specific synergy from D being best alone.

The completed B/D test outputs must not trigger changes in models, settings or checkpoint-selection rules. If the deadline prevents40-fit completion, preserve partial validation results and do not run ablation test evaluation. Stop starting new training stages after2026-10-10 10:46:34UTC. Geometry/rendering/units/geographic-overlap limitations remain.
