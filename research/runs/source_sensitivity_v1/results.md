# Source sensitivity v1 validation results

Separate source-budget and source-seed sensitivity axes; shared scratch controls. Validation-selected results; test untouched.

| Source maximum updates | Source seed | Selected updates | Source MAE |
| --- | --- | --- | --- |
| 1500 | 0 | 1500 | 0.061281 |
| 500 | 0 | 500 | 0.079118 |
| 4500 | 0 | 4500 | 0.048985 |
| 1500 | 1 | 1500 | 0.054738 |
| 1500 | 2 | 1500 | 0.062718 |

| Source updates | Source seed | Target data | D MAE ± target-seed SD | Reduction vs B | D better pairs |
| --- | --- | --- | --- | --- | --- |
| 1500 | 0 | 25% | 0.075337 ± 0.001973 | 3.47% | 3/3 |
| 1500 | 0 | 100% | 0.068737 ± 0.000578 | 2.86% | 3/3 |
| 500 | 0 | 25% | 0.076213 ± 0.000814 | 2.35% | 3/3 |
| 500 | 0 | 100% | 0.069537 ± 0.000280 | 1.73% | 3/3 |
| 4500 | 0 | 25% | 0.074562 ± 0.002295 | 4.47% | 3/3 |
| 4500 | 0 | 100% | 0.068201 ± 0.000061 | 3.61% | 3/3 |
| 1500 | 1 | 25% | 0.075172 ± 0.000337 | 3.69% | 3/3 |
| 1500 | 1 | 100% | 0.068634 ± 0.000287 | 3.00% | 3/3 |
| 1500 | 2 | 25% | 0.074655 ± 0.001697 | 4.35% | 3/3 |
| 1500 | 2 | 100% | 0.067931 ± 0.000871 | 4.00% | 3/3 |

Target epoch-cap runs: 0/30 (includes six reused fits).

Source seed0/1500 and scratch controls are reused from prior runs. Shared controls and source conditions are not independent statistical replicates. Source-budget selection uses a fixed validation probe; actual selected updates can be lower than the maximum. No significance or data-saving claim is established. Geometry/rendering/geographic-overlap limitations remain.

All checkpoint and pairing checks passed. Per-target results are in analysis.json; histories and configs remain in individual run folders.

[Validation sensitivity figure](validation_source_sensitivity.png) · [Source histories](source_learning_curves.png) · [Final verification](final_verification.json)
