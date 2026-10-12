# Convergence v1 validation results

Longer fresh B/D fits with fixed pilot LR; validation-selected results. Test data was not opened.

| Budget | Model | MAE mean ± seed SD | RMSE mean | Epoch-cap runs |
| --- | --- | --- | --- | --- |
| 25% | B | 0.078049 ± 0.000623 | 0.121043 | 0/3 |
| 25% | D | 0.075337 ± 0.001973 | 0.116849 | 0/3 |
| 100% | B | 0.070759 ± 0.000785 | 0.111565 | 0/3 |
| 100% | D | 0.068737 ± 0.000578 | 0.109002 | 0/3 |

| Run | Selected/total epoch | MAE | Reduction from pilot |
| --- | --- | --- | --- |
| B_budget25_seed0 | 66/91 | 0.078765 | 7.80% |
| D_budget25_seed0 | 45/70 | 0.073587 | 3.62% |
| B_budget100_seed0 | 48/73 | 0.071220 | 2.75% |
| D_budget100_seed0 | 32/57 | 0.069169 | 1.84% |
| B_budget25_seed1 | 70/95 | 0.077629 | 6.63% |
| D_budget25_seed1 | 51/76 | 0.077476 | 2.97% |
| B_budget100_seed1 | 39/64 | 0.071203 | 1.82% |
| D_budget100_seed1 | 22/47 | 0.068081 | 0.03% |
| B_budget25_seed2 | 98/123 | 0.077753 | 8.85% |
| D_budget25_seed2 | 81/106 | 0.074949 | 9.71% |
| B_budget100_seed2 | 31/56 | 0.069853 | 0.35% |
| D_budget100_seed2 | 26/51 | 0.068960 | 0.02% |

| Budget | Seed | B−D |
| --- | --- | --- |
| 25% | 0 | 0.005178 |
| 25% | 1 | 0.000153 |
| 25% | 2 | 0.002804 |
| 100% | 0 | 0.002051 |
| 100% | 1 | 0.003122 |
| 100% | 2 | 0.000893 |

Epoch cap: 0/12. Best in final25 epochs: 0/12.

These flags are conservative diagnostics; early stopping does not prove convergence. Preserve source seed/budget, geometry proxy and geographic-overlap limitations. No significance or data-saving claim is established.

Checkpoint verification passed for all12 runs. Full histories and analysis.json accompany this report.

[Paired learning curves](validation_learning_curves.png) · [Budget curve with target-seed SD](validation_budget_curve.png). Figures use saved validation histories only.
