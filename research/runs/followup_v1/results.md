# Required follow-up results

Supporting/exploratory evaluations after original main test was known. Models selected only by validation; geometry bootstrap intervals are descriptive.

## Parameter-count-matched scalar-only information controls

| Group budget | Scalar-only B MAE | Scalar-only D MAE | D reduction | B scalar−full | D scalar−full |
| --- | --- | --- | --- | --- | --- |
| 10% | 0.091143 | 0.089090 | 2.25% | -0.001290 | -0.000858 |
| 25% | 0.083916 | 0.081773 | 2.55% | 0.001138 | 0.002068 |
| 50% | 0.083572 | 0.081924 | 1.97% | 0.004344 | 0.005048 |
| 100% | 0.079973 | 0.078335 | 2.05% | 0.006208 | 0.006196 |

Positive scalar−full values mean lower error with pattern information. The pattern encoder remains present with zero normalized pattern inputs; this is an information ablation, not a smaller-architecture comparison.

## Configuration-density axis

| Configurations per spreadsheet row | Train samples | B MAE | D MAE | D reduction |
| --- | --- | --- | --- | --- |
| 1 | 178 | 0.079061 | 0.078203 | 1.09% |
| 2 | 356 | 0.075247 | 0.073639 | 2.14% |
| 3 | 534 | 0.073765 | 0.072139 | 2.20% |

All60 geometry groups and178 spreadsheet rows are retained. Antenna-number entries can have different Tx coordinates/heights, so this is not fixed-Tx angular variation. The534-sample references are frozen main100% results, not new independent replicates.

## Adapted prior architecture

| Group budget | HRT scratch MAE | HRT source-encoder MAE | Reduction |
| --- | --- | --- | --- |
| 25% | 0.071946 | 0.067238 | 6.54% |
| 100% | 0.062226 | 0.056937 | 8.50% |

HRFormer-S+multistage scalar FiLM+pattern cross-attention uses audited B/T inputs, cyclic scalars, full-raster L1 and group split. Published7-channel/beam/outdoor/random-split/TTA scores are not directly compared. Source replication is limited to seed0 for this expensive reference study. Geometry/renderer/absolute-unit/geographic-overlap limitations remain.
