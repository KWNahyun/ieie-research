# Spatial ablation v1 held-out image-group results

Supporting preregistered comparison; C and D both use source seed0. Five target seeds. A/C spatial-only; B/D scalar/pattern-conditioned.

| Budget | A MAE | C MAE | A−C | B−D_source0 | Interaction | Interaction geometry95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| 10% | 0.096158 | 0.097316 | -0.001158 | 0.001947 | 0.003104 | [-0.000028, 0.006062] |
| 25% | 0.094426 | 0.094214 | 0.000212 | 0.002749 | 0.002537 | [0.000442, 0.004500] |
| 50% | 0.092762 | 0.092269 | 0.000493 | 0.002613 | 0.002120 | [0.000616, 0.004033] |
| 100% | 0.091091 | 0.091177 | -0.000087 | 0.001400 | 0.001487 | [0.000508, 0.002438] |

Interaction=(B−D_source0)−(A−C). Positive values indicate larger conditioned transfer improvement for these trained models; this is not proof of mechanism or significance. Intervals are descriptive paired14-geometry-group bootstrap intervals conditional on this split and trained models. C was not replicated across source seeds. Geometry/rendering/absolute-unit/geographic-overlap limitations remain. No test-based setting or checkpoint changes were made.

[Independent final audit](final_audit.json) · [A–D and interaction figure](test_spatial_and_interaction.png)
