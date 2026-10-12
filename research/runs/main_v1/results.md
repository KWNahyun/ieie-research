# Main v1 independent held-out image-group results

B: scratch conditioned U-Net; D: source encoder transfer with identical target inputs/architecture. Five target seeds and three1500-update source seeds; test checkpoints frozen before evaluation. Primary endpoint:25% target data.

| Target group budget | B test MAE | D test MAE | Relative reduction | B−D paired geometry95% interval |
| --- | --- | --- | --- | --- |
| 10% | 0.092433 | 0.089948 | 2.69% | [0.000463, 0.004642] |
| 25% | 0.082779 | 0.079705 | 3.71% | [0.001823, 0.004248] |
| 50% | 0.079228 | 0.076876 | 2.97% | [0.001557, 0.003220] |
| 100% | 0.073765 | 0.072139 | 2.20% | [0.000754, 0.002568] |

Intervals resample14 image geometry groups, after averaging model-seed errors. They condition on this split and trained models; pixel counts and shared controls are not independent replicates. Source and target-seed variation are reported separately in test_analysis.json. Secondary-budget intervals are descriptive, with no multiple-comparison significance claim.

Verified training checkpoints:80. Unresolved final epoch-cap runs:0.

The percentages are fractions of60 train geometry groups, not exact sample fractions. Actual per-seed base counts are in training_records.json. This is normalized-image whole-raster evaluation with image geometry proxies. Absolute EMF, independent outdoor masks, physical-scene provenance and geographic/source-target overlap remain unresolved; no physical/geographic generalization or exact data-saving claim.

All80 test metric files preserve per-sample/group errors and checkpoint hashes. Test results were not used to retrain or choose checkpoints. Source budget sensitivity is reported separately.

[Independent final audit](final_audit.json) · [Statistics verification](statistics_verification.json) · [Budget curve](test_budget_curve.png)
