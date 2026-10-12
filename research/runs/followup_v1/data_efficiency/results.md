# Descriptive equal-error data-efficiency analysis

Post-hoc analysis of frozen main_v1 test metrics. Values interpolate train geometry-group counts within four observed budgets; no extrapolation or new empirical data-saving claim.

| Target normalized MAE | B interpolated groups | D interpolated groups | Estimated group savings | Joint bootstrap identifiable fraction | Conditional95% savings interval |
| --- | --- | --- | --- | --- | --- |
| 0.080 | 26.74 | 14.74 | 44.87% | 97.28% | [10.45%, 56.51%] |
| 0.085 | 12.93 | 10.35 | 19.97% | 96.65% | [9.89%, 58.29%] |

Bootstrap intervals condition on both requirements being identifiable inside observed budgets. Censored draws are counted and explicitly reported in analysis.json; the interval is not an unconditional confidence interval. Fractional group counts are interpolation estimates, not actually trained experiments. A prespecified fresh evaluation is needed for a confirmatory equal-error data-saving claim.

Group count axis:6/15/30/60. Exact sample counts vary across target seeds. Source pretraining cost is additional and is reported separately.
