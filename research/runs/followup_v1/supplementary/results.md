# Supplementary frozen-checkpoint evaluation

All120 main/spatial checkpoints evaluated with fixed SSIM policy; repeated MAE/RMSE matched prior outputs at1e-6. No model selection/retraining.

| Budget | Model | Test group-macro SSIM |
| --- | --- | --- |
| 10% | A | 0.829362 |
| 10% | B | 0.804183 |
| 10% | C | 0.777507 |
| 10% | D | 0.805979 |
| 25% | A | 0.852251 |
| 25% | B | 0.852067 |
| 25% | C | 0.836189 |
| 25% | D | 0.849985 |
| 50% | A | 0.862375 |
| 50% | B | 0.859118 |
| 50% | C | 0.858118 |
| 50% | D | 0.858427 |
| 100% | A | 0.863677 |
| 100% | B | 0.865867 |
| 100% | C | 0.866209 |
| 100% | D | 0.864783 |

Compute uses model-only CUDA events under current shared-GPU load. FLOPs include supported conv/matmul only; synthetic batch4 memory is not historical training peak VRAM. Parameter counts and timing/memory details are in analysis.json. Representative figures use metadata-only selection and identical color scales.
