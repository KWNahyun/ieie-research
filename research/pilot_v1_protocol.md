# Pilot v1 — 고정 실행 protocol

2026-10-08. Test를 열지 않는 validation-only source/target pilot이다. Raw building 파일이 없다는 사용자 확인을 반영한 image-proxy/full-raster 범위를 유지한다.

| 항목 | 고정 설정 |
| --- | --- |
| GPU | idle RTX4090 1/2, 동시 process 최대 2; GPU0 제외 |
| Backbone | U-Net, base channels 16, native 256/500 해상도 |
| Precision | float32, CUDA matmul/cuDNN TF32 허용 |
| Source | DPM seed0, batch16, AdamW LR3e-4/weight_decay1e-4, 1500 updates |
| Source validation | validation map 각각 Tx0/40; 250updates마다 map macro MAE로 encoder checkpoint 선택 |
| LR 후보 | 각 A–D에 1e-4 및 3e-4, 같은 25% budget/seed0, 최대10epoch/patience3 |
| Target pilot | A–D × 25/100% budget × seed0/1/2, 24 fits |
| Target training | batch4, AdamW/weight_decay1e-4, 최대30epoch/patience8 |
| Transfer | source encoder만 strict load, decoder/head/conditioning 새 초기화; 처음부터 full fine-tuning |
| Validation | 전체 target validation 108개 base/12 image groups, whole-raster group macro MAE/RMSE |
| Test | sample/label 접근 및 평가 없음 |

LR 탐색은 모델별로 동일한 예산을 제공하고 가장 낮은 validation group macro MAE를 기록한 후보를 선택한다. Source 선택은 source validation만 사용한다. Final paired fit은 같은 subset 및 seed를 공유하고 동일 target 구조의 A/C와 B/D를 비교한다.

Source 고정 update 예산과 target 최대 epoch는 pilot 계산 한도다. 충분한 수렴을 보장하지 않으며 예산 소진으로 계속 개선 중인 run은 결과에서 표시한다. Source seed 민감도 및 본실험 5개 target seed는 이 단계에 포함하지 않는다. Geographic overlap 미확인과 renderer 차이는 한계로 남긴다.

실행: `python IEIE/research/run_pilot.py`. 각 run은 overwrite를 거부한다. Protocol, LR 선택, configs, statistics, checkpoints, histories, summaries와 console log를 `runs/pilot_v1/`에 저장한다. 진행과 실패는 작업로그.md에 기록한다.
