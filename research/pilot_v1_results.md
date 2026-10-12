# Pilot v1 validation 결과

주 비교 D/B의 평균 validation MAE는 25% budget에서 0.084626 → 0.079737(5.78% 감소), 100%에서 0.071953 → 0.069178(3.86% 감소)다. 6개 paired budget/seed 모두 D가 B보다 낮았다. A/C는 100% seed2에서 C가 더 나쁜 결과도 있어 보존한다. 이 결과는 다음 convergence 검증을 진행할 근거이며 test 효과나 통계적 유의성의 증거는 아니다.

모델 정의: A=scratch 공간 입력, B=scratch 공간+scalar/pattern, C=source encoder 공간 입력, D=source encoder 공간+scalar/pattern.

Source pretraining, 동일 LR screen, 24 target fits를 완료했다. 이 수치는 validation에서 checkpoint와 LR를 선택한 pilot 결과이며 독립 test 결과가 아니다.

| Budget | Model | Validation MAE mean ± seed SD | Selected RMSE mean | 30epoch 도달 runs |
| --- | --- | --- | --- | --- |
| 25% | A | 0.088279 ± 0.000568 | 0.128703 | 3/3 |
| 25% | B | 0.084626 ± 0.001285 | 0.124720 | 3/3 |
| 25% | C | 0.087116 ± 0.001673 | 0.126063 | 3/3 |
| 25% | D | 0.079737 ± 0.003331 | 0.119154 | 3/3 |
| 100% | A | 0.083744 ± 0.000905 | 0.124943 | 2/3 |
| 100% | B | 0.071953 ± 0.001645 | 0.112652 | 3/3 |
| 100% | C | 0.083269 ± 0.000255 | 0.123688 | 1/3 |
| 100% | D | 0.069178 ± 0.001196 | 0.109197 | 2/3 |

동일 seed/subset에서 positive B−D는 conditioned transfer의 낮은 validation 오차를 의미한다. 아래 차이는 이미 validation 선택을 거친 수치이므로 유의성 또는 test 개선의 근거로 사용하지 않는다.

| Budget | Seed | A−C | B−D | Interaction |
| --- | --- | --- | --- | --- |
| 25% | 0 | 0.000579 | 0.009079 | 0.008499 |
| 25% | 1 | 0.002523 | 0.003293 | 0.000770 |
| 25% | 2 | 0.000387 | 0.002296 | 0.001910 |
| 100% | 0 | 0.001096 | 0.002773 | 0.001677 |
| 100% | 1 | 0.001114 | 0.004425 | 0.003311 |
| 100% | 2 | -0.000785 | 0.001126 | 0.001911 |

최대30epoch에 도달한 run은 20/24이다. Source는 1500update, target은 max30epoch 제한이므로 convergence를 확인하기 전에는 최종 정확도 또는 negative transfer의 원인을 단정하지 않는다.

재현 파일은 runs/pilot_v1의 protocol.json, selected_learning_rates.json, source checkpoint, 24개 config/history/normalization/best.pt/summary와 analysis.json/selected_metrics.csv다. [Validation plot](runs/pilot_v1/validation_mae.png).

다음 단계는 동일 validation protocol에서 convergence/LR screen의 민감도를 검토하고 필요한 경우 학습을 연장하는 것이다. 본 실험의 예산/5개 seed와 source seed 민감도 protocol을 확정한 뒤 독립 test를 한 번 평가한다. 현재 pilot에서는 test image/label을 열지 않았다.

실행 비용: source/LR-screen/24 target fit의 기록된 training-stage runtime 합은 50.86 device-run minutes이다. 이는 두 GPU에서 병렬 실행한 각 stage의 wall-time 합이며 GPU-active time, 준비/프로세스 시작 시간을 포함한 end-to-end 시간과 구분한다. TF32를 허용한 float32 실행이다.

현재 config의 seed SD는 25%에서 subset 선택과 optimizer 변동이 함께 포함되고, 100%에서는 동일 train group에 대한 optimizer 변동이다. Source seed는 0 하나이며 source-seed 민감도는 미검증이다. Best epoch와 epoch-limit 여부는 runs/pilot_v1/convergence_flags.json에 공개한다.
