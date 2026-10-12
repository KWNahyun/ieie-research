# 2026-10-10 자율 실험 최종 보고서

작성 및 독립 검증:2026-10-11. 핵심 GPU 실험은2026-10-10 승인된10시간 안에 완료됐다.

주 비교는 동일한 공간·scalar·pattern 입력과 구조를 사용하는 scratch B 대 source encoder transfer D다. D의 평균 held-out image-group test MAE는 네 데이터 예산 모두 B보다 낮았다. 사전 지정한 주 평가인25% 예산에서는0.082779→0.079705, 상대3.71% 감소다. 이는 normalized-image full-raster 결과이며 독립 physical/geographic scene 일반화나 절대 EMF 정확도 주장이 아니다.

## 본실험 결과

Target seed5개, source seed3개(최대1500 updates),60개 train geometry group의10/25/50/100% 예산이다. D는 각 모델의 오차를 source·target seed에 걸쳐 평균한 결과이며 prediction ensemble이 아니다. Test는126개 base sample/14개 image geometry group이다.

| Train group budget | Scratch B MAE | Transfer D MAE | 상대 MAE 감소 | B−D geometry bootstrap95% 구간 |
| --- | --- | --- | --- | --- |
| 10% | 0.092433 | 0.089948 | 2.69% | [0.000463, 0.004642] |
| 25% | 0.082779 | 0.079705 | 3.71% | [0.001823, 0.004248] |
| 50% | 0.079228 | 0.076876 | 2.97% | [0.001557, 0.003220] |
| 100% | 0.073765 | 0.072139 | 2.20% | [0.000754, 0.002568] |

구간은 trained-seed 평균 오차를 계산한 뒤14개 geometry group을 paired resampling한10,000회 percentile bootstrap이다(RNG20261010). Pixel을 독립 표본으로 세지 않는다. 해당 split과 학습 모델에 조건부인 구간이며 source/target seed 변동은 별도로 보존한다.25%에서14개 group 중13개가 D의 낮은 평균 오차를 보였다. 나머지 예산은 secondary/descriptive 분석이며 다중 비교 유의성 주장을 하지 않는다.

[본실험 상세 결과](runs/main_v1/results.md) · [Budget curve](runs/main_v1/test_budget_curve.png) · [Test JSON](runs/main_v1/test_analysis.json) · [독립 감사](runs/main_v1/final_audit.json) · [통계 재계산](runs/main_v1/statistics_verification.json)

## Source 민감도 및 공간 입력 보조 실험

Source budget500/1500/4500은 seed0에서 비교했고, source seed0/1/2는1500-update 예산에서 비교했다. Fully crossed design이 아니다. 모든30개 paired validation case에서 D가 공유된 B control보다 낮은 오차를 보였으며 test 개선으로 해석하지 않는다.

| Source max updates | Source seed | 25% validation 감소 | 100% validation 감소 |
| --- | --- | --- | --- |
| 500 | 0 | 2.35% | 1.73% |
| 1500 | 0 | 3.47% | 2.86% |
| 4500 | 0 | 4.47% | 3.61% |
| 1500 | 1 | 3.69% | 3.00% |
| 1500 | 2 | 4.35% | 4.00% |

[Source 민감도 보고서](runs/source_sensitivity_v1/results.md) · [Validation 그림](runs/source_sensitivity_v1/validation_source_sensitivity.png) · [Source 학습곡선](runs/source_sensitivity_v1/source_learning_curves.png) · [Source 최종 검증](runs/source_sensitivity_v1/final_verification.json)

A/C는 공간 입력만 사용하는 비교다. C의 전이 효과는 일관되지 않았다:10%와100%에서는 C의 오차가 A보다 높았고,25%와50%의 개선은 작았다. Matched source0 비교의 interaction=(B−D_source0)−(A−C)는 양수였지만,10%의 descriptive 구간은0을 포함했다. 메커니즘이나 synergy의 증명으로 해석하지 않는다. C의 source seed는0 하나다.

[A/C 상세 결과](runs/spatial_ablation_v1/results.md) · [A–D/source0와 interaction 그림](runs/spatial_ablation_v1/test_spatial_and_interaction.png) · [A/C 독립 감사](runs/spatial_ablation_v1/final_audit.json)

## 실행 및 검증

- 자율 실행 시작:2026-10-10 00:46:34UTC(09:46:34KST). Source 민감도 완료02:31:17UTC; main/test 완료05:53:25UTC; A/C/test 완료08:07:11UTC(17:07:11KST). 핵심 작업 경과7시간20분37초; 승인된 신규 학습 한도10:46:34UTC 안에 종료했다. 최종 독립 감사와 보고서는 다음 날 read-only 산출물 분석으로 수행했다.
- 새 training stage128개: source4개와 target124개. Main의56개 신규 fit 및3개 cap 재실행, A/C의40개 fit 및1개 cap 재실행을 포함한다. 공유 control/checkpoint는 재사용했으며 추가 독립 반복으로 세지 않았다.
- 기록된 training-stage runtime 합은430.57 device-run minutes다. GPU-active time 또는 end-to-end 시간과 다르며, 원래 capped run과 재실행을 포함하고 이전 control/source의 학습 비용은 제외한다.
- Main80개 및 A/C40개 checkpoint의 strict/finite load·synthetic prediction, 코드/input/snapshot hash, source seed/budget/hash, paired subset/normalization, train-only normalization, validation checkpoint 선택, test manifest 고정과 sample/group 대응을 검사했다.120개 모델의15,120개 sample 지표 및 group/macro MAE/RMSE 재계산이 통과했다. 모든 최종 모델은 epoch cap에 도달하지 않았다.
- 네 capped 원본 checkpoint를 그대로 보존하고 검증했다. 규칙에 따른300-epoch fresh 재실행에서도 동일 seed가 bitwise trajectory를 보장하지 않는다. TF32 및 nondeterministic CUDA 차이는 기록했다. 실패/불리한 결과를 삭제하지 않았다.
- Test checkpoint는 raw test sample 접근 전에 고정했다. Test 결과로 모델·source·LR·checkpoint를 다시 선택하거나 재학습하지 않았다. 모든 평가 지표와 provenance를 저장했다.
- 추가 fully crossed source grid는 optional이었다. 실행 과정의 시간 한도가 이미 지나 신규 source/target stage0개로 종료했다. Balanced grid 결과가 생성된 것으로 주장하지 않는다. 이 추가 실험은 별도 후속 승인 범위에서 진행할 수 있다.

[실행 비용](runs/execution_cost_audit.json) · [Cap 원본/재실행 감사](runs/cap_extension_audit.json) · [환경 버전](runs/environment_20261010.json) · [작업로그](작업로그.md)

## 해석 한계와 다음 연구

Image geometry proxy, renderer 차이, physical footprint/outdoor semantics, absolute EMF units 및 geographic/source-target overlap은 아직 해결되지 않았다. 현재 결과는 이 조건에서의 normalized-image empirical evidence다. Data-efficiency는 네 실제 예산 점으로 보고하며 정확한 데이터 절감률이나 노출량 정확도를 추정하지 않는다.

후속 우선순위는 독립 geographic/physical-scene provenance 검증, source budget×seed fully crossed validation 확대, scalar-only conditioning control이다. 추가 test 결과를 보고 설정을 고르면 새로운 독립 평가가 필요하므로 현재 held-out 결과와 구분해야 한다.
