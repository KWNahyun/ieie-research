# Scratch A/B 구현 및 engineering smoke 결과

2026-10-08. [Baseline protocol](baseline_protocol.md)에 따라 dataset, 모델, 학습 경로를 구현했다. Full pilot과 source pretraining은 아직 수행하지 않았다.

후속 상태: 이후 source pretraining과 24개 target fit을 완료했다. 위 문장은 smoke 당시의 상태이며 현재 결과는 [pilot_v1_results.md](pilot_v1_results.md)를 참고한다.

## 완료한 구현

- data_pipeline.py: read-only memory mapping, base-only group budget, label-free image proxy/Tx 입력, train-only condition/pattern normalization, 명시적인 whole-raster/legacy label-positive 평가 구분.
- baseline_model.py: 원본 해상도를 유지하는 U-Net A와 scalar/pattern bottleneck FiLM을 추가한 B. 두 모델의 같은 seed encoder/decoder/head 초기값은 동일하다.
- train_baseline.py: config와 code/contract hash, exact sample IDs, normalization, epoch/update history, validation-selected checkpoint와 runtime 기록. Test sample은 학습이나 validation에서 읽지 않는다.
- geometry_proxy_manifest.csv와 geometry_proxy/*.png: 86개 building 입력 PNG에서 label을 읽지 않고 추출한 black-pixel geometry proxy와 raw file SHA256.

Original geometry나 rendering code가 없으므로 target B는 `B-channel==0 AND G-channel==0`으로 정의한 image proxy다. 이는 물리 footprint 복원 결과가 아니다. Main metric은 전체 raster의 normalized-image error이며 geographic scene 일반화 또는 outdoor-only 정확도로 표현하지 않는다.

## 검증 결과

test_pipeline.py의 6개 테스트가 통과했다. Label을 읽지 않아도 model input이 동일하며, read-only mmap, train-only statistics, 잘못된 subset/normalization 거부, 좌표 floor/axis/범위, base-only 선택, source sample, A/B 동일 초기화, 500/256 output 크기와 finite gradient를 검사했다.

Smoke는 같은 seed 0, 25% train-group budget의 normalization을 사용하고 실제 train 2개와 validation 2개에 대해 수행했다. Base channel 수는 4, batch size는 1, CPU threads는 2, optimizer는 AdamW, LR는 3e-4, 총 update는 8이다. Test image/label은 읽지 않았다.

| Model | 첫 epoch train MAE | 마지막 epoch train MAE | Updates | CPU runtime |
| --- | --- | --- | --- | --- |
| A | 0.237903 | 0.185847 | 8 | 2.59 s |
| B | 0.237881 | 0.185648 | 8 | 3.04 s |

두 실행의 loss와 gradient가 finite이고 train loss가 감소했다. Best checkpoint를 다시 읽어 strict load 후 기록된 소수 validation sample metric을 재현했다. B의 pattern encoder parameter도 초기값에서 변했다. 이 확인은 데이터 전달, optimization 및 저장/복원 경로가 작동한다는 증거다.

위의 MAE 차이는 conditioning 효과나 최종 정확도 차이의 증거가 아니다. 소수 sample, 8 update와 같은 geometry에 속할 수 있는 validation 2개는 연구 성능 비교에 충분하지 않다. Source encoder를 아직 학습·전이하지 않았으므로 transfer 결과도 없다.

Run 산출물:

- [A summary](runs/smoke_A_budget25_seed0/summary.json), [A history](runs/smoke_A_budget25_seed0/history.json)
- [B summary](runs/smoke_B_budget25_seed0/summary.json), [B history](runs/smoke_B_budget25_seed0/history.json)
- [Checkpoint verification](baseline_smoke_verification.json)

각 run 폴더에 config.json, normalization.json, best.pt와 실제 smoke ID를 명시한 run_metadata_addendum.json도 있다. Config의 train_base_ids는 statistics fit에 사용한 전체 budget ID이며 addendum은 실제 반복 학습 2개 ID를 구분한다. 앞으로의 run은 config 자체에 actual_train_base_ids와 actual_validation_base_ids를 기록한다.

## 다음 연구 실행

다음은 정식 25/100% 예산의 scratch A/B pilot과 DPM source pretraining이다. 그 전에 본 실행의 base channel 수, 최대 update/epoch 수, early stopping과 동일 validation LR 탐색 예산을 고정한다. 현재 실행 환경의 PyTorch는 CUDA unavailable이며 GPU가 보이는 실행 환경 또는 실제 CPU 처리량에 맞춘 일정을 사용해야 한다.

Source checkpoint는 encoder만 전이하고 target decoder/head와 conditioning은 새로 초기화한다. B/D는 같은 target 입력과 구조, 같은 subset을 사용한다. Full pilot은 geometry-proxy normalized-image 연구로 진행하며 residual rendering domain gap과 geographic overlap 미확인은 함께 보고한다.
