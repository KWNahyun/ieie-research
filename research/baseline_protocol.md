# Image geometry proxy를 사용하는 초기 baseline protocol

2026-10-08. 사용자는 target의 원본 building vector 및 rendering 코드가 존재하지 않는다고 확인했다. 기존 PNG/배열만으로 재현 가능한 empirical normalized-image 실험을 진행한다.

## 입력과 평가 범위

Target B는 input PNG의 OpenCV B/G가 모두 정확히 0인 pixel의 indicator다. Source B는 binary building PNG의 nonzero indicator다. Target의 이 규칙은 이미지에서 관찰되는 black geometry를 추출하는 운영상 정의이며 physical footprint의 복원을 주장하지 않는다. 두 renderer의 경계 및 표현 차이는 domain gap에 포함된다.

공통 입력은 `[B_proxy,T]`의 두 채널이다. Target T는 original double-precision coordinate의 `column=floor(x),row=floor(y)`에 한 pixel을 표시한다. Float32로 numeric을 변환하기 전에 좌표를 rasterize한다. Source T는 감사한 `column=x,row=255-y`를 사용한다. 범위 밖 좌표는 clipping 대신 오류로 처리한다.

Target encoded B/G 자체는 loader가 별도로 보존하지만 이번 A/B model에는 입력하지 않는다. Building mask와 beam 역시 모델 입력에 넣지 않는다. `include_target=False`에서도 모든 model input을 생성할 수 있다.

Primary loss는 전체 원본 500×500 pixel의 MAE다. Primary validation metric은 샘플별 MAE/RMSE를 계산하고 image-geometry-group별 평균 후 group macro average다. Label은 원래의 PNG/255를 사용한다. 실험명은 normalized-image empirical pilot이며 outdoor-only error 또는 absolute EMF error라고 부르지 않는다. Geographical scene와 tile overlap 미확인도 한계로 남긴다.

## 모델과 conditioning

Fully convolutional U-Net은 세 번 pooling하는 encoder, skip decoder와 sigmoid head로 구성한다. Input은 필요한 8의 배수로 zero pad하고 prediction은 원래 H/W로 crop한다. 256×256 source와 500×500 target을 resize하지 않는다.

A는 공간 입력만 사용한다. B는 height/downtilt/power와 위치 x/y, azimuth sin/cos의 7 feature 및 horizontal/vertical 360-angle pattern을 사용한다. Pattern encoder는 각도별 정보를 보존하는 36-bin pooling과 projection을 사용하고 scalar MLP와 결합해 bottleneck FiLM을 만든다. FiLM output은 0으로 초기화한다.

같은 seed의 A/B는 encoder와 decoder 및 output head가 exact 동일한 값으로 초기화된다. B의 conditioning parameter는 별도다. 최종 transfer 실험에서는 source encoder만 가져오고 target decoder/head를 새로 초기화해야 한다.

## Split과 normalization

기존 seed 42의 60/12/14 image group split을 유지한다. Train/validation/test는 모두 base-only다. 저장된 horizontal/vertical flip은 비대칭 local pattern을 그대로 반복하므로 사용하지 않는다.

Budget은 train group의 10/25/50/100%로 정의하고 각 seed의 nested subset을 사용한다. 매 run의 normalization은 해당 budget에 포함된 train base에서만 fit한다. 같은 seed와 budget의 A/B는 같은 subset과 normalization을 사용한다. Validation 또는 test를 포함한 statistics는 loader가 거부한다.

Building과 label scaling은 PNG/255로 고정한다. Condition feature는 train mean/std, pattern은 channel별 train mean/std를 사용한다. 상수 feature의 std는 1로 둔다. Label statistics는 fit하지 않는다.

## 실행과 산출물

데이터와 모델 검증:

```bash
python IEIE/research/test_pipeline.py
```

Engineering smoke:

```bash
python IEIE/research/train_baseline.py --model A --smoke --base-channels 4 --steps 8 --threads 2
python IEIE/research/train_baseline.py --model B --smoke --base-channels 4 --steps 8 --threads 2
```

Full scratch run의 예시 명령은 다음과 같다. 이는 예약하거나 실행한 작업이 아니며 epochs, LR 탐색 및 compute protocol을 본 pilot 전에 고정한다.

```bash
python IEIE/research/train_baseline.py --model A --budget 25 --seed 0 --base-channels 16 --epochs 100 --device cpu
python IEIE/research/train_baseline.py --model B --budget 25 --seed 0 --base-channels 16 --epochs 100 --device cpu
```

Trainer는 test sample을 읽지 않는다. Validation으로 checkpoint를 선택하며 config, exact train IDs, train-only normalization, code/data-contract hash, parameter count, epoch/update history, runtime와 best checkpoint를 run 폴더에 저장한다. 기존 run은 overwrite하지 않는다. 시작/완료를 작업로그.md에 추가한다.

Smoke는 2개 train/2개 validation sample과 8 optimizer update의 engineering 검사다. Finite loss/gradient, 데이터 전달, checkpoint와 loss 변화만 확인한다. 이 수치로 A/B 정확도나 conditioning 효과를 비교하지 않는다.
