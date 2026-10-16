# 동료 DINO 조건에 맞춘 CNN 재실험

사용자 승인: 기존 CNN의 무작위 초기화와 인코더 전이를 동료 실험 조건에서 재학습한다. 소스 CNN L1 시드142 1회, 타깃 base/encoder×시드42/43/44=6회. 이전 모델·결과는 보존한다. DINO는 GPU1에서 계속 실행하고 CNN은 GPU2를 사용한다.

데이터: dino_replication_v1의 동일한 target/source manifest를 바이트 그대로 복사한다. 타깃68/9/9그룹,609/78/81표본; 소스550/50/100지도와700×80표본. 동일원본/Tx/마스크/train-only Q/H/V정규화와S=[occupancy,Gaussian,log distance],Q5,P=[H,V,n2i]를 같은 로더로 받는다. 전체 학습 그룹100% 사용. target group budget의 이전10/25/50%는 이번 비교에 포함하지 않는다.

모델: 기존 U-Net의16/32/64/128 인코더와 skip decoder를 유지한다. 입력을2→3채널로 조정하고, 기존1D패턴 대신3채널 패턴 영상의zero-init1×1 투영을 첫 인코더 특징에 더한다. 스칼라FiLM은7→5개 입력으로 조정하며 bottleneck zero-init을 유지한다. sigmoid를 제거해 동료처럼 선형 출력/no clamp를 사용한다. DINO의토큰embedding/LayerNorm/LoRA와 CNN의GroupNorm/전체미세조정은 계열별 구조 차이로 남는다. 이 실험은 동일 정보·데이터·손실·평가 조건이며 백본만의 인과 비교는 아니다. 기존 CNN 체크포인트와는 입력 구조가 달라 호환되지 않는다.

소스는CNN 무작위 초기화/전체미세조정,타깃은CNN base무작위초기화 또는 새소스encoder만 복사/전체미세조정. 타깃decoder/head/FiLM/side 초기값은 대응 seed별로 같고 source전이가 이들을 바꾸지 않았음을 검사한다. 공개 DINOv2는 일반영상 사전학습에서 출발하므로 CNN과 초기학습이력도 다르다.

양쪽 공통설정:maskedL1,FP32,AdamW lr1e-4/wd0,clip5,batch8/accum1,cosine minlr1e-5,증강/EMA/clamp off. source max50/warmup5/patience15/min_epoch10;target max200/warmup10/patience50/min_epoch30. CNN 고유구조에DINO의drop_path를 억지로 적용하지 않으며 해당차이를 기록한다. 동일 검증global maskedMAE 선택,MAE/RMSE/globalPSNR/image maskedSSIM/groupmacro를동일함수로계산한다.

target6선택체크포인트/설정/데이터목록해시를고정한뒤test81표본9건물을평가한다. 기존test는알려져있으므로탐색적검증이다. source1시드에조건부인타깃반복이고지리/물리단위보장을하지않는다. 모든조건·불리한결과·상한도달·실패를보존한다. 초기화순서·source복사범위·패턴/FiLMgradient·native500·재로드검증을확인하고groupmacro쌍체bootstrap과pixel-global paired seed차이를구분한다.
