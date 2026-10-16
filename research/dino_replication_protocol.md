# DINOv2 중심 논문 추가 검증 프로토콜

2026-10-13 사용자 승인 범위: DINOv2 기준, 소스 인코더 전이, 인코더·디코더 전이+LoRA 반복과 디코더 전이 후 전체 학습 대조. 기존 CNN/HRFormer 결과와 팀원 파일은 변경하지 않는다.

타깃 4조건×3시드(42,43,44)=12회. 타깃·소스 masked L1을 고정한다. 팀원의 n2i 입력, split_seed42, 건물68/9/9와 표본609/78/81, train-only 정규화, P=[H,V,n2i], DINOv2 ViT-B/14 reg4+DPT, 인코더 LoRA r8/alpha16을 재사용한다. 주 선택 지표는 검증 pixel-global masked MAE. 보조 지표는 RMSE/SSIM/PSNR 및 건물 macro MAE. 시그모이드/clamp/증강/EMA 미사용. 기존 테스트는 알려져 있으므로 supporting/exploratory 검증이다.

조건: base(공개 DINOv2+새 디코더 전체 학습), encoder(소스 인코더+새 디코더 전체 학습), decoder_lora(소스 인코더·디코더+디코더 base 동결/24Conv LoRA/출력층 학습), decoder_full(같은 소스 인코더·디코더+디코더 전체 학습). 인코더 적응·새 FiLM·패턴 side embedding 초기화는 대응한다. decoder_full은 decoder_lora의 디코더 재사용을 유지한 채 적응 정책을 비교한다. LoRA와 동결을 개별적으로 분리한 인과 효과는 주장하지 않는다.

팀원 소스 체크포인트를 찾지 못하므로 소스 L1을 시드142로 새로 학습한다. 700지도,550/50/100 분할; 공개 DINOv2 초기화+새 디코더, 전체 미세조정. 기존 소스 결과의 수치 재현을 미리 보장하지 않는다. 소스 반복은 1개이므로 타깃 반복은 이 소스에 조건부이다.

공통 타깃 설정: batch8/accum1,AdamW lr1e-4/wd0,minlr1e-5,FP32,clip5,drop/drop_path0.3,max200,warmup10,patience50,min_epoch30. 소스 batch8/max50/warmup5/patience15/min_epoch10. 조건별 검증 최소 MAE 체크포인트를 선택한다. 상한 도달은 별도 표시하고 자동으로 예산을 늘리지 않는다. 초기 테스트 평가 전에 모든 타깃 선택 체크포인트·설정·정규화 해시를 고정한다.

공개 사전학습 가중치는 로컬 캐시를 사용하고 SHA256을 기록한다. 데이터 원본 읽기 전용. 새 실행은 research/runs/dino_replication_v1 아래에 저장한다. 팀원 train_one_epoch/validate/스케줄러/데이터 로더/체크포인트 함수를 research 어댑터에서 재사용한다. 팀원 프로젝트 내부 구현이나 GitHub Issue는 변경하지 않는다.

검증: 동일 초기화·데이터 순서, 모든 조건의 입력/분할/정규화 일치, 전이 키·해시, 디코더 전체 학습 및 LoRA 학습 가능 파라미터, 패턴/FiLM gradient, native500 출력, strict 저장·재로드, 검증 로그/선택 체크포인트 일치. 합성 GPU 검증과 실제 학습을 구분한다. 실패·부분 실행은 보존하고 명시적 재개로 처리한다.

완료 후 12조건 전체를 테스트 평가하며 조건별 평균/시드 변동, 대응 건물 오차 변화와 조건부 그룹 부트스트랩을 보고한다. 결과를 보고 유리한 조건만 남기거나 테스트 최저 조건으로 대표를 변경하지 않는다. 논문은 결과 검증 후 업데이트한다. 이 작업은 기존 결과 공개를 위한 GitHub 푸시와 별개이며 데이터·가중치는 업로드하지 않는다.
