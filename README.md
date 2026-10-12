# ieie-research

ddd

# IEIE research

정규화 전자기장(EMF) 영상 예측을 위한 안테나 조건부 인코더 전이 연구입니다. 소스 전파 지도에서 사전학습한 인코더와 동일 구조의 무작위 초기화 모델을 형상 그룹 단위로 비교합니다.

## 논문

- [한글 PDF](research/manuscript/manuscript_ko_review.pdf) · [한글 Word](research/manuscript/manuscript_ko.docx) · [한글 본문](research/manuscript/manuscript_ko.md)
- [English PDF](research/manuscript/manuscript_review.pdf) · [English Word](research/manuscript/manuscript.docx) · [English manuscript](research/manuscript/manuscript.md)
- [영문 LaTeX](research/manuscript/manuscript.tex) · [한글 LaTeX](research/manuscript/manuscript_ko.tex)

PDF는 검토용이며 저자·소속 및 최종 투고 양식은 미정입니다. LaTeX 원본은 아직 컴파일하지 않았습니다.

## 결과 및 검증

[IEIE와 EMF-DINOv2 비교 및 논문 통합 보고서](research/reports/experiment_comparison_integration_ko.md) · [보고서 PDF](research/reports/experiment_comparison_integration_ko.pdf) · [보고서 Word](research/reports/experiment_comparison_integration_ko.docx)

- 조건부 CNN 인코더 전이는 네 학습 규모 모두에서 평균 테스트 MAE를 2.20–3.71% 줄였습니다.
- 수정한 조건부 HRFormer에서는 25%/100% 조건의 MAE가 6.54%/8.50% 줄었습니다. 이 추가 비교는 타깃 시드 3개와 소스 시드 하나를 사용했습니다.
- 패턴 정보는 25·50·100%에서 평균 MAE를 낮췄으나 10%에서는 불리했습니다.
- 추가 비교 152개와 표본 지표 19,152개에 대한 [최종 검증](research/runs/followup_v1/final_audit.json)이 통과했습니다.

[본 실험 결과](research/runs/main_v1/results.md), [추가 실험 결과](research/runs/followup_v1/results.md), [소스 민감도](research/runs/source_sensitivity_v1/results.md), [공간 입력 대조군](research/runs/spatial_ablation_v1/results.md).

정답은 정규화 영상입니다. 물리 단위 복원, 독립적인 실외 영역 및 소스·타깃의 지리적 독립성은 확인되지 않았습니다. 추가 비교는 본 실험 테스트 결과를 확인한 후 설계한 보조적·탐색적 평가입니다. 동일 오차 보간에 따른 44.87% 추정은 실제 확인한 데이터 절감률이 아닙니다.

## 구성

`research/`에는 데이터 점검, 학습, 평가, 통계 분석 및 논문 생성 코드가 있습니다. `research/splits/`와 목록 CSV는 고정된 분할·표본 ID를 기록합니다. `research/runs/`에는 체크포인트를 제외한 실행 설정, 선택 이력, 정규화 통계, 지표, 결과 및 그림이 있습니다. 원본 데이터와 대용량 체크포인트, 로컬 인증 정보 및 설치된 패키지 사본은 포함하지 않습니다.

## 실행 조건

Python, PyTorch/CUDA, NumPy, OpenCV, matplotlib, scikit-image, timm, einops가 필요합니다. 문서 생성에는 python-docx, reportlab, fonttools가 추가로 필요합니다. 기록된 환경은 [환경 기록](research/runs/environment_20261010.json)을 참조하십시오. GPU와 원본 데이터가 없는 환경에서 전체 학습을 재현할 수는 없습니다.

데이터 접근 코드는 당시의 로컬 경로를 사용합니다. 외부 환경에서는 `research/data_pipeline.py`의 `TARGET_ROOT`와 목록·계약에 기록된 원본 데이터 경로를 실제 자료 위치에 맞춰야 합니다. 일부 실행·검증 코드는 저장된 절대 경로와 체크포인트 해시를 참조하므로 그대로 실행하기보다 프로토콜과 인자를 먼저 확인하십시오. 이 저장소는 완성된 실험의 기록이며 자동 설치 패키지가 아닙니다.

```bash
python research/run_required_followup.py --dry-run
```

학습을 다시 시작하기 전에 [본 실험 프로토콜](research/main_protocol.md) 및 [추가 실험 프로토콜](research/followup_protocol.md)을 읽으십시오. 기존 실행 기록을 덮어쓰지 않고 별도의 출력 폴더를 사용해야 합니다.

## 외부 코드

`research/vendor/hrformer/`는 MIT 라이선스의 HRFormer 계열 백본이며 [라이선스](research/vendor/hrformer/LICENSE)와 [출처 기록](research/vendor/hrformer/provenance.json)을 포함합니다. 저장소 전체에 별도의 라이선스는 아직 지정하지 않았습니다.
