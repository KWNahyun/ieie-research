# 전체 데이터 감사 보고서

작성일: 2026-10-08. 연구 실행계획의 첫 데이터 검증 단계 결과다. 원본 데이터는 변경하지 않았고 학습은 실행하지 않았다. 검사 결과는 [data_contract.json](data_contract.json)에 계약 형태로 정리했다.

후속 상태: 사용자가 원본 building 파일과 PNG 생성 코드의 부재를 확인했다. 따라서 이후에는 [baseline_protocol.md](baseline_protocol.md)의 **label-independent image geometry proxy 및 whole-raster empirical pilot** 범위를 적용한다. 아래 미완료 physical geometry 조건은 실제 outdoor/footprint 주장에 대한 한계이며 이 명시적 proxy pilot을 막는 조건으로 두지 않는다. Scratch A/B의 engineering smoke 결과는 baseline_smoke_report.md에 별도로 기록한다.

## 판단

Target 원본과 저장된 augmentation의 배열 대응, source Tx 좌표, candidate manifest와 그룹 분할은 검증됐다. 다만 **독립 building occupancy와 outdoor mask, 물리적 pattern augmentation, 지리 중복 검증은 아직 미완료**다. 현재 파일을 곧바로 학습에 연결하면 label 기반 mask를 공간 입력으로 오인하거나 비대칭 pattern의 잘못된 flip을 학습할 수 있다.

첫 학습 protocol의 안전한 기본값은 base-only, beam 제외, label-derived mask의 입력 사용 금지다. 이를 manifest에 명시했다. Primary outdoor 평가를 시작하려면 독립 footprint 또는 검증된 geometry encoding과 nodata 의미를 먼저 확보해야 한다.

## Target 원본과 전체 배열

읽은 원본은 `/home/ailab/Desktop/data/emf_data`의 configuration spreadsheet, building PNG, field-strength PNG와 pattern 파일이다. Spreadsheet는 256 configuration row와 각 row의 3개 안테나 설정을 가진다.

| 검사 | 범위 | 결과 |
| --- | --- | --- |
| 8개 preprocessing 배열 finite/range | 모든 3072 entry | nonfinite entry 0 |
| Building B/G PNG 대응 | 768 base | exact 일치 768 |
| Label grayscale PNG /255 대응 | 768 base | exact 일치 768 |
| Scalar spreadsheet 대응 | 768 base | atol 1e-10에서 일치 768 |
| 수평·수직 pattern 파일 대응 | 각각 768 base | atol 1e-10에서 모두 일치 |
| Building 두 채널, y, mask, 3채널 image flip | 각각 2304 변형 | exact 일치 |
| Numeric 좌표·azimuth flip | 2304 변형 | atol 1e-10에서 모두 일치 |
| 수평·수직 pattern 반복 | 각각 2304 변형 | base 값과 exact 일치 |
| Beam과 base의 단순 영상 flip | 2304 변형 | 2300 불일치 |

Base index는 0–767, 안테나 번호는 `base//256+1`, spreadsheet row는 `base%256`이다. 저장된 변형은 `768+3*base+j`이며 j=0 horizontal, j=1 vertical, j=2 combined다. 이 관계는 전체 배열에서 검사됐다.

Building ID는 86개다. 두 채널을 8-bit로 환산한 뒤 회전·반사를 포함한 D4 exact equality로 묶어도 86개이며 building ID와의 union도 86개다. 기존 geometry_groups.json의 172개 회전 그룹과 다른 결과는 **반사까지 묶었기 때문**이다. 86은 원본 파일 identity/영상 동등 그룹 수이지 검증된 독립 물리 환경 수가 아니다.

전체 범위는 [augmentation_audit.json](augmentation_audit.json)에 있다. 초기 세 샘플의 낮은 building 채널 범위를 전체 범위로 사용하면 안 된다. 전체 x_image와 3채널 image의 최대는 약 0.988235다.

## Scalar, 좌표와 pattern

검증한 numeric 순서는 x, y, height, downtilt, power, azimuth다. Spreadsheet coordinate 문자열의 두 번째 값을 x로 사용하고 첫 번째 값은 `500-value`로 변환해 y로 사용한다. 높이는 세 번째 값이다. 근거는 `emf_project/ViT/utils/data.py`의 load_data와 전체 spreadsheet 대응이다. Power와 height의 물리 단위 및 GIS 원점은 별도 근거가 필요하다.

Tx raster의 후보 convention은 `column=floor(x), row=floor(y)`다. Loader 주석은 azimuth를 시계방향으로 설명하며 angle 변환은 image에서 north=0, east=90인 방식을 사용한다. 연속 좌표 반사 `500-coordinate`와 pixel index 반사 `499-index`는 정수 경계에서 다르므로 향후 변환 함수는 경계 처리를 명시해야 한다.

Pattern 원본 파일은 각도를 0–359도로 명시한다. Parser는 행 순서를 보존하고 attenuation의 부호를 바꾼 뒤 gain을 더한다. 저장 배열은 `GAIN − attenuation + 40`과 전체 일치한다. 40을 빼면 파일에서 정의한 값으로 돌아가지만 spatial gain 지도나 절대 EMF 변환이 되는 것은 아니다. Local 기준 방향과 vertical angle의 물리 convention은 아직 별도 검증이 필요하다.

모든 768개 horizontal pattern은 `theta → −theta` 반사에 대해 비대칭이다. 영상과 azimuth를 반사하면서 같은 local pattern을 반복한 기존 augmentation은 물리적으로 검증된 flip으로 간주할 수 없다. 초기 학습은 base-only로 두고, 안테나 local frame에서 반사할 pattern index와 좌표 변환을 함께 검증한 뒤 augmentation을 추가한다.

## Mask와 beam의 중요한 차이

**building_mask는 모든 3072 entry에서 y>0과 exact 일치한다.** 동일 building ID의 설정 사이에서도 85/86개 building의 mask가 달라진다. 생성 코드 전체를 발견한 것은 아니므로 저장 당시의 생성 경로를 확정했다고 주장하지는 않지만, 독립 geometry-only mask로 사용할 근거는 없다.

기존 notebook는 mask=0을 건물, mask=1을 배경으로 설명한다. 그러나 단순 building channel zero/nonzero와는 모든 base에서 불일치한다. y의 0에 building, nodata, 낮은 값의 양자화가 포함되는지는 아직 미확인이다. 이 mask는 모델 입력에 넣지 않는다. 기존 label-positive 평가를 재현할 경우 그렇게 명명하고 outdoor-only 평가와 구분한다.

Beam은 3072개 모두 legacy polygon 생성 코드를 재현했을 때 exact 일치한다. 따라서 flip 불일치의 원인은 base beam의 영상 반사가 아니라 변환 metadata로 beam을 다시 생성한 저장 방식이다. 이 생성 경로는 y를 사용하지 않는다.

다만 legacy 코드는 origin marker를 `tmp[int(x),int(y)]`에 넣고 polygon은 OpenCV의 x/y 좌표로 그린다. Marker의 row/column을 바로잡으면 1836개 beam entry가 바뀐다. Core pilot에서는 beam을 제외한다. 이를 사용할 때는 legacy 재현과 corrected beam을 버전으로 분리하고 모든 모델에 같은 입력을 제공한다.

검사 결과는 [conditioning_audit.json](conditioning_audit.json)에 있다. [Target coordinate overlay](figures/coordinate_candidates.png)는 index 0의 정상/교환 좌표를 비교한다. 시각 검사 한 장만으로 전체 물리 convention을 확정하지 않는다.

## Source DPM

로컬 데이터는 map ID 0–700의 701 map, map당 80 Tx, 총 56080 DPM pair다. 이 개수 차이는 701번째 로컬 map으로 설명된다. 공개 원본 배포와 map 700의 생성 provenance까지 증명한 것은 아니다.

모든 DPM pair의 gain header shape, 대응 binary building PNG, antenna PNG와 JSON 좌표를 검사했다. 모든 pair에서 Tx는 `column=x,row=255−y`에 exact 정합한다. 실패 pair는 0이다. Gain은 header와 파일 연결 검사이며 모든 gain pixel의 물리 의미를 검증한 결과는 아니다.

[Source alignment overlay](figures/source_alignment.png)는 map 0의 building polygon, binary raster와 Tx를 비교한다. 경계는 시각적으로 정합하지만 간단한 fillPoly raster와 5921 pixel이 달라 원본 rasterization의 완전 재현으로 주장하지 않는다. Nonzero building occupancy는 근거가 있는 후보이며 최종 계약은 rasterization/좌표 규칙을 명시해야 한다.

Source/target 원본 raster 크기는 256/500으로 다르므로 exact hash가 일치하지 않는다는 사실로 지리 중복 부재를 판단할 수 없다. Target 0을 64×64로 정규화한 보조 D4 비교의 최고 source IoU는 0.245였다. 한 샘플의 근사 비교이며 전체 topology나 겹치는 tile 검증을 대신하지 않는다.

## Manifest와 provisional split

Target split seed는 42다. Raw building identity와 두 채널 D4 equality의 union을 단위로 분리했다.

| Split | Group 수 | Base sample 수 |
| --- | --- | --- |
| Train | 60 | 534 |
| Validation | 12 | 108 |
| Test | 14 | 126 |

3072개 전체 관계를 [target_manifest.csv](target_manifest.csv)에 보존하지만 `use_for_training`은 train identity 534개에만 true다. Validation/test도 identity만 평가한다. 실제 physical scene ID는 unknown으로 표시했다. [Target split](splits/target_groups.json)은 tile overlap 검증 전의 provisional 결과다.

5개 seed의 중첩 10/25/50/100% subset을 [target_subsets.json](splits/target_subsets.json)에 생성했다. 백분율의 분모는 train group 수다. 불균등한 group 크기 때문에 실제 base sample 백분율과 다를 수 있으며 두 수를 함께 보고한다.

[source_manifest.csv](source_manifest.csv)는 모든 DPM triple을 map/Tx ID로 연결하고 [source split](splits/source_maps.json)은 map/D4 geometry를 분리한다. Physical provenance와 geographic overlap이 해결되기 전에는 candidate/provisional 상태를 유지한다.

`verify_audit.py`는 finite/provenance, beam 재생성, base의 4개 entry 관계, building/group split 분리, training flag, seed별 subset 중첩과 source map 분리를 검사하여 통과했다. 이는 미확인 물리 조건의 검증을 대신하지 않는다.

## 다음 작업과 완료 조건

1. Target PNG 생성 코드 또는 simulator footprint/vector 자료를 찾아 건물 encoding의 의미와 geometry-only occupancy를 확정한다. 현재 emf_data에는 PNG와 metadata만 확인됐다. Label이나 label-positive mask로 입력 occupancy를 만들지 않는다.
2. 독립 geometry mask와 nodata 정의를 확보한다. 확보할 수 없다면 outdoor-only 주장을 제거하고 전체 raster 평가 또는 명시적인 legacy label-positive 재현으로 연구 범위를 수정한다. 그 경우에도 input occupancy는 독립적으로 확정해야 한다.
3. Raw building ID의 지리 좌표, tile extent 또는 원본 simulator scene 정보를 찾아 겹치는 tile와 source-target overlap을 검사한다. 정보가 없으면 geometry-group 평가로 명명하고 unseen physical topology 주장 범위를 제한한다.
4. Base-only 입력/평가 계약이 확정되면 memory-mapped dataset과 scratch A/B를 구현한다. Flip 도입은 local pattern reflection convention을 별도 검증한 뒤 수행한다.

재실행 명령은 `python IEIE/research/audit_target.py`, `python IEIE/research/audit_source.py`, `python IEIE/research/audit_conditioning.py`, `python IEIE/research/verify_audit.py`다. Target 감사는 대용량 배열 전체를 읽는다. Scripts는 reports/manifests/splits와 작업로그를 갱신하고 원본에는 쓰지 않는다.
