# 03. 누수 방지 CV 분할 프롬프트
목적: 모든 모델이 공유할 fold 배정을 확정. UATR 최대 함정(세그먼트 누수) 차단.

===== 프롬프트 시작 =====

너는 Kaggle Grandmaster다. 수중 음향 분류 대회에서 가장 흔한 실패는 같은 원본 녹음에서 잘린 클립이 train/valid에 동시에 들어가 CV 점수가 가짜로 높아지는 것이다(DeepShip 등 공개 벤치마크에서 반복 확인된 문제). 누수 없는 CV 분할 코드를 작성하라.

[입력]
- 파일명 패턴 예시: {예: classA_ship03_rec02_seg007.wav 등 실제 패턴 5~10개}
- 메타데이터 컬럼: {있으면 붙여넣기, 없으면 "없음"}
- 클래스 분포: {01번 결과}

[요구사항]
1. 그룹 키 추출 함수: 파일명에서 정규식으로 녹음/선박 ID를 추출.
   ID가 명시적으로 없으면 아래 휴리스틱으로 같은 세션을 추정하는 함수를 함께 작성:
   a. 파일명 prefix + 연속 번호 패턴
   b. 파일 길이/sr이 동일하고 생성 시각이 인접한 그룹
   c. (강력) 각 파일의 저주파 평균 스펙트럼 벡터 간 코사인 유사도 > 0.98 클러스터링
      (같은 녹음의 클립은 배경 잡음 지문이 거의 동일함)
2. StratifiedGroupKFold(n_splits=5, seed=42)로 fold 배정,
   결과를 folds.csv (file, label, group, fold)로 저장 — 이후 모든 모델이 이 파일만 사용
3. 검증 출력:
   - fold별 클래스 분포 표 (불균형 fold 경고)
   - 같은 group이 두 fold에 걸치지 않음을 assert로 증명
   - 그룹 수가 너무 적어(예: 클래스당 < n_splits) 그룹 분할이 불가능한 클래스 경고
4. 비교 실험 장치: 동일 데이터로 (a) 세그먼트 랜덤 KFold (b) 그룹 KFold
   두 가지 fold를 모두 저장하고, 이후 베이스라인에서 두 점수 차이를 출력하게 하라.
   차이가 크면 누수가 실재한다는 증거다.
5. Adversarial Validation 코드 포함: 가벼운 통계 피처로 train vs test 분류 AUC 출력,
   AUC > 0.7이면 "분포 이동 — 도메인 적응 검토" 경고.
6. 시계열성 데이터(녹음 시각 존재)라면 TimeSeriesSplit 버전도 함께 제공.

[제약]
- sklearn StratifiedGroupKFold / GroupKFold / TimeSeriesSplit 사용
- folds.csv가 단일 진실 공급원(single source of truth)임을 코드 주석에 명시

===== 프롬프트 끝 =====

## 체크리스트
- [ ] folds.csv를 팀 전체가 공유했는가 (모델별로 fold 다르면 앙상블 OOF 무효)
- [ ] 랜덤 split 대비 그룹 split 점수 차이를 확인했는가
- [ ] Adversarial AUC를 확인했는가
