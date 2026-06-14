# 05. LightGBM/CatBoost 베이스라인 프롬프트
목적: 첫 제출(시작 3시간 내). 빠르고 디버깅 쉬운 백업 모델 + feature importance 확보.

===== 프롬프트 시작 =====

너는 Kaggle Grandmaster다. 04번에서 추출한 수치 피처(features.parquet)와 공용 fold(folds.csv)로 LightGBM 베이스라인을 작성하라. 목적은 (1) 파이프라인 끝까지 검증, (2) 첫 제출, (3) 누수 점검(랜덤 vs 그룹 fold 점수 비교)이다.

[입력]
- 평가지표: {예: Macro-F1 / Accuracy / AUC}
- 클래스 수/분포: {01번 결과}
- 제출 형식: {sample_submission 컬럼}

[요구사항]
1. folds.csv 기반 5-fold OOF 학습 (StratifiedGroupKFold 배정 그대로 사용)
2. LightGBM multiclass (objective=multiclass, metric=multi_logloss),
   기본 파라미터: n_estimators=3000, lr=0.05, early_stopping=200,
   num_leaves=63, feature_fraction=0.8, bagging 0.8/1, seed=42
3. 클래스 불균형 시 class_weight 또는 is_unbalance 적용 — 둘 다 실험해 OOF로 선택
4. 점수 보고 (반드시 둘 다):
   a. 세그먼트 단위 OOF {평가지표}
   b. 녹음(group) 단위 집계(확률 평균) OOF {평가지표} <- 의사결정 기준
5. CatBoost 동일 구조로 추가 학습, 두 모델 확률 평균 앙상블 점수까지 보고
6. feature importance (gain) 상위 30개 출력 + 누수 의심 점검:
   duration/sr 계열 피처가 1위권이면 경고 출력
7. Optuna 튜닝 함수 포함 (n_trials=50, 녹음 단위 OOF 점수를 objective로) — 단,
   기본 파라미터 제출을 먼저 만들고 튜닝은 백그라운드 실행 가능하게 분리
8. test 추론 -> 제출 파일 생성: submission_lgbm_cv{score:.4f}_{timestamp}.csv
   + OOF 확률(oof_lgbm.npy)과 test 확률(test_lgbm.npy) 저장 — 이후 앙상블용
9. 실험 로그 json append (exp_id, model, cv_seg, cv_group, params, time)

[제약]
- 완전한 실행 가능 코드, placeholder 금지
- 분류 threshold가 필요한 지표(F1 이진 등)면 OOF 기반 threshold 탐색 포함 (0.5 고정 금지)

===== 프롬프트 끝 =====

## 체크리스트
- [ ] 세그먼트 점수와 녹음 단위 점수 갭을 확인했는가 (갭 크면 누수/집계 이슈)
- [ ] OOF/test 확률을 저장했는가 (앙상블 필수 재료)
- [ ] 첫 제출을 07번 검증기로 통과시켰는가
