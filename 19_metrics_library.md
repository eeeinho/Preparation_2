# 19. 평가지표 구현 + 지표별 최적화 전략 프롬프트
목적: 지표를 코드로 정확히 재현해야 CV를 믿을 수 있다. 지표별 공략법을 사전 장착.

===== 프롬프트 시작 =====

너는 평가지표 최적화 전문가다. metrics.py 모듈을 작성하라. 대회 지표가 무엇이든 즉시 대응할 수 있도록 아래 전부를 구현하고, 각 지표의 최적화 전략을 docstring에 명시하라.

[구현 지표]
1. 분류: accuracy, macro/micro/weighted F1, balanced accuracy, MCC,
   multiclass logloss, macro AUC(OvR), top-k accuracy, Cohen kappa, QWK(순서형)
2. 이진/탐지: F1@threshold, PR-AUC, ROC-AUC, FPR@TPR95
3. 회귀: RMSE, MAE, SMAPE, R2, RMSLE
4. 신호 복원: SI-SNR, SNR improvement, log-spectral distance
5. 이벤트 탐지(SED): segment-based F1, event-based F1 (onset tolerance 파라미터)
6. 공통 유틸: 세그먼트 확률 -> 녹음 단위 집계(mean/max/trimmed/rank-mean) 후 지표 계산

[지표별 최적화 전략 — docstring + 전략표 출력 함수]
- macro-F1: 소수 클래스가 지배. 클래스별 threshold 개별 최적화(이진 분해),
  argmax 대신 클래스별 보정 계수 곱 후 argmax 탐색. 소수 클래스 오버샘플/가중.
- accuracy: 다수 클래스 우선, calibration 영향 적음, argmax 그대로.
- logloss: 캘리브레이션이 전부. temperature scaling, 확률 클리핑(1e-15),
  앙상블은 기하평균도 비교.
- AUC: threshold 무관, 순위만 중요. rank-mean 앙상블이 단순 평균보다 나을 수 있음.
- QWK: 순서형 — 회귀로 풀고 threshold 경계 최적화(OptimizedRounder 구현 포함).
- F1(이진): OOF 기반 threshold 그리드(0.05~0.95) + 미세 탐색, PR 곡선 출력.
- SI-SNR: 시간영역 손실로 직접 학습. 스케일 불변 주의.

[추가 구현]
1. optimize_argmax_weights(oof, y): 클래스별 확률 보정 계수를 Nelder-Mead로 탐색해
   macro-F1/QWK 등 비미분 지표를 직접 최대화 (multiclass의 threshold 대체 기법)
2. OptimizedRounder (순서형 경계 최적화)
3. aggregate_and_score(oof_seg, groups, y_group, metric, methods=[...]) 비교 표
4. 단위 테스트: sklearn 결과와 일치 검증 (지표 구현 버그는 대회 전체를 망침)

===== 프롬프트 끝 =====

## 사용 규칙
- 대회 지표 공개 즉시: metrics.py에서 해당 지표 + 전략표 확인 -> 전 파이프라인의
  early stopping/모델 선택 기준을 그 지표(녹음 단위)로 통일
- 지표 재현 검증: 가능하면 첫 제출로 public 점수와 자체 계산 일치 확인
