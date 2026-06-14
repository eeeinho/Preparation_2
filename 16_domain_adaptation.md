# 16. 도메인 적응 프롬프트 (train/test 분포 이동 대응)
목적: Adversarial Validation AUC > 0.7일 때 발동하는 대응 카드 3종.
수중 음향은 해역/계절/수온/녹음 장비 차이로 도메인 갭이 흔하다.

===== 프롬프트 시작 =====

너는 도메인 적응(DA) 전문가다. 수중 음향 분류에서 train/test 분포 이동이 확인되었다 (Adversarial Validation AUC = {AUC값}). 아래 3단계 대응을 비용 낮은 순서로 구현하라.

[입력]
- 기존 베스트 모델: {06/08번 모델}, folds.csv, OOF/test 확률 보유
- Adversarial Validation에서 분포 이탈이 큰 피처/특성: {결과 붙여넣기}

[레벨 1 — 저비용 (30분, 항상 먼저)]
1. 입력 정규화 점검: per-sample 표준화가 적용되어 있는지 (녹음 게인 차이 제거 효과)
2. 분포 이탈 피처 제거 (LGBM 계열): adversarial importance 상위 피처 drop 후 재학습 비교
3. test 유사도 가중 학습: adversarial 모델의 P(test|x)를 train 샘플 가중치로 사용
4. (CNN) test-time BN 통계 재추정: BN layer를 test 배치 통계로 갱신 후 추론
   (학습 불필요, momentum=None forward pass만으로 구현)

[레벨 2 — pseudo-labeling (1.5시간)]
1. 현 앙상블의 test 예측 중 신뢰도 상위(예: max prob > 0.95, 클래스 균형 유지) 선별
2. train fold에 추가해 재학습 (valid에는 절대 추가 금지, 라운드 1회만 — 확증 편향 방지)
3. 추가 전/후 녹음 단위 OOF + (가능하면) public 점수 비교로 채택 결정
4. 안전장치: pseudo 샘플 비율 상한 30%, 클래스당 최소/최대 개수 제한

[레벨 3 — DANN (2~3시간, 갭이 클 때만)]
1. 기존 CNN 백본에 GRL(Gradient Reversal Layer) + 도메인 판별기(train vs test) 부착
   - GRL 구현: forward identity, backward에 -lambda 곱 (torch.autograd.Function)
   - lambda 스케줄: 0 -> 1 점진 증가 (2/(1+exp(-10p))-1)
2. 손실: 분류 CE(train만) + 도메인 BCE(train+test, 레이블 불필요)
3. 평가: DANN 적용 전/후 녹음 단위 OOF + adversarial AUC 재측정
   (AUC가 0.5에 가까워지면 도메인 정렬 성공 신호)
4. 주의: OOF가 하락하면 즉시 롤백 — DA는 CV가 개선을 보장하지 못하는 영역이므로
   제출 기회로 직접 검증하고, 최종 제출 2개 중 1개에만 반영(리스크 분리)

[공통 제약]
- 각 레벨 종료 시 비교 표 (적용 전/후 OOF, adversarial AUC) 출력
- 기존 체크포인트/산출물 덮어쓰기 금지 (별도 이름: *_da.npy)

===== 프롬프트 끝 =====

## 체크리스트
- [ ] 레벨 1을 건너뛰고 DANN부터 가지 않았는가
- [ ] pseudo-labeling을 valid에 섞지 않았는가
- [ ] DA 결과를 최종 제출 1개에만 반영해 리스크를 분리했는가
