# 09. Mel+LOFAR 듀얼브랜치 + 앙상블 가중치 최적화 프롬프트
목적: 수상권 차별화 카드. 도메인 특화 다중 표현 융합(CAF-ViT 근사) + 최종 앙상블.

===== 프롬프트 시작 (A. 듀얼브랜치 모델) =====

너는 수중 음향 딥러닝 연구자다. CAF-ViT(LOFAR/Mel/Wavelet 다중 표현 -> Cross-Attention 융합) 아이디어를 28.5시간 해커톤 내 구현 가능한 형태로 단순화한 듀얼브랜치 CNN을 PyTorch로 작성하라.

[설계]
- 입력 1: log-Mel (128 x T) — 광대역 음색
- 입력 2: LOFAR-gram (0~1000Hz 고해상도, 배경 차감, 128 x T로 리사이즈) — 저주파 라인
- 브랜치: 각각 timm efficientnet_b0 (사전학습, 1ch 입력)
- 융합: 두 브랜치 GAP 임베딩 concat -> (옵션 플래그) 단순 concat vs
  cross-attention 1층 vs gated fusion(시그모이드 게이트) 비교 가능하게
- 헤드: LayerNorm -> Dropout 0.3 -> Linear(num_classes)

[요구사항]
1. LOFAR-gram 생성 함수 포함 (n_fft=4096, 0~1000Hz, median 배경 차감, 사전 캐싱)
2. folds.csv 5-fold, bf16, 06번과 동일 학습/평가/산출물 규격
   (oof_dual.npy / test_dual.npy / submission_dual_cv{score}_{ts}.csv)
3. ablation 자동 실행: Mel-only vs LOFAR-only vs Dual(3가지 융합) 의 fold0 점수 표 출력
   -> 최선 구성만 5-fold 완주
4. 두 입력의 시간축 정렬(같은 세그먼트, 같은 hop) 보장

===== 프롬프트 끝 (A) =====

===== 프롬프트 시작 (B. 최종 앙상블) =====

너는 Kaggle Grandmaster다. 지금까지 저장된 OOF/test 확률들(oof_lgbm, oof_cnn, oof_ast, oof_dual / test_*)로 최종 앙상블을 구성하라.

[요구사항]
1. 입력 검증: 모든 OOF가 같은 folds.csv 기반인지, 같은 행 순서/클래스 순서인지 assert
2. 모델 간 OOF 상관행렬 출력 (확률 평탄화 후 피어슨) — 0.95+ 쌍은 한쪽 제외 검토
3. 가중치 최적화: scipy minimize(Nelder-Mead)로 녹음 단위 집계 {평가지표}를 직접 최대화
   - 제약: 가중치 합=1, 음수 금지(클리핑) — 음수가 나오면 OOF 과적합 경고 출력
   - 초기값 균등, multi-start 5회로 국소해 회피
4. 비교 표 출력: 개별 모델 / 단순 평균 / 최적 가중 / 로지스틱 스태킹(OOF 로짓 -> LR)
   각각의 세그먼트 점수 + 녹음 단위 점수
5. 안정성 검사: fold-out 가중치 재추정(각 fold 제외 후 최적화) 시 가중치 분산 출력
   — 분산 크면 단순 평균 채택 권고
6. 최종 제출 2개 생성 전략:
   a. 안전: 단순 평균 또는 안정 가중
   b. 공격: 최적 가중 + (확률 temperature 보정 옵션)
   submission_ensemble_cv{score:.4f}_{ts}.csv 형식, 검증기(07번) 자동 호출
7. threshold/argmax 결정이 필요한 지표면 OOF 기반 최적화 포함

===== 프롬프트 끝 (B) =====

## 체크리스트
- [ ] 모든 OOF가 동일 fold 배정 기반인가 (아니면 앙상블 점수 자체가 무효)
- [ ] 음수 가중치 신호를 확인했는가
- [ ] 최종 제출 2개(안전/공격)를 구분해 준비했는가
