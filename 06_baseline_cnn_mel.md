# 06. CNN + Mel-spectrogram 베이스라인 프롬프트
목적: 딥러닝 1차 주력. Practice6 경험과 직결, H200에서 빠른 학습.

===== 프롬프트 시작 =====

너는 오디오 딥러닝 전문 Kaggle Grandmaster다. 함정 수중 음향 분류를 위한 Mel-spectrogram + CNN 파이프라인을 PyTorch로 작성하라. 환경은 H200 GPU(bf16), 인터넷 가능(timm/torchaudio 사용 가능)이다.

[입력]
- 평가지표/클래스/제출형식: {01번 결과}
- 권장 전처리: {02번 EDA 결론 — sr, 세그먼트 길이, f_min/f_max}
- 공용 fold: folds.csv

[요구사항]
1. 전처리:
   - 리샘플 {SR}Hz, 세그먼트 {N}초 (train: 50% overlap, valid/test: non-overlap)
   - log-Mel: n_fft=2048, hop=320, n_mels=128, f_min=10, f_max={EDA 결론},
     per-sample 표준화 (fold 통계 사용 금지 — 누수 방지)
   - Mel을 사전 계산해 .npy 캐싱 (학습 중 CPU 병목 제거), float16 저장
2. 모델: timm 사전학습 백본 2종을 같은 코드로 스위치 가능하게
   - efficientnet_b0 (빠른 검증용) / convnext_small (성능용)
   - 입력 1채널 -> 3채널 복제 또는 conv1 수정, num_classes 교체
3. 증강 (수중 음향 안전 증강만):
   - time shift (roll), gain ±2dB, 가우시안 노이즈(SNR 15~30dB)
   - SpecAugment: time mask 2개(폭<=세그먼트의 10%), freq mask 폭 F<=12
     (LOFAR 라인 보존 위해 freq mask 작게 — 주석으로 이유 명시)
   - mixup(alpha=0.2) 옵션 플래그
   - 금지: pitch shift, 큰 speed 변화 (함정 음향 지문 훼손)
4. 학습: 5-fold(folds.csv), AdamW lr=3e-4(백본)/1e-3(헤드), cosine schedule,
   epochs=15 + early stopping(녹음 단위 macro-F1), bf16 autocast, batch 256,
   CrossEntropy(label_smoothing=0.1) + 불균형 시 class weight
5. 평가: 세그먼트 OOF + 녹음 단위 집계(logit 평균) OOF 둘 다 보고,
   집계 방식 비교(mean/max/trimmed-mean 20%) 후 최선 채택
6. 추론: test 모든 세그먼트 예측 평균(자연 TTA), fold 앙상블(5모델 평균)
7. 산출물: oof_cnn.npy / test_cnn.npy / fold별 체크포인트(.pt) /
   submission_cnn_mel_cv{score:.4f}_{timestamp}.csv / 실험 로그 append
8. confusion matrix + 가장 헷갈리는 클래스 쌍 출력, 오분류 상위 파일 목록 저장
   (사람이 직접 들어볼 수 있게)

[제약]
- 완전한 실행 가능 코드, 학습 1 fold당 예상 시간 print
- seed 고정, 체크포인트 이름 체계: cnn_{backbone}_fold{k}.pt (가중치 제출 대비)

===== 프롬프트 끝 =====

## 체크리스트
- [ ] Mel 캐싱으로 GPU 사용률 90%+ 유지되는가
- [ ] freq mask가 LOFAR 라인을 지우지 않는지 증강 샘플 시각화로 확인했는가
- [ ] LGBM OOF와 상관계수 확인 (낮을수록 앙상블 가치 높음)
