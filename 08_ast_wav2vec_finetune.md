# 08. AST / Wav2Vec2 Fine-tuning 프롬프트 (주력 모델)
목적: 최신 SOTA 접근 — 대형 사전학습 오디오/음성 모델 전이. 최근 연구에서 음성 대형모델
전이가 DeepShip/ShipsEar F1 99%+로 도메인 특화 구조를 능가함이 확인됨.

===== 프롬프트 시작 =====

너는 HuggingFace 생태계에 능숙한 오디오 딥러닝 전문가다. 함정 수중 음향 분류를 위해 사전학습 모델 fine-tuning 파이프라인을 작성하라. 환경: H200 GPU(bf16, 메모리 충분), 인터넷 가능, transformers/torchaudio 설치됨.

[입력]
- 클래스/평가지표/fold: {01·03번 결과}, folds.csv 공용 사용
- 세그먼트: {N}초, 리샘플 16kHz (사전학습 모델 호환)

[요구사항]
1. 모델 추상화: 아래 3종을 같은 학습 루프에서 config 스위치로 교체 가능하게
   a. AST: "MIT/ast-finetuned-audioset-10-10-0.4593" (스펙트로그램 입력)
   b. Wav2Vec2/WavLM: "microsoft/wavlm-large" 또는 "facebook/wav2vec2-large"
      (원시 파형 입력, mean pooling + linear head)
   c. (옵션) Whisper encoder: "openai/whisper-small" encoder + attention pooling head
2. 학습 전략:
   - 1 epoch: 백본 freeze, 헤드만 lr=1e-3 warmup
   - 이후 전체 unfreeze, layer-wise LR decay(0.9), base lr=1e-5~3e-5
   - bf16 autocast, gradient clipping 1.0, epochs 4~8 + early stopping(녹음 단위 F1)
   - batch는 GPU 메모리 한도까지 (AST는 64+, wav2vec2-large는 32+ 시작)
3. 증강: gain, time shift, 소량 노이즈만 (pitch/speed 금지 — 음향 지문 보존)
4. 평가/산출물: 06번과 동일 규격 —
   세그먼트+녹음단위 OOF, oof_ast.npy/test_ast.npy, fold 체크포인트,
   submission_ast_cv{score:.4f}_{timestamp}.csv, 실험 로그 append
5. 시간 관리: fold0만 먼저 완주해 녹음 단위 점수를 CNN 베이스라인과 비교 출력,
   유의미하게 높을 때만 5-fold 전체 진행하는 게이트 로직 포함
6. 추론 전용 inference.py 분리: 체크포인트 로드 -> test 예측 -> 제출 생성
   (심사측 재현 제출용, 학습 코드와 독립 실행 가능)

[알아둘 것]
- 16kHz 리샘플로 8kHz 이상 정보가 잘리므로, 원본 sr이 높고 HF 대역에 정보가 있다면
  (02번 EDA 확인) CNN(원 sr Mel) 모델과의 앙상블이 보완 역할을 한다 — 주석으로 명시
- AST 입력 길이 초과 시 feature extractor의 max_length 조정 필요

===== 프롬프트 끝 =====

## 체크리스트
- [ ] fold0 게이트에서 CNN 대비 향상을 확인하고 진행했는가
- [ ] inference.py 단독 실행으로 제출 재현이 되는가
- [ ] OOF 상관: AST vs CNN vs LGBM (다양성 확인)
