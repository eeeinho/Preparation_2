# 22. 트러블슈팅 치트시트 (사람용 — AI 주입 불필요, 막혔을 때 즉시 참조)
목적: 28.5시간 중 디버깅으로 새는 시간을 분 단위로 줄인다.

## A. GPU / 학습
- CUDA OOM: batch 절반 -> gradient accumulation 2배 (유효 batch 유지).
  AST/wav2vec2-large는 입력 길이가 메모리 지배 — 세그먼트 단축이 batch 축소보다 효과적.
  torch.cuda.empty_cache()는 임시방편, 데이터로더 worker가 GPU 텐서를 만들지 않는지 확인.
- 학습이 비정상적으로 느림: GPU util 확인(nvidia-smi). util 낮으면 CPU 병목 —
  Mel 사전 캐싱(06번), num_workers 증가, pin_memory=True, persistent_workers=True.
- loss=NaN: lr 1/10, gradient clipping 1.0, bf16 유지(에러 시 fp32 fallback),
  입력에 NaN/inf 검사 (깨진 WAV가 0길이/inf RMS를 만드는 경우 흔함).
- bf16 vs fp16: H200은 bf16 — fp16 GradScaler 불필요, autocast(dtype=bfloat16)만.
- 재현 안 됨: DataLoader worker seed(worker_init_fn), shuffle seed, cudnn.benchmark=True는
  속도 위해 허용하되 최종 재학습만 deterministic 고려.

## B. 오디오 I/O
- librosa.load 느림: soundfile.read + scipy/torchaudio resample이 수배 빠름.
  대량 처리는 반드시 멀티프로세싱 + 사전 캐싱.
- sr 불일치 버그: 일부 파일만 다른 sr -> 전 파일 강제 리샘플 함수 단일화 (conf.py 경유).
- mp3/이상 포맷: torchaudio backend 에러 시 ffmpeg-python으로 wav 일괄 변환 후 시작.
- 짧은 파일: 세그먼트보다 짧으면 zero-pad가 아니라 tile(반복)이 보통 더 나음 — 실험.
- int16/float 스케일: soundfile은 float64(-1~1), scipy.io.wavfile은 int16 — 혼용 금지.

## C. HuggingFace / 다운로드
- 다운로드 실패/지연: HF_HOME 캐시 확인, export HF_HUB_ENABLE_HF_TRANSFER=1,
  미러: export HF_ENDPOINT=https://hf-mirror.com (공식 막힐 때).
- ignore_mismatched_sizes=True 필요: 분류 헤드 클래스 수 교체 시.
- AST 입력 길이 에러: feature extractor의 max_length(기본 1024 frame=약 10.24s) 조정.

## D. CV / 점수 이상
- CV 95%+인데 public 폭락: 99% 누수. 03번 그룹 분할 재점검, 파일명/메타 피처 제거.
- fold 간 점수 분산 큼: 그룹 수 부족 — fold 수 축소(3) 또는 그룹 정의 완화 검토.
- OOF와 제출 점수 계산 불일치: 클래스 순서(label encoder) 불일치가 1순위 의심 —
  label_encoder.pkl 단일화(07번).
- 앙상블이 개별보다 나쁨: fold 배정 불일치 또는 클래스 컬럼 순서 불일치 — assert 추가.

## E. 제출
- 제출 직전 체크 3종: 07번 검증기 통과 / 행수 / 클래스 분포 sanity.
- 마감 30분 전: 새 학습 금지, 베스트 2개 재검증 후 제출. zip 패키징은 11-D 체크리스트.

## F. 막혔을 때 LLM 디버깅 프롬프트 (복붙용)
"아래 에러를 디버깅하라. 환경: H200, torch {버전}, 코드 목적: {1줄}.
전체 traceback: {붙여넣기}. 관련 코드: {붙여넣기}.
가능한 원인을 확률 순으로 3개 제시하고, 각각 1분 내 검증 가능한 확인 방법을 함께 제시하라.
수정 코드는 변경 부분만 diff 형태로."
