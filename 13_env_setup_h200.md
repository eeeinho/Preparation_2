# 13. H200 원격 SSH 환경 셋업 프롬프트
목적: 대회 시작 후 첫 30분 내 환경 완비. 학습 중단/다운로드 지연 사고 예방.

===== 프롬프트 시작 =====

너는 MLOps 전문가다. 해커톤용 H200 GPU 원격 SSH 서버를 빠르게 셋업하는 스크립트와 운영 가이드를 작성하라.

[요구사항]
1. setup.sh 작성:
   - 시스템 확인: nvidia-smi, CUDA 버전, 디스크 용량, CPU 코어, RAM
   - tmux 세션 구성: train / monitor / jupyter 3개 윈도우 자동 생성 스크립트
     (SSH 끊겨도 학습 유지 — 모든 장기 작업은 tmux 안에서만 실행하는 규칙 명시)
   - 패키지 일괄 설치 (버전 핀 고정):
     torch torchaudio(CUDA 호환 버전 확인 로직), transformers, timm, librosa,
     soundfile, scipy, scikit-learn, lightgbm, catboost, optuna, pandas, pyarrow,
     audiomentations, noisereduce, tqdm, matplotlib, seaborn
   - pip 실패 대비: 핵심 패키지 재시도 + 미러 fallback
2. 사전학습 가중치 선다운로드 스크립트 (대회 시작 직후 백그라운드 실행):
   - timm: efficientnet_b0, efficientnet_b4, convnext_small, resnet50
   - HF: MIT/ast-finetuned-audioset-10-10-0.4593, microsoft/wavlm-large,
     facebook/wav2vec2-large-960h, openai/whisper-small
   - PANNs CNN14 체크포인트 (zenodo URL)
   - HF_HOME을 프로젝트 내 경로로 고정 (캐시 위치 통일, 패키징 시 제외 목록 작성)
3. GPU 모니터링: watch -n2 nvidia-smi 별칭 + gpustat, 학습 로그 tail 헬퍼
4. 디렉토리 표준 생성: data/ outputs/ outputs/checkpoints/ eda/ logs/ submissions/
5. 공용 conf.py 생성: SEED, 경로 상수, N_FOLDS — 모든 스크립트가 import
6. 팀 공유: 코딩 2명이 같은 서버를 쓸 때 GPU 메모리 분할 규칙
   (CUDA_VISIBLE_DEVICES 또는 메모리 fraction, 실험 큐는 한 명이 관리)
7. 백업: outputs/를 1시간마다 로컬로 rsync하는 cron 또는 watch 스크립트
   (서버 사고 대비 — 체크포인트/OOF 유실은 복구 불가)

[제약]
- 셋업 전체 15분 내 완료 목표, 각 단계 실패 시에도 다음 단계 진행되게
- 다운로드는 nohup 백그라운드 + 로그 파일

===== 프롬프트 끝 =====

## 체크리스트
- [ ] tmux 밖에서 학습을 돌리고 있지 않은가
- [ ] 사전학습 가중치가 전부 로컬 캐시에 있는가 (오프라인에서도 로드 확인)
- [ ] outputs/ 백업이 도는가
