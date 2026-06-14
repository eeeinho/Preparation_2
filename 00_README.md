# 해군 AI 경진대회 — 프롬프트 세트 v2 사용 가이드 (전 24파일)

## 사용 순서 요약
대회 전: 23번 숙지, 11번 팀 공유, 마스터 프롬프트(AI_Prompt_UATR_Naval_v1.md) 각 AI 세션에 주입
대회 시작: 23(규정) -> 13(환경) -> 01(구조) -> 02(EDA) -> 03(CV) -> 19(지표 확정)
베이스라인: 04(피처) -> 05(LGBM 첫 제출, 07로 검증) / 14(임베딩, 병렬)
주력: 06(CNN) + 08(AST) 병렬 -> 15(증강 ablation)
중반: 17(오류 분석/진단 트리) -> 필요시 16(도메인 적응)
후반: 09(듀얼브랜치+앙상블) -> 12(보고서) + 18(생성형AI 문서화)
상시: 22(트러블슈팅 치트시트)

## 파일 분류

### [0단계: 사전/시작 직후 — 사람용 문서]
| 파일 | 용도 |
|------|------|
| 23_day_zero_checklist | 규정 파악 질문, 외부 데이터 전략, 첫 1시간 게이트 |
| 11_team_ops | 실험 시트, 4인 역할 분담, 시간 예산표, 패키징 체크리스트 |
| 22_troubleshooting | OOM/오디오IO/HF/CV이상/제출 사고 치트시트 + 디버깅 프롬프트 |

### [1단계: 기반 구축 — ★★★ 필수]
| 파일 | 용도 |
|------|------|
| 13_env_setup_h200 | SSH/tmux/패키지/사전학습 가중치 선다운로드/백업 |
| 01_data_structure_check | 10분 구조 점검, 누수 위험 탐지 |
| 02_audio_eda | 수중 음향 7종 시각화 + 전처리 결론 자동화 |
| 03_cv_leakfree | folds.csv 확정 (단일 진실 공급원), Adversarial Val |
| 19_metrics_library | 전 지표 구현 + 지표별 공략 전략표 |

### [2단계: 베이스라인 — ★★★ 첫 제출]
| 파일 | 용도 |
|------|------|
| 04_features_lofar_demon | 통계+LOFAR+DEMON 피처 추출 |
| 05_baseline_lgbm | LGBM/CatBoost 첫 제출 |
| 14_pretrained_embeddings | frozen 임베딩(PANNs/AST/WavLM)+경량분류기 — 가성비 최강 |
| 07_submission_validator | 제출 검증기 (모든 제출 통과 필수) |

### [3단계: 주력 모델 — ★★]
| 파일 | 용도 |
|------|------|
| 06_baseline_cnn_mel | Mel CNN (timm, 안전 증강 규칙 내장) |
| 08_ast_wav2vec_finetune | 대형모델 fine-tune (SOTA, fold0 게이트) |
| 15_augmentation_synthesis | 3등급 증강 + SNR 합성 + 지문 보존 검증 |

### [4단계: 점수 압착 — ★★]
| 파일 | 용도 |
|------|------|
| 17_error_analysis_debug | 오류 해부 + 점수 정체 진단 트리 |
| 16_domain_adaptation | 분포 이동 대응 3레벨 (BN재추정/pseudo/DANN) |
| 09_dual_branch_ensemble | Mel+LOFAR 듀얼브랜치 + 앙상블 가중 최적화 |

### [태스크 변형 대응 — 출제 확인 후 선택]
| 파일 | 대응 태스크 |
|------|------|
| 10_task_variants | A 분류(ordinal 포함) / B 이상탐지 / C 잡음제거 / D 시계열·기동의도 |
| 20_sed_detection | 시간 구간 탐지(SED): CFAR 고전 탐지 + CRNN |
| 21_multichannel_doa | 다채널 어레이 처리 / 방위(sin·cos)·거리 추정 |

### [마무리/평가 항목 대응 — ★]
| 파일 | 용도 |
|------|------|
| 18_genai_strategy | 생성형 AI 활용 (점수 기여 + 심사 문서화, API 코드 포함) |
| 12_report_writeup | 보고서 + 슬라이드 10장 구성안 |

## 공통 절대 원칙
1. 세그먼트 랜덤 split 금지 — folds.csv(녹음/선박 그룹) 단일 진실 공급원
2. 의사결정은 녹음 단위 집계 점수 (19번 aggregate_and_score)
3. 모든 모델 산출물 규격 통일: oof_*.npy / test_*.npy / 체크포인트 이름 규칙
4. fold0 게이트: 고비용 모델은 fold0 향상 확인 후 5-fold 진행
5. seed 고정, 실험 로그 기록, placeholder 있는 코드는 채택 금지
6. 최종 제출 2개: 안전(단순 평균) / 공격(최적 가중·DA 반영) 분리
