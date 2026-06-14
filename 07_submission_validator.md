# 07. 제출 파일 자동 검증 프롬프트
목적: 대회 막판 가장 자주 터지는 submission 오류를 기계적으로 차단.

===== 프롬프트 시작 =====

너는 꼼꼼한 ML 엔지니어다. 모든 제출 파일이 통과해야 하는 검증기 validate_submission.py를 작성하라. 검증 실패 시 구체적 사유와 함께 비정상 종료(exit 1)해야 한다.

[입력]
- sample_submission 경로: {경로}
- 제출 형식: {클래스명 문자열? 클래스 인덱스? 확률 컬럼?}

[검증 항목 — 전부 구현]
1. shape 일치: 행 수/컬럼 수/컬럼명/컬럼 순서가 sample_submission과 동일
2. id 검증: sample_submission의 id 집합과 완전 일치, 중복 없음, 순서 동일 여부 보고
3. 결측/inf 값 0개
4. 값 도메인:
   - 클래스 레이블 제출이면: 예측값이 train 레이블 집합의 부분집합인지
   - 확률 제출이면: 0~1 범위, 행 합 ~1 (multiclass), 음수 없음
5. dtype 검증 (id가 숫자로 변환되며 leading zero가 깨지는 사고 탐지)
6. 인코딩/구분자: utf-8, 콤마, 인덱스 컬럼 미포함 확인
7. 예측 분포 sanity check: 클래스별 예측 비율 출력, 단일 클래스 95%+ 쏠림이면 경고
8. (있다면) 직전 베스트 제출과 예측 불일치율 출력 — 급변이면 경고

[추가 구현]
- LabelEncoder 저장/복원 유틸 (label_encoder.pkl) — 학습/추론 간 클래스 순서 불일치 사고 방지
- 제출 파일명 생성기: submission_{model}_cv{score:.4f}_{YYYYMMDD_HHMM}.csv
- 제출 이력 로그: submissions_log.csv (파일명, cv, 제출시각, 메모) append 함수
- 사용법: python validate_submission.py sub.csv 처럼 CLI로 동작 (argparse)

===== 프롬프트 끝 =====

## 운영 규칙 (팀 공유)
- 검증기 미통과 파일은 절대 제출 금지
- 제출 직후 submissions_log.csv에 public 점수 기록 (보조 팀원 담당)
- CV vs public 점수 산점도를 주기적으로 확인 — 상관 무너지면 CV 재점검
