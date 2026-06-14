# 20. 음향 이벤트 탐지(SED) 태스크 프롬프트
목적: "긴 녹음 속 어느 시간 구간에 함정/소나 핑이 존재하는가" 형태 출제 대비.
분류(10-A)와 다른 별도 파이프라인이 필요한 유일한 태스크 — 사전 준비 가치 높음.

===== 프롬프트 시작 =====

너는 DCASE(음향 이벤트 탐지 챌린지) 수상 경험이 있는 전문가다. 수중 음향에서 표적 신호의 시간 구간(onset/offset)을 탐지하는 SED 파이프라인을 작성하라.

[입력]
- 레이블 형식: {강한 레이블(구간 주석) / 약한 레이블(클립 단위 존재 여부)}
- 평가지표: {event-based F1? segment-based F1? IoU 기준?} — 19번 metrics.py 사용

[요구사항]
1. 베이스라인 (1시간 내): 에너지/스펙트럼 기반 고전 탐지
   - 대역 제한 에너지(표적 대역) 이동 평균 + 적응 threshold(배경 잡음 추적,
     CFAR 스타일: 셀 평균 대비 배수) -> 구간 병합/최소 길이 필터
   - 소나 핑 탐지라면: matched filter(템플릿 상관) 옵션 추가
2. 주력: CRNN frame-level 모델
   - 입력 log-Mel(전체 클립), CNN 인코더(시간축 보존) -> BiGRU -> frame별 sigmoid
   - 강한 레이블: frame BCE / 약한 레이블: linear-softmax pooling로 클립 손실 +
     frame 출력은 그대로 탐지에 사용 (weakly-supervised SED 표준)
   - median filter 후처리 + threshold/최소 구간 길이를 검증셋 event-F1로 그리드 탐색
3. CV: 녹음 단위 그룹 분할 (03번 folds 재사용), 평가는 event-based 지표로
4. 제출 형식 대응: (file, onset, offset, class) 또는 frame 마스크 — sample_submission
   구조를 보고 자동 변환 함수 작성
5. 앙상블: 고전 탐지와 CRNN의 frame 확률 평균 + 구간 병합 규칙 통일
6. 시각화: 스펙트로그램 위에 GT 구간 vs 예측 구간 오버레이 플롯 (오탐/미탐 검수용)

[도메인 팁 — 주석 포함]
- 함정 통과(CPA) 이벤트는 에너지가 서서히 증가/감소 — onset 정의가 모호하므로
  평가 tolerance를 먼저 확인하고 후처리 스무딩 강도를 그에 맞출 것
- 소나 핑은 짧고 협대역 — hop을 작게(10ms급), 핑 반복 주기(PRI)를 후처리 검증에 활용

===== 프롬프트 끝 =====

## 체크리스트
- [ ] 평가지표의 onset/offset tolerance를 정확히 재현했는가
- [ ] 고전 탐지 베이스라인을 먼저 제출했는가 (SED에서 의외로 강함)
- [ ] threshold/최소 길이/병합 갭을 지표로 튜닝했는가
