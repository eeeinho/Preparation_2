# 15. 데이터 증강 + 합성 데이터 프롬프트
목적: 수중 음향에 안전한 증강 체계 + SNR 합성으로 강건성/소수 클래스 보강.

===== 프롬프트 시작 =====

너는 오디오 증강 전문가이자 수중 음향 도메인 전문가다. 함정 음향 분류용 증강 파이프라인을 작성하라. 대원칙: 함정의 음향 지문(LOFAR 라인 주파수, DEMON 변조 주파수)은 함종 정보 그 자체이므로, 주파수 구조를 바꾸는 증강은 레이블을 훼손한다.

[증강 분류 — 코드에 3단계 등급으로 구현]
SAFE (기본 적용):
- gain ±3dB, time shift(circular roll), 세그먼트 내 랜덤 크롭
- 가우시안/핑크 노이즈 추가 (SNR 10~30dB 랜덤)
- 배경 잡음 mixing: train의 다른 클래스가 아닌 "잡음 전용 소스"와 합성
  (데이터에 background/noise 클래스가 있으면 그것을, 없으면 신호 없는 구간을
   에너지 기준으로 자동 탐지해 잡음 뱅크 구축)
RISKY (CV 검증 통과 시만):
- SpecAugment: time mask(폭<=10%) 2개, freq mask F<=12 (라인 보존 한계 명시)
- mixup(alpha=0.2) / 같은 클래스 내 cutmix(시간축)
- time stretch ±5% 이내 (속도-음향 상관을 약간만 교란)
FORBIDDEN (구현하되 기본 차단, 사용 시 경고):
- pitch shift, 큰 time stretch, 주파수 왜곡 계열

[요구사항]
1. audiomentations 기반 Compose 3종(safe/risky/full) + 커스텀 배경 mixing 클래스
2. 합성 데이터 생성기:
   - (clean 신호, 잡음, 목표 SNR) -> 합성 샘플. SNR -5/0/5/10/15dB 버킷
   - 소수 클래스 오버샘플링: 소수 클래스 신호 x 다양한 잡음/SNR 조합으로 N배 생성
   - 합성 샘플은 train fold에만 추가 (valid 오염 금지 — assert 포함)
3. 증강 검증 장치:
   - 증강 전/후 LOFAR-gram, DEMON 스펙트럼 비교 플롯 자동 생성
   - DEMON 최대 피크 주파수가 증강 후에도 ±1Hz 내 보존되는지 수치 검사 함수
4. ablation 러너: 06번 CNN fold0 기준으로 none/safe/safe+risky 3조건 점수 비교 표
5. test-time 강건성 실험(옵션): valid에 SNR 0dB 잡음을 입혀 평가 ->
   잡음 강건성이 낮으면 SNR 증강 강화 권고 출력

===== 프롬프트 끝 =====

## 체크리스트
- [ ] 증강 후에도 DEMON/LOFAR 지문이 보존되는지 수치로 확인했는가
- [ ] 합성 샘플이 valid fold에 새지 않는가 (assert)
- [ ] ablation 수치로 채택을 결정했는가 (감으로 켜지 말 것)
