# 02. 오디오 EDA 프롬프트
목적: 수중 음향 특화 시각화 7종을 한 번에 생성. 전처리/모델 결정 근거 확보.

===== 프롬프트 시작 =====

너는 수중 음향 신호처리 전문가이자 Kaggle Grandmaster다. 함정 수중 음향 데이터의 EDA 코드를 작성하라. 수중 음향은 정보가 저주파(20~1000Hz)에 집중되고, LOFAR(선스펙트럼)와 DEMON(프로펠러 변조)이 함종 식별의 핵심이다.

[입력]
- 데이터 경로/구조: {01번 점검 결과 요약 붙여넣기}
- 클래스 목록: {클래스명들}
- sample rate: {SR}

[요구사항]
클래스별로 무작위 3개 샘플씩 선택하여 아래 시각화를 모두 생성하는 코드를 작성하라.

1. Waveform (시간축, 초 단위)
2. STFT spectrogram (n_fft=2048, dB 스케일)
3. Mel-spectrogram — 단, 수중 음향용 설정: f_min=10, f_max=min(8000, sr/2), n_mels=128
4. LOFAR-gram: 저주파(0~1000Hz) 고해상도 스펙트로그램(n_fft=4096)
   + 행 방향 median filter로 배경 추정 후 차감하여 선스펙트럼(라인) 강조
5. DEMON envelope spectrum: 2~8kHz 밴드패스 -> Hilbert 포락선 -> FFT(0~100Hz)
   + 상위 피크에 추정 축회전수(Hz, RPM 환산) 주석 표시
6. 클래스별 평균 Welch 스펙트럼 (모든 클래스 한 그림에 겹쳐서, log-freq 축)
7. 클래스별 duration / sample rate / RMS 분포 (boxplot)

[분석 출력]
시각화 후 아래 질문에 답하는 자동 요약을 print하라:
- 클래스 간 스펙트럼 차이가 어느 대역에서 가장 큰가 (대역별 평균 에너지 차이 수치)
- LOFAR 라인이 뚜렷한 클래스 / DEMON 피크가 뚜렷한 클래스
- SNR이 낮아 보이는 샘플 비율 (spectral flatness 기준)
- train/test 평균 스펙트럼 비교 — 분포 이동 의심 여부
- 권장 전처리: 리샘플 sr, 세그먼트 길이, 밴드패스 범위

[제약]
- librosa + scipy.signal + matplotlib, 그림은 eda/ 폴더에 png 저장
- 한 클래스당 1개의 종합 figure(subplot)로 묶어 비교 용이하게
- 전체 실행 5분 이내 (샘플링 기반)

===== 프롬프트 끝 =====

## 체크리스트
- [ ] 클래스 구분 정보가 어느 대역/표현에 있는지 결론 냈는가
- [ ] Mel 파라미터(f_min/f_max/n_mels)를 데이터 근거로 정했는가
- [ ] train/test 스펙트럼 차이(도메인 갭) 여부를 확인했는가
