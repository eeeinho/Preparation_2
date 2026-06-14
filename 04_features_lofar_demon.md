# 04. 통계 + LOFAR + DEMON 피처 추출 프롬프트
목적: LGBM/CatBoost용 수치 피처 일괄 추출. 딥러닝의 강력한 백업이자 앙상블 다양성 원천.

===== 프롬프트 시작 =====

너는 수중 음향 신호처리 전문가다. 함정 음향 WAV에서 아래 피처를 모두 추출해 features.parquet으로 저장하는 멀티프로세싱 스크립트를 작성하라. 함정 음향의 물리: 기계 소음은 저주파 선스펙트럼(LOFAR 라인), 프로펠러 캐비테이션은 광대역 소음의 진폭 변조(DEMON, 축회전수x날수=BPF)로 나타난다.

[입력]
- 파일 목록/sr: {01번 결과}
- 세그먼트 정책: {파일 전체 1행 또는 N초 세그먼트당 1행 — 선택}

[추출 피처 (그룹별)]
A. 시간 영역: duration, rms_mean/std, zcr_mean/std, peak_amplitude, crest_factor
B. 스펙트럼 통계 (librosa): spectral_centroid, bandwidth, rolloff(0.85), flatness
   — 각각 mean/std
C. MFCC: n_mfcc=20, 각 계수의 mean/std/min/max (총 80개) + delta mean/std
D. Mel band energy: n_mels=64 각 밴드 mean/std (저주파 해상도 위해 f_min=10, f_max=8000)
E. 대역 에너지 비율 (Welch PSD 기반, 전체 에너지로 정규화):
   0~20Hz(INFRA), 20~100Hz, 100~1000Hz(LF), 1~10kHz(MF), 10kHz+(HF)
   + lf_ratio = (20~1000Hz)/전체 — 함정 수동 탐지 핵심 대역
F. LOFAR 라인 피처:
   - 0~1000Hz 고해상도 PSD(n_fft=8192)에서 median filter 배경 차감
   - top-5 라인 피크의 (주파수, 강도, 배경 대비 prominence)
   - 라인 개수(prominence 임계 초과), 라인 주파수 간 최소 공약 간격(고조파 기본주파수 추정)
G. DEMON 피처:
   - 2~8kHz 밴드패스 -> Hilbert 포락선 -> 변조 스펙트럼(0~100Hz)
   - top-3 변조 피크 (주파수, 강도)
   - 추정 축회전수(최대 피크 Hz), 추정 BPF 후보(피크 고조파 간격),
     blade_count 후보 = BPF/축회전수 (정수 근접도 포함)
H. 하모닉/타악 분리: librosa.effects.hpss 후 harmonic_ratio = E_h/(E_h+E_p)

[요구사항]
1. 함수 단위로 분리(피처 그룹별), 단일 파일 처리 함수 -> ProcessPoolExecutor 병렬화
2. 실패 파일은 NaN 행으로 기록하고 목록 저장 (전체 중단 금지)
3. dtype 최적화(float32), 피처 이름은 그룹 prefix 부여 (예: demon_pk1_hz)
4. 완료 후: 피처 수, NaN 비율, 클래스별 주요 피처 평균 차이 상위 10개 출력
   (mutual_info_classif 빠른 스코어링)
5. seed 고정, tqdm 진행 표시

[제약]
- librosa + scipy.signal, H200 서버이므로 CPU 코어 수만큼 병렬
- 세그먼트 모드일 경우 (file, seg_idx, group, label) 키 유지 — folds.csv와 조인 가능하게

===== 프롬프트 끝 =====

## 체크리스트
- [ ] features.parquet이 folds.csv와 file 키로 조인되는가
- [ ] DEMON/LOFAR 피처가 실제로 클래스 분리력을 보이는가 (MI 상위권 여부)
- [ ] NaN 비율이 높은 피처는 원인 파악했는가 (짧은 파일, 저sr 등)
