# 진행한 실험과 결과

## 데이터 구성

head 영상에서 판정한 원본 30개 에피소드는 수령자 기준 무회전 10개, 왼쪽 12개, 오른쪽 8개였습니다. 기존과 같은 방향별 검증 2개를 유지했습니다. 왼쪽 학습 에피소드 `7`, `10`은 동작 구간 EMG 형태가 같은 방향 분포에서 멀어 제외했습니다. 학습에서 무회전과 왼쪽은 고유 에피소드 각각 8개, 오른쪽은 6개를 사용했고, 재사용을 포함해 각 방향에 **10개 학습 슬롯**을 배정했습니다. 구체적인 에피소드와 순서는 [균형 분할](../data/balanced_splits.json)에 있습니다.

320ms 캐시는 200ms 캐시의 같은 영상·행동 행을 사용합니다. 유효한 이력이 부족한 학습 행 105개와 검증 행 23개가 제외됐습니다. 수령 구간의 1,200개 학습 방향 라벨과 255개 검증 방향 라벨은 유지됐습니다. [캐시 정렬 검증](../reports/window_320_cache_alignment.json)

## 학습 목록

공통 설정은 seed `1000`, 배치 크기 `8`, AdamW 학습률 `1e-5`, weight decay `1e-4`, ResNet18 및 ACT 행동 청크 30 step입니다. Best로 표시하는 오차는 정규화된 행동의 `val_l1_zero_latent`입니다.

| 실험 | 입력과 변경점 |
| --- | --- |
| ACT Baseline | 카메라 + 관절 상태. 균형 분할로 원본 설정 재학습 |
| ACT Concat | Baseline 입력의 state에 EMG 50ms RMS 4개 추가. 균형 분할로 원본 설정 재학습 |
| ACT Wave 200ms | 2kHz EMG 400×4를 1D CNN으로 인코딩해 ACT의 environment-state 토큰으로 사용. 원본 Wave 균형 재학습에도 이 결과를 재사용 |
| ACT Spectrogram 200ms | 64ms STFT 프레임, 16ms hop, `n_fft=256`, 20–450Hz log-power를 2D CNN으로 인코딩 |
| ACT Wave 200ms + 방향 분류 | 수령 구간에서 EMG 토큰으로 무회전·왼쪽·오른쪽을 분류해 보조 손실 적용 |
| ACT Wave 320ms | EMG 파형 길이를 640×4로 확장 |
| ACT Wave 320ms + 방향 분류 | 같은 320ms 이력으로 Wave와 수령 구간 방향 분류를 함께 학습 |

## 서버 상태 스냅샷

<!-- RUN_STATUS_START -->
**2026-11-23 06:46:27 KST 기준**

| 실험 | 상태 | 마지막 step | Best step | Best validation L1 | 모델·로그 |
| --- | --- | ---: | ---: | ---: | --- |
| ACT Wave 200ms | 완료 | 100,000 | 47,000 | 0.166926 | [Hub](https://huggingface.co/HYHero/kroc-act-emg-balanced-20261123/tree/main/runs/window/8f0c57366a08414aa9ef17471832c4ed) |
| ACT Spectrogram 200ms | 완료 | 100,000 | 98,000 | 0.158947 | [Hub](https://huggingface.co/HYHero/kroc-act-emg-balanced-20261123/tree/main/runs/spectrogram/6df28b3ee7e640bb9a30e811d35e4b23) |
| ACT Wave 200ms + 방향 분류 | 완료 | 100,000 | 47,000 | 0.166224 | [Hub](https://huggingface.co/HYHero/kroc-act-emg-balanced-20261123/tree/main/runs/window_direction/e89c1d6f8a344bb8b4e140970eac6d6f) |
| ACT Wave 320ms | 진행 중 | 59,400 | 56,000 | 0.171450 | [Hub](https://huggingface.co/HYHero/kroc-act-emg-balanced-20261123/tree/main/runs/window_320/65d0132c25f34b4880085ce8cf1b9d99) |
| ACT Wave 320ms + 방향 분류 | 진행 중 | 59,300 | 32,000 | 0.169391 | [Hub](https://huggingface.co/HYHero/kroc-act-emg-balanced-20261123/tree/main/runs/window_direction_320/7e3614de15f045ee9edcd46d11b86c64) |
| ACT Baseline | 예약 | — | — | — | — |
| ACT Concat | 예약 | — | — | — | — |
<!-- RUN_STATUS_END -->

[`experiment_snapshot.json`](../reports/experiment_snapshot.json)은 확인한 시점의 결과입니다. 실시간에 가까운 상세 로그와 체크포인트는 각 실행의 Hugging Face 경로에서 확인할 수 있습니다. 별도로 배포된 원본 Baseline·Concat·Wave 추론 체크포인트의 provenance는 [추론 매니페스트](../configs/inference_models.json)에 있습니다.

## 방향 분석과 200ms·320ms 비교

- 영상에서 수령 구간을 표시하고, 카메라 수신 시각으로 EMG와 맞췄습니다. 사람의 손 방향 검증에는 프레임 단위 macro-F1과 혼동행렬을 사용합니다.
- 시연에서 로봇 `left_joint5`의 명령 변화 중앙값은 무회전 `0.0°`, 왼쪽 `+56.1°`, 오른쪽 `−71.8°`였습니다. [손목 관절 분석](../reports/rotation_action_diagnostic.json)
- 이전 EMG 분석에서 왼쪽은 A0·A2 RMS, 오른쪽은 A1 RMS와 A0 120–250Hz 에너지 비율이 상대적으로 높았습니다. [EMG 형태 분석](../reports/emg_balance_analysis.json)
- 동일 검증 에피소드 6개에서 EMG-only Wave 인코더를 두 초기 seed로 비교했습니다. 200ms의 macro-F1은 전체 `0.785`, 첫 400ms `0.698`이었고 320ms는 전체 `0.816`, 첫 400ms `0.764`였습니다. 진폭과 주파수 특징을 쓴 별도 선형 분류기는 200ms에서 첫 400ms macro-F1 `0.822`를 얻었습니다. 이 비교는 분류 평가이며, 실제 건네기 성능은 로봇 행동으로 평가해야 합니다. [길이별 원본 결과](../reports/emg_window_diagnostic.json)
