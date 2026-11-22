# 학습 재현과 서버 운영

실험 서버는 RTX 5090에서 Python 3.12, PyTorch `2.11.0+cu128`과 ROBOTIS LeRobot Cyclo `240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b`를 사용했습니다. [`requirements-lock.txt`](../requirements-lock.txt)는 학습 종속성의 버전을 고정합니다.

## 환경과 데이터

GPU 서버에서 저장소를 내려받은 뒤 루트에서 실행합니다. 서버 이미지에 호환 NVIDIA 드라이버, `ffmpeg`, `git`, Python과 Supervisor가 필요합니다.

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/src"
export TORCH_HOME="$PWD/.cache/torch"

python scripts/build_cache.py --config configs/window_balanced.json --workers 4
python scripts/build_direction_targets.py --config configs/window_direction_balanced.json
python scripts/extend_window_cache.py \
  --base configs/window_balanced.json \
  --config configs/window_direction_320_balanced.json
```

부트스트랩은 버전이 고정된 LeRobot을 설치하고, [원본 데이터셋의 고정 리비전](https://huggingface.co/datasets/HYHero/red_pen_adaptive_handover_202611141907/tree/8953cd81be67e3605f3b0ecdcc3e1875464250d6)을 변환합니다. 이미 변환된 동일한 데이터가 있으면 그대로 사용합니다. 환경만 만들 때는 `bash scripts/bootstrap.sh --env-only`를 사용합니다.

이 절차는 새 체크아웃에서 데이터를 재구축하는 흐름입니다. 이미 저장한 optimizer 상태에서 재개할 때는 해당 체크포인트에 기록된 실험 설정, 데이터 경로와 입력 파일 해시를 사용합니다. 공개 주석의 수집자 표기는 `participant_01`로 통일했습니다.

[`balanced_splits.json`](../data/balanced_splits.json)은 고유 학습 에피소드와 검증 에피소드를 서로 분리합니다. 학습에서 재사용하는 에피소드는 `train_slots`에 반복해서 적혀 있습니다. 200ms 캐시에 이 실험의 30 step 행동 청크, 두 카메라, 4채널 EMG와 훈련 분할에서 계산한 정규화 통계를 보관합니다. 320ms 캐시는 기존 카메라와 행동 배열을 공유하며, 더 긴 인과적 EMG 이력이 확보된 행을 사용합니다.

방향 분류에는 [`receiving_annotations.json`](../data/receiving_annotations.json)의 head 프레임 범위를 기록된 수신 시각으로 변환해 적용합니다. 분류 헤드는 영상 결합 전에 200ms 또는 320ms Wave 토큰만 읽습니다. 세 방향의 빈도 역수로 보정한 cross-entropy에 `0.03`을 곱해 기존 ACT 손실에 더합니다. 체크포인트는 방향 헤드를 학습 상태에 따로 저장합니다.

## 서버 서비스와 실행 순서

```bash
python scripts/render_supervisor.py --output .cache/supervisor
sudo install -m 0644 .cache/supervisor/*.conf /etc/supervisor/conf.d/
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start kroc-balanced-uploader-session
```

노트북에서는 SSH 호스트 별칭과 서버의 체크아웃 경로를 지정합니다. Hugging Face에서 대상 저장소에 쓰기 권한이 있는 fine-grained 토큰을 발급해 숨김 프롬프트에 입력합니다. 서버 프로세스가 종료되면 메모리에 보관한 토큰이 해제됩니다.

```bash
python3 scripts/enroll_uploader.py --ssh-target kroc-gpu --remote-root /opt/kroc-emg-server
```

현재 세션의 공개 아카이브와 실행 이름은 [`data/upload_visibility.json`](../data/upload_visibility.json), [`data/upload_runs.json`](../data/upload_runs.json)에 있습니다. 다른 Hugging Face 계정으로 재현할 경우 등록 전에 설정 파일들의 `checkpoint_archive.repo_id`, `upload_visibility.json`과 `training_mirror_session.py`의 `REPO`, `server_uploader_session.py`의 계정 검사를 해당 저장소에 맞춥니다. 이전 실험의 대기열에는 당시의 `private` 플래그가 남아 있으며, 실제 공개 범위는 Hub 저장소와 `upload_visibility.json`으로 확인합니다.

| 순서 | 시작할 Supervisor 프로그램 |
| --- | --- |
| 초기 균형 비교 | `kroc-balanced-window`, `kroc-balanced-spectrogram`, `kroc-continue-window`, `kroc-continue-spectrogram` |
| 200ms 방향 분류 | `kroc-window-direction`, `kroc-continue-window-direction` |
| 320ms 비교 | `kroc-window-320`, `kroc-window-direction-320`, `kroc-continue-window-320`, `kroc-continue-window-direction-320` |
| 원본 Baseline·Concat | `kroc-original-baseline`, `kroc-original-concat` |

각 행의 프로그램은 `sudo supervisorctl start`에 이름을 함께 넘기면 됩니다. 200ms 방향 분류 실행기는 초기 균형 두 실험의 완료를 기다립니다. 320ms 두 실험은 200ms 방향 분류가 실행 중일 때 병행했습니다. 원본 Baseline과 Concat 실행기는 320ms 두 실험이 체크포인트 검증을 마치고 종료할 때까지 대기합니다. 설정 파일은 [`configs/`](../configs/)에 있습니다.

## 종료와 체크포인트 검증

- Window, Spectrogram, Window + 방향 분류 및 320ms 비교의 기본 한도는 100,000 step입니다. 70,001–80,000 step과 90,001–100,000 step의 중앙값을 비교해 학습 ACT loss가 5% 이상, 세 방향 균등 validation loss가 2% 이상 낮아지고 마지막 validation 구간이 직전 구간보다 낮을 때만 해당 실행을 200,000 step까지 연장합니다. 연장에는 100,000 step의 AdamW 상태와 난수 상태를 복원하고 고정 학습률을 유지합니다. 로더는 새 에폭에서 셔플을 시작합니다.
- 원본 설정으로 재학습하는 Baseline과 Concat은 100,000 step 고정입니다. 체크포인트 주기는 이 두 실험에서 10,000 step, 앞선 균형 실험에서 20,000 step입니다. 최종 best는 전체 검증 프레임의 `val_l1_zero_latent`로 선택합니다.
- 업로더는 Git/LFS 파일 무결성을 해당 커밋에서 확인한 뒤 서버 체크포인트를 정리합니다. `StorageLimitReached`가 발생하면 가중치를 보존하며 재시도합니다. CSV, JSONL 및 실험 메타데이터는 진행 중에도 동기화합니다.

서버의 실행 목록과 검증된 업로드를 갱신하려면 아래를 사용합니다.

```bash
python scripts/report_runs.py > reports/experiment_snapshot.json
```
