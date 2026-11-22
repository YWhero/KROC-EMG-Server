# GPU 추론과 노트북 Cyclo 연결

배포한 서비스는 원본 Baseline·Concat·Window의 100,000-step 모델을 한 GPU 프로세스에 올립니다. 노트북의 Cyclo는 관절과 두 카메라를 읽고, EMG 모델에는 같은 노트북에서 구성한 과거 200ms 파형도 함께 보냅니다. GPU는 30 step 청크를 예측해 앞 15 step을 전달하고, Cyclo의 기존 런타임이 미리보기 또는 로봇 제어를 담당합니다. 320ms와 균형 재학습 체크포인트는 [실험 아카이브](https://huggingface.co/HYHero/kroc-act-emg-balanced-20261123)에 있으며, 문서화한 활성 RPC 계약은 원본 200ms 모델용입니다.

## GPU 서버

```bash
bash scripts/bootstrap.sh --env-only
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/src"
python scripts/fetch_inference_models.py
python -m kroc_remote.server --checkpoints checkpoints
```

[`inference_models.json`](../configs/inference_models.json)은 공개 저장소의 고정 리비전, 경로, 크기와 SHA-256을 기록합니다. 다운로드 스크립트는 파일 검증을 거친 뒤 각 모드의 체크포인트를 `checkpoints/`에 배치합니다. Supervisor로 실행할 때는 [서비스 정의 생성 절차](TRAINING.md#서버-서비스와-실행-순서)를 사용한 뒤 `sudo supervisorctl start kroc-inference`를 실행합니다.

서버는 `127.0.0.1:17665`에 바인딩합니다. `GET /v1/models`가 모드와 입력 계약을 제공하고, `POST /v1/actions/{baseline,concat,window}`가 little-endian float32 행동 청크를 반환합니다. 이진 요청은 지속 HTTP/1.1 연결을 사용합니다. 노트북에서 BGR 카메라를 `320×180`으로 줄여 기본 JPEG 품질 90으로 전송합니다. 6개 왼팔 관절의 순서는 [`wire.py`](../kroc_remote/wire.py)에 있습니다. Concat은 같은 2kHz 파형의 끝 50ms RMS 4개를 사용합니다.

## 노트북

OpenSSH에 접속 계정과 서버 주소가 들어 있는 `kroc-gpu` 별칭을 설정합니다. 다음처럼 로컬 loopback에 터널을 엽니다.

```bash
ssh -NT -o ExitOnForwardFailure=yes -L 127.0.0.1:17665:127.0.0.1:17665 kroc-gpu
curl -fsS http://127.0.0.1:17665/v1/models
```

패치는 [ROBOTIS Cyclo Intelligence](https://github.com/ROBOTIS-GIT/cyclo_intelligence)의 `e648425e1f724ba61a8060369e200bd3e0f2a877`을 기준으로 합니다. 해당 버전에서 아래 명령으로 런타임 변경분, 원격 클라이언트와 로봇 프로필을 설치하고, 사용하는 Cyclo 브레인 환경을 다시 빌드합니다.

```bash
bash integrations/cyclo/apply.sh /path/to/cyclo_intelligence
```

Docker를 쓰는 Cyclo 브레인은 호스트 네트워크에서 로컬 SSH 포워드를 읽습니다. 적용한 왼팔 프로필의 카메라와 ROS 토픽은 [프로필 YAML](../integrations/cyclo/omx_ai_left_emg_config.yaml)에 있습니다. URDF 미리보기에 필요한 메시 파일은 ROS의 `open_manipulator_description` 패키지가 제공합니다.

EMG 모드는 기존 4채널 RAW 수집 서비스의 `emg_acq.processing.Processor`와 연결합니다. 수집 컨테이너와 Cyclo 컨테이너에 같은 작업 디렉터리를 `/workspace`로 마운트하고, 수집 서비스에 `CYCLO_KROC_EMG_WINDOW=/workspace/kroc_emg/window.bin`을 설정합니다. [`window_publisher.py`](../integrations/cyclo/window_publisher.py)와 `src/kroc_act/emg.py`를 수집기 Python 경로에 제공합니다. 수집 서비스는 마운트된 파일로 최신 유효 파형을 교체합니다. 파일 형식은 magic `KEM1`, monotonic ns 타임스탬프, `[400, 4]` little-endian float32입니다. 기존 수집기의 notch warm-up과 누락 처리 이후 약 3.5초와 200ms 이력이 필요합니다.

Cyclo Inference에서 `omx_ai_left_emg`, LeRobot, Dataset FPS 30, Async를 선택한 뒤 Policy Path에 `kroc://baseline`, `kroc://concat`, `kroc://window` 중 하나를 입력합니다. Simulation 모드에서 먼저 확인할 수 있습니다. 센서 타임스탬프가 100ms를 넘거나 카메라에서 응답까지 150ms를 넘으면 요청을 거절합니다. 관측을 거절하면 런타임은 해당 요청의 남은 행동 버퍼를 비웁니다.

## 로봇 연결 없이 테스트

노트북의 Python 환경에 NumPy와 OpenCV가 있어야 합니다. 서버 또는 터널 양쪽에서 합성 관측을 보낼 수 있습니다.

```bash
PYTHONPATH="$PWD" python3 scripts/benchmark_inference.py --samples 10 --codec both
```

벤치마크가 요청 크기, 왕복 시간과 GPU 계산 시간을 모드별로 출력합니다. 기존 합성 측정의 요청 크기는 JPEG 약 69–75kB, raw 약 346–352kB였습니다.

게시 전 검증에서 공개 체크포인트 세 개의 SHA-256을 확인했습니다. 서버 loopback에서 모델별 합성 JPEG 요청을 두 번씩 실행해 세 모드 모두 유효한 행동 청크를 반환했습니다. Loopback 수치는 노트북에서 SSH 터널을 거치는 지연과 구분합니다.
