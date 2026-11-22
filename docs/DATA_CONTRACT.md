# 원시 데이터와 공통 학습 데이터의 계약

2026-11-15 실제 원본 감사 및 전체 변환 완료. 현재 원본에는 `convert_raw.py`를 사용합니다.
아래 aligned intermediate 규약은 다른 수집 데이터를 연결할 때의 대안 경로입니다.
실제 적용 정책: 30Hz, RGB 320×180, 카메라/로봇 수신시각 기준 과거값 선택,
EMG scan-end 시각의 host clock 변환과 sample availability 적용.
host clock 일관성은 확인했지만 MCU↔host mapping의 추정 오차를 제거한 것은 아닙니다.

Hugging Face 원본은 MCAP + 별도 MP4 + EMG HDF5/manifest입니다.
원본을 보존하고 다음 구조를 한 번 생성합니다.

```text
data/
  lerobot_v3/               # 표준 v3.0; observation.state, action, 카메라
    meta/info.json
    meta/stats.json
    data/...
    videos/...
  emg/
    manifest.json
    episode_000000.npz      # 고주파 신호; 윈도우를 프레임마다 복제하지 않음
  splits.json               # 세 모델 공통 에피소드 분할
```

EMG summary는 로더에서 동일한 규칙으로 계산하므로 v3 Parquet에 별도 저장할 필요가 없습니다.
추후 성능상 캐시를 만든다면 전처리/시계/윈도우 설정 해시와 함께 보관해야 합니다.

## EMG sidecar (프로젝트 확장 형식; 표준 LeRobot 파일이 아님)

`manifest.json` 예시의 숫자/파일은 스키마 설명용이며 실제 데이터가 아닙니다.

```json
{
  "schema": "kroc-emg-v1",
  "clock_alignment_verified": true,
  "sample_rate_hz": 2000,
  "channels": ["A0", "A1", "A2", "A3"],
  "signal_processing": {"causal": true, "variant": "bandpass_notch", "units": "ADC counts"},
  "episodes": {
    "0": {
      "file": "episode_000000.npz",
      "robot_origin_ns": "1794651015784711907",
      "source_episode": "0"
    }
  }
}
```

各 NPZ は `allow_pickle=False` で読める以下の配列を持ちます。

| 配列 | shape / dtype | 意味 |
|---|---|---|
| `time_ns` | `[M] int64` | ロボットと同じ時計に変換した取得時刻。厳密昇順 |
| `available_ns` | `[M] int64` | そのサンプルを推論側で利用可能になった時刻。同じ時計 |
| `values` | `[M,4] float32` | 同一の causal 前処理をした信号。未整流の bandpass/notch 信号を想定 |
| `valid` | `[M] bool` | warmup/欠損/品質問題を除いた有効サンプル |

`robot_origin_ns + round(row.timestamp * 1e9)` を各 v3 フレームの絶対時刻とします。
`robot_origin_ns` は **変換後の最初のフレーム**に対応する時刻です。
bag 開始と変換後フレーム開始が違う場合に bag 開始をそのままコピーしてはいけません。
epoch ns を float に変換して差を求めず、int64のまま扱います。
`clock_alignment_verified=true` は開発者が実測/解析で確認して初めて設定してください。
原本の `clock_mapping_is_estimated=true` をこの値へ機械的に読み替えてはいけません。

## 원본 → aligned intermediate를 사용하는 경우의 절차

1. MCAPの関節名順、state位置単位、action指令の意味を確認する。
   actionは `/left/leader/joint_trajectory` の目標。follower stateをactionとして代用しない。
2. HDF5内部の取得/到着時刻、raw/derived配列、valid/warmup maskを確認する。
   EMGは原則としてMCAPの `/emg/status` から作らない。
3. ロボット・画像・EMGのclock mapping、オフセット、ドリフト、可用時刻を検証する。
4. 学習FPS（暫定30などを無断で固定せず実際の動画と制御設計から決定）で画像/state/actionを揃える。
   入力state/画像は未来参照しない。action targetの時刻規約も明記する。
5. causalityを守った同一bandpass/notch信号とvalid maskを保存する。
   既存derivedを使う際も実際のvariant名とフィルタ初期状態/遅延を確認する。
6. 下記intermediateを作り、`prepare`で標準v3とsidecarを生成する。

manifest上の `electrodes_attached_declared=false` や `signal_quality=not_validated` は
筋電信号の有効性を保証していないため、信号の分布・飽和・動作との対応も別途確認する。

## prepare.py가 받는 aligned intermediate

`aligned/manifest.json`:

```json
{
  "clock_alignment_verified": true,
  "fps": 30,
  "robot_type": "omx_ai_left_emg",
  "sample_rate_hz": 2000,
  "channels": ["A0", "A1", "A2", "A3"],
  "state_names": ["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_gripper_joint_1"],
  "action_names": ["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_gripper_joint_1"],
  "camera_shapes": {"observation.images.top": [3,224,224], "observation.images.wrist": [3,224,224]},
  "signal_processing": {"causal": true, "variant": "bandpass_notch", "units": "ADC counts"},
  "episodes": [
    {"source_episode": "0", "robot_file": "robot_000000.npz", "emg_file": "emg_000000.npz", "robot_origin_ns": "1794651015784711907", "task": "Pick up the red pen and hand it to the person"},
    {"source_episode": "1", "robot_file": "robot_000001.npz", "emg_file": "emg_000001.npz", "robot_origin_ns": "1794651115784711907", "task": "Pick up the red pen and hand it to the person"}
  ]
}
```

この例のFPS・画像解像度・時刻は説明用です。実データを調べて確定してください。
robot NPZ: `timestamp` `[T]`（0から1/fps間隔）、`state` `[T,6]`、`action` `[T,6]`、
カメラ名ごとのRGB uint8 `[T,3,H,W]`。EMG NPZは上記sidecarと同じ配列です。
圧縮NPZへの画像展開は検証しやすさを優先した中間形式で、大規模変換では
動画を逐次デコードするreaderへ置き換えることを推奨します。

```bash
.venv/bin/python -m kroc_act.prepare --input aligned/manifest.json \
  --output data --repo-id HYHero/red_pen_adaptive_handover_kroc_v3
```

新しい `data` フォルダにだけ書き込みます。既存フォルダには上書きしません。
20%のvalidationをseed 1000でエピソード単位に割り当てます。
試行/被験者/セッションが重なるデータでは、その単位で分割し直してから全実験を開始してください。
`repo-id` はローカルの識別名であり、Hugging Faceへの作成・アップロードはしません。
