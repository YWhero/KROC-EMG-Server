"""Audited Cyclo MCAP/MP4/EMG -> common LeRobot v3, without giant image intermediates."""
import argparse
import hashlib
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from .emg import EMGStream


JOINTS = [f"left_joint{i}" for i in range(1, 6)] + ["left_gripper_joint_1"]


def read_robot(episode):
    streams = {"state": [], "action": []}
    with AnyReader([episode], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        connections = [c for c in reader.connections if c.topic in
                       ["/left/joint_states", "/left/leader/joint_trajectory"]]
        for connection, timestamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic == "/left/joint_states":
                names, values, key = message.name, message.position, "state"
            else:
                if len(message.points) != 1:
                    raise ValueError("Expected one commanded trajectory point")
                names, values, key = message.joint_names, message.points[0].positions, "action"
            values = np.asarray(values)[[names.index(k) for k in JOINTS]].astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite robot position/command")
            streams[key].append((int(timestamp), values))
    return {k: (np.array([r[0] for r in rows], dtype=np.int64), np.stack([r[1] for r in rows]))
            for k, rows in streams.items()}


class Camera:
    def __init__(self, sidecar, width, height):
        data = pd.read_parquet(sidecar)
        if not np.array_equal(data.frame_index, np.arange(len(data))):
            raise ValueError("Nonsequential video sidecar frame indices")
        self.times = data.recv_ns.to_numpy(dtype=np.int64)
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Video receive times must increase")
        self.path = sidecar.with_name(sidecar.name.replace("_timestamps.parquet", ".mp4"))
        self.container = av.open(str(self.path))
        self.decoder = iter(self.container.decode(video=0))
        self.index = -1
        self.frame = None
        self.width, self.height = width, height
        self.delay_ms = (data.recv_ns.to_numpy()-data.header_stamp_ns.to_numpy()) / 1e6

    def at(self, now):
        index = int(np.searchsorted(self.times, now, side="right")) - 1
        if index < 0 or now - self.times[index] > 100_000_000:
            raise ValueError("Missing or stale causal camera frame")
        while self.index < index:
            frame = next(self.decoder)
            self.index += 1
            if self.index == index:
                image = frame.reformat(width=self.width, height=self.height, format="rgb24").to_ndarray()
                self.frame = image.transpose(2, 0, 1).copy()
        return self.frame

    def close(self):
        self.container.close()


def read_emg(episode):
    folder = episode / "emg/segments/0"
    manifest = json.loads((folder / "manifest.json").read_text())
    processing = json.loads((folder / "derived/emg_all_variants_v1.h5.json").read_text())
    if not manifest["same_kernel_clock_domain"] or not processing["settings"]["causal"]:
        raise ValueError("Shared host clock / causal processing is required")
    if len(list((episode / "emg/segments").iterdir())) != 1:
        raise ValueError("Multiple EMG segments need explicit stitching")
    source = folder / "derived/emg_all_variants_v1.h5"
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if checksum != manifest["derived_sha256"]:
        raise ValueError("Derived EMG checksum mismatch")
    with h5py.File(source, "r") as f:
        receive = f["native/host_receive_ns"][:]
        wall = f["native/host_receive_system_ns"][:]
        # Map scan start into the same wall clock as bag and camera receipt times.
        # Timestamp each 4-channel vector at END of scan, conservatively causal.
        time = f["native/mapped_host_ns"][:] + (wall-receive) + f["native/scan_duration_us"][:].astype(np.int64)*1000
        values = f["processed/bandpass_notch60_counts"][:]
        valid = f["quality/bandpass_warmup_valid"][:] & f["quality/notch_warmup_valid"][:]
        valid &= ~f["quality/gap_before"][:]
        uncertainty = f["native/clock_uncertainty_ns"][:]
    keep = np.zeros(len(time), dtype=bool)
    last = -1
    for i, t in enumerate(time):
        if t > last and t <= wall[i]:
            keep[i] = True
            last = t
    stream = EMGStream(time[keep], wall[keep], values[keep], valid[keep])
    report = {"raw_samples": len(time), "retained_samples": int(keep.sum()),
              "dropped_clock_order_or_noncausal_samples": int((~keep).sum()),
              "clock_mapping_is_estimated": manifest["clock_mapping_is_estimated"],
              "clock_uncertainty_ns_max": int(uncertainty.max()),
              "availability_delay_ms_p50_p99_max": np.percentile((wall[keep]-time[keep])/1e6, [50,99,100]).tolist(),
              "signal_quality": processing["signal_quality"], "source_sha256": checksum}
    return stream, processing["settings"], report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="dorii22/red_pen_adaptive_handover_lerobot")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args.output.mkdir(parents=True)
    emg_root = args.output / "emg"
    emg_root.mkdir()
    camera_keys = ["observation.images.top", "observation.images.wrist"]
    features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
                "action": {"dtype": "float32", "shape": (6,), "names": JOINTS}}
    features.update({key: {"dtype": "video", "shape": (3,args.height,args.width),
                           "names": ["channels","height","width"]} for key in camera_keys})
    dataset = LeRobotDataset.create(args.repo_id, args.fps, features, root=args.output / "lerobot_v3",
                                    robot_type="omx_ai_left_emg", video_backend="pyav",
                                    rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast", g=2),
                                    streaming_encoding=True, encoder_threads=2)
    output_manifest = {"schema": "kroc-emg-v1", "clock_alignment_verified": True,
                       "clock_alignment_scope": "Recorded host clock consistency and causal receipt alignment; device mapping remains estimated",
                       "sample_rate_hz": 2000, "channels": ["A0","A1","A2","A3"], "episodes": {}}
    report = {"fps": args.fps, "image_shape": [3,args.height,args.width],
              "camera_timestamp": "recv_ns; latest received image at decision time",
              "state_action_timestamp": "MCAP record timestamp; latest command and state at decision time",
              "emg_time": "mapped_host_ns + receive wall/monotonic offset + scan_duration_us",
              "emg_availability": "host_receive_system_ns; derived computation cost not included",
              "episodes": {}}
    episodes = sorted((p for p in args.raw.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    try:
        for ep, path in enumerate(episodes):
            stream, processing, emg_report = read_emg(path)
            robot = read_robot(path)
            camera_paths = [next(path.glob(f"videos/*/{name}_timestamps.parquet")) for name in ["top","wrist"]]
            cameras = [Camera(p,args.width,args.height) for p in camera_paths]
            origin = max(robot["state"][0][0],robot["action"][0][0],*(c.times[0] for c in cameras))
            stop = min(robot["state"][0][-1],robot["action"][0][-1],*(c.times[-1] for c in cameras))
            decisions = origin + np.rint(np.arange(int((stop-origin)*args.fps//1_000_000_000)+1)
                                         * (1e9/args.fps)).astype(np.int64)
            info = json.loads((path / "episode_info.json").read_text())
            max_robot_age = 0
            try:
                for now in decisions:
                    frame = {"task": info["task_instruction"]}
                    for key in ["state","action"]:
                        times, values = robot[key]
                        index = int(np.searchsorted(times,now,side="right"))-1
                        age = int(now-times[index])
                        max_robot_age = max(max_robot_age,age)
                        if index < 0 or age > 100_000_000:
                            raise ValueError("Missing/stale robot input")
                        frame["observation.state" if key=="state" else "action"] = values[index].copy()
                    for key,camera in zip(camera_keys,cameras,strict=True):
                        frame[key] = camera.at(now)
                    dataset.add_frame(frame)
                dataset.save_episode()
            finally:
                for camera in cameras:
                    camera.close()
            np.savez_compressed(emg_root / f"episode_{ep:06d}.npz", time_ns=stream.time_ns,
                                available_ns=stream.available_ns, values=stream.values, valid=stream.valid)
            output_manifest["episodes"][str(ep)] = {"file": f"episode_{ep:06d}.npz",
                "robot_origin_ns": str(origin), "source_episode": path.name}
            output_manifest["signal_processing"] = {**processing, "variant": "bandpass_notch60_counts",
                                                     "filter_reset": "episode boundaries, as provided by source derived artifact"}
            report["episodes"][str(ep)] = {"source_episode": path.name,"frames":len(decisions),
                "robot_origin_ns":str(origin),"max_robot_age_ms":max_robot_age/1e6,
                "camera_delay_ms_p50": {k:float(np.median(c.delay_ms)) for k,c in zip(camera_keys,cameras)},
                "emg":emg_report}
            print(f"episode {ep}: {len(decisions)} frames, {len(stream.time_ns)} EMG samples",flush=True)
    finally:
        dataset.finalize()
    (emg_root / "manifest.json").write_text(json.dumps(output_manifest,indent=2))
    ids = np.random.default_rng(1000).permutation(len(episodes)).tolist()
    n_val = max(1,round(len(ids)*0.2))
    (args.output / "splits.json").write_text(json.dumps({"train":sorted(ids[n_val:]),"val":sorted(ids[:n_val])},indent=2))
    (args.output / "conversion_report.json").write_text(json.dumps(report,indent=2))
    print("Conversion complete. No Hub upload or training.",flush=True)


if __name__ == "__main__":
    main()
