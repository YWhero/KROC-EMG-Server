"""Convert explicitly aligned intermediate episodes to common v3 + EMG sidecars.

NOT a guessed MCAP/HDF5 decoder: source-schema/clock mapping must be audited first.
See docs/DATA_CONTRACT.md for the exact intermediate schema and responsibility boundary.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Audited aligned/manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrites")
    parser.add_argument("--repo-id", required=True, help="Local metadata name; does not upload")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not source.get("clock_alignment_verified"):
        raise ValueError("Clock alignment must be verified before conversion")
    if not source.get("signal_processing", {}).get("causal"):
        raise ValueError("This experiment requires causal EMG processing")
    if args.output.exists():
        raise FileExistsError(args.output)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .emg import EMGStream

    src_root = args.input.resolve().parent
    def source_path(name):
        p = (src_root / name).resolve()
        if not p.is_relative_to(src_root):
            raise ValueError("Input files must remain inside aligned data directory")
        return p
    records = source["episodes"]
    if len(records) < 2:
        raise ValueError("At least two episodes required for episode-level validation")
    with np.load(source_path(records[0]["robot_file"]), allow_pickle=False) as first:
        features = {
            "observation.state": {"dtype": "float32", "shape": tuple(first["state"].shape[1:]),
                                  "names": source["state_names"]},
            "action": {"dtype": "float32", "shape": tuple(first["action"].shape[1:]),
                       "names": source["action_names"]}}
        for camera, shape in source["camera_shapes"].items():
            features[camera] = {"dtype": "video", "shape": tuple(shape),
                                "names": ["channels", "height", "width"]}
    args.output.mkdir(parents=True)
    emg_root = args.output / "emg"
    emg_root.mkdir()
    dataset = LeRobotDataset.create(args.repo_id, fps=source["fps"], features=features,
                                    root=args.output / "lerobot_v3", use_videos=True,
                                    robot_type=source["robot_type"])
    emg_manifest = {"schema": "kroc-emg-v1", "clock_alignment_verified": True,
                    "channels": source["channels"], "sample_rate_hz": source["sample_rate_hz"],
                    "signal_processing": source["signal_processing"], "episodes": {}}
    # Sequential IDs match LeRobot writer episode_index; source IDs stay in provenance.
    try:
        for ep, record in enumerate(records):
            with np.load(source_path(record["robot_file"]), allow_pickle=False) as robot:
                state, action = robot["state"], robot["action"]
                n = len(state)
                if not n or len(action) != n or not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError("Invalid robot episode arrays")
                expected = np.arange(n) / source["fps"]
                if not np.allclose(robot["timestamp"], expected, atol=1e-6, rtol=0):
                    raise ValueError("Robot/camera rows must already be aligned to the declared FPS")
                for key, shape in source["camera_shapes"].items():
                    if robot[key].shape != (n, *shape) or robot[key].dtype != np.uint8:
                        raise ValueError(f"Expected aligned uint8 RGB CHW images: {key}")
                for i in range(n):
                    dataset.add_frame({"observation.state": state[i].astype(np.float32),
                                       "action": action[i].astype(np.float32),
                                       **{k: robot[k][i] for k in source["camera_shapes"]},
                                       "task": record["task"]})
                dataset.save_episode()
            with np.load(source_path(record["emg_file"]), allow_pickle=False) as raw:
                stream = EMGStream(*(raw[k] for k in ["time_ns", "available_ns", "values", "valid"]))
                if stream.values.shape[1] != len(source["channels"]):
                    raise ValueError("EMG channel order/count mismatch")
                np.savez_compressed(emg_root / f"episode_{ep:06d}.npz",
                                    time_ns=stream.time_ns, available_ns=stream.available_ns,
                                    values=stream.values, valid=stream.valid)
            emg_manifest["episodes"][str(ep)] = {
                "file": f"episode_{ep:06d}.npz", "robot_origin_ns": record["robot_origin_ns"],
                "source_episode": record["source_episode"]}
        (emg_root / "manifest.json").write_text(json.dumps(emg_manifest, indent=2), encoding="utf-8")
    finally:
        dataset.finalize()
    # Fixed split is shared by ALL variants. Replace deliberately before any runs.
    rng = np.random.default_rng(1000)
    ids = rng.permutation(len(records)).tolist()
    n_val = max(1, round(len(ids) * 0.2))
    splits = {"train": sorted(ids[n_val:]), "val": sorted(ids[:n_val])}
    (args.output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (args.output / "source_manifest.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Prepared shared dataset at {args.output}; no training or upload")


if __name__ == "__main__":
    main()
