"""Common v3 rows + high-rate EMG sidecars, shared exclusions and train statistics."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .emg import EMGStream, rms
from .paths import project_path
from .study import validate_study


def load_json(path):
    return json.loads(project_path(path).read_text(encoding="utf-8"))


class SharedDataset(Dataset):
    def __init__(self, cfg, split: str):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        self.manifest = load_json(cfg["manifest"])
        self.root = Path(cfg["manifest"]).resolve().parent
        if self.manifest.get("schema") != "kroc-emg-v1":
            raise ValueError("Expected kroc-emg-v1 manifest")
        if not self.manifest.get("clock_alignment_verified", False):
            raise ValueError("Verify EMG/robot clock mapping before training")
        if not self.manifest.get("signal_processing", {}).get("causal", False):
            raise ValueError("Causal EMG signal processing must be declared")
        self.rate = int(self.manifest["sample_rate_hz"])
        if self.rate != cfg["sample_rate_hz"]:
            raise ValueError("EMG rate differs from experiment/inference configuration")
        self.samples = round(cfg["window_ms"] * self.rate / 1000)
        self.rms_samples = round(cfg["rms_ms"] * self.rate / 1000)
        if not 0 < self.rms_samples <= self.samples:
            raise ValueError("RMS interval must fit inside EMG window")
        splits = load_json(cfg["splits"])
        train, val = splits["train"], splits["val"]
        if not train or not val or set(train) & set(val):
            raise ValueError("Nonempty disjoint train/val episode lists required")
        if len(train) != len(set(train)) or len(val) != len(set(val)):
            raise ValueError("Duplicate episodes in split")
        self.episodes = set(splits[split])
        multiplicity = Counter(splits.get("train_slots", train) if split == "train" else val)
        self.annotations = load_json(cfg["annotations"]) if cfg.get("annotations") else None
        if self.annotations:
            self.split_condition_counts = validate_study(self.annotations, splits, self.manifest["episodes"])
        info = load_json(Path(cfg["dataset_root"]) / "meta/info.json")
        if info["codebase_version"] != "v3.0":
            raise ValueError("Expected LeRobotDataset v3.0")
        self.features, self.fps = info["features"], info["fps"]
        self.camera_keys = sorted(k for k in self.features if k.startswith("observation.images."))
        if not self.camera_keys:
            raise ValueError("These experiments require the camera inputs")
        self.base = LeRobotDataset(
            cfg["repo_id"], root=cfg["dataset_root"],
            delta_timestamps={"action": [i / self.fps for i in range(cfg["chunk_size"])]},
            video_backend=cfg.get("video_backend", "pyav"))
        self.streams = {}
        self.indices = []
        self.excluded = 0
        # Raw metadata access avoids decoding every image while checking alignment.
        for i in range(len(self.base)):
            row = self.base.get_raw_item(i)
            ep = int(row["episode_index"])
            if ep not in self.episodes:
                continue
            try:
                self.emg_for_row(row)
            except ValueError:
                self.excluded += 1
                continue
            self.indices.extend([i] * multiplicity[ep])
        if not self.indices:
            raise ValueError(f"No valid frames for {split}; inspect clocks, warmup and splits")

    def stream(self, ep):
        if ep not in self.streams:
            entry = self.manifest["episodes"][str(ep)]
            path = (self.root / entry["file"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Sidecar path must remain under manifest directory")
            with np.load(path, allow_pickle=False) as data:
                self.streams[ep] = EMGStream(*(data[k] for k in
                    ["time_ns", "available_ns", "values", "valid"]))
            if self.streams[ep].values.shape[1] != len(self.manifest["channels"]):
                raise ValueError("Channel count mismatch")
        return self.streams[ep]

    def emg_for_row(self, row):
        ep = int(row["episode_index"])
        origin = int(self.manifest["episodes"][str(ep)]["robot_origin_ns"])
        now = origin + round(float(row["timestamp"]) * 1e9)
        window = self.stream(ep).window(now, self.samples, self.rate, self.cfg["max_gap_ms"],
                                       self.cfg.get("max_emg_age_ms", 100.0))
        return window, rms(window, self.rms_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = dict(self.base[self.indices[index]])
        window, summary = self.emg_for_row(self.base.get_raw_item(self.indices[index]))
        result = {**{k: item[k] for k in ["observation.state", "action", "action_is_pad", *self.camera_keys]},
                "observation.emg_window": torch.from_numpy(window),
                "observation.emg": torch.from_numpy(summary)}
        if self.annotations:
            ep = int(self.base.get_raw_item(self.indices[index])["episode_index"])
            result.update(episode_index=torch.tensor(ep), study_condition=self.annotations["episode_conditions"][str(ep)])
        return result

    def train_statistics(self):
        """Fit only selected train frames, using online moments to bound memory."""
        sums, squares, counts = {}, {}, {}
        def add(key, x):
            x = np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1])
            sums[key] = sums.get(key, 0) + x.sum(0)
            squares[key] = squares.get(key, 0) + (x * x).sum(0)
            counts[key] = counts.get(key, 0) + len(x)
        for i in self.indices:
            row = self.base.get_raw_item(i)
            window, summary = self.emg_for_row(row)
            add("state", row["observation.state"])
            add("action", row["action"])
            add("emg", summary)
            add("window", window)
        result = {}
        for key in sums:
            mean = sums[key] / counts[key]
            var = np.maximum(squares[key] / counts[key] - mean**2, 0)
            result[key] = {"mean": mean.tolist(), "std": np.maximum(np.sqrt(var), 1e-6).tolist()}
        return result
