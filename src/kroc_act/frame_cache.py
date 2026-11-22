"""Read a verified, shared cache of the pinned LeRobot/PyAV training frames."""
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .data import load_json
from .study import sha256_file


class CachedDataset(Dataset):
    def __init__(self, cfg, split):
        root = Path(cfg["frame_cache"]) / split
        meta = load_json(root / "cache.json")
        sources = {key: sha256_file(cfg[key]) for key in ("manifest", "splits", "annotations")}
        if (meta["cache_inputs"] != sources or meta["steps_per_observation"] != cfg["chunk_size"]
                or meta["window_ms"] != cfg["window_ms"]):
            raise ValueError("Cache does not match the configured split or observation window")
        self.manifest = load_json(cfg["manifest"])
        self.features = load_json(Path(cfg["dataset_root"]) / "meta/info.json")["features"]
        self.annotations = load_json(cfg["annotations"])
        self.conditions = self.annotations["episode_conditions"]
        self.camera_keys = meta["camera_keys"]
        self.split_condition_counts = meta["split_condition_counts"]
        self.excluded = meta["excluded"]
        self.statistics = meta.get("statistics")
        self.arrays = {key: np.load(root / file, mmap_mode="r", allow_pickle=False)
                       for key, file in meta["arrays"].items()}
        self.indices = np.load(root / "sampling_indices.npy", allow_pickle=False)
        self.direction_targets = None
        if cfg.get("direction_targets"):
            from .direction import CLASSES, load_targets

            self.direction_targets = load_targets(cfg, split, root, len(self.arrays["episode_index"]))
            self.direction_counts = {name: int(np.count_nonzero(self.direction_targets[self.indices] == index))
                                     for index, name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        item = {key: data[idx].copy() for key, data in self.arrays.items()}
        item["study_condition"] = self.conditions[str(int(item["episode_index"]))]
        if self.direction_targets is not None:
            item["emg_direction_target"] = self.direction_targets[idx]
        return item

    def train_statistics(self):
        return self.statistics
