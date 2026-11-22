"""Bind head-camera receiving annotations to the existing valid cached rows."""
import argparse
import json
from pathlib import Path

import numpy as np

from kroc_act.data import SharedDataset, load_json
from kroc_act.direction import CLASSES, IGNORE, receiving_intervals, target_provenance
from kroc_act.paths import atomic_json, project_path
from kroc_act.train import resolve_config


def label_split(cfg, split, intervals):
    ds = SharedDataset(cfg, split)
    unique, repeated = np.unique(ds.indices, return_inverse=True)
    cache = Path(cfg["frame_cache"]) / split
    metadata = load_json(cache / "cache.json")
    cached_episodes = np.load(cache / metadata["arrays"]["episode_index"], allow_pickle=False)
    cached_state = np.load(cache / metadata["arrays"]["observation.state"], mmap_mode="r", allow_pickle=False)
    if not np.array_equal(repeated, np.load(cache / "sampling_indices.npy", allow_pickle=False)):
        raise ValueError("Cache sampling does not match the original dataset order")
    targets = np.full(len(unique), IGNORE, dtype=np.int64)
    seen = set()
    for index, row_index in enumerate(unique):
        row = ds.base.get_raw_item(int(row_index))
        episode = int(row["episode_index"])
        if episode != cached_episodes[index] or not np.array_equal(cached_state[index], row["observation.state"]):
            raise ValueError(f"Cache row identity mismatch in {split} at {index}")
        origin = int(ds.manifest["episodes"][str(episode)]["robot_origin_ns"])
        decision_ns = origin + round(float(row["timestamp"]) * 1e9)
        start, stop, target = intervals[episode]
        if start <= decision_ns < stop:
            targets[index] = target
            seen.add(episode)
    if seen != ds.episodes:
        raise ValueError(f"Missing eligible labeled {split} frames for episodes {ds.episodes - seen}")
    counts = {name: int(np.count_nonzero(targets[repeated] == idx)) for idx, name in enumerate(CLASSES)}
    summary = {"sampled_frames": len(repeated), "labeled_frames": sum(counts.values()),
               "unique_labeled_frames": int(np.count_nonzero(targets != IGNORE)), "class_frames": counts}
    print(split, json.dumps(summary), flush=True)
    return targets, target_provenance(cfg, cache), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    cfg = resolve_config(parser.parse_args().config)
    intervals = receiving_intervals(cfg)
    payload, proofs, summary = {}, {}, {}
    for split in ("train", "val"):
        payload[split], proofs[split], summary[split] = label_split(cfg, split, intervals)
    path = project_path(Path(cfg["direction_targets"]))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload, provenances=json.dumps(proofs))
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), summary)


if __name__ == "__main__":
    main()
