"""Extend the existing causal Wave history while retaining aligned camera/action rows."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from kroc_act.data import SharedDataset, load_json
from kroc_act.direction import CLASSES, IGNORE, receiving_intervals, target_provenance
from kroc_act.paths import atomic_json, project_path
from kroc_act.train import resolve_config


WAVEFORM = "observation.emg_window"


def extend_split(base, version, split, output, intervals):
    dataset = SharedDataset(base, split)
    source = Path(base["frame_cache"]) / split
    metadata = load_json(source / "cache.json")
    prior_indices, prior_sampling = np.unique(dataset.indices, return_inverse=True)
    if not np.array_equal(prior_sampling, np.load(source / "sampling_indices.npy", allow_pickle=False)):
        raise ValueError(f"The {split} source rows no longer match the original Wave cache")
    paths = metadata["arrays"]
    old_wave = np.load(source / paths[WAVEFORM], mmap_mode="r", allow_pickle=False)
    episodes = np.load(source / paths["episode_index"], allow_pickle=False)
    if len(episodes) != len(prior_indices) or old_wave.shape[:2] != (len(prior_indices), dataset.samples):
        raise ValueError(f"The {split} source cache shape differs from its raw rows")
    output.mkdir()
    for name, filename in paths.items():
        if name != WAVEFORM:
            os.link(project_path(source / filename), output / filename)
    old_samples = dataset.samples
    dataset.samples = round(version["window_ms"] * dataset.rate / 1000)
    shape = (len(prior_indices), dataset.samples, old_wave.shape[-1])
    wave = np.lib.format.open_memmap(output / paths[WAVEFORM], mode="w+", dtype=old_wave.dtype, shape=shape)
    valid = np.zeros(len(prior_indices), dtype=bool)
    labels = np.full(len(prior_indices), IGNORE, dtype=np.int64)
    for index, raw_id in enumerate(prior_indices):
        row = dataset.base.get_raw_item(int(raw_id))
        episode = int(row["episode_index"])
        if episode != int(episodes[index]):
            raise ValueError(f"Raw episode order differs from cache at {split} row {index}")
        try:
            longer, _ = dataset.emg_for_row(row)
        except ValueError:
            continue
        if not np.array_equal(longer[-old_samples:], old_wave[index]):
            raise ValueError(f"Causal Wave tail changed at {split} row {index}")
        valid[index] = True
        wave[index] = longer
        origin = int(dataset.manifest["episodes"][str(episode)]["robot_origin_ns"])
        stamp = origin + round(float(row["timestamp"]) * 1e9)
        start, stop, label = intervals[episode]
        if start <= stamp < stop:
            labels[index] = label
    wave.flush()
    sampling = prior_sampling[valid[prior_sampling]]
    if set(episodes[labels >= 0]) != dataset.episodes or set(episodes[sampling]) != dataset.episodes:
        raise ValueError(f"The {split} dataset lost an entire episode or all its classified frames")
    np.save(output / "sampling_indices.npy", sampling.astype(np.int32), allow_pickle=False)
    metadata.update(window_ms=version["window_ms"], excluded=dataset.excluded + int((~valid).sum()))
    atomic_json(output / "cache.json", metadata)
    counts = {name: int(np.count_nonzero(labels[sampling] == i)) for i, name in enumerate(CLASSES)}
    summary = {"original_sampled_frames": len(prior_sampling), "retained_sampled_frames": len(sampling),
               "original_unique_frames": len(prior_indices), "retained_unique_frames": int(valid.sum()),
               "labeled_sampled_frames": sum(counts.values()), "labeled_frames_by_class": counts}
    print(split, json.dumps(summary), flush=True)
    return labels, target_provenance(version, output), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    base, version = resolve_config(args.base), resolve_config(args.config)
    if not version["window_ms"] > base["window_ms"] or any(
        base[key] != version[key] for key in ("dataset_root", "manifest", "splits", "annotations", "sample_rate_hz",
                                             "chunk_size", "max_gap_ms", "max_emg_age_ms", "rms_ms")
    ):
        raise ValueError("The extended cache must keep the existing dataset and causal sampling rules")
    cache = project_path(version["frame_cache"])
    targets = project_path(version["direction_targets"])
    if cache.exists() or targets.exists():
        raise FileExistsError("The extended Wave cache or targets already exist")
    intervals = receiving_intervals(version)
    rows, provenance, summary = {}, {}, {}
    with TemporaryDirectory(prefix=".new-wave-cache-", dir=cache.parent) as temp:
        work = Path(temp)
        stage = work / "cache"
        stage.mkdir()
        for split in ("train", "val"):
            rows[split], provenance[split], summary[split] = extend_split(base, version, split, stage / split, intervals)
        sidecar = work / "targets.npz"
        np.savez_compressed(sidecar, **rows, provenances=json.dumps(provenance))
        stage.replace(cache)
        sidecar.replace(targets)
    atomic_json(cache / "alignment.json", {
        "source_window_ms": base["window_ms"], "window_ms": version["window_ms"], "splits": summary,
        "verified": "Exact equality of retained 200ms tails. Original camera, action, phase labels, splits and normalization.",
    })


if __name__ == "__main__":
    main()
