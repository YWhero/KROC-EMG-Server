"""Cache existing KROC targets, valid windows, and lossless uint8 camera tensors."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from kroc_act.data import SharedDataset
from kroc_act.paths import atomic_json, project_path
from kroc_act.study import sha256_file
from kroc_act.train import resolve_config


def cache_split(cfg, root, split, workers):
    dataset = SharedDataset(cfg, split)
    unique, repeated = np.unique(dataset.indices, return_inverse=True)
    lookup = {int(raw): idx for idx, raw in reversed(list(enumerate(dataset.indices)))}
    loader = DataLoader(Subset(dataset, [lookup[int(raw)] for raw in unique]),
                        batch_size=32, num_workers=workers, shuffle=False)
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / "sampling_indices.npy", repeated.astype(np.int32), allow_pickle=False)
    arrays, paths = {}, {}
    start = 0
    for batch in loader:
        batch.pop("study_condition")
        size = len(batch["episode_index"])
        for key, values in batch.items():
            array = values.numpy()
            if key in dataset.camera_keys and array.dtype != np.uint8:
                quantized = np.rint(array * 255).astype(np.uint8)
                if not np.allclose(array, quantized.astype(np.float32)/255, atol=1e-7, rtol=0):
                    raise ValueError("Unexpected non-uint8-equivalent camera values")
                array = quantized
            if key not in arrays:
                paths[key] = f"{key}.npy"
                arrays[key] = np.lib.format.open_memmap(root / paths[key], mode="w+",
                            dtype=array.dtype, shape=(len(unique), *array.shape[1:]))
            arrays[key][start:start+size] = array
        start += size
        if start % 2048 == 0 or start == len(unique):
            print(f"{split}: cached {start}/{len(unique)} unique frames", flush=True)
    for array in arrays.values():
        array.flush()
    atomic_json(root / "cache.json", {
        "arrays": paths, "camera_keys": dataset.camera_keys,
        "split_condition_counts": dataset.split_condition_counts, "excluded": dataset.excluded,
        "statistics": dataset.train_statistics() if split == "train" else None,
        "cache_inputs": {key: sha256_file(cfg[key]) for key in ("manifest", "splits", "annotations")},
        "steps_per_observation": cfg["chunk_size"], "window_ms": cfg["window_ms"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = resolve_config(args.config)
    root = project_path(cfg["frame_cache"])
    stage = root.with_name(root.name + ".building")
    for split in ("train", "val"):
        cache_split(cfg, stage / split, split, args.workers)
    if root.exists():
        raise FileExistsError(root)
    stage.replace(root)


if __name__ == "__main__":
    main()
