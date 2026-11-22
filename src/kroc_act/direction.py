"""Direction targets for the visually annotated receiving interval."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_json
from .paths import PROJECT_ROOT, project_path
from .study import sha256_file


CLASSES = ("as_is", "wrist_left", "wrist_right")
IGNORE = -1


def receiving_intervals(cfg):
    import pyarrow.parquet as pq

    annotations = load_json(cfg["receiving_annotations"])
    study = load_json(cfg["annotations"])
    if (annotations["schema"] != "kroc-receive-phase-v1"
            or tuple(annotations["class_order"]) != CLASSES
            or annotations["excluded_class_index"] != IGNORE
            or set(annotations["episodes"]) != set(study["episode_conditions"])):
        raise ValueError("Receiving-phase annotations do not match the study")
    raw = project_path(PROJECT_ROOT / "raw" / study["source_dataset"].split("/")[-1])
    result = {}
    for episode, span in annotations["episodes"].items():
        sidecar, = (raw / episode).glob("videos/*/top_timestamps.parquet")
        rows = pq.read_table(sidecar, columns=["frame_index", "recv_ns"])
        indices, times = (column.to_numpy() for column in rows.columns)
        start, stop = span["first_frame"], span["stop_frame_exclusive"]
        if (not np.array_equal(indices, np.arange(len(indices))) or np.any(np.diff(times) <= 0)
                or not 0 <= start < stop < len(times)):
            raise ValueError(f"Bad head-camera timestamps or receiving bounds for episode {episode}")
        result[int(episode)] = (int(times[start]), int(times[stop]),
                                CLASSES.index(study["episode_conditions"][episode]))
    return result


def target_provenance(cfg, cache_split):
    files = {"receiving_annotations": cfg["receiving_annotations"], "study": cfg["annotations"],
             "cache": cache_split / "cache.json", "sample_order": cache_split / "sampling_indices.npy"}
    return {key: sha256_file(path) for key, path in files.items()}


def load_targets(cfg, split, cache_split, length):
    import json

    with np.load(project_path(Path(cfg["direction_targets"])), allow_pickle=False) as packed:
        provenances = json.loads(packed["provenances"].item())
        targets = packed[split]
    if (provenances[split] != target_provenance(cfg, cache_split) or targets.shape != (length,)
            or not np.issubdtype(targets.dtype, np.integer)
            or not np.isin(targets, [IGNORE, *range(len(CLASSES))]).all()):
        raise ValueError(f"Direction targets do not match the {split} cache")
    return targets


def classification_loss(logits, targets, class_weights):
    valid = targets != IGNORE
    count = valid.sum()
    total = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=IGNORE, reduction="sum")
    return total / count.clamp_min(1), count


def report_confusion(confusion):
    confusion = np.asarray(confusion)
    per_class = confusion.sum(axis=1)
    denom = per_class + confusion.sum(axis=0)
    correct = confusion.diagonal()
    f1 = np.divide(2 * correct, denom, out=np.zeros(len(CLASSES)), where=denom > 0)
    recall = np.divide(correct, per_class, out=np.zeros(len(CLASSES)), where=per_class > 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(correct.sum() / confusion.sum()),
            "recall": dict(zip(CLASSES, recall.tolist(), strict=True)), "confusion": confusion.tolist()}
