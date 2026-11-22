"""Validated collection annotations and a frozen, episode-stratified split."""
import hashlib
import json
import random
from collections import Counter

from .paths import project_path


def read_json(path):
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def validate_study(annotations, splits, available_episodes):
    labels = annotations["episode_conditions"]
    available = set(map(int, available_episodes))
    if set(map(int, labels)) != available:
        raise ValueError("Study annotations must cover exactly the dataset episodes")
    if not annotations.get("participant_id") or not annotations.get("electrodes_attached_declared"):
        raise ValueError("Confirmed participant and electrode declaration required")
    if annotations.get("emg_source_role") != "recipient":
        raise ValueError("This study requires recipient EMG")
    active_splits = {name: splits[name] for name in ("train", "val")}
    excluded = splits.get("excluded", [])
    flattened = [ep for values in (*active_splits.values(), excluded) for ep in values]
    if not splits.get("train") or not splits.get("val"):
        raise ValueError("Nonempty train and validation episodes required")
    if len(flattened) != len(set(flattened)) or set(flattened) != available:
        raise ValueError("Splits must be disjoint and cover exactly the available episodes")
    counts = {name: dict(sorted(Counter(labels[str(ep)] for ep in episodes).items()))
              for name, episodes in active_splits.items()}
    expected = set(annotations["conditions"])
    if any(set(values) != expected for values in counts.values()):
        raise ValueError("Every split must contain every study condition")
    if "train_slots" in splits:
        slots = splits["train_slots"]
        if set(slots) != set(active_splits["train"]):
            raise ValueError("Every training episode must appear in the sampling slots")
        slot_counts = Counter(labels[str(ep)] for ep in slots)
        if set(slot_counts) != expected or set(slot_counts.values()) != {10}:
            raise ValueError("The balanced experiment requires ten training slots per condition")
        counts["train_slots"] = dict(sorted(slot_counts.items()))
    return counts


def stratified_split(annotations, seed=1000, val_per_condition=2):
    """Shuffle each condition once; never use validation outcomes to pick episodes."""
    rng = random.Random(seed)
    train, val = [], []
    for condition in sorted(annotations["conditions"]):
        episodes = sorted(annotations["conditions"][condition]["source_episode_indices"])
        if len(episodes) <= val_per_condition:
            raise ValueError("Not enough episodes to preserve training examples")
        rng.shuffle(episodes)
        val.extend(episodes[:val_per_condition])
        train.extend(episodes[val_per_condition:])
    result = {"train": sorted(train), "val": sorted(val)}
    validate_study(annotations, result, annotations["episode_conditions"])
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with project_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
