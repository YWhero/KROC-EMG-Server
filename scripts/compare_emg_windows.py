"""Compare causal history lengths on shared observation times, with episode-level holdout."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_emg import features
from kroc_act.data import load_json
from kroc_act.direction import CLASSES, receiving_intervals, report_confusion
from kroc_act.emg import EMGStream
from kroc_act.paths import atomic_json, project_path
from kroc_act.train import resolve_config


DURATIONS = (120, 200, 250, 320, 500)
TICK_NS = 100_000_000


def history_features(signal):
    amplitude, bands = features(signal)
    return np.r_[np.log1p(amplitude), np.log(bands + .02).ravel()]


def collect(cfg, episodes, transform=history_features):
    paths = Path(cfg["manifest"]).parent
    manifest = load_json(cfg["manifest"])["episodes"]
    intervals = receiving_intervals(cfg)
    samples = [(duration * cfg["sample_rate_hz"]) // 1000 for duration in DURATIONS]
    result = {}
    for episode in episodes:
        with np.load(paths / manifest[str(episode)]["file"], allow_pickle=False) as data:
            stream = EMGStream(*(data[key] for key in ("time_ns", "available_ns", "values", "valid")))
        start, stop, target = intervals[episode]
        rows, offsets = [], []
        for now in range(start, stop, TICK_NS):
            try:
                windows = [stream.window(now, count, cfg["sample_rate_hz"], cfg["max_gap_ms"],
                                         cfg["max_emg_age_ms"]) for count in samples]
            except ValueError:
                continue
            rows.append([transform(window) for window in windows])
            offsets.append((now - start) // TICK_NS)
        if not rows:
            raise ValueError(f"All observation times excluded in episode {episode}")
        result[episode] = (target, np.asarray(offsets), tuple(np.stack(column) for column in zip(*rows, strict=True)))
    return result


def scores(data, fit, episode, duration_index):
    classes = Counter(data[key][0] for key in fit)
    x = np.concatenate([data[key][2][duration_index] for key in fit])
    y = np.concatenate([np.repeat(data[key][0], len(data[key][1])) for key in fit])
    weights = np.concatenate([np.full(len(data[key][1]), 1 / (len(data[key][1]) * classes[data[key][0]]))
                              for key in fit])
    weights /= weights.sum()
    mean = np.average(x, axis=0, weights=weights)
    std = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)).clip(.01)
    x = (x - mean) / std
    y = np.eye(len(CLASSES))[y]
    matrix = (x.T * weights) @ x + .08 * np.eye(x.shape[1])
    coefficients = np.linalg.solve(matrix, (x.T * weights) @ (y - 1 / len(CLASSES)))
    return ((data[episode][2][duration_index] - mean) / std) @ coefficients


def evaluate(data, fit, tests, duration_index):
    episode_matrix = np.zeros((3, 3), dtype=int)
    frame_matrix = np.zeros_like(episode_matrix)
    early_matrix = np.zeros_like(episode_matrix)
    details = {}
    for episode in tests:
        target, offsets, _ = data[episode]
        output = scores(data, [key for key in fit if key != episode], episode, duration_index)
        predicted = output.argmax(1)
        np.add.at(frame_matrix[target], predicted, 1)
        np.add.at(early_matrix[target], predicted[offsets < 4], 1)
        episode_matrix[target, output.mean(axis=0).argmax()] += 1
        stable = next((int(offsets[i]) * .1 for i in range(2, len(offsets))
                       if offsets[i] - offsets[i-2] == 2 and np.all(predicted[i-2:i+1] == target)), None)
        details[str(episode)] = {"correct_class": CLASSES[target], "frames": len(offsets),
                                 "episode_prediction": CLASSES[output.mean(axis=0).argmax()],
                                 "first_3_consecutive_correct_seconds": stable}
    return {"episode": report_confusion(episode_matrix), "frame": report_confusion(frame_matrix),
            "first_400ms": report_confusion(early_matrix), "per_episode": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    split = load_json(cfg["splits"])
    fit, holdout = split["train"], split["val"]
    data = collect(cfg, [*fit, *holdout])
    results = {str(ms): {"held_out": evaluate(data, fit, holdout, i),
                         "leave_one_train_episode_out": evaluate(data, fit, fit, i)}
               for i, ms in enumerate(DURATIONS)}
    atomic_json(project_path(args.output), {
        "method": "Fixed regularized linear EMG-only probe on causal log RMS and band fractions; train scaled by class and by episode. Same valid observation times for every duration. First 400 ms measured from visually annotated hand-opening/approach.",
        "observation_interval_ms": 100, "class_order": CLASSES,
        "train_episodes": fit, "held_out_episodes": holdout, "windows_ms": results,
    })
    for ms, row in results.items():
        print(ms, {set_name: [round(result[key]["macro_f1"], 3) for key in ("episode", "frame", "first_400ms")]
                   for set_name, result in row.items()}, flush=True)


if __name__ == "__main__":
    main()
