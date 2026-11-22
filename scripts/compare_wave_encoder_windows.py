"""Repeat the shared-timestamp history comparison with the actual Wave encoder on CPU."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from compare_emg_windows import DURATIONS, collect
from kroc_act.data import load_json
from kroc_act.direction import CLASSES, report_confusion
from kroc_act.model import WindowEncoder
from kroc_act.paths import atomic_json, project_path
from kroc_act.train import resolve_config


SEEDS = (1000, 2000)
EPOCHS = 20
BATCH = 32


def fit_wave(data, fit, index, width, seed):
    train = np.concatenate([data[ep][2][index] for ep in fit])
    targets = np.repeat([data[ep][0] for ep in fit], [len(data[ep][1]) for ep in fit])
    labels = torch.from_numpy(targets)
    counts = Counter(data[ep][0] for ep in fit)
    weights = np.concatenate([np.full(len(data[ep][1]), 1 / (len(data[ep][1]) * counts[data[ep][0]]))
                              for ep in fit])
    weights *= len(train) / weights.sum()
    weights = torch.from_numpy(weights).float()
    center = train.mean(axis=(0, 1))
    scale = train.std(axis=(0, 1)).clip(.01)
    train = torch.from_numpy(np.asarray((train - center) / scale, dtype=np.float32))
    torch.manual_seed(seed)
    model = torch.nn.Sequential(WindowEncoder(train.shape[2], width), torch.nn.Linear(width, len(CLASSES)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(train)).split(BATCH):
            logits = model(train[idx])
            loss = (torch.nn.functional.cross_entropy(logits, labels[idx], reduction="none") * weights[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), center, scale


@torch.no_grad()
def evaluate(model, center, scale, data, holdout, index):
    matrices = {key: np.zeros((len(CLASSES), len(CLASSES)), dtype=int) for key in ("episode", "frame", "first_400ms")}
    for ep in holdout:
        target, offsets, histories = data[ep]
        x = torch.from_numpy(np.asarray((histories[index] - center) / scale, dtype=np.float32))
        logits = model(x).numpy()
        classes = logits.argmax(1)
        matrices["episode"][target, logits.mean(axis=0).argmax()] += 1
        np.add.at(matrices["frame"][target], classes, 1)
        np.add.at(matrices["first_400ms"][target], classes[offsets < 4], 1)
    return {key: report_confusion(value) for key, value in matrices.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = resolve_config(args.config)
    split = load_json(cfg["splits"])
    fit, holdout = split["train"], split["val"]
    data = collect(cfg, [*fit, *holdout], transform=lambda window: window)
    windows = {}
    for index, duration in enumerate(DURATIONS):
        fits = [evaluate(*fit_wave(data, fit, index, cfg["dim_model"], seed), data, holdout, index) for seed in SEEDS]
        means = {key: float(np.mean([result[key]["macro_f1"] for result in fits])) for key in fits[0]}
        print(duration, means, flush=True)
        windows[str(duration)] = {"macro_f1_mean": means, "seed_results": dict(zip(map(str, SEEDS), fits, strict=True))}
    output = project_path(args.output)
    report = load_json(output)
    report["wave_encoder_check"] = {"method": "EMG-only 1D Wave encoder and linear direction head. Fixed 20 epochs, two seeds, identical held-out episode split, and no ACT or image inputs.",
                                    "epochs": EPOCHS, "windows_ms": windows}
    atomic_json(output, report)


if __name__ == "__main__":
    main()
