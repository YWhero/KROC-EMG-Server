"""Rank train-only episode EMG shapes around visually reviewed hand movement."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kroc_act.emg import EMGStream
from kroc_act.paths import atomic_json
from kroc_act.study import validate_study

ROOT = Path.cwd()
RATE = 2000
CLASSES = ("as_is", "wrist_left", "wrist_right")
BANDS = ((20, 60), (60, 120), (120, 250), (250, 450))
# Approximate mid-movement timestamps read from the top-view frame series, in seconds.
VISUAL_ANCHORS = [
    15.0, 15.1, 12.4, 11.7, 12.2, 12.0, 14.0, 13.2, 12.5, 12.0,
    12.0, 10.2, 11.7, 11.2, 11.9, 12.1, 12.5, 12.8, 11.2, 11.1,
    11.4, 11.9, 10.5, 14.1, 13.4, 13.8, 13.6, 12.8, 11.6, 15.0,
]


def load_stream(episode, metadata):
    path = ROOT / "data/emg" / metadata[str(episode)]["file"]
    with np.load(path, allow_pickle=False) as data:
        return EMGStream(*(data[k] for k in ("time_ns", "available_ns", "values", "valid")))


def features(signal):
    amplitudes = np.sqrt(np.mean(signal.astype(float) ** 2, axis=0))
    segments = np.lib.stride_tricks.sliding_window_view(signal, 128, axis=0)[::64]
    centered = segments - segments.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered * np.hanning(128), n=256, axis=-1)
    power = np.mean(np.abs(spectrum) ** 2, axis=0).T
    freq = np.fft.rfftfreq(256, d=1/RATE)
    bands = np.asarray([power[(freq >= low) & (freq < high)].sum(axis=0) for low, high in BANDS])
    band_fraction = bands / np.maximum(bands.sum(axis=0), 1e-9)
    return amplitudes, band_fraction


def episode_features(stream, anchor):
    phases, quality = [], []
    for center in (-1.25, -.625, 0., .625, 1.25):
        left, right = np.searchsorted(stream.time_ns, anchor + np.array([center-.32, center+.32]) * 1e9)
        valid = stream.valid[left:right] & (stream.available_ns[left:right] <= anchor + (center+.45)*1e9)
        quality.append(min(1., valid.sum() / 1280))
        clean = stream.values[left:right][valid]
        if len(clean) < 128:
            raise ValueError("Movement phase is missing")
        phases.append(features(clean))
    amps = np.stack([phase[0] for phase in phases])
    fractions = np.stack([phase[1] for phase in phases])
    log_amp = np.log1p(amps)
    # Give equal weight to RMS envelope and normalized band shape.
    vector = np.concatenate((log_amp.ravel(), np.log(fractions + .02).ravel() / 2))
    return vector, amps, fractions, float(np.mean(quality))


def probe(stream, anchor, window_ms):
    results = []
    for end in np.arange(-1.4, 1.01, .15):
        try:
            history = stream.window(round(anchor + end*1e9), round(window_ms*RATE/1000), RATE)
        except ValueError:
            continue
        a, b = features(history)
        results.append(np.r_[np.log1p(a), np.log(b+.02).ravel()])
    return np.asarray(results)


def robust_fit(values):
    location = np.median(values, axis=0)
    spread = np.maximum(1.4826*np.median(np.abs(values-location), axis=0), .08)
    return location, spread


def diagnostic_probe(data, labels, originals):
    results = {}
    for ms in (120, 200, 250, 320):
        predicted = {}
        for hold in originals:
            fit = [ep for ep in originals if ep != hold]
            loc, scale = robust_fit(np.vstack([data[ep][ms] for ep in fit]))
            center = {}
            for condition in CLASSES:
                samples = [np.median(data[ep][ms], axis=0) for ep in fit if labels[str(ep)] == condition]
                center[condition] = (np.median(samples, axis=0)-loc)/scale
            x = (np.median(data[hold][ms], axis=0)-loc)/scale
            predicted[hold] = min(center, key=lambda condition: np.linalg.norm(x-center[condition]))
        results[str(ms)] = {
            "correct": sum(label == labels[str(ep)] for ep, label in predicted.items()),
            "unique_train_episodes": len(originals),
            "confusion": {actual: dict(Counter(predicted[ep] for ep in originals if labels[str(ep)] == actual))
                          for actual in CLASSES},
        }
    return results


def main():
    raw = next((ROOT/"raw").glob("red_pen*"))
    annotated = json.loads((ROOT/"data/study_annotations.json").read_text())
    labels = annotated["episode_conditions"]
    split = json.loads((ROOT/"data/splits.json").read_text())
    manifest = json.loads((ROOT/"data/emg/manifest.json").read_text())["episodes"]
    raw_results, probes, anchor_rows = {}, {}, {}
    for ep in split["train"]:
        ts_file = next((raw/str(ep)).glob("videos/*/top_timestamps.parquet"))
        ns = int(pd.read_parquet(ts_file, columns=["recv_ns"]).recv_ns.iloc[0])
        anchor = ns + round(VISUAL_ANCHORS[ep]*1e9)
        anchor_rows[ep] = VISUAL_ANCHORS[ep]
        stream = load_stream(ep, manifest)
        raw_results[ep] = episode_features(stream, anchor)
        probes[ep] = {ms: probe(stream, anchor, ms) for ms in (120, 200, 250, 320)}
    _, scale = robust_fit(np.stack([raw_results[ep][0] for ep in split["train"]]))
    stats, excluded, slots, prior_balance = {}, [], [], {}
    for condition in CLASSES:
        candidates = [ep for ep in split["train"] if labels[str(ep)] == condition]
        val = [ep for ep in split["val"] if labels[str(ep)] == condition]
        center = np.median([raw_results[ep][0] for ep in candidates], axis=0)
        ranking = []
        for ep in candidates:
            vector, amps, fractions, coverage = raw_results[ep]
            z = (vector-center) / scale
            rms_distance = np.sqrt(np.mean(z[:20] ** 2))
            band_distance = np.sqrt(np.mean(z[20:] ** 2))
            score = (rms_distance + band_distance) / 2 + 3*(1-coverage)
            ranking.append((float(score), ep, float(rms_distance), float(band_distance), coverage))
        ranked = sorted(ranking)
        source_count = len(candidates) + len(val)
        drop = max(0, source_count-10)
        selected = ranked if not drop else ranked[:-drop]
        excluded.extend(row[1] for row in ranked[len(selected):])
        repeat = [row[1] for row in selected[:max(0,10-source_count)]]
        full_pool_train = [row[1] for row in selected] + repeat
        prior_balance[condition] = full_pool_train + val
        # Preserve validation disjointness and restore the requested ten training slots.
        extra_order = sorted([row[1] for row in selected], key=lambda ep: (full_pool_train.count(ep),
                             next(rank[0] for rank in selected if rank[1] == ep)))
        full_pool_train.extend(extra_order[:10-len(full_pool_train)])
        slots.extend(full_pool_train)
        amps = np.stack([raw_results[ep][1] for ep in candidates])
        fraction = np.stack([raw_results[ep][2] for ep in candidates])
        stats[condition] = {
            "source_episode_count": source_count, "validation_ids": val,
            "ranking": [{"episode": ep, "score": score, "rms_distance": rmsd,
                         "band_distance": bandd, "valid_coverage": coverage}
                        for score, ep, rmsd, bandd, coverage in ranked],
            "original_pool_repeats": repeat, "train_slots": full_pool_train,
            "median_rms_by_phase_and_channel": np.median(amps, axis=0).round(2).tolist(),
            "median_rms_counts": np.median(amps, axis=(0,1)).round(2).tolist(),
            "median_band_fraction_band_by_channel": np.median(fraction, axis=(0,1)).round(3).tolist(),
        }
    balanced = {"train": sorted(set(slots)), "val": split["val"], "excluded": sorted(excluded), "train_slots": slots}
    validate_study(annotated, balanced, manifest)
    atomic_json(ROOT/"data/balanced_splits.json", balanced)
    atomic_json(ROOT/"data/emg_balance_analysis.json", {
        "visual_anchor_seconds": VISUAL_ANCHORS, "anchor_interpretation": "Approximate movement center; seconds in top clip",
        "bands_hz": BANDS, "phase_centers_seconds": [-1.25,-.625,0,.625,1.25],
        "conditions": stats, "balanced_source_slots_with_fixed_validation": prior_balance,
        "train_only_window_probe": diagnostic_probe(probes, labels, split["train"]),
    })
    print(json.dumps(balanced, indent=2), flush=True)


if __name__ == "__main__":
    main()
