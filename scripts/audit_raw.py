"""Read-only schema audit of the actual Cyclo raw recording."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def summarize(array):
    a = np.asarray(array)
    result = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.size and a.dtype.kind in "biuf":
        result.update(first=a[:3].tolist(), last=a[-3:].tolist())
        if a.ndim == 1 and len(a) > 1:
            diff = np.diff(a.astype(np.float64))
            result["diff_percentiles"] = np.percentile(diff, [0, 1, 50, 99, 100]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    episode = args.root / "0"
    report = {}
    for path in sorted((episode / "emg").rglob("*.h5")):
        content = {}
        with h5py.File(path, "r") as f:
            def visit(name, node):
                if isinstance(node, h5py.Dataset):
                    content[name] = summarize(node[:]) if node.shape else str(node[()])[:300]
            f.visititems(visit)
            content["root_attributes"] = {k: str(v)[:1500] for k, v in f.attrs.items()}
        report[str(path.relative_to(args.root))] = content
    for path in sorted((episode / "videos").rglob("*.parquet")):
        df = pd.read_parquet(path)
        report[str(path.relative_to(args.root))] = {
            "columns": {k: str(v) for k, v in df.dtypes.items()},
            "first": df.head(3).to_dict(orient="records"),
            "last": df.tail(3).to_dict(orient="records")}
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
