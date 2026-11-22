"""Snapshot complete, structured KROC training metrics."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

DEFAULT_RUNS = ("window", "spectrogram", "window_direction")
CURVES = (
    "step", "loss", "l1_loss", "kld_loss", "grad_norm", "emg_grad_norm",
    "val_l1_zero_latent", "val_condition_macro_l1", "val_frames", "val_valid_elements",
    "best_step", "best_val_l1",
    "policy_loss", "direction_loss", "direction_frames", "val_direction_loss", "val_direction_frames",
    "val_direction_macro_f1", "val_direction_accuracy",
)
JOINTS = ("j1", "j2", "j3", "j4", "j5", "gripper")
CONDITIONS = ("as_is", "wrist_left", "wrist_right")
REPO = "HYHero/kroc-act-emg-balanced-20261123"


def configured_runs(root):
    registry = root / "data" / "upload_runs.json"
    if not registry.is_file():
        return DEFAULT_RUNS
    names = json.loads(registry.read_text())
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Invalid archived run registry")
    return tuple(names)


def json_lines(path):
    data = path.read_bytes()
    lines = data.splitlines()
    if data and not data.endswith(b"\n"):
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def jsonl(rows):
    return "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def metrics_csv(rows):
    columns = (*CURVES, *(f"val_{condition}_l1" for condition in CONDITIONS),
               *(f"val_mae_{joint}_rad" for joint in JOINTS),
               *(f"val_direction_recall_{condition}" for condition in CONDITIONS))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values = {key: row.get(key) for key in CURVES}
        values.update({f"val_{condition}_l1": row.get("val_by_condition", {}).get(condition)
                       for condition in CONDITIONS})
        values.update({f"val_mae_{joint}_rad": value for joint, value in
                       zip(JOINTS, row.get("val_action_mae_original_units") or [])})
        values.update({f"val_direction_recall_{condition}": row.get("val_direction_recall", {}).get(condition)
                       for condition in CONDITIONS})
        writer.writerow(values)
    return buffer.getvalue()


def snapshot(root):
    files = {}
    for variant in configured_runs(root):
        run = root / "outputs" / f"{variant}_balanced"
        if not (run / "run_identity.json").is_file() or not (run / "metrics.jsonl").is_file():
            continue
        identity = json.loads((run / "run_identity.json").read_text())
        run_id = identity["run_id"]
        if identity.get("archive_variant", identity["variant"]) != variant or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid checkpoint run identity")
        prefix = f"runs/{variant}/{run_id}/training/"
        rows = json_lines(run / "metrics.jsonl")
        files[prefix + "metrics.jsonl"] = jsonl(rows)
        files[prefix + "metrics.csv"] = metrics_csv(rows)
        files[prefix + "validation.jsonl"] = jsonl(row for row in rows if "val_frames" in row)
        for name in ("experiment.json", "provenance.json", "split_summary.json", "splits.json",
                     "code_provenance.json", "continuation.json", "completed_100k.json", "completed.json",
                     "receiving_annotations.json", "direction_supervision.json"):
            path = run / name
            if path.is_file():
                files[prefix + name] = path.read_text()
        diagnostic = root / "data" / "emg_window_diagnostic.json"
        if variant in {"window_direction", "window_direction_320"} and diagnostic.is_file():
            files[prefix + diagnostic.name] = diagnostic.read_text()
        if variant.endswith("_320"):
            alignment = root / "data" / "frame_cache_320ms" / "alignment.json"
            files[prefix + "cache_alignment.json"] = alignment.read_text()
    return files


def upload_logs(api, files, state=None):
    from huggingface_hub import CommitOperationAdd

    contents = {path: body.encode("utf-8") for path, body in files.items()}
    digests = {path: hashlib.sha256(data).hexdigest() for path, data in contents.items()}
    previous = state or {}
    changed = {path: data for path, data in contents.items() if digests[path] != previous.get(path)}
    if not changed:
        return None, digests
    commit = api.create_commit(
        repo_id=REPO, repo_type="model", commit_message="Sync KROC training metrics",
        operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)
                    for path, data in changed.items()],
    )
    paths = api.get_paths_info(REPO, list(changed), repo_type="model", revision=commit.oid)
    by_path = {item.path: item for item in paths}
    for path, body in changed.items():
        item = by_path.get(path)
        blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        lfs = item.lfs if item is not None else None
        if lfs is not None:
            actual = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            expected = digests[path]
        else:
            actual = item.blob_id if item is not None else None
            expected = blob
        if item is None or item.size != len(body) or actual != expected:
            raise ValueError("Hugging Face training log did not match the local snapshot")
    return commit.oid, digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--upload-json", type=Path)
    parser.add_argument("--public", action="store_true", help="Confirm the archive is intended to be public")
    args = parser.parse_args()
    if args.upload_json:
        from huggingface_hub import HfApi, get_token

        api = HfApi(token=get_token())
        if api.whoami().get("name", "").casefold() != "hyhero" or api.repo_info(REPO).private == args.public:
            raise SystemExit("Expected HYHero and the configured training repository visibility")
        files = json.loads(args.upload_json.read_text())
        if any(not re.fullmatch(r"runs/[a-z][a-z0-9_]{0,39}/[a-f0-9]{32}/training/[a-z_0-9.]+", p)
               for p in files):
            raise SystemExit("Unexpected training-log path")
        commit, _ = upload_logs(api, files)
        print("Verified metric commit:", commit)
    else:
        print(json.dumps(snapshot(args.root), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
