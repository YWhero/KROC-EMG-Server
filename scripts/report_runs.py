"""Export a status snapshot without copying training logs or host paths."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from training_mirror_session import REPO, configured_runs, json_lines


def optional_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def summarize(root):
    result = []
    for name in configured_runs(root):
        run = root / "outputs" / f"{name}_balanced"
        identity = optional_json(run / "run_identity.json")
        complete = optional_json(run / "completed.json")
        decision = optional_json(run / "continuation.json")
        metrics = json_lines(run / "metrics.jsonl") if (run / "metrics.jsonl").is_file() else []
        best = min((row for row in metrics if "val_l1_zero_latent" in row),
                   key=lambda row: row["val_l1_zero_latent"], default={})
        latest_direction = next((row for row in reversed(metrics) if "val_direction_macro_f1" in row), {})
        receipts = [path for path in sorted((run / "upload_receipts").glob("*.json"))
                    if optional_json(path).get("cleanup_complete")]
        run_id = identity.get("run_id")
        result.append({
            "archive_variant": name,
            "status": ("complete" if complete and complete["steps"] == decision.get("target_steps", complete["steps"])
                       else "started" if metrics else "queued"),
            "last_logged_step": metrics[-1]["step"] if metrics else None,
            "target_steps": decision.get("target_steps", 100_000),
            "best_validation_step": best.get("step"),
            "best_validation_action_l1": best.get("val_l1_zero_latent"),
            "latest_direction_validation_step": latest_direction.get("step"),
            "latest_direction_macro_f1": latest_direction.get("val_direction_macro_f1"),
            "verified_checkpoint_steps": [int(path.stem) for path in receipts],
            "hub_tree": f"https://huggingface.co/{REPO}/tree/main/runs/{name}/{run_id}" if run_id else None,
        })
    return {"captured_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
            "checkpoint_repository": REPO, "runs": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
