"""Apply the validation gate once, then resume eligible 100k runs at the same LR."""
import argparse
import json
import os
import re
import statistics
import sys
import time

from .paths import PROJECT_ROOT, atomic_json, project_path
from .study import sha256_file


MILESTONE = 100_000
EXTENDED_STEPS = 200_000
MIN_TRAIN_DROP = 0.05
MIN_VALIDATION_DROP = 0.02
CONTINUATION_UPLOAD = "/dev/shm/kroc-balanced-hf-session/enable"


def evaluate_trend(rows):
    if not rows or rows[-1]["step"] != MILESTONE:
        raise ValueError("The 100,000-step validation record is required")
    summaries = []
    for end in (80_000, 90_000, MILESTONE):
        selected = [row for row in rows if end - 10_000 < row["step"] <= end]
        validation = [row["val_condition_macro_l1"] for row in selected
                      if "val_condition_macro_l1" in row]
        if len(validation) != 10 or len(selected) != 100:
            raise ValueError(f"Incomplete training or validation logs ending at step {end}")
        summaries.append({"end_step": end, "train_loss_median": statistics.median(
            row.get("policy_loss", row["loss"]) for row in selected),
            "validation_macro_l1_median": statistics.median(validation)})
    early, middle, late = summaries
    train_drop = 1 - late["train_loss_median"] / early["train_loss_median"]
    validation_drop = 1 - late["validation_macro_l1_median"] / early["validation_macro_l1_median"]
    extend = (train_drop >= MIN_TRAIN_DROP and validation_drop >= MIN_VALIDATION_DROP
              and late["validation_macro_l1_median"] < middle["validation_macro_l1_median"])
    return {"extend": extend, "target_steps": EXTENDED_STEPS if extend else MILESTONE,
            "windows": summaries, "train_relative_drop": train_drop,
            "validation_relative_drop": validation_drop,
            "minimum_train_drop": MIN_TRAIN_DROP, "minimum_validation_drop": MIN_VALIDATION_DROP}


def pin_eligible_checkpoint(run, checkpoint):
    """Called before the uploader can remove the local checkpoint after verification."""
    decision = run / "continuation.json"
    pinned = run / ".resume_100k.pt"
    if decision.is_file():
        result = json.loads(decision.read_text())
        if result["extend"] and not pinned.is_file():
            os.link(checkpoint, pinned)
        return
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    result = evaluate_trend(rows)
    if result["extend"] and not pinned.is_file():
        os.link(checkpoint, pinned)
    atomic_json(decision, result)
    print(json.dumps({"run": run.name, "continuation": result}), flush=True)


def run_finished(run):
    try:
        target = json.loads((run / "continuation.json").read_text())["target_steps"]
        steps = json.loads((run / "completed.json").read_text())["steps"]
    except FileNotFoundError:
        return False
    return steps == target


def launch(variant):
    run = PROJECT_ROOT / "outputs" / f"{variant}_balanced"
    decision = run / "continuation.json"
    milestone_complete = run / "completed_100k.json"
    completed = run / "completed.json"
    milestone_checkpoint = run / "checkpoints" / f"{MILESTONE:08d}" / "checkpoint.pt"
    missing_checkpoint_reported = False
    print(f"Waiting for the {variant} 100,000-step decision.", flush=True)
    while not (decision.is_file() and (completed.is_file() or milestone_complete.is_file())):
        # Keep the compute decision available while the replacement uploader awaits credentials.
        if (not os.path.isfile(CONTINUATION_UPLOAD) and not decision.is_file()
                and (milestone_checkpoint.is_file() or completed.is_file())):
            try:
                pin_eligible_checkpoint(run, milestone_checkpoint)
            except FileNotFoundError:
                if not missing_checkpoint_reported:
                    print("Waiting for the verified 100,000-step checkpoint to resume.", flush=True)
                    missing_checkpoint_reported = True
        time.sleep(1)
    result = json.loads(decision.read_text())
    if not result["extend"] or run_finished(run):
        print(f"{variant} complete at step {result['target_steps']}.", flush=True)
        return
    with (run / "metrics.jsonl").open() as stream:
        if any(json.loads(line)["step"] > MILESTONE for line in stream if line.strip()):
            raise RuntimeError("Continuation logs already exist. Resume from the latest verified checkpoint.")
    checkpoint = project_path(run / ".resume_100k.pt")
    receipt = json.loads((run / "upload_receipts" / f"{MILESTONE:08d}.json").read_text())
    proof, = (item for name, item in receipt["files"].items() if name.endswith("/checkpoint.pt"))
    if (not receipt.get("cleanup_complete") or not receipt.get("verified")
            or checkpoint.stat().st_size != proof["size"] or sha256_file(checkpoint) != proof["sha256"]):
        raise ValueError("The pinned continuation checkpoint does not match the verified 100k upload")
    if not milestone_complete.exists():
        completed.replace(milestone_complete)
    command = [sys.executable, "-u", "-m", "kroc_act.train", "--config", f"configs/{variant}_balanced.json",
               "--resume", str(checkpoint), "--extend-to", str(EXTENDED_STEPS)]
    print(f"Continuing {variant} from step {MILESTONE} to {EXTENDED_STEPS}.", flush=True)
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    variant = parser.parse_args().variant
    registry = json.loads(project_path(PROJECT_ROOT / "data/upload_runs.json").read_text())
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", variant) or variant not in registry:
        parser.error("Expected a run registered for archival")
    launch(variant)


if __name__ == "__main__":
    main()
