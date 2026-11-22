"""Atomic checkpoints and a token-free server queue for a trusted PC relay."""
import json
import math
import random
import re
import time
import uuid
from pathlib import Path

from .paths import PROJECT_ROOT, atomic_json, checkpoint_pointer_lock, project_path
from .study import sha256_file


def save_checkpoint(run, step, model, optimizer, cfg, statistics, provenance, generator,
                    probe=None, training_state=None, is_best=False, validation=None,
                    wait_for_archive=True, direction_head=None):
    import numpy as np
    import torch
    run = project_path(run)
    target = project_path(run / "checkpoints" / f"{step:08d}")
    if cfg.get("checkpoint_archive") and probe is not None:
        raise ValueError("Raw input probes must not be included in archived model bundles")
    target.mkdir(parents=True, exist_ok=False)
    temporary = project_path(target / "checkpoint.tmp")
    verification = None
    if probe is not None:
        was_training = model.training
        prediction = model.predict_action_chunk(probe).detach().cpu()
        verification = {"batch": {k: v.detach().cpu() for k, v in probe.items()
                                  if torch.is_tensor(v)}, "prediction": prediction}
        model.train(was_training)
    state = dict(training_state or {})
    auxiliary = {"direction_head": direction_head.state_dict()} if direction_head is not None else {}
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "experiment": cfg, "statistics": statistics, "provenance": provenance,
                "training_state": state, "verification": verification,
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [], "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "loader_rng": generator.get_state(), **auxiliary}, temporary)
    temporary.replace(target / "checkpoint.pt")
    metadata = {"step": step, "is_best": bool(is_best), "training_state": state,
                "validation": validation, "experiment": cfg, "statistics": statistics,
                "provenance": provenance}
    for name in ["study_annotations.json", "splits.json", "split_summary.json", "code_provenance.json",
                 "receiving_annotations.json", "direction_supervision.json"]:
        source = project_path(run / name)
        if source.is_file():
            value = json.loads(source.read_text(encoding="utf-8"))
            if name == "study_annotations.json":
                # Public exports keep collection facts and placements, not the local participant ID.
                value["participant_id"] = "participant_01"
                value["declaration_source"] = "Retrospective collector confirmation"
                value["public_export_note"] = "Local participant identifier pseudonymized for export"
            metadata[name.removesuffix(".json")] = value
    atomic_json(target / "metadata.json", metadata)
    pointer = {"step": step, "checkpoint": str((target / "checkpoint.pt").relative_to(run)),
               "selection_metric": "val_l1_zero_latent", "validation": validation}
    previous_best = None
    with checkpoint_pointer_lock(run):
        atomic_json(run / "latest.json", pointer)
        if is_best:
            best_path = project_path(run / "best.json")
            if best_path.is_file():
                previous_best = json.loads(best_path.read_text())["step"]
            atomic_json(best_path, pointer)
    archive = cfg.get("checkpoint_archive", {})
    if archive.get("mode") == "relay":
        if archive.get("final_best_only"):
            if step % cfg["save_freq"] == 0 or step == cfg["steps"]:
                queue_checkpoint(run, target, step, cfg, is_best)
            if previous_best is not None and previous_best != step:
                old_name = f"{previous_best:08d}"
                old_queue = project_path(run / "transfer_queue" / f"{old_name}.json")
                if not old_queue.exists():
                    project_path(run / "checkpoints" / old_name / "checkpoint.pt").unlink(missing_ok=True)
        else:
            job = queue_checkpoint(run, target, step, cfg, is_best)
            if wait_for_archive:
                wait_for_receipt(run, job, archive)
    else:
        # Offline/local experiments keep the best in addition to the rolling window.
        completed = sorted(p for p in (run / "checkpoints").iterdir()
                           if project_path(p).is_dir() and project_path(p / "checkpoint.pt").is_file())
        best_step = None
        if project_path(run / "best.json").is_file():
            best_step = json.loads((run / "best.json").read_text())["step"]
        for old in completed[:-cfg["keep_checkpoints"]]:
            if best_step is None or old.name != f"{best_step:08d}":
                # Only the model bundle is removed. No recursive directory deletion.
                project_path(old / "checkpoint.pt").unlink()
    return target / "checkpoint.pt"


def run_identity(run, cfg):
    identity_file = project_path(run / "run_identity.json")
    if identity_file.is_file():
        return json.loads(identity_file.read_text())
    identity = {"run_id": uuid.uuid4().hex, "variant": cfg["variant"]}
    name = cfg.get("checkpoint_archive", {}).get("run_name")
    if name is not None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name) or run.name != f"{name}_balanced":
            raise ValueError("Archive run name must match the experiment directory")
        identity["archive_variant"] = name
    elif "direction_loss_weight" in cfg:
        identity["archive_variant"] = "window_direction"
    atomic_json(identity_file, identity)
    return identity


def queue_checkpoint(run, target, step, cfg, is_best, *, update_latest=True):
    archive = cfg["checkpoint_archive"]
    if not archive.get("repo_id") or not isinstance(archive.get("private"), bool):
        raise ValueError("Configure a relay repository and its visibility explicitly")
    identity = run_identity(run, cfg)
    prefix = f"runs/{identity.get('archive_variant', identity['variant'])}/{identity['run_id']}"
    files = []
    for name in ["checkpoint.pt", "metadata.json"]:
        path = project_path(target / name)
        files.append({"project_path": str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                      "name": name, "remote_path": f"{prefix}/checkpoints/{step:08d}/{name}",
                      "size": path.stat().st_size, "sha256": sha256_file(path)})
    pointer = {"step": step, "checkpoint": files[0]["remote_path"],
               "sha256": files[0]["sha256"], "size": files[0]["size"],
               "selection_metric": "val_l1_zero_latent"}
    job = {"schema": "kroc-relay-job-v1", "job_id": f"{identity['run_id']}-{step:08d}",
           "run_path": str(run), "step": step, "repo_id": archive["repo_id"],
           "private": archive["private"], "remote_prefix": prefix, "files": files,
           "pointers": {f"{prefix}/latest.json": pointer} if update_latest else {},
           "is_best": bool(is_best)}
    if is_best:
        job["pointers"][f"{prefix}/best.json"] = pointer
    queue = project_path(run / "transfer_queue")
    queue.mkdir(exist_ok=True)
    atomic_json(queue / f"{step:08d}.json", job)
    print(json.dumps({"checkpoint_pending_relay": job["job_id"], "step": step}), flush=True)
    return job


def finalize_archive(run, cfg, best_step):
    """Publish only the final best in addition to the periodic checkpoints."""
    archive = cfg.get("checkpoint_archive", {})
    if not archive.get("final_best_only"):
        return
    queue = project_path(run / "transfer_queue")
    best_name = f"{best_step:08d}"
    if not (queue / f"{best_name}.json").exists():
        target = project_path(run / "checkpoints" / best_name)
        queue_checkpoint(run, target, best_step, cfg, True, update_latest=False)
    jobs = [json.loads(path.read_text()) for path in sorted(queue.glob("*.json"))]
    for job in jobs:
        wait_for_receipt(run, job, archive)
    atomic_json(run / "completed.json", {"steps": cfg["steps"], "best_step": best_step,
                                        "verified_uploads": len(jobs)})


def wait_for_receipt(run, job, archive):
    timeout = float(archive.get("wait_timeout_seconds", 3600))
    poll = float(archive.get("poll_seconds", 5))
    if not math.isfinite(timeout) or timeout <= 0 or not 0 < poll <= 60:
        raise ValueError("Positive bounded relay timeout/poll required")
    receipt_path = project_path(run / "upload_receipts" / f"{job['step']:08d}.json")
    started = time.monotonic()
    while True:
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("cleanup_complete"):
                break
        if time.monotonic() - started >= timeout:
            raise TimeoutError("PC relay has not confirmed upload. Local checkpoint retained; resume after relay recovery.")
        time.sleep(max(0.001, min(poll, timeout - (time.monotonic() - started))))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("job_id") != job["job_id"] or not receipt.get("verified"):
        raise RuntimeError("Invalid upload receipt; local checkpoint must be retained")
    print(json.dumps({"checkpoint_archived": job["job_id"],
                      "commit": receipt["commit"], "local_checkpoint_removed": True}), flush=True)
