"""Single-GPU ACT training with full validation and verified PC-relay archival."""
import argparse
import json
import math
import random
from pathlib import Path

from .checkpoints import finalize_archive, run_identity, save_checkpoint
from .paths import PROJECT_ROOT, atomic_json, project_path
from .study import sha256_file


def resolve_config(path):
    path = project_path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ["dataset_root", "manifest", "splits", "output_dir", "annotations", "frame_cache",
                "direction_targets", "receiving_annotations"]:
        if cfg.get(key) is not None:
            cfg[key] = str(project_path(path.parent / cfg[key]))
    if cfg["keep_checkpoints"] < 1 or cfg["save_freq"] < 1:
        raise ValueError("Positive checkpoint frequency/retention required")
    for key in ["steps", "batch_size", "eval_freq", "log_freq"]:
        if cfg[key] < 1:
            raise ValueError(f"Positive {key} required")
    if cfg.get("val_batches") is not None and cfg["val_batches"] < 1:
        raise ValueError("val_batches must be null (full validation) or positive")
    if cfg.get("checkpoint_archive", {}).get("mode", "relay") != "relay":
        raise ValueError("Only token-free PC relay archival is supported")
    if cfg.get("checkpoint_archive") and cfg.get("verify_checkpoints"):
        raise ValueError("Public archived checkpoints must not include raw verification probes")
    if "direction_loss_weight" in cfg:
        weight = cfg["direction_loss_weight"]
        if (not math.isfinite(weight) or weight <= 0 or cfg["variant"] != "window"
                or not all(cfg.get(key) for key in ("frame_cache", "receiving_annotations", "direction_targets"))):
            raise ValueError("Direction supervision requires Wave, aligned phase targets, and a positive weight")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, help="Trusted local checkpoint.pt; new epoch boundary")
    parser.add_argument("--extend-to", type=int, help="Raise only the completed run's step limit")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    steps = cfg["steps"]
    if args.extend_to is not None:
        if cfg.get("checkpoint_archive", {}).get("fixed_step_limit"):
            parser.error("This experiment has a fixed training limit")
        if args.resume is None or args.extend_to <= steps:
            parser.error("--extend-to requires --resume and a higher step limit")
        steps = args.extend_to
    end_step = min(steps, args.stop_after) if args.stop_after is not None else steps
    if end_step <= 0:
        raise ValueError("Positive --stop-after required")
    if args.dry_run:
        print(json.dumps(dict(cfg, steps=steps), indent=2, ensure_ascii=False))
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .data import SharedDataset
    from .evaluation import evaluate
    from .model import ACTExperiment

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if cfg.get("frame_cache"):
        from .frame_cache import CachedDataset
        dataset_cls = CachedDataset
    else:
        dataset_cls = SharedDataset
    train, val = dataset_cls(cfg, "train"), dataset_cls(cfg, "val")
    cfg.update(state_dim=train.features["observation.state"]["shape"][0],
               action_dim=train.features["action"]["shape"][0],
               emg_channels=len(train.manifest["channels"]),
               camera_shapes={k: train.features[k]["shape"] for k in train.camera_keys})
    provenance = {key: sha256_file(cfg[key]) for key in ["manifest", "splits", "annotations",
                  "direction_targets", "receiving_annotations"] if cfg.get(key)}
    provenance.update(lerobot_commit="240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b",
                      train_frames=len(train), val_frames=len(val),
                      train_excluded=train.excluded, val_excluded=val.excluded)
    run = project_path(cfg["output_dir"])
    resume = None
    if args.resume:
        resume = torch.load(project_path(args.resume), map_location="cpu", weights_only=False)
        if resume["experiment"] != cfg or resume["provenance"] != provenance:
            raise ValueError("Resume configuration/data provenance differs")
        if args.extend_to is not None and resume["step"] != cfg["steps"]:
            raise ValueError("The step limit can only be extended from a completed run")
        statistics = resume["statistics"]
    else:
        if run.exists():
            raise FileExistsError(f"Choose a new output_dir: {run}")
        statistics = train.train_statistics()
        run.mkdir(parents=True)
    cfg["steps"] = steps
    model = ACTExperiment(cfg, statistics).to(device)
    groups = model.optimizer_groups()
    direction_head = None
    class_weights = None
    parameters = list(model.parameters())
    if "direction_loss_weight" in cfg:
        from .direction import CLASSES, classification_loss

        counts = torch.tensor([train.direction_counts[name] for name in CLASSES], dtype=torch.float32)
        if not bool((counts > 0).all()):
            raise ValueError("Direction supervision requires labeled training frames for all classes")
        class_weights = (counts.sum() / (len(CLASSES) * counts)).to(device)
        direction_head = torch.nn.Linear(cfg["dim_model"], len(CLASSES)).to(device)
        direction_parameters = list(direction_head.parameters())
        groups.append({"params": direction_parameters})
        parameters.extend(direction_parameters)
    optimizer = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    step = 0
    best_value, best_step = None, None
    resume_to_clean = None
    if resume:
        model.load_state_dict(resume["model"])
        if direction_head is not None:
            direction_head.load_state_dict(resume["direction_head"])
        optimizer.load_state_dict(resume["optimizer"])
        step = resume["step"]
        torch.set_rng_state(resume["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        generator.set_state(resume["loader_rng"])
        best_value = resume.get("training_state", {}).get("best_val_l1")
        best_step = resume.get("training_state", {}).get("best_step")
        if args.extend_to is not None and args.resume == run / ".resume_100k.pt":
            resume_to_clean = args.resume
        del resume
    records = [("experiment.json", cfg), ("statistics.json", statistics), ("provenance.json", provenance),
               ("splits.json", json.loads(project_path(cfg["splits"]).read_text()))]
    if cfg.get("checkpoint_archive", {}).get("fixed_step_limit"):
        records.append(("continuation.json", {"extend": False, "target_steps": cfg["steps"],
                                              "reason": "fixed_step_limit"}))
    if train.annotations:
        records += [("study_annotations.json", train.annotations),
                    ("split_summary.json", {"episode_counts": train.split_condition_counts,
                     "selection_metric": "val_l1_zero_latent", "validation": "all eligible frames",
                     "test_set": "not present; validation selects models and is not independent testing"})]
    if direction_head is not None:
        records += [("receiving_annotations.json", json.loads(Path(cfg["receiving_annotations"]).read_text())),
                    ("direction_supervision.json", {"classes": CLASSES, "train_frames_by_class": train.direction_counts,
                     "val_frames_by_class": val.direction_counts, "weight": cfg["direction_loss_weight"],
                     "train_class_weights": dict(zip(CLASSES, class_weights.tolist(), strict=True)),
                     "input": "Wave EMG token before vision fusion", "inference": "unchanged ACT action model"})]
    sources = sorted((PROJECT_ROOT / "src/kroc_act").glob("*.py"))
    records.append(("code_provenance.json", {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sources}))
    for name, content in records:
        atomic_json(run / name, content)
    if cfg.get("checkpoint_archive"):
        run_identity(run, cfg)
    workers = cfg.get("loader_workers", 0)
    opts = dict(num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
                        drop_last=False, **opts)
    val_loader = DataLoader(val, batch_size=cfg["batch_size"], shuffle=False, **opts)
    def move(batch):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
    iterator = iter(loader)
    with project_path(run / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while step < end_step:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            moved = move(batch)
            if direction_head is None:
                loss, metrics = model(moved)
            else:
                prepared = model.prepare(moved, training=True)
                policy_loss, metrics = model.policy(prepared)
                logits = direction_head(prepared["observation.environment_state"])
                direction_loss, direction_frames = classification_loss(logits, moved["emg_direction_target"], class_weights)
                loss = policy_loss + cfg["direction_loss_weight"] * direction_loss
                metrics.update(policy_loss=policy_loss.item(), direction_loss=direction_loss.item(),
                               direction_frames=int(direction_frames))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            record = {"step": step, "loss": loss.item(), "grad_norm": float(norm), **metrics}
            if cfg["variant"] in {"window", "spectrogram"}:
                record["emg_grad_norm"] = sum(p.grad.detach().square().sum().item()
                    for p in model.emg_encoder.parameters() if p.grad is not None) ** 0.5
            validation, improved = None, False
            if step % cfg["eval_freq"] == 0 or step == end_step:
                validation = evaluate(model, val_loader, device, cfg.get("val_batches"), direction_head)
                value = validation["val_l1_zero_latent"]
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite validation metric")
                improved = best_value is None or value < best_value
                if improved:
                    best_value, best_step = value, step
                record.update(validation)
                record.update(best_val_l1=best_value, best_step=best_step)
            if step % cfg["log_freq"] == 0 or validation is not None:
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                if resume_to_clean is not None:
                    resume_to_clean.unlink(missing_ok=True)
                    resume_to_clean = None
            if step % cfg["save_freq"] == 0 or step == end_step or improved:
                save_checkpoint(run, step, model, optimizer, cfg, statistics, provenance, generator,
                                probe=moved if cfg.get("verify_checkpoints", False) else None,
                                training_state={"best_val_l1": best_value, "best_step": best_step},
                                is_best=improved, validation=validation,
                                direction_head=direction_head)
        if end_step == cfg["steps"]:
            finalize_archive(run, cfg, best_step)


if __name__ == "__main__":
    main()
