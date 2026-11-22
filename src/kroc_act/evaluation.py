"""Full zero-latent validation, including episode and condition aggregates."""
import torch


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, direction_head=None):
    model.eval()
    episodes = {}
    raw_sums = None
    valid_steps = 0
    frames = 0
    total_abs, total_elements = 0.0, 0
    if direction_head is not None:
        from .direction import CLASSES, IGNORE, report_confusion

        size = len(CLASSES)
        confusion = torch.zeros(size * size, dtype=torch.int64, device=device)
        direction_total = torch.zeros((), device=device)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        conditions = batch.get("study_condition")
        episode_ids = batch.get("episode_index")
        moved = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        # Prepare normalizes targets without sampling the training VAE posterior.
        prepared = model.prepare(moved, training=True)
        if direction_head is not None:
            target = moved["emg_direction_target"]
            logits = direction_head(prepared["observation.environment_state"])
            direction_total += torch.nn.functional.cross_entropy(logits, target, ignore_index=IGNORE, reduction="sum")
            offset = target.clamp_min(0) * size + logits.argmax(1)
            confusion.scatter_add_(0, offset, target.ne(IGNORE).long())
        prediction = model.policy.predict_action_chunk(prepared)
        mask = ~moved["action_is_pad"].bool()
        errors = (prediction - prepared["action"]).abs()
        masked = errors * mask.unsqueeze(-1)
        totals = masked.sum((1, 2)).cpu()
        counts = (mask.sum(1) * errors.shape[-1]).cpu()
        raw = masked * model.action_std
        sums = raw.sum((0, 1)).cpu()
        raw_sums = sums if raw_sums is None else raw_sums + sums
        valid_steps += int(mask.sum())
        total_abs += float(totals.sum())
        total_elements += int(counts.sum())
        frames += len(mask)
        if episode_ids is not None:
            for index, ep in enumerate(episode_ids.tolist()):
                row = episodes.setdefault(str(ep), {"absolute_error_sum": 0.0,
                    "valid_elements": 0, "frames": 0,
                    "condition": conditions[index] if conditions is not None else "unknown"})
                row["absolute_error_sum"] += float(totals[index])
                row["valid_elements"] += int(counts[index])
                row["frames"] += 1
    if not total_elements:
        raise ValueError("Validation has no valid target elements")
    condition_values = {}
    for row in episodes.values():
        row["l1"] = row["absolute_error_sum"] / row["valid_elements"]
        condition_values.setdefault(row["condition"], []).append(row["l1"])
    condition_means = {key: sum(values) / len(values) for key, values in condition_values.items()}
    result = {"val_l1_zero_latent": total_abs / total_elements,
            "val_condition_macro_l1": (sum(condition_means.values()) / len(condition_means)
                                       if condition_means else total_abs / total_elements),
            "val_frames": frames, "val_valid_elements": total_elements,
            "val_action_mae_original_units": (raw_sums / valid_steps).tolist(),
            "val_by_episode": episodes, "val_by_condition": condition_means}
    if direction_head is not None:
        rows = confusion.reshape(size, size).cpu().numpy()
        count = int(rows.sum())
        if count:
            direction = report_confusion(rows)
            result.update(val_direction_loss=float(direction_total) / count, val_direction_frames=count,
                          val_direction_macro_f1=direction["macro_f1"], val_direction_accuracy=direction["accuracy"],
                          val_direction_recall=direction["recall"], val_direction_confusion=direction["confusion"])
    return result
