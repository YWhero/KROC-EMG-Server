"""Restore a real smoke checkpoint and compare against its pre-save prediction."""
import argparse
import json
from pathlib import Path

import torch

from kroc_act.model import ACTExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(saved["experiment"], device=args.device, backbone_weights=None)
    model = ACTExperiment(cfg, saved["statistics"]).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    probe = saved["verification"]
    if probe is None:
        raise ValueError("Use verify_checkpoints=true for smoke runs")
    batch = {k: v.to(args.device) for k, v in probe["batch"].items()}
    predicted = model.predict_action_chunk(batch).cpu()
    torch.testing.assert_close(predicted, probe["prediction"], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(predicted).all()
    assert predicted.shape[1:] == (30, 6)
    assert saved["optimizer"]["state"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print(json.dumps({"status": "PASS", "variant": cfg["variant"], "step": saved["step"],
                      "prediction_shape": list(predicted.shape),
                      "max_reload_prediction_error": (predicted - probe["prediction"]).abs().max().item(),
                      "checkpoint_bytes": args.checkpoint.stat().st_size,
                      "emg_encoder_saved": any(k.startswith("emg_encoder.") for k in saved["model"])}, indent=2))


if __name__ == "__main__":
    main()
