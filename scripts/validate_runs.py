"""Run 3 short real-data experiments, resume each, verify every saved prediction."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

import torch

from kroc_act.train import resolve_config


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_dir = root / "configs" / f"verification_{stamp}"
    config_dir.mkdir()
    output_root = root / "outputs" / f"verification_{stamp}"
    reports = []
    for original in sorted((root / "configs").glob("0*.json")):
        cfg = resolve_config(original)
        cfg.update(steps=20, batch_size=2, save_freq=5, eval_freq=5,
                   val_batches=2, log_freq=1, verify_checkpoints=True,
                   output_dir=str(output_root / original.stem))
        path = config_dir / original.name
        path.write_text(json.dumps(cfg,indent=2))
        run = Path(cfg["output_dir"])
        log = root / "logs" / f"verify_{original.stem}_{stamp}.log"
        command = [sys.executable,"-m","kroc_act.train","--config",str(path)]
        print(f"Starting {cfg['variant']} steps 1-10",flush=True)
        with log.open("w") as stream:
            subprocess.run([*command,"--stop-after","10"],stdout=stream,stderr=subprocess.STDOUT,check=True)
            middle = run / "checkpoints/00000010/checkpoint.pt"
            subprocess.run([*command,"--resume",str(middle)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        checkpoints = sorted((run / "checkpoints").glob("*/checkpoint.pt"))
        assert [p.parent.name for p in checkpoints] == ["00000010","00000015","00000020"]
        comparisons = []
        for checkpoint in checkpoints:
            output = subprocess.check_output([sys.executable,str(root / "scripts/verify_checkpoint.py"),str(checkpoint)],text=True)
            comparisons.append(json.loads(output))
        initial = torch.load(middle,map_location="cpu",weights_only=False)
        final = torch.load(checkpoints[-1],map_location="cpu",weights_only=False)
        assert final["step"] == 20 and initial["step"] == 10
        assert any(not torch.equal(initial["model"][k],final["model"][k])
                   for k in initial["model"] if k.startswith("policy.") and initial["model"][k].is_floating_point())
        optimizer_steps = sorted({int(s["step"].item()) for s in final["optimizer"]["state"].values()})
        assert optimizer_steps == [20], optimizer_steps
        encoder_updated = None
        if cfg["variant"] == "window":
            encoder_updated = any(not torch.equal(initial["model"][k],final["model"][k])
                                  for k in initial["model"] if k.startswith("emg_encoder."))
            assert encoder_updated
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert [r["step"] for r in metrics] == list(range(1,21))
        assert all(torch.isfinite(torch.tensor(r["loss"])) for r in metrics)
        if cfg["variant"] == "window":
            assert all(r["emg_grad_norm"] > 0 for r in metrics)
        report = {"variant":cfg["variant"],"status":"PASS","steps":20,
                  "chunk_size":cfg["chunk_size"],"n_action_steps":cfg["n_action_steps"],
                  "resume_from_step":10,"optimizer_steps":optimizer_steps,
                  "first_loss":metrics[0]["loss"],"final_loss":metrics[-1]["loss"],
                  "val_l1":metrics[-1]["val_l1_zero_latent"],"emg_encoder_updated_after_resume":encoder_updated,
                  "checkpoints":comparisons,"run":str(run),"log":str(log)}
        reports.append(report)
        (root / "logs/verification_report.json").write_text(json.dumps(reports,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        del initial, final
    print("ALL THREE REAL-DATA TRAINING/RESUME/CHECKPOINT CHECKS PASSED",flush=True)


if __name__ == "__main__":
    main()
