"""Render relocatable Supervisor definitions for the deployed GPU processes."""
import argparse
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECT = {
    "window": "kroc-balanced-window",
    "spectrogram": "kroc-balanced-spectrogram",
    "window_320": "kroc-window-320",
    "window_direction_320": "kroc-window-direction-320",
}


def programs(checkpoints):
    result = {
        "kroc-inference": (["-m", "kroc_remote.server", "--checkpoints", str(checkpoints)], "unexpected"),
        "kroc-balanced-uploader-session": (["scripts/server_uploader_session.py"], "unexpected"),
        "kroc-window-direction": (["scripts/start_window_direction.py"], "false"),
        "kroc-original-baseline": (["scripts/start_original_balanced.py", "baseline"], "false"),
        "kroc-original-concat": (["scripts/start_original_balanced.py", "concat"], "false"),
    }
    for variant, program in DIRECT.items():
        result[program] = (["-m", "kroc_act.train", "--config", f"configs/{variant}_balanced.json"], "false")
    for variant in (*DIRECT, "window_direction"):
        result[f"kroc-continue-{variant.replace('_', '-')}"] = (
            ["-m", "kroc_act.continuation", variant], "false")
    return result


def configure(root, python, output, checkpoints):
    output.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    env = {"PYTHONPATH": f"{root}:{root / 'src'}", "TORCH_HOME": str(root / ".cache/torch"),
           "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env_line = ",".join(f'{key}="{value}"' for key, value in env.items()).replace("%", "%%")
    for name, (arguments, restart) in programs(checkpoints).items():
        nice = [] if name in {"kroc-inference", "kroc-balanced-uploader-session"} else ["nice", "-n", "5"]
        command = shlex.join([*nice, str(python), "-u", *arguments]).replace("%", "%%")
        config = f"""[program:{name}]
directory={str(root).replace('%', '%%')}
command={command}
environment={env_line}
autostart=false
autorestart={restart}
startsecs={5 if name == 'kroc-inference' else 1}
stopasgroup=true
killasgroup=true
stopwaitsecs=20
redirect_stderr=true
stdout_logfile={str(root / 'logs' / (name + '.log')).replace('%', '%%')}
stdout_logfile_maxbytes=10MB
"""
        (output / f"{name}.conf").write_text(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    configure(root, args.python or root / ".venv/bin/python", args.output, args.checkpoints or root / "checkpoints")


if __name__ == "__main__":
    main()
