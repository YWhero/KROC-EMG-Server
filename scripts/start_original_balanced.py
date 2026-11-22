"""Start one original KROC experiment after both 320ms experiments exit."""
import argparse
import os
import subprocess
import sys
import time

from kroc_act.continuation import run_finished
from kroc_act.paths import PROJECT_ROOT


PREREQUISITES = ("window_320", "window_direction_320")


def gpu_training_finished():
    outputs = PROJECT_ROOT / "outputs"
    if not all(run_finished(outputs / f"{run}_balanced") for run in PREREQUISITES):
        return False
    services = [f"kroc-{prefix}{run.replace('_', '-')}" for prefix in ("", "continue-") for run in PREREQUISITES]
    status = subprocess.run(["supervisorctl", "status", *services], capture_output=True, text=True)
    states = dict(line.split()[:2] for line in status.stdout.splitlines() if len(line.split()) >= 2)
    return all(states.get(name) in {"EXITED", "STOPPED"} for name in services)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", "concat"))
    variant = parser.parse_args().variant
    print(f"{variant} waits for both registered 320ms experiments to finish.", flush=True)
    while not gpu_training_finished():
        time.sleep(5)
    print(f"Starting the balanced original {variant} run.", flush=True)
    command = [sys.executable, "-u", "-m", "kroc_act.train", "--config", f"configs/{variant}_balanced.json"]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
