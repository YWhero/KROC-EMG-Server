"""Start Wave direction training once the two current jobs have finished."""
import os
import subprocess
import sys
import time

from kroc_act.continuation import run_finished
from kroc_act.paths import PROJECT_ROOT


def main():
    runs = ("window", "spectrogram")
    services = [f"kroc-balanced-{name}" for name in runs] + [f"kroc-continue-{name}" for name in runs]
    print("Waiting for Window and Spectrogram to finish before using the shared GPU.", flush=True)
    while True:
        if all(run_finished(PROJECT_ROOT / "outputs" / f"{name}_balanced") for name in runs):
            result = subprocess.run(["supervisorctl", "status", *services], capture_output=True, text=True)
            states = {line.split()[0]: line.split()[1] for line in result.stdout.splitlines() if len(line.split()) > 1}
            if all(states.get(name) in {"EXITED", "STOPPED"} for name in services):
                break
        time.sleep(5)
    cmd = [sys.executable, "-u", "-m", "kroc_act.train", "--config", "configs/window_direction_balanced.json"]
    print("Starting the Wave direction training run.", flush=True)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
