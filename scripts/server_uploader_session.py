"""Archive the KROC queue directly using a temporary, repo-scoped credential."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import socket
import stat
import sys
import tempfile
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from kroc_act.continuation import MILESTONE, pin_eligible_checkpoint, run_finished
from kroc_act.paths import PROJECT_ROOT, atomic_json
from kroc_act.relay_server import acknowledge, pending_jobs, read_job
from kroc_act.study import sha256_file
from relay_client import upload_and_verify
from training_mirror_session import REPO, configured_runs, snapshot, upload_logs

RAM = Path("/dev/shm/kroc-balanced-hf-session")
ENROLL = RAM / "enroll.sock"
READY = RAM / "ready"
ENABLE = RAM / "enable"
STATUS = PROJECT_ROOT / "logs" / "hub-upload-state.json"
ARCHIVE_SETTINGS = PROJECT_ROOT / "data" / "upload_visibility.json"


def report_error(error):
    http = getattr(getattr(error, "response", None), "status_code", None)
    print(f"Hub retry: {type(error).__name__} (HTTP {http})", flush=True)


def expected_private():
    if not ARCHIVE_SETTINGS.is_file():
        return True
    settings = json.loads(ARCHIVE_SETTINGS.read_text())
    if settings.get("repo_id") != REPO or not isinstance(settings.get("private"), bool):
        raise ValueError("Invalid checkpoint repository visibility setting")
    return settings["private"]


def authorized_client(token):
    api = HfApi(token=token)
    info = api.whoami()
    if info.get("name", "").casefold() != "hyhero":
        raise ValueError("Wrong Hugging Face account")
    access = info.get("auth", {}).get("accessToken", {})
    if str(access.get("role", "")).replace("-", "").casefold() != "finegrained":
        raise ValueError("Expected a fine-grained token")
    scope = access.get("fineGrained") or access.get("fine_grained") or {}
    if scope.get("global"):
        raise ValueError("Expected a token without global permissions")
    if scope.get("canReadGatedRepos") or scope.get("canReadGatedRepositories"):
        raise ValueError("Expected a token restricted to the checkpoint repository")
    scoped = scope.get("scoped") or []
    active = [entry for entry in scoped if entry.get("permissions")]
    accepted = [
        entry for entry in active
        if entry.get("entity", {}).get("name", "").casefold() == REPO.casefold()
        and entry.get("entity", {}).get("type", "model") in {"repo", "model"}
    ]
    if not accepted or len(accepted) != len(active):
        raise ValueError(
            f"Expected permissions only for {REPO}. "
            f"Hub reported {len(scoped)} scope entries, {len(active)} with permissions, "
            f"{len(accepted)} matching the training repo."
        )
    permissions = {permission for entry in accepted for permission in entry["permissions"]}
    if not permissions.intersection({"repo.content.write", "repo.write"}):
        raise ValueError("Enable content write access for the selected training repository")
    if api.repo_info(REPO, repo_type="model").private != expected_private():
        raise ValueError("Hugging Face repository visibility differs from the configured archive")
    return api


def private_ram_dir():
    RAM.mkdir(mode=0o700, exist_ok=True)
    metadata = RAM.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700):
        raise PermissionError("Unexpected credential directory permissions")


def install():
    token = sys.stdin.read(512).strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9]{20,200}", token):
        raise ValueError("Invalid Hugging Face token format")
    private_ram_dir()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(35)
        client.connect(str(ENROLL))
        client.sendall(token.encode("ascii"))
        client.shutdown(socket.SHUT_WR)
        reply = client.recv(2048).decode("utf-8")
    if reply != "OK":
        raise SystemExit(reply or "The upload worker did not confirm authorization.")
    print("Repo-scoped Hugging Face upload credential loaded into the worker.", flush=True)


def receive_client():
    READY.unlink(missing_ok=True)
    ENABLE.unlink(missing_ok=True)
    ENROLL.unlink(missing_ok=True)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(ENROLL))
            ENROLL.chmod(0o600)
            listener.listen(1)
            print("Waiting for repository-scoped authentication.", flush=True)
            while True:
                stream, _ = listener.accept()
                with stream:
                    stream.settimeout(4)
                    try:
                        with stream.makefile("rb") as request:
                            token = request.read(512).decode("ascii").strip()
                        if not re.fullmatch(r"hf_[A-Za-z0-9]{20,200}", token):
                            raise ValueError("Invalid token format")
                        api = authorized_client(token)
                    except ValueError as error:
                        stream.sendall(str(error).encode())
                        continue
                    except Exception as error:
                        report_error(error)
                        stream.sendall(b"Could not verify the training repository permissions.")
                        continue
                    stream.sendall(b"OK")
                    READY.touch(mode=0o600)
                    return api
    finally:
        ENROLL.unlink(missing_ok=True)


@contextmanager
def worker_lock():
    private_ram_dir()
    with (RAM / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def all_complete():
    return all(run_finished(PROJECT_ROOT / "outputs" / f"{name}_balanced")
               for name in configured_runs(PROJECT_ROOT))


def recover_milestone(api, run):
    if (run / "continuation.json").is_file():
        return
    name = f"{MILESTONE:08d}"
    receipt_file = run / "upload_receipts" / f"{name}.json"
    if not receipt_file.is_file():
        return
    receipt = json.loads(receipt_file.read_text())
    if not receipt["cleanup_complete"] or not receipt["verified"] or receipt["repo_id"] != REPO:
        return
    try:
        pin_eligible_checkpoint(run, run / "checkpoints" / name / "checkpoint.pt")
    except FileNotFoundError:
        filename, = (path for path in receipt["files"] if path.endswith(f"/{name}/checkpoint.pt"))
        proof = receipt["files"][filename]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".cache") as directory:
            restored = Path(hf_hub_download(REPO, filename, revision=receipt["commit"],
                                           token=api.token, local_dir=directory))
            if restored.stat().st_size != proof["size"] or sha256_file(restored) != proof["sha256"]:
                raise ValueError("Recovered checkpoint failed immutable Hub receipt verification")
            pin_eligible_checkpoint(run, restored)


def sync(api, state, *, logs_due, checkpoints_due):
    archived = False
    names = configured_runs(PROJECT_ROOT)
    directories = {name + "_balanced" for name in names}
    private = expected_private()
    jobs = []
    for queued in pending_jobs():
        _, run, job = read_job(queued)
        if run.name not in directories:
            continue
        if (job["repo_id"] != REPO or not isinstance(job["private"], bool)
                or private and not job["private"]):
            raise ValueError("Checkpoint destination differs from the configured run repository")
        files = {item["name"]: PROJECT_ROOT / item["project_path"] for item in job["files"]}
        if job["step"] == MILESTONE:
            pin_eligible_checkpoint(run, files["checkpoint.pt"])
        jobs.append((queued, job, files))
    if checkpoints_due:
        block = None
        for queued, job, files in jobs:
            try:
                receipt = upload_and_verify(api, job, files)
            except HfHubHTTPError as error:
                if error.response.headers.get("X-Error-Code") != "StorageLimitReached":
                    raise
                block = "StorageLimitReached"
                print("Hub private storage limit reached. Pending checkpoints retained.", flush=True)
                break
            result = acknowledge(queued, receipt)
            if not result["checkpoint_removed"]:
                raise RuntimeError("Verified checkpoint was not removed")
            print(json.dumps({"uploaded": job["job_id"], "commit": receipt["commit"],
                              "server_checkpoint_removed": True}), flush=True)
            archived = True
        if block != state.get("checkpoint_block"):
            state["checkpoint_block"] = block
            atomic_json(STATUS, state)
    for name in names:
        recover_milestone(api, PROJECT_ROOT / "outputs" / f"{name}_balanced")
    if logs_due or archived or all_complete():
        commit, digests = upload_logs(api, snapshot(PROJECT_ROOT), state.get("digests"))
        if commit:
            state.update(digests=digests, last_log_commit=commit)
            atomic_json(STATUS, state)
            print(f"Verified metric commit: {commit}", flush=True)
        return True
    return False


def run():
    with worker_lock():
        api = receive_client()
        print("Credential verified. Waiting for checkpoint handoff.", flush=True)
        while not ENABLE.exists():
            time.sleep(1)
        print(f"Independent checkpoint uploader ready: {REPO}", flush=True)
        state = json.loads(STATUS.read_text()) if STATUS.is_file() else {}
        next_logs = 0.
        next_checkpoints = 0.
        while True:
            try:
                now = time.monotonic()
                due = now >= next_checkpoints
                if sync(api, state, logs_due=now >= next_logs, checkpoints_due=due):
                    next_logs = time.monotonic() + 300
                if due and state.get("checkpoint_block"):
                    next_checkpoints = time.monotonic() + 120
                if all_complete() and not pending_jobs():
                    commit, digests = upload_logs(api, snapshot(PROJECT_ROOT), state.get("digests"))
                    state.update(digests=digests, completed=True)
                    if commit:
                        state["last_log_commit"] = commit
                    atomic_json(STATUS, state)
                    READY.unlink(missing_ok=True)
                    ENABLE.unlink(missing_ok=True)
                    print("All training runs archived. In-memory credential released.", flush=True)
                    return
            except Exception as error:
                report_error(error)
                time.sleep(30)
                continue
            time.sleep(8)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    try:
        if args.install:
            install()
        else:
            run()
    except Exception as error:
        report_error(error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
