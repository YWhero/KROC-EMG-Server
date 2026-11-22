"""Restricted SSH endpoint: enumerate jobs, transfer approved files, confirm cleanup.

No Hugging Face credentials or client code ever run on this server.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from .paths import PROJECT_ROOT, atomic_json, checkpoint_pointer_lock, project_path
from .study import sha256_file


def job_path(value):
    path = project_path(PROJECT_ROOT / value)
    relative = path.relative_to(PROJECT_ROOT)
    if (len(relative.parts) != 4 or relative.parts[0] != "outputs"
            or relative.parts[2] != "transfer_queue"
            or not re.fullmatch(r"\d{8}\.json", relative.parts[3])):
        raise ValueError("Only direct experiment transfer-queue jobs are accepted")
    return path


def read_job(value):
    path = job_path(value)
    job = json.loads(path.read_text(encoding="utf-8"))
    run = project_path(path.parent.parent)
    if project_path(job["run_path"]) != run or job["step"] != int(path.stem):
        raise ValueError("Job/run identity mismatch")
    expected = project_path(run / "checkpoints" / path.stem)
    if [item["name"] for item in job["files"]] != ["checkpoint.pt", "metadata.json"]:
        raise ValueError("Unexpected transfer file allowlist")
    for item in job["files"]:
        if project_path(PROJECT_ROOT / item["project_path"]) != expected / item["name"]:
            raise ValueError("Transfer path must match this checkpoint")
    return path, run, job


def pending_jobs():
    result = []
    outputs = project_path(PROJECT_ROOT / "outputs")
    for run in sorted(outputs.iterdir()):
        if run.is_symlink() or not run.is_dir():
            continue
        queue = project_path(run / "transfer_queue")
        if not queue.is_dir():
            continue
        for path in sorted(queue.iterdir()):
            if path.is_symlink() or not re.fullmatch(r"\d{8}\.json", path.name):
                continue
            receipt = project_path(run / "upload_receipts" / path.name)
            if not receipt.is_file() or not json.loads(receipt.read_text()).get("cleanup_complete"):
                result.append(str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'))
    return result


def acknowledge(value, receipt):
    _, run, job = read_job(value)
    if (receipt.get("job_id") != job["job_id"] or receipt.get("repo_id") != job["repo_id"]
            or receipt.get("verified") is not True or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("commit", ""))):
        raise ValueError("Receipt identity/commit is invalid")
    expected = {item["remote_path"]: {"sha256": item["sha256"], "size": item["size"]}
                for item in job["files"]}
    if receipt.get("files") != expected:
        raise ValueError("Receipt does not verify every queued file")
    checkpoint = project_path(PROJECT_ROOT / job["files"][0]["project_path"])
    if checkpoint.is_file() and (checkpoint.stat().st_size != job["files"][0]["size"]
                                or sha256_file(checkpoint) != job["files"][0]["sha256"]):
        raise ValueError("Local checkpoint changed; refusing cleanup")
    # Persist proof before deleting. A retry can safely complete interrupted cleanup.
    directory = project_path(run / "upload_receipts")
    directory.mkdir(exist_ok=True)
    receipt = dict(receipt, cleanup_complete=False)
    atomic_json(directory / f"{job['step']:08d}.json", receipt)
    if checkpoint.is_file():
        checkpoint.unlink()
    receipt["cleanup_complete"] = True
    with checkpoint_pointer_lock(run):
        for name in ["latest.json", "best.json"]:
            path = project_path(run / name)
            if path.is_file():
                pointer = json.loads(path.read_text())
                if pointer["step"] == job["step"]:
                    pointer["remote"] = receipt
                    atomic_json(path, pointer)
    atomic_json(directory / f"{job['step']:08d}.json", receipt)
    return {"acknowledged": job["job_id"], "checkpoint_removed": not checkpoint.exists()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["list", "job", "file", "ack"])
    parser.add_argument("--job")
    parser.add_argument("--name", choices=["checkpoint.pt", "metadata.json"])
    args = parser.parse_args()
    if args.action == "list":
        print("KROC_JSON=" + json.dumps(pending_jobs()))
        return
    _, _, job = read_job(args.job)
    if args.action == "job":
        print("KROC_JSON=" + json.dumps(job))
    elif args.action == "file":
        item = next(item for item in job["files"] if item["name"] == args.name)
        path = project_path(PROJECT_ROOT / item["project_path"])
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError("Queued file changed")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                sys.stdout.buffer.write(block)
    else:
        receipt = json.loads(sys.stdin.read())
        print("KROC_JSON=" + json.dumps(acknowledge(args.job, receipt)))


if __name__ == "__main__":
    main()
