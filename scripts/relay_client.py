"""Run on the personal PC. Never install/copy this token to the GPU server.

python scripts/relay_client.py --repo HYHero/kroc-act-emg-checkpoints
Token is read without echo, kept in this process, and never written to disk.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

# Prefer ordinary HTTPS/LFS uploads. Some PC networks cannot reach Xet CAS.
# Set before importing huggingface_hub so both paths and pointer uploads use HTTP.
os.environ['HF_HUB_DISABLE_XET'] = '1'


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    if algorithm == "sha1":
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ssh_arguments(args, action, job=None, name=None):
    command = [args.remote_python, "-B", "-m", "kroc_act.relay_server", action]
    if job is not None:
        command += ["--job", job]
    if name is not None:
        command += ["--name", name]
    root = shlex.quote(args.remote_root)
    python_path = shlex.quote(args.remote_root.rstrip("/") + "/src")
    remote = (f"cd {root} && env PYTHONPATH={python_path} "
              "PYTHONDONTWRITEBYTECODE=1 " + " ".join(shlex.quote(x) for x in command))
    return ["ssh", "-F", "/dev/null", "-T", "-p", str(args.port), "-o", "BatchMode=yes", "-o",
            "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", args.host, remote]


def child_environment():
    # Even a broad local SSH SendEnv setting must never forward an HF credential.
    return {key: value for key, value in os.environ.items()
            if key.upper() not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KROC_RELAY_HF_TOKEN"}}


def remote_json(args, action, job=None, payload=None):
    result = subprocess.run(ssh_arguments(args, action, job),
        input=json.dumps(payload).encode() if payload is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, env=child_environment())
    lines = result.stdout.decode("utf-8").splitlines()
    marked = [line[len("KROC_JSON="):] for line in lines if line.startswith("KROC_JSON=")]
    if not marked:
        raise RuntimeError("No valid response from KROC relay endpoint")
    return json.loads(marked[-1])


def ensure_repository(api, repo_id, private=False):
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    if bool(info.private) != private:
        raise RuntimeError("Existing repository visibility differs; refusing to change it automatically")


def upload_and_verify(api, job, local_files):
    from huggingface_hub import CommitOperationAdd
    operations = []
    expected = {}
    for item in job["files"]:
        local = local_files[item["name"]]
        if local.stat().st_size != item["size"] or digest(local) != item["sha256"]:
            raise RuntimeError("Transferred file does not match the queued SHA256")
        operations.append(CommitOperationAdd(path_in_repo=item["remote_path"], path_or_fileobj=str(local)))
        expected[item["remote_path"]] = {"size": item["size"], "sha256": item["sha256"],
                                          "git_blob": digest(local, "sha1")}
    for path, value in job["pointers"].items():
        body = json.dumps(value, sort_keys=True).encode()
        operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=body))
        expected[path] = {"size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                         "git_blob": hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest()}
    commit = api.create_commit(repo_id=job["repo_id"], repo_type="model", operations=operations,
                               commit_message=f"Archive {job['job_id']} (verified relay)")
    # Read metadata from the exact immutable commit, never from a moving main branch.
    remote = api.get_paths_info(repo_id=job["repo_id"], paths=list(expected), repo_type="model",
                                revision=commit.oid)
    actual = {item.path: item for item in remote}
    for path, wanted in expected.items():
        item = actual.get(path)
        if item is None or item.size != wanted["size"]:
            raise RuntimeError("Remote file missing or wrong size; server copy must stay")
        if item.lfs is not None:
            remote_sha = item.lfs.get("sha256") if isinstance(item.lfs, dict) else item.lfs.sha256
            if remote_sha != wanted["sha256"]:
                raise RuntimeError("Remote LFS SHA256 mismatch; server copy must stay")
        elif item.blob_id != wanted["git_blob"]:
            raise RuntimeError("Remote Git object hash mismatch; server copy must stay")
    return {"job_id": job["job_id"], "repo_id": job["repo_id"], "commit": commit.oid,
            "verified": True, "files": {item["remote_path"]:
                {"sha256": item["sha256"], "size": item["size"]} for item in job["files"]}}


def process_job(args, api, relative_job, spool):
    job = remote_json(args, "job", relative_job)
    if job["repo_id"] != args.repo or job["private"] is not args.private:
        raise RuntimeError("Queue destination differs from the selected repository and visibility")
    with tempfile.TemporaryDirectory(prefix="checkpoint_", dir=spool) as directory:
        workspace = Path(directory).resolve()
        if not workspace.is_relative_to(spool.resolve()):
            raise RuntimeError("Temporary transfer path escaped the relay workspace")
        local_files = {}
        for item in job["files"]:
            if item["name"] not in ["checkpoint.pt", "metadata.json"]:
                raise RuntimeError("Unexpected transfer file")
            local = workspace / item["name"]
            with local.open("wb") as output:
                subprocess.run(ssh_arguments(args, "file", relative_job, item["name"]),
                    stdout=output, stderr=subprocess.DEVNULL, check=True, env=child_environment())
            local_files[item["name"]] = local
        receipt = upload_and_verify(api, job, local_files)
        acknowledgment = remote_json(args, "ack", relative_job, receipt)
        if not acknowledgment.get("checkpoint_removed"):
            raise RuntimeError("Server has not confirmed cleanup")
        print(json.dumps({"uploaded": job["job_id"], "commit": receipt["commit"],
                          "server_checkpoint_removed": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="HYHero/kroc-act-emg-checkpoints")
    parser.add_argument("--host", required=True, help="SSH destination in user@host form")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="List server jobs without requesting an HF token")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--use-local-login", action="store_true")
    parser.add_argument("--remote-root", default="/workspace/KROC")
    parser.add_argument("--remote-python", default=".venv/bin/python")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(remote_json(args, "list"), indent=2))
        return
    from huggingface_hub import HfApi, get_token
    token = get_token() if args.use_local_login else (
        os.environ.get("KROC_RELAY_HF_TOKEN") or getpass.getpass("HF token (PC memory only, not saved): ")
    )
    if not token:
        raise ValueError("A write token for the selected repository is required")
    api = HfApi(token=token)
    try:
        if api.whoami().get("name", "").casefold() != args.repo.split("/")[0].casefold():
            raise ValueError("Local Hugging Face login belongs to a different account")
        ensure_repository(api, args.repo, private=args.private)
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(f"Local Hugging Face login not ready: {type(error).__name__} HTTP {status}", flush=True)
        raise SystemExit(1) from None
    print(f"Relay ready: https://huggingface.co/{args.repo}; no token sent to server", flush=True)
    spool = Path(__file__).resolve().parents[1] / ".relay_tmp"
    spool.mkdir(exist_ok=True)
    if spool.is_symlink():
        raise ValueError("Relay spool must not be a symlink")
    while True:
        try:
            jobs = remote_json(args, "list")
            for job in jobs:
                process_job(args, api, job, spool)
        except Exception as error:
            # Do not echo HTTP headers, token material, or credential-bearing exceptions.
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            detail = f" HTTP {status}" if status is not None else ''
            print(f"Relay pending: {type(error).__name__}{detail}; server files retained; retrying", flush=True)
            if args.once:
                raise SystemExit(1) from None
        if args.once:
            break
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
