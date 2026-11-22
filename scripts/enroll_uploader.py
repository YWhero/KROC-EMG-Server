"""Send a repository-scoped token to the GPU uploader through SSH stdin."""
import argparse
import getpass
import re
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", required=True, help="OpenSSH alias or user@host")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", help="Defaults to the repository's .venv/bin/python")
    args = parser.parse_args()
    root = args.remote_root.rstrip("/")
    python = args.remote_python or f"{root}/.venv/bin/python"
    command = shlex.join(["env", f"PYTHONPATH={root}/src", python, "-u",
                          "scripts/server_uploader_session.py", "--install"])
    remote = (f"cd {shlex.quote(root)} && {command} && "
              "test -f /dev/shm/kroc-balanced-hf-session/ready && "
              "touch /dev/shm/kroc-balanced-hf-session/enable")
    token = getpass.getpass("Repository-scoped Hugging Face token (hidden): ").strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9]{20,200}", token):
        parser.error("Expected a Hugging Face fine-grained token")
    subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "--", args.ssh_target, remote], input=token.encode("ascii"), check=True)


if __name__ == "__main__":
    main()
