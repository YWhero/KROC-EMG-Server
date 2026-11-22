"""Download and verify the public checkpoints used by the original GPU service."""
import argparse
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

from kroc_act.paths import PROJECT_ROOT
from kroc_act.study import sha256_file


def fetch(spec, destination):
    modes = ("baseline", "concat", "window")
    if len(spec["files"]) != len(modes):
        raise ValueError("Expected exactly the three deployed inference modes")
    destination.mkdir(parents=True, exist_ok=True)
    for mode, file in zip(modes, spec["files"], strict=True):
        origin = PurePosixPath(file["path"])
        if origin.parts[:2] != ("runs", mode) or origin.name != "checkpoint.pt":
            raise ValueError(f"Unexpected checkpoint path for {mode}")
        target = destination / mode
        target.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=target) as temporary:
            options = dict(repo_id=spec["repo_id"], revision=spec["revision"], token=False, local_dir=temporary)
            bundle = target / "checkpoint.pt"
            if bundle.is_file():
                checkpoint = bundle
            else:
                checkpoint = Path(hf_hub_download(filename=str(origin), **options))
            if checkpoint.stat().st_size != file["size"] or sha256_file(checkpoint) != file["sha256"]:
                raise ValueError(f"Verification failed for {mode}")
            metadata = Path(hf_hub_download(filename=str(origin.with_name("metadata.json")), **options))
            info = json.loads(metadata.read_text())
            if info["step"] != 100_000 or info["experiment"]["variant"] != mode:
                raise ValueError(f"Unexpected checkpoint metadata for {mode}")
            if checkpoint != bundle:
                checkpoint.replace(bundle)
            metadata.replace(target / "metadata.json")
        print(f"Verified {mode} checkpoint", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs/inference_models.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints")
    args = parser.parse_args()
    fetch(json.loads(args.manifest.read_text()), args.output)


if __name__ == "__main__":
    main()
