"""Restore an uploaded public checkpoint on the server without any HF token."""
import argparse
import json
import tempfile
from pathlib import Path

from kroc_act.paths import PROJECT_ROOT, project_path
from kroc_act.study import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--which', choices=['latest','best'], default='latest')
    args = parser.parse_args()
    run = project_path(args.run)
    pointer = json.loads(project_path(run/f'{args.which}.json').read_text())
    receipt = pointer['remote']
    filename, proof = next((p, v) for p, v in receipt['files'].items() if p.endswith('/checkpoint.pt'))
    target = project_path(run/pointer['checkpoint'])
    if target.is_file():
        if sha256_file(target) != proof['sha256']:
            raise ValueError('Existing checkpoint differs; refusing overwrite')
        print(target)
        return
    temporary_root = project_path(PROJECT_ROOT/'.cache/tmp')
    temporary_root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory(prefix='kroc_restore_',dir=temporary_root) as directory:
        project_path(directory)
        source = project_path(hf_hub_download(repo_id=receipt['repo_id'],repo_type='model',
                    filename=filename,revision=receipt['commit'],token=False,local_dir=directory))
        if source.stat().st_size != proof['size'] or sha256_file(source) != proof['sha256']:
            raise ValueError('Downloaded checkpoint failed hash verification')
        target.parent.mkdir(parents=True,exist_ok=True)
        source.replace(target)
    print(target)


if __name__ == '__main__':
    main()
