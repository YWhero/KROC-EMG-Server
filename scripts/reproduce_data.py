"""Rebuild the shared LeRobot + EMG dataset without overwriting tracked study files."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from kroc_act.paths import project_path

SOURCE = 'HYHero/red_pen_adaptive_handover_202611141907'
REVISION = '8953cd81be67e3605f3b0ecdcc3e1875464250d6'


def main():
    os.chdir(ROOT)
    parts = ['lerobot_v3', 'emg', 'conversion_report.json']
    destinations = [project_path(ROOT/'data'/name) for name in parts]
    if all(path.exists() for path in destinations):
        print('Converted artifacts already exist; preserving data, annotations and frozen split.')
        return
    if any(path.exists() for path in destinations):
        raise RuntimeError('Partial data export exists; refusing overwrite. Inspect data/ first.')
    from huggingface_hub import snapshot_download
    raw = project_path(ROOT/'raw/red_pen_adaptive_handover_202611141907')
    if raw.exists():
        for parent, directories, files in os.walk(raw, followlinks=False):
            for name in directories + files:
                if (Path(parent)/name).is_symlink():
                    raise ValueError('Refusing symlink in the raw dataset tree')
    snapshot_download(repo_id=SOURCE,repo_type='dataset',revision=REVISION,
                      local_dir=str(raw),token=False)
    # The existing converter requires a nonexistent output directory.
    staging = project_path(ROOT/'.cache/data_conversion_staging')
    if staging.exists():
        raise RuntimeError('Conversion staging already exists; inspect interrupted conversion before retrying.')
    subprocess.run([sys.executable,'-m','kroc_act.convert_raw','--raw',str(raw),
                    '--output',str(staging),'--repo-id','dorii22/red_pen_adaptive_handover_lerobot'],check=True)
    for source, destination in zip(parts,destinations):
        project_path(staging/source).replace(destination)
    # The converter's historical random split is not the approved research split.
    project_path(staging/'splits.json').unlink()
    staging.rmdir()
    print('Shared dataset rebuilt; tracked study annotations and stratified split preserved.')


if __name__ == '__main__':
    main()
