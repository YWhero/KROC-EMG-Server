"""Read-only installation/data checks; never starts training or uploads."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.chdir(ROOT)
import torch
from kroc_act.train import resolve_config
from kroc_act.study import read_json,validate_study

cfg=resolve_config('configs/01_baseline.json')
annotations=read_json(cfg['annotations']);splits=read_json(cfg['splits'])
counts=validate_study(annotations,splits,annotations['episode_conditions'])
result={'python':sys.version.split()[0],'torch':torch.__version__,
        'cuda_available':torch.cuda.is_available(),'split_condition_counts':counts,
        'training_started':False}
if (ROOT/'data/lerobot_v3/meta/info.json').is_file():
    from kroc_act.data import SharedDataset
    train=SharedDataset(cfg,'train');val=SharedDataset(cfg,'val')
    result.update(train_frames=len(train),val_frames=len(val))
else:
    result['data_status']='not rebuilt yet; run scripts/reproduce_data.py'
print(json.dumps(result,indent=2))
