"""脚本共用：把 src 放进路径、默认远端/本地路径。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

REMOTE_HOST = '10.160.17.102'
REMOTE_USER = 'u2025140002'
REMOTE_PORTS = (20654, 20655, 20627, 20653)          # 四个容器，各一张 GPU，共享 /home
REMOTE_PY = '/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python'
REMOTE_ROOT = '/home/u2025140002/s2lab'               # repo/ data/ runs/ logs/
REMOTE_S2FIX_DATA = '/home/u2025140002/s2fix_20261018/data'
LOCAL_OUT = Path('D:/paper/outputs/experiments/s2lab_20261024')
DATASET_NAME = 'v1'


def default_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'
