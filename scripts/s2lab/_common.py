"""脚本共用：把 src 放进路径、默认远端/本地路径。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

REMOTE_HOST = '10.160.17.102'
REMOTE_USER = 'u2025140002'
REMOTE_PORTS = (20654, 20655, 20627, 20653, 20780, 20781, 20782, 20783)   # 八个容器，各一张 4090D，共享 /home（后四个 10-24 13:00 加入）
REMOTE_PY = '/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python'
REMOTE_ROOT = '/home/u2025140002/s2lab'               # repo/ data/ runs/ logs/
REMOTE_S2FIX_DATA = '/home/u2025140002/s2fix_20261018/data'
LOCAL_OUT = Path('D:/paper/outputs/experiments/s2lab_20261024')
DATASET_NAME = 'v1'

# 第二轮两个账号（同一主机、同一口令 BUPT_PASSWORD、/home 各自独立）
ACCOUNTS = {
    'A': dict(user='u2025140002', ports=(20654, 20655, 20627, 20653, 20780, 20781, 20782, 20783),
              py='/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python', root='/home/u2025140002/s2lab'),
    # 第四轮（10-28）可用：20802、20804—20809、20878（20803 不在其中）
    'B': dict(user='u2024018003', ports=(20802, 20804, 20805, 20806, 20807, 20808, 20809, 20878),
              py='/opt/conda/envs/pytorch-2.1.1/bin/python', root='/home/u2024018003/s2lab'),
}
LOCAL_OUT2 = Path('D:/paper/outputs/experiments/s2lab_round2_20261025')


def default_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'
