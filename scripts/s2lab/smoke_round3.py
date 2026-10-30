"""第三轮本机冒烟：从第三轮矩阵挑代表配置，在冒烟数据上各训 2400 步（含开发集评价、训完自动评价），看能否跑通与速度。"""
import json
import subprocess
import sys
from pathlib import Path

from _common import ROOT
from satellite_rl.s2lab.matrix import build_matrix_round3

OUT = Path('D:/paper/outputs/experiments/s2lab_round3_20261027/smoke')
DATA = 'D:/paper/outputs/experiments/s2lab_20261024/smoke_data'
PICK = sys.argv[1:] or ['r3-r0', 'r3-hist', 'r3-legal', 'r3-cond', 'r3-condtop8', 'r3-p1', 'r3-p2', 'r3-p3', 'r3-wide', 'r3-f-all',
                        'r3-x-noglobal', 'r3-x-nointf', 'r3-auto', 'r3-ppo', 'r3-dqn', 'r3-mlpsac']
runs = {r['run_id']: r for r in build_matrix_round3()}
OUT.mkdir(parents=True, exist_ok=True)
for name in PICK:
    run = dict(runs[f'{name}-s1'], total_steps=2400, dev_every=1200, log_every=600)
    run['agent'] = dict(run['agent'], warmup=600)
    cfg = OUT / f'{name}.json'
    cfg.write_text(json.dumps(run), encoding='utf-8')
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/s2lab/train_run.py'), '--runs-dir', str(OUT), '--dataset-dir', DATA,
                           '--config', str(cfg), '--device', 'cuda'], capture_output=True, text=True, encoding='utf-8', errors='replace')
    lines = (proc.stdout + proc.stderr).splitlines()
    sps = [l.split('sps=')[1].split()[0] for l in lines if 'sps=' in l]
    tail = [l for l in lines if any(k in l for k in ('eval confirm best_sgm', 'parameters', 'Error', 'FAILED', 'Traceback'))]
    print(f'== {name} rc={proc.returncode} sps={sps[-2:]}', *tail[-4:], sep='\n  ', flush=True)
