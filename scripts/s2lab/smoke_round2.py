"""第二轮本机冒烟：从第二轮矩阵挑代表运行，在冒烟数据上各训 2400 步（含开发集评价、检查点、训完自动评价），只看能否跑通与速度。"""
import json
import subprocess
import sys
from pathlib import Path

from _common import ROOT
from satellite_rl.s2lab.matrix import build_matrix_round2

OUT = Path('D:/paper/outputs/experiments/s2lab_round2_20261025/smoke')
DATA = 'D:/paper/outputs/experiments/s2lab_20261024/smoke_data'
PICK = sys.argv[1:] or ['t14-dbw-paper-s1', 'avoid-topk-s1', 'avoid-intf-s1', 'power100-s1', 'nomask-tsac-s1', 'fix-cnn-auto-s1',
                        'token-beams-s1', 'main-ppo-s1', 'main-dqn-s1', 'avoid-rule-s1']
runs = {r['run_id']: r for r in build_matrix_round2()}
OUT.mkdir(parents=True, exist_ok=True)
for run_id in PICK:
    run = dict(runs[run_id], total_steps=2400, dev_every=1200, log_every=600)
    run['agent'] = dict(run['agent'], warmup=600)
    cfg = OUT / f'{run_id}.json'
    cfg.write_text(json.dumps(run), encoding='utf-8')
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/s2lab/train_run.py'), '--runs-dir', str(OUT), '--dataset-dir', DATA,
                           '--config', str(cfg), '--device', 'cuda'], capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = [l for l in (proc.stdout + proc.stderr).splitlines() if any(k in l for k in ('sps', 'eval ', 'done', 'Error', 'FAILED', 'Traceback'))]
    print(f'== {run_id} rc={proc.returncode}', *tail[-6:], sep='\n  ', flush=True)
