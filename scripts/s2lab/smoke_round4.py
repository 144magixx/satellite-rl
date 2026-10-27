"""第四轮本机冒烟：13 种配置各训 2400 步（含开发集评价与训完自动评价），看能否跑通与速度。"""
import json
import subprocess
import sys
from pathlib import Path

from _common import ROOT
from satellite_rl.s2lab.matrix import build_matrix_round4

OUT = Path('D:/paper/outputs/experiments/s2lab_round4_20261028/smoke')
DATA = 'D:/paper/outputs/experiments/s2lab_20261024/smoke_data'
runs = {r['run_id']: r for r in build_matrix_round4()}
OUT.mkdir(parents=True, exist_ok=True)
for name in sys.argv[1:] or sorted({k.rsplit('-s', 1)[0] for k in runs}, key=list(dict.fromkeys(k.rsplit('-s', 1)[0] for k in runs)).index):
    run = dict(runs[f'{name}-s1'], total_steps=2400, dev_every=1200, log_every=600)
    run['agent'] = dict(run['agent'], warmup=600)
    if 'bc_updates' in run['agent']:
        run['agent'].update(bc_demos=3000, bc_updates=300, actor_freeze=1200)
    cfg = OUT / f'{name}.json'
    cfg.write_text(json.dumps(run), encoding='utf-8')
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/s2lab/train_run.py'), '--runs-dir', str(OUT), '--dataset-dir', DATA,
                           '--config', str(cfg), '--device', 'cuda'], capture_output=True, text=True, encoding='utf-8', errors='replace')
    lines = (proc.stdout + proc.stderr).splitlines()
    sps = [l.split('sps=')[1].split()[0] for l in lines if 'sps=' in l]
    tail = [l for l in lines if any(k in l for k in ('eval confirm best_sgm', 'parameters', '[bc]', 'Error', 'FAILED', 'Traceback'))]
    print(f'== {name} rc={proc.returncode} sps={sps[-2:]}', *tail[-5:], sep='\n  ', flush=True)
