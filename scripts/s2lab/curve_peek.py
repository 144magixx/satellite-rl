"""看若干运行的训练曲线快照（训练场景采样策略的 W/SGM/奖励）。用法：python scripts/s2lab/curve_peek.py run_id [run_id ...]"""
import json
import sys
from pathlib import Path

root = Path('D:/paper/outputs/experiments/s2lab_20261024/runs')
for r in sys.argv[1:]:
    path = root / r / 'curve.jsonl'
    if not path.exists():
        print(r, 'no curve'); continue
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    pts = [x for x in rows if 'W' in x][::10]
    print(r, ' '.join(f"{x['step'] // 1000}k:W{x['W']:.2f}/S{x['SGM']:.2f}/r{x['reward']:.2f}" for x in pts))
