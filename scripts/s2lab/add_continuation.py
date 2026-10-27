"""给某个第三轮配置补排"续到 30 万步"：先在远端把 20 万步目录复制成 <运行>-c30（去掉状态与评价），
再加入该账号的追加运行并重写队列，最后在指定卡上起排队器。顺序不能反，否则排队器会抢在复制前从零开训。
  python scripts/s2lab/add_continuation.py --name r3-ppo --account B --ports 20802 20803
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ['S2LAB_ROUND'] = '3'
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import remote2  # noqa: E402
from satellite_rl.s2lab.matrix import ROUND3, SEEDS, r3_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--account', required=True)
ap.add_argument('--ports', type=int, nargs='+', required=True)
ap.add_argument('--window-hours', type=float, default=2.)
a = ap.parse_args()
grp, label, parent, agent, size, other = ROUND3[a.name]
other = dict(other)
algo = other.pop('algo', 'sac3')
runs = []
for s in SEEDS:
    run = r3_run(a.name, s, grp, label, parent, agent, size, algo=algo, **other)
    run.update(run_id=f'{a.name}-s{s}-c30', total_steps=300000, continued_from=f'{a.name}-s{s}', account=a.account, label=f'{label}（30万步）')
    runs.append(run)
p = remote2.paths(a.account)
with remote2.connect(remote2.first_port(a.account)) as c:
    print(remote2.run(c, f"cd {p['runs']} && for s in 1 2 3; do d={a.name}-s$s-c30; [ -e $d ] && mv $d _wrong/$d.old_$(date +%H%M%S); "
                         f"cp -r {a.name}-s$s $d && rm -f $d/status.json $d/eval_* $d/.claim; done; ls -d {a.name}-s*-c30", check=False))
tmp = Path(tempfile.mkdtemp()) / 'add.json'
tmp.write_text(json.dumps(runs, ensure_ascii=False), encoding='utf-8')
subprocess.run([sys.executable, str(HERE / 'edit_extra.py'), '--account', a.account, '--add-json', str(tmp)], check=True)
remote2.cmd_launch(argparse.Namespace(accounts=[a.account], window_hours=a.window_hours, parallel=2, ports=a.ports, only=None))
