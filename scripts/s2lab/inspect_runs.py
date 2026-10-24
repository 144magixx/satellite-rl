"""巡检表：每个运行/规则在开发集（或确认集）的全部直出指标，按组排列，附运行状态。

  python scripts/s2lab/inspect_runs.py [--root ...] [--split dev]
数据来源：<root>/eval/<split>/summary.json（evaluate.py 产物，policy_best）；没有评价结果的运行用 dev.jsonl 最后一条（训练中的快照）。
输出到终端并写 <root>/report/inspect_<split>.md。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401

COLS = [('W', '满足率W', 3), ('SI', '束均SI', 3), ('SGM', 'SGM', 3), ('worst10', '最差10%', 3), ('power_w', '功率W', 0), ('slots', '用槽', 0),
        ('copol_overlap', '同极重叠槽', 0), ('interference_loss', '干扰损失', 3), ('mean_in_db', 'I/N dB', 1), ('adj_ratio_pooled', '相邻共槽÷随机', 2),
        ('unserved', '未服务束', 1), ('timing', '单景秒', 2)]


def fmt(v, d):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ''
    return f'{v:.{d}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/paper/outputs/experiments/s2lab_20261024')
    ap.add_argument('--split', default='dev')
    args = ap.parse_args()
    root = Path(args.root)
    summary_path = root / 'eval' / args.split / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    matrix = {r['run_id']: r for r in json.loads((root / 'runs' / 'matrix.json').read_text(encoding='utf-8'))}
    rows = []
    for name, s in summary.items():
        if name.startswith('rule-') or name.startswith('meta-'):
            a = s['all']
            rows.append(dict(name=name, group='rule' if name.startswith('rule-') else 'literature', state='-', step='-', source='eval',
                             timing=(s.get('timing_s') or [None])[0], **{k: a.get(k) for k, _, _ in COLS if k != 'timing'}))
    for run_id, cfg in matrix.items():
        st_path = root / 'runs' / run_id / 'status.json'
        st = json.loads(st_path.read_text(encoding='utf-8')) if st_path.exists() else {}
        if run_id in summary:
            a = summary[run_id]['all']
            rows.append(dict(name=run_id, group=cfg['group'], state=st.get('state', '?'), step=st.get('step', 0), source='eval(best)',
                             timing=(summary[run_id].get('timing_s') or [None])[0], **{k: a.get(k) for k, _, _ in COLS if k != 'timing'}))
            continue
        dev = root / 'runs' / run_id / 'dev.jsonl'
        if dev.exists():
            last = json.loads(dev.read_text(encoding='utf-8').splitlines()[-1])
            rows.append(dict(name=run_id, group=cfg['group'], state=st.get('state', '?'), step=st.get('step', 0), source=f"dev@{last['step']}",
                             timing=None, **{k: last.get(k) for k, _, _ in COLS if k != 'timing'}))
        else:
            rows.append(dict(name=run_id, group=cfg['group'], state=st.get('state', 'new'), step=st.get('step', 0), source='-', timing=None,
                             **{k: None for k, _, _ in COLS if k != 'timing'}))
    order = {'rule': 0, 'literature': 1, 'main': 2, 'reward': 3, 'token': 4, 'group': 5, 'size': 6, 'nomask': 7}
    rows.sort(key=lambda r: (order.get(r['group'], 9), r['name']))
    head = '| 名称 | 组 | 状态 | 步数 | 数据 | ' + ' | '.join(c for _, c, _ in COLS) + ' |'
    sep = '| --- | --- | --- | ---: | --- | ' + ' | '.join('---:' for _ in COLS) + ' |'
    lines = [f'# 巡检表（{args.split}，直出，60 景均值；eval(best)=policy_best 的评价脚本结果，dev@步数=训练中快照）', '', head, sep]
    for r in rows:
        lines.append(f"| {r['name']} | {r['group']} | {r['state']} | {r['step']} | {r['source']} | " + ' | '.join(fmt(r.get(k), d) for k, _, d in COLS) + ' |')
    # 分组均值（学习方法按 label 聚合）
    lines += ['', '## 按方法聚合（种子均值）', '', '| 方法 | n | ' + ' | '.join(c for _, c, _ in COLS[:11]) + ' |', '| --- | ---: | ' + ' | '.join('---:' for _ in COLS[:11]) + ' |']
    groups = {}
    for r in rows:
        if r['group'] in ('rule', 'literature'):
            key = r['name']
        else:
            key = f"{r['group']}/{matrix[r['name']].get('label', '')}"
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        vals = []
        for k, _, d in COLS[:11]:
            xs = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
            vals.append(fmt(float(np.mean(xs)), d) if xs else '')
        lines.append(f'| {key} | {len(rs)} | ' + ' | '.join(vals) + ' |')
    text = '\n'.join(lines) + '\n'
    out = root / 'report' / f'inspect_{args.split}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
