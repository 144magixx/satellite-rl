"""结果汇总：把 fetch 下来的评价 CSV、开发曲线整理成表与图。

  python scripts/s2lab/report.py --root D:/paper/outputs/experiments/s2lab_20261024 --split confirm
输出 <root>/report/<split>/{summary.md, tables.json, figures/*.png}，并复制 summary.md 到 docs/Claude维护/S2-LAB/02_结果汇总_<split>.md。
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import ROOT

METRICS = ('W', 'SGM', 'worst10', 'power_w', 'slots', 'interference_loss', 'adj_ratio', 'unserved', 'seconds_per_scene')
LABELS = dict(tsac='T-SAC（新令牌、自己选组）', tsac_legacy='T-SAC（历史最好版）', cnnsac='CNN-SAC', mlpsac='MLP-SAC', ppo='PPO', dqn='DQN',
              greedy='容量感知贪心', fixed='固定分配', random='随机分配', sa32='模拟退火32次', sa320='模拟退火320次', ga32='遗传32次', ga320='遗传320次',
              beam_tokens='每束一令牌', pointer='块指针头', mod8='%8定组', greedy_group='贪心定组', delta='delta奖励')


def read_csv(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def load_eval(root, split):
    out = {}
    for path in sorted((Path(root) / 'eval' / split).glob('*.csv')):
        rows = read_csv(path)
        if not rows:
            continue
        name = path.stem
        out[name] = dict(rows=rows, group=rows[0].get('group', ''), label=rows[0].get('label', name), seed=rows[0].get('seed', 0),
                         mask_mode=rows[0].get('mask_mode', ''))
    return out


def per_scene(rows, metric):
    return {(r['key'], r['per_aircraft_bps']): r[metric] for r in rows}


def mean_over_seeds(entries, metric):
    """同 label 多种子：先每景对种子平均，返回 {scene: value}。"""
    acc = defaultdict(list)
    for e in entries:
        for k, v in per_scene(e['rows'], metric).items():
            if isinstance(v, float) and not np.isnan(v):
                acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def bootstrap_diff(a, b, n=2000, seed=0):
    keys = sorted(set(a) & set(b))
    if not keys:
        return float('nan'), (float('nan'), float('nan')), 0
    d = np.asarray([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))), len(keys)


def build_tables(evals):
    by_label = defaultdict(list)
    for name, e in evals.items():
        by_label[(e['group'], e['label'], e['mask_mode'])].append(e)
    table = []
    strata_rows = []
    for (group, label, mask), entries in by_label.items():
        row = dict(group=group, label=label, mask=mask, seeds=len(entries), name=LABELS.get(label, label))
        for m in METRICS:
            vals = mean_over_seeds(entries, m)
            row[m] = float(np.mean(list(vals.values()))) if vals else float('nan')
        table.append(row)
        for s in ('ample', 'marginal', 'short'):
            sub = [dict(e, rows=[r for r in e['rows'] if r.get('stratum') == s]) for e in entries]
            srow = dict(group=group, label=label, mask=mask, stratum=s, name=LABELS.get(label, label))
            for m in ('W', 'SGM', 'worst10', 'power_w', 'interference_loss', 'adj_ratio'):
                vals = mean_over_seeds(sub, m)
                srow[m] = float(np.mean(list(vals.values()))) if vals else float('nan')
            strata_rows.append(srow)
    return table, strata_rows, by_label


def paired(by_label, reference_label, metric):
    ref = [v for (g, l, m), v in by_label.items() if l == reference_label and m in ('full', '')]
    if not ref:
        return []
    ref_vals = mean_over_seeds(ref[0], metric)
    out = []
    for (g, l, m), entries in by_label.items():
        if l == reference_label:
            continue
        vals = mean_over_seeds(entries, metric)
        d, ci, n = bootstrap_diff(vals, ref_vals)
        out.append(dict(label=l, name=LABELS.get(l, l), group=g, mask=m, diff=d, lo=ci[0], hi=ci[1], n=n))
    return out


def fmt(v, digits=3):
    return '' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{digits}f}'


def render(table, strata_rows, pairs_sgm, pairs_w, split, dev_rows):
    lines = [f'# S2-LAB 结果汇总（{split}，直出，自动生成）', '',
             '## 总表（各档合并；学习方法按种子平均）', '',
             '| 方法 | 组 | 掩码 | 种子 | SGM | 满足率W | 最差10% | 功率W | 用槽 | 干扰损失 | 相邻共槽÷随机 | 未服务 | 单景秒 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(table, key=lambda r: (-r['SGM'] if not np.isnan(r['SGM']) else 0)):
        lines.append(f"| {r['name']} | {r['group']} | {r['mask']} | {r['seeds']} | {fmt(r['SGM'])} | {fmt(r['W'])} | {fmt(r['worst10'])} | {fmt(r['power_w'], 0)} | {fmt(r['slots'], 0)} | {fmt(r['interference_loss'])} | {fmt(r['adj_ratio'], 2)} | {fmt(r['unserved'], 1)} | {fmt(r['seconds_per_scene'], 2)} |")
    lines += ['', '## 分档（宽裕 / 勉强 / 不够）', '', '| 方法 | 档 | SGM | W | 最差10% | 功率 | 干扰损失 | 相邻共槽÷随机 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(strata_rows, key=lambda r: (r['stratum'], -(r['SGM'] if not np.isnan(r['SGM']) else 0))):
        lines.append(f"| {r['name']} | {r['stratum']} | {fmt(r['SGM'])} | {fmt(r['W'])} | {fmt(r['worst10'])} | {fmt(r['power_w'], 0)} | {fmt(r['interference_loss'])} | {fmt(r['adj_ratio'], 2)} |")
    for title, pairs in (('SGM 配对差（对容量感知贪心，95% 自举区间）', pairs_sgm), ('满足率 W 配对差（对容量感知贪心）', pairs_w)):
        lines += ['', f'## {title}', '', '| 方法 | 差 | 区间 | 场景数 |', '| --- | ---: | --- | ---: |']
        for p in sorted(pairs, key=lambda p: -p['diff'] if not np.isnan(p['diff']) else 0):
            lines.append(f"| {p['name']}{'（无掩码）' if p['mask'] == 'none' else ''} | {fmt(p['diff'], 4)} | [{fmt(p['lo'], 4)}, {fmt(p['hi'], 4)}] | {p['n']} |")
    if dev_rows:
        lines += ['', '## 训练过程（开发集最好分 SGM+W、达到最好分的步数、用时）', '', '| 运行 | 最好分 | 步数 | 最终W | 最终SGM | 用时h |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for r in dev_rows:
            lines.append(f"| {r['run_id']} | {fmt(r['best'])} | {r['best_step']} | {fmt(r['W'])} | {fmt(r['SGM'])} | {fmt(r['hours'], 2)} |")
    return '\n'.join(lines) + '\n'


def dev_summary(root):
    rows = []
    for run_dir in sorted((Path(root) / 'runs').glob('*/')):
        dev = run_dir / 'dev.jsonl'
        status = run_dir / 'status.json'
        if not dev.exists():
            continue
        records = [json.loads(l) for l in dev.read_text(encoding='utf-8').splitlines() if l.strip()]
        if not records:
            continue
        best = max(records, key=lambda r: r['score'])
        st = json.loads(status.read_text(encoding='utf-8')) if status.exists() else {}
        rows.append(dict(run_id=run_dir.name, best=best['score'], best_step=best['step'], W=records[-1]['W'], SGM=records[-1]['SGM'],
                         hours=st.get('elapsed_hours', float('nan')), records=records))
    return rows


def figures(root, split, dev_rows, table):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
            if any(f.name == name for f in font_manager.fontManager.ttflist):
                plt.rcParams['font.family'] = name
                break
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        return []
    out = Path(root) / 'report' / split / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    if dev_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for r in dev_rows:
            if r['run_id'].startswith('main-'):
                steps = [x['step'] for x in r['records']]
                axes[0].plot(steps, [x['SGM'] for x in r['records']], label=r['run_id'])
                axes[1].plot(steps, [x['W'] for x in r['records']], label=r['run_id'])
        axes[0].set_title('开发集 SGM（直出）'); axes[1].set_title('开发集满足率 W（直出）')
        for ax in axes:
            ax.set_xlabel('环境步'); ax.grid(alpha=.3)
        axes[1].legend(fontsize=6, ncol=2)
        fig.tight_layout(); p = out / 'dev_curves_main.png'; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    if table:
        rows = [r for r in table if r['mask'] in ('full', '')]
        rows = sorted(rows, key=lambda r: -(r['SGM'] if not np.isnan(r['SGM']) else 0))
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(rows))
        ax.bar(x - .2, [r['SGM'] for r in rows], .4, label='SGM')
        ax.bar(x + .2, [r['W'] for r in rows], .4, label='满足率W')
        ax.set_xticks(x); ax.set_xticklabels([r['name'] for r in rows], rotation=30, ha='right', fontsize=8)
        ax.legend(); ax.grid(axis='y', alpha=.3); ax.set_title(f'{split} 直出指标')
        fig.tight_layout(); p = out / 'bars.png'; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/paper/outputs/experiments/s2lab_20261024')
    ap.add_argument('--split', default='confirm')
    args = ap.parse_args()
    evals = load_eval(args.root, args.split)
    table, strata_rows, by_label = build_tables(evals)
    pairs_sgm = paired(by_label, 'greedy', 'SGM')
    pairs_w = paired(by_label, 'greedy', 'W')
    dev_rows = dev_summary(args.root)
    text = render(table, strata_rows, pairs_sgm, pairs_w, args.split, dev_rows)
    paths = figures(args.root, args.split, dev_rows, table)
    if paths:
        text += '\n## 图\n\n' + '\n'.join(f'![{p.stem}]({p.as_posix()})' for p in paths) + '\n'
    out = Path(args.root) / 'report' / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.md').write_text(text, encoding='utf-8')
    (out / 'tables.json').write_text(json.dumps(dict(table=table, strata=strata_rows, pairs_sgm=pairs_sgm, pairs_w=pairs_w), ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    doc = ROOT / 'docs' / 'Claude维护' / 'S2-LAB' / f'02_结果汇总_{args.split}.md'
    doc.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
