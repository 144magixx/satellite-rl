"""第四轮报告的数字与图（S2LAB_ROUND=4 的 remote2.py fetch --accounts B 结果）。

  python scripts/s2lab/report_figures4.py
每组对父组（matrix.ROUND4 中定义）、对本轮重跑的探索组合 r4-base、对容量感知贪心做同种子/逐景配对；
另把 r4-base 与第三轮探索组合比（复现性）；新确认集（c2）结果来自 <root>/eval_c2/confirm/*.csv。
写 <root>/report/{numbers.json, figures/*.png}。
"""
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from report_figures2 import STRATA, STRATA_ZH, read_csv, summary, paired, read_jsonl, smooth
from satellite_rl.s2lab.matrix import ROUND4, ROUND4_COND

R = Path('D:/paper/outputs/experiments/s2lab_round4_20261028')
R3 = Path('D:/paper/outputs/experiments/s2lab_round3_20261027')
RULES = Path('D:/paper/outputs/experiments/s2lab_round2_20261025/account_A/eval2/confirm')
SEEDS = (1, 2, 3)


def run_dirs():
    base = R / 'account_B' / 'runs4'
    return {d.name: d for d in base.iterdir() if d.is_dir() and not d.name.startswith('_')}


def main():
    dirs = run_dirs()
    defs = {n: v[:3] for n, v in ROUND4.items()}
    defs.update({n: v[:3] for n, v in ROUND4_COND.items()})
    for d in dirs:                                           # 条件组 r4-ptrg*（从配置读标签）
        name = d.rsplit('-s', 1)[0]
        if name not in defs:
            cfg = json.loads((dirs[d] / 'config.json').read_text(encoding='utf-8'))
            defs[name] = (cfg['group'], cfg['label'], cfg.get('parent'))

    def rows(rid, stratum=None, which='best_sgm', split='confirm'):
        p = dirs.get(rid)
        f = p / f'eval_{split}_{which}.csv' if p else None
        if f is None or not f.exists():
            return None
        return [r for r in read_csv(f) if not stratum or r['stratum'] == stratum]

    groups = {}
    for name, (grp, label, parent) in defs.items():
        seeds = [s for s in (1, 2, 3, 4, 5, 6) if rows(f'{name}-s{s}') is not None]
        if seeds:
            groups[name] = dict(group=grp, label=label, parent=parent, seeds=seeds)
    rule_rows = {k: read_csv(RULES / f'{v}.csv') for k, v in
                 {'容量感知贪心': 'rule-greedy', '分散摆放规则': 'rule-dispersed', '贪心（功率100档）': 'rule-greedy-p0.5'}.items()}
    n = dict(groups={}, rules={}, paired_parent={}, paired_base={}, paired_greedy={}, strata={}, convergence={}, timing={},
             repro={}, newconf={}, bc={})
    for name, g in groups.items():
        s3 = [s for s in g['seeds'] if s <= 3]
        rs = [rows(f'{name}-s{s}') for s in s3]
        n['groups'][name] = dict(summary(rs), label=g['label'], group=g['group'], parent=g['parent'], seeds=s3)
        n['strata'][name] = {st: summary([rows(f'{name}-s{s}', st) for s in s3]) for st in STRATA}
        for ref, key in ((g['parent'], 'paired_parent'), ('r4-base', 'paired_base')):
            if ref and ref in groups and ref != name:
                common = [s for s in s3 if s in groups[ref]['seeds']]
                a, b = [rows(f'{name}-s{s}') for s in common], [rows(f'{ref}-s{s}') for s in common]
                n[key][name] = {c: paired(a, b, c) for c in ('SGM', 'W', 'worst10')}
                n[key][name]['seed_diff'] = [float(np.mean([r['SGM'] for r in x]) - np.mean([r['SGM'] for r in y])) for x, y in zip(a, b)]
                n[key][name]['by_stratum'] = {st: paired([rows(f'{name}-s{s}', st) for s in common], [rows(f'{ref}-s{s}', st) for s in common], 'SGM')
                                              for st in STRATA}
                dev_a = [np.mean([r['SGM'] for r in rows(f'{name}-s{s}', split='dev')]) for s in common]
                dev_b = [np.mean([r['SGM'] for r in rows(f'{ref}-s{s}', split='dev')]) for s in common]
                n[key][name]['dev_seed_diff'] = [float(x - y) for x, y in zip(dev_a, dev_b)]
        n['paired_greedy'][name] = {c: paired(rs, [rule_rows['容量感知贪心']], c) for c in ('SGM', 'W', 'worst10')}
        n['paired_greedy'][name]['by_stratum'] = {st: paired([rows(f'{name}-s{s}', st) for s in s3],
                                                             [[r for r in rule_rows['容量感知贪心'] if r['stratum'] == st]], 'SGM') for st in STRATA}
        gains, hrs = [], []
        for s in s3:
            d = dirs[f'{name}-s{s}']
            dev = list({r['step']: r for r in read_jsonl(d / 'dev.jsonl')}.values())
            sg = np.array([r['SGM'] for r in dev])
            gains.append(float(sg[-5:].mean() - sg[-10:-5].mean()) if len(sg) >= 10 else float('nan'))
            st = json.loads((d / 'status.json').read_text(encoding='utf-8'))
            h = st.get('elapsed_hours', 0)
            if (d / 'bc.json').exists():
                bc = json.loads((d / 'bc.json').read_text(encoding='utf-8'))
                h += (bc['demo_seconds'] + bc['clone_seconds']) / 3600
                n['bc'][f'{name}-s{s}'] = bc
            hrs.append(h)
        n['convergence'][name], n['timing'][name] = gains, hrs
        extra = [s for s in g['seeds'] if s > 3]                 # 补种子 4—6
        if extra:
            n['groups'][name]['extra_seeds'] = extra
            n['groups'][name]['extra_SGM'] = [float(np.mean([r['SGM'] for r in rows(f'{name}-s{s}')])) for s in extra]
            n['groups'][name]['extra_dev_SGM'] = [float(np.mean([r['SGM'] for r in rows(f'{name}-s{s}', split='dev')])) for s in extra]
    for k, r in rule_rows.items():
        n['rules'][k] = summary([r])
        n['strata'][k] = {s: summary([[x for x in r if x['stratum'] == s]]) for s in STRATA}
    # 复现性：本轮重跑的探索组合 vs 第三轮探索组合（同种子号，不同代码提交、同口径）
    r3 = {s: R3 / f"account_{'A' if s == 2 else 'B'}" / 'runs3' / f'r3-xcombo-s{s}' / 'eval_confirm_best_sgm.csv' for s in SEEDS}
    if all(p.exists() for p in r3.values()) and 'r4-base' in groups:
        old = [read_csv(r3[s]) for s in SEEDS]
        n['repro'] = dict(r3=summary(old), r4=n['groups']['r4-base'], diff={c: paired([rows(f'r4-base-s{s}') for s in SEEDS], old, c)
                                                                         for c in ('SGM', 'W', 'worst10')})
    # 新确认集
    c2 = R / 'eval_c2' / 'confirm'
    if c2.exists():
        for f in sorted(c2.glob('*.csv')):
            rr = read_csv(f)
            n['newconf'][f.stem] = dict(summary([rr]), strata={st: summary([[x for x in rr if x['stratum'] == st]]) for st in STRATA if any(x['stratum'] == st for x in rr)})
    out = R / 'report'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'numbers.json').write_text(json.dumps(n, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(n, dirs, out / 'figures')
    print({k: len(v) for k, v in n.items()})


def plot(n, dirs, figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    G, Rr = n['groups'], n['rules']
    order = sorted(G, key=lambda k: -G[k]['SGM'])
    # F1 全部组
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, .36 * (len(order) + 3))), sharey=True)
    names = [G[k]['label'] for k in order] + list(Rr)
    y = np.arange(len(names))[::-1]
    for ax, col, t in zip(axes, ('SGM', 'W', 'worst10'), ('SGM（主指标）', '满足率 W', '最差10%飞机')):
        vals = [G[k][col] for k in order] + [Rr[k][col] for k in Rr]
        errs = [G[k][col + '_std'] for k in order] + [0] * len(Rr)
        ax.barh(y, vals, xerr=errs, color=['#e67e22' if k == 'r4-base' else '#2980b9' for k in order] + ['#555555'] * len(Rr), capsize=2)
        for yi, v in zip(y, vals):
            ax.text(v + .003, yi, f'{v:.3f}', va='center', fontsize=7)
        ax.axvline(Rr['容量感知贪心'][col], color='k', ls='--', lw=1)
        if 'r4-base' in G:
            ax.axvline(G['r4-base'][col], color='#e67e22', ls=':', lw=1.2)
        ax.set_title(t); ax.grid(axis='x', alpha=.3)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlim(max(0, min(G[k]['SGM'] for k in G) - .05), 1.0)
    fig.suptitle('第四轮旧确认集 287 景（3 种子均值±种子标准差；黑虚线＝容量感知贪心，橙点线＝探索组合本轮重跑）')
    fig.tight_layout(); fig.savefig(figs / 'F1_全部组.png', dpi=140); plt.close(fig)
    # F2 对父组
    P = n['paired_parent']
    keys = [k for k in G if k in P]
    fig, ax = plt.subplots(figsize=(10, max(4, .45 * len(keys))))
    yy = np.arange(len(keys))[::-1]
    d = [P[k]['SGM']['diff'] for k in keys]
    ax.barh(yy, d, xerr=[[P[k]['SGM']['diff'] - P[k]['SGM']['lo'] for k in keys], [P[k]['SGM']['hi'] - P[k]['SGM']['diff'] for k in keys]],
            color=['#27ae60' if v > 0 else '#c0392b' for v in d], capsize=3)
    for yi, v in zip(yy, d):
        ax.text(v, yi + .25, f'{v:+.3f}', ha='center', fontsize=8)
    ax.set_yticks(yy); ax.set_yticklabels([f"{G[k]['label']}（对 {G[G[k]['parent']]['label'] if G[k]['parent'] in G else G[k]['parent']}）" for k in keys], fontsize=8)
    ax.axvline(0, color='k', lw=1); ax.grid(axis='x', alpha=.3)
    ax.set_title('每组相对父组的旧确认集 SGM 变化（同种子逐景配对，误差棒＝95% 区间）')
    fig.tight_layout(); fig.savefig(figs / 'F2_对父组.png', dpi=140); plt.close(fig)
    # F3 分档
    sel = order[:5] + (['r4-base'] if 'r4-base' not in order[:5] else [])
    keys3 = sel + ['容量感知贪心', '分散摆放规则']
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(3); w = .8 / len(keys3); pal = plt.cm.tab10(np.arange(len(keys3)))
    for ax, col in zip(axes, ('SGM', 'W')):
        for i, k in enumerate(keys3):
            ax.bar(x + (i - len(keys3) / 2 + .5) * w, [n['strata'][k][s][col] for s in STRATA], w, label=G[k]['label'] if k in G else k, color=pal[i])
        ax.set_xticks(x); ax.set_xticklabels([STRATA_ZH[s] for s in STRATA]); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=.3); ax.set_title(f'分档 {col}')
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout(); fig.savefig(figs / 'F3_分档.png', dpi=140); plt.close(fig)
    # F4 开发集曲线与训练奖励（每组三种子平均）
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    pal = plt.cm.tab20(np.arange(len(G)))
    grid = np.arange(2000, 200001, 2000, dtype=float)
    for i, k in enumerate(G):
        dev = [list({r['step']: r for r in read_jsonl(dirs[f'{k}-s{s}'] / 'dev.jsonl')}.values()) for s in G[k]['seeds']]
        steps = sorted(set.intersection(*[set(r['step'] for r in d) for d in dev]))
        v = np.mean([[next(r['SGM'] for r in d if r['step'] == st) for st in steps] for d in dev], 0)
        style = dict(color='k', lw=2.5, ls='--') if k == 'r4-base' else dict(color=pal[i], lw=1.3)
        axes[0].plot(steps, v, label=G[k]['label'], **style)
        cur = []
        for s in G[k]['seeds']:
            rows_ = list({r['step']: r for r in read_jsonl(dirs[f'{k}-s{s}'] / 'curve.jsonl')}.values())
            rows_.sort(key=lambda r: r['step'])
            cur.append(np.interp(grid, [r['step'] for r in rows_], [r['reward'] for r in rows_]))
        axes[1].plot(grid, np.mean(cur, 0), label=G[k]['label'], **style)
    axes[0].set_title('开发集 SGM（三种子平均）'); axes[1].set_title('训练奖励（每步 delta 奖励，三种子平均）')
    for ax in axes:
        ax.set_xlabel('环境步'); ax.grid(alpha=.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(figs / 'F4_曲线.png', dpi=140); plt.close(fig)
    # F5 新旧确认集对比
    NC = n['newconf']
    if NC:
        methods = [k for k in order if any(f'{k}-s{s}' in NC for s in SEEDS)]
        rows_ = []
        for k in methods:
            vals = [NC[f'{k}-s{s}']['SGM'] for s in SEEDS if f'{k}-s{s}' in NC]
            rows_.append((G[k]['label'], G[k]['SGM'], float(np.mean(vals))))
        for k, name in (('容量感知贪心', 'rule-greedy'), ('分散摆放规则', 'rule-dispersed')):
            if name in NC:
                rows_.append((k, Rr[k]['SGM'], NC[name]['SGM']))
        fig, ax = plt.subplots(figsize=(10, .5 * len(rows_) + 2))
        yy = np.arange(len(rows_))[::-1]
        ax.barh(yy + .2, [r[1] for r in rows_], .4, label='旧确认集（1—2 月）', color='#95a5a6')
        ax.barh(yy - .2, [r[2] for r in rows_], .4, label='新确认集（6、8、10 月）', color='#2980b9')
        for yi, r in zip(yy, rows_):
            ax.text(r[1] + .003, yi + .2, f'{r[1]:.3f}', va='center', fontsize=7)
            ax.text(r[2] + .003, yi - .2, f'{r[2]:.3f}', va='center', fontsize=7)
        ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows_], fontsize=8); ax.legend(); ax.grid(axis='x', alpha=.3)
        ax.set_xlim(max(0, min(min(r[1], r[2]) for r in rows_) - .05), 1.0)
        ax.set_title('新旧确认集 SGM 对比')
        fig.tight_layout(); fig.savefig(figs / 'F5_新旧确认集.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
