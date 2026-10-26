"""第三轮报告的数字与图：从 S2LAB_ROUND=3 的 remote2.py fetch 结果生成。

  python scripts/s2lab/report_figures3.py [--root D:/paper/outputs/experiments/s2lab_round3_20261027]
每组对其父组做同种子逐景配对（父组见 matrix.ROUND3），另给对 R0、对贪心（第二轮评过的规则 CSV）的配对差。
写 <root>/report/{numbers.json, figures/*.png}。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from report_figures2 import COLS, STRATA, STRATA_ZH, read_csv, summary, paired, read_jsonl, smooth
from satellite_rl.s2lab.matrix import ROUND3

RULES = Path('D:/paper/outputs/experiments/s2lab_round2_20261025/account_A/eval2/confirm')
SEEDS = (1, 2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/paper/outputs/experiments/s2lab_round3_20261027')
    args = ap.parse_args()
    root = Path(args.root)
    run_dirs = {}
    for acc in ('A', 'B'):
        base = root / f'account_{acc}' / 'runs3'
        if base.exists():
            for d in base.iterdir():
                if d.is_dir() and not d.name.startswith('_'):
                    run_dirs[d.name] = d

    def rows(run_id, split='confirm', stratum=None, which='best_sgm'):
        p = run_dirs.get(run_id)
        if p is None or not (p / f'eval_{split}_{which}.csv').exists():
            return None
        return [r for r in read_csv(p / f'eval_{split}_{which}.csv') if not stratum or r['stratum'] == stratum]

    def rid(name, s):
        """组键 → 运行目录名：'<组>@30' 是续训到 30 万步的副本，'r3-combo' 是组合版。"""
        return f"{name[:-3]}-s{s}-c30" if name.endswith('@30') else f'{name}-s{s}'

    defs = {name: (grp, label, parent) for name, (grp, label, parent, *_rest) in ROUND3.items()}
    defs['r3-combo'] = ('combo', '组合版', 'r3-r0')
    for name, (grp, label, parent) in list(defs.items()):
        if any(f'{name}-s{s}-c30' in run_dirs for s in SEEDS):
            defs[f'{name}@30'] = ('cont', f'{label}（30万步）', f'{parent}@30' if parent and any(f'{parent}-s{s}-c30' in run_dirs for s in SEEDS) else parent)
    groups = {}
    for name, (grp, label, parent) in defs.items():
        have = [s for s in SEEDS if rows(rid(name, s)) is not None]
        if have:
            groups[name] = dict(group=grp, label=label, parent=parent, seeds=have)
    rules = {'容量感知贪心': 'rule-greedy', '分散摆放规则': 'rule-dispersed', '贪心（功率100档）': 'rule-greedy-p0.5'}
    rule_rows = {k: read_csv(RULES / f'{v}.csv') for k, v in rules.items() if (RULES / f'{v}.csv').exists()}

    n = dict(groups={}, rules={}, paired_parent={}, paired_r0={}, paired_greedy={}, strata={}, checkpoint={}, convergence={})
    for name, g in groups.items():
        rs = [rows(rid(name, s)) for s in g['seeds']]
        n['groups'][name] = dict(label=g['label'], group=g['group'], parent=g['parent'], seeds=g['seeds'], **summary(rs))
        n['strata'][name] = {s: summary([rows(rid(name, x), 'confirm', s) for x in g['seeds']]) for s in STRATA}
        other = [rows(rid(name, s), which='best') for s in g['seeds']]
        if all(o is not None for o in other):
            n['checkpoint'][name] = dict(best_sgm=n['groups'][name]['SGM'], best=summary(other)['SGM'])
        for ref, key in ((g['parent'], 'paired_parent'), ('r3-r0', 'paired_r0')):
            if ref and ref in groups and ref != name:
                common = [s for s in g['seeds'] if s in groups[ref]['seeds']]
                a = [rows(rid(name, s)) for s in common]
                b = [rows(rid(ref, s)) for s in common]
                n[key][name] = {c: paired(a, b, c) for c in ('SGM', 'W', 'worst10')}
                n[key][name]['seed_diff'] = [float(np.mean([r['SGM'] for r in x]) - np.mean([r['SGM'] for r in y])) for x, y in zip(a, b)]
                n[key][name]['by_stratum'] = {st: paired([rows(rid(name, s), 'confirm', st) for s in common],
                                                         [rows(rid(ref, s), 'confirm', st) for s in common], 'SGM') for st in STRATA}
        if '容量感知贪心' in rule_rows:
            n['paired_greedy'][name] = {c: paired(rs, [rule_rows['容量感知贪心']], c) for c in ('SGM', 'W', 'worst10')}
            n['paired_greedy'][name]['by_stratum'] = {st: paired([rows(rid(name, s), 'confirm', st) for s in g['seeds']],
                                                                 [[r for r in rule_rows['容量感知贪心'] if r['stratum'] == st]], 'SGM') for st in STRATA}
        # 收敛：去重后的开发集末段变化
        gains = []
        for s in g['seeds']:
            dev = list({r['step']: r for r in read_jsonl(run_dirs[rid(name, s)] / 'dev.jsonl')}.values())
            sg = np.array([r['SGM'] for r in dev])
            if len(sg) >= 10:
                gains.append(float(sg[-5:].mean() - sg[-10:-5].mean()))
        n['convergence'][name] = gains
    for k, r in rule_rows.items():
        n['rules'][k] = summary([r])
        n['strata'][k] = {s: summary([[x for x in r if x['stratum'] == s]]) for s in STRATA}
    out = root / 'report'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'numbers.json').write_text(json.dumps(n, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(n, run_dirs, out / 'figures')
    print(json.dumps({k: len(v) for k, v in n.items()}, ensure_ascii=False))


def plot(n, run_dirs, figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    G, R = n['groups'], n['rules']
    greedy = R.get('容量感知贪心')
    order = sorted(G, key=lambda k: -G[k]['SGM'])

    # F1 全部组 SGM / W / 最差10%
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, .32 * (len(order) + 3))), sharey=True)
    names = [G[k]['label'] for k in order] + list(R)
    y = np.arange(len(names))[::-1]
    for ax, col, t in zip(axes, ('SGM', 'W', 'worst10'), ('SGM（主指标）', '满足率 W', '最差10%飞机')):
        vals = [G[k][col] for k in order] + [R[k][col] for k in R]
        errs = [G[k][col + '_std'] for k in order] + [0] * len(R)
        colors = ['#2980b9' if G[k]['group'] != 'algo' else '#8e44ad' for k in order] + ['#555555'] * len(R)
        ax.barh(y, vals, xerr=errs, color=colors, capsize=2)
        for yi, v in zip(y, vals):
            ax.text(v + .005, yi, f'{v:.3f}', va='center', fontsize=7)
        if greedy:
            ax.axvline(greedy[col], color='k', ls='--', lw=1)
        if 'r3-r0' in G:
            ax.axvline(G['r3-r0'][col], color='#e67e22', ls=':', lw=1.2)
        ax.set_title(t); ax.grid(axis='x', alpha=.3)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    lo = min(min(G[k]['SGM'] for k in G), min(R[k]['SGM'] for k in R)) - .05 if R else .3
    axes[0].set_xlim(max(0, lo), 1.0); axes[1].set_xlim(max(0, min(G[k]['W'] for k in G) - .05), .9)
    fig.suptitle('第三轮确认集 287 景（3 种子均值±种子标准差；黑虚线＝容量感知贪心，橙点线＝R0）')
    fig.tight_layout(); fig.savefig(figs / 'F1_全部组.png', dpi=140); plt.close(fig)

    # F2 对父组的配对差（SGM，95% 区间）
    P = n['paired_parent']
    keys = [k for k in ROUND3 if k in P]
    fig, ax = plt.subplots(figsize=(10, max(5, .35 * len(keys))))
    yy = np.arange(len(keys))[::-1]
    d = [P[k]['SGM']['diff'] for k in keys]
    lo = [P[k]['SGM']['diff'] - P[k]['SGM']['lo'] for k in keys]
    hi = [P[k]['SGM']['hi'] - P[k]['SGM']['diff'] for k in keys]
    ax.barh(yy, d, xerr=[lo, hi], color=['#27ae60' if v > 0 else '#c0392b' for v in d], capsize=3)
    for yi, v, k in zip(yy, d, keys):
        ax.text(v + (.002 if v >= 0 else -.002), yi, f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
    ax.set_yticks(yy); ax.set_yticklabels([f"{G[k]['label']}（对 {G[G[k]['parent']]['label'] if G[k]['parent'] in G else G[k]['parent']}）" for k in keys], fontsize=8)
    ax.axvline(0, color='k', lw=1); ax.grid(axis='x', alpha=.3)
    ax.set_title('每组相对父组的确认集 SGM 变化（同种子逐景配对，误差棒＝95% 区间）')
    fig.tight_layout(); fig.savefig(figs / 'F2_对父组.png', dpi=140); plt.close(fig)

    # F3 分档（最好几组＋贪心）
    sel = [k for k in order[:5]] + (['r3-r0'] if 'r3-r0' in G and 'r3-r0' not in order[:5] else [])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    labels = [G[k]['label'] for k in sel] + [k for k in ('容量感知贪心', '分散摆放规则') if k in R]
    keys3 = sel + [k for k in ('容量感知贪心', '分散摆放规则') if k in R]
    x = np.arange(3); w = .8 / len(keys3); pal = plt.cm.tab10(np.arange(len(keys3)))
    for ax, col in zip(axes, ('SGM', 'W')):
        for i, (k, lab) in enumerate(zip(keys3, labels)):
            ax.bar(x + (i - len(keys3) / 2 + .5) * w, [n['strata'][k][s][col] for s in STRATA], w, label=lab, color=pal[i])
        ax.set_xticks(x); ax.set_xticklabels([STRATA_ZH[s] for s in STRATA]); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=.3)
        ax.set_title(f'分档 {col}')
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout(); fig.savefig(figs / 'F3_分档.png', dpi=140); plt.close(fig)

    # F4 开发集曲线（动作/功率/训练三类代表）
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, members, title in zip(axes, (['r3-r0', 'r3-legal', 'r3-cond', 'r3-condtop8'], ['r3-cond', 'r3-p1', 'r3-p2', 'r3-p3'],
                                         ['r3-r0', 'r3-hist', 'r3-wide', 'r3-f-pneed', 'r3-f-all', 'r3-x-noglobal']),
                                  ('动作结构', '功率三组递进', '训练与状态')):
        pal = plt.cm.tab10(np.arange(len(members)))
        for i, k in enumerate([m for m in members if m in G]):
            recs = [list({r['step']: r for r in read_jsonl(run_dirs[rr] / 'dev.jsonl')}.values()) for rr in G[k]['runs']]
            steps = sorted(set.intersection(*[set(r['step'] for r in rec) for rec in recs]))
            v = np.array([[next(r['SGM'] for r in rec if r['step'] == st) for st in steps] for rec in recs])
            ax.plot(steps, v.mean(0), color=pal[i], lw=2, label=G[k]['label'])
            ax.fill_between(steps, v.min(0), v.max(0), color=pal[i], alpha=.12)
        ax.set_title(f'开发集 SGM：{title}'); ax.set_xlabel('环境步'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / 'F4_开发集曲线.png', dpi=140); plt.close(fig)

    # F5 最好组训练过程
    best = order[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8)); pal = plt.cm.Set1(np.arange(3))
    for i, (s, rr) in enumerate(zip(G[best]['seeds'], G[best]['runs'])):
        d = run_dirs[rr]
        rows_ = read_jsonl(d / 'curve.jsonl'); st = [x['step'] for x in rows_]
        axes[0, 0].plot(st, smooth([x['reward'] for x in rows_]), color=pal[i], label=f'种子 {s}')
        axes[0, 1].plot(st, smooth([x.get('loss_critic_loss', np.nan) for x in rows_]), color=pal[i], label=f'评论家 {s}')
        axes[0, 1].plot(st, smooth([x.get('loss_actor_loss', np.nan) for x in rows_]), color=pal[i], ls='--', label=f'演员 {s}')
        dv = list({r['step']: r for r in read_jsonl(d / 'dev.jsonl')}.values())
        for ax, col in ((axes[1, 0], 'SGM'), (axes[1, 1], 'SI')):
            ax.plot(st, smooth([x.get(col, np.nan) for x in rows_]), color=pal[i])
            ax.plot([r['step'] for r in dv], [r[col] for r in dv], 'o', ms=3, color=pal[i], alpha=.6)
    for ax, t in zip(axes.flat, ('每步奖励', '损失', 'SGM（线：训练场景；点：开发集）', '束平均满足率 SI')):
        ax.set_title(t); ax.set_xlabel('环境步'); ax.grid(alpha=.3)
    axes[0, 1].set_yscale('symlog'); axes[0, 1].legend(fontsize=7, ncol=2); axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{G[best]['label']} 的训练过程（滑动平均 2 万步）")
    fig.tight_layout(); fig.savefig(figs / 'F5_最好组训练过程.png', dpi=140); plt.close(fig)

    # F6 干扰与截断
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for k in G:
        c = G[k]
        axes[0].scatter(c['adj_ratio'], c['interference_loss'], color='#c0392b', s=40)
        axes[0].annotate(c['label'], (c['adj_ratio'], c['interference_loss']), fontsize=7, xytext=(3, 2), textcoords='offset points')
    for k, c in R.items():
        axes[0].scatter(c['adj_ratio'], c['interference_loss'], color='k', marker='s', s=50)
        axes[0].annotate(k, (c['adj_ratio'], c['interference_loss']), fontsize=7, xytext=(3, 2), textcoords='offset points')
    axes[0].axvline(1, color='gray', ls=':'); axes[0].set_xlabel('相邻同频共槽÷随机预期'); axes[0].set_ylabel('被干扰吃掉的业务量占比')
    axes[0].set_title('躲干扰（方块＝规则）'); axes[0].grid(alpha=.3)
    trunc = {k: G[k].get('truncated', np.nan) for k in G}
    ks = [k for k in order if np.isfinite(trunc.get(k, np.nan))]
    axes[1].barh(np.arange(len(ks))[::-1], [trunc[k] for k in ks], color='#7f8c8d')
    axes[1].set_yticks(np.arange(len(ks))[::-1]); axes[1].set_yticklabels([G[k]['label'] for k in ks], fontsize=8)
    axes[1].set_title('每景被截短的波束数（申请槽数 ≠ 实际槽数）'); axes[1].grid(axis='x', alpha=.3)
    fig.tight_layout(); fig.savefig(figs / 'F6_干扰与截断.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
