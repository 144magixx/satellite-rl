"""第三轮全部学习组的训练奖励曲线（规则方法不训练，没有曲线）。

奖励＝每个环境步的 delta 奖励（放下这一束后整景目标的增量），curve.jsonl 每 2000 步记一次，
每个点是最近 200 个批次步（约 1.3 万个波束决策）的平均，本身已平滑。
续训到 30 万步的组画完整 0—30 万步（续训目录里含前 20 万步的原始记录），20 万步处画竖线。
  python scripts/s2lab/plot_rewards3.py
输出：<root>/report/figures/R1_每组训练奖励.png、R2_训练奖励分类对比.png
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('D:/paper/outputs/experiments/s2lab_round3_20261027')
SEEDS = (1, 2, 3)
# 与报告主表同顺序：组键 → 显示名
GROUPS = [
    ('r3-xcombo', '探索组合（七项补充＋回放池10万）'), ('r3-f-all', '＋全部补充'), ('r3-f-gspec', '＋全局频谱'), ('r3-replay', '回放池10万'),
    ('r3-f-share', '＋功率份额'), ('r3-f-pneed', '＋所需功率'), ('r3-auto', '自动温度'), ('r3-f-outint', '＋出向干扰'),
    ('r3-f-future', '＋后续邻居'), ('r3-f-sens', '＋邻束敏感度'), ('r3-x-nointf', '−干扰输入'), ('r3-hist', '历史双Q'),
    ('r3-f-snr', '＋信噪比'), ('r3-r0', 'R0（基线）'), ('r3-condtop8', 'B′ 低干扰前8'), ('r3-legal', 'L 合法落位'),
    ('r3-wide', '加宽 d256'), ('r3-lr', '演员学习率3e-5'), ('r3-cond', 'A 槽数条件化'), ('r3-p1', 'P1 评论家功率项看槽数'),
    ('r3-p2', 'P2 连续功率'), ('r3-p3', 'P3 连续功率看起点'), ('r3-dqn', 'DQN（14令牌）'), ('r3-x-noglobal', '−全局令牌'),
    ('r3-mlpsac', 'MLP-SAC（同字段）'), ('r3-ppo', 'PPO（14令牌）'),
]
LABEL = dict(GROUPS)
CATEGORIES = [
    ('状态补充', ['r3-r0', 'r3-f-pneed', 'r3-f-share', 'r3-f-outint', 'r3-f-snr', 'r3-f-sens', 'r3-f-future', 'r3-f-gspec', 'r3-f-all', 'r3-xcombo']),
    ('状态去除', ['r3-r0', 'r3-x-nointf', 'r3-x-noglobal']),
    ('动作与功率', ['r3-r0', 'r3-legal', 'r3-cond', 'r3-condtop8', 'r3-p1', 'r3-p2', 'r3-p3']),
    ('训练设置', ['r3-r0', 'r3-hist', 'r3-wide', 'r3-replay', 'r3-lr', 'r3-auto']),
    ('同口径算法', ['r3-r0', 'r3-dqn', 'r3-mlpsac', 'r3-ppo']),
]


def run_dirs():
    out = {}
    for acc in ('A', 'B'):
        for d in (ROOT / f'account_{acc}' / 'runs3').iterdir():
            if d.is_dir() and not d.name.startswith('_'):
                out[d.name] = d
    return out


def curve(d):
    rows = [json.loads(l) for l in (d / 'curve.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    rows = list({r['step']: r for r in rows}.values())          # 续训接续处可能有重复步数
    rows.sort(key=lambda r: r['step'])
    return np.array([r['step'] for r in rows], float), np.array([r['reward'] for r in rows])


def load():
    dirs = run_dirs()
    data = {}
    for key, _ in GROUPS:
        seeds = []
        for s in SEEDS:
            name = f'{key}-s{s}-c30' if f'{key}-s{s}-c30' in dirs else f'{key}-s{s}'
            seeds.append(curve(dirs[name]))
        data[key] = seeds
    return data


def mean_curve(seeds, grid):
    return np.mean([np.interp(grid, x, y, right=np.nan) for x, y in seeds], 0)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    data = load()
    figs = ROOT / 'report' / 'figures'
    grid = np.arange(2000, 300001, 2000, dtype=float)
    r0 = mean_curve(data['r3-r0'], grid)
    allv = np.concatenate([y for k in data for _, y in data[k]])
    lo, hi = np.percentile(allv, .5), np.percentile(allv, 99.9)
    pad = .05 * (hi - lo)

    # R1：每组一格，三个种子细线＋均值粗线，灰虚线为 R0 均值
    cols, rows = 5, int(np.ceil(len(GROUPS) / 5))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), sharex=True, sharey=True)
    pal = plt.cm.Set1(np.arange(3))
    for ax, (key, name) in zip(axes.flat, GROUPS):
        for i, (x, y) in enumerate(data[key]):
            ax.plot(x, y, color=pal[i], lw=.7, alpha=.6)
        m = mean_curve(data[key], grid)
        ax.plot(grid, m, color='k', lw=1.6)
        if key != 'r3-r0':
            ax.plot(grid, r0, color='gray', lw=1.2, ls='--')
        if max(x.max() for x, _ in data[key]) > 200001:
            ax.axvline(200000, color='#e67e22', lw=.8, ls=':')
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=.3)
        ax.set_ylim(lo - pad, hi + pad)
    for ax in axes.flat[len(GROUPS):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('环境步')
    for ax in axes[:, 0]:
        ax.set_ylabel('每步奖励')
    handles = [plt.Line2D([], [], color=pal[i], lw=.8, label=f'种子 {s}') for i, s in enumerate(SEEDS)]
    handles += [plt.Line2D([], [], color='k', lw=1.6, label='三种子平均'), plt.Line2D([], [], color='gray', ls='--', label='R0 平均'),
                plt.Line2D([], [], color='#e67e22', ls=':', label='20 万步（之后为续训）')]
    fig.legend(handles=handles, loc='upper center', ncol=6, bbox_to_anchor=(.5, 1.0))
    fig.suptitle('第三轮各组训练奖励（每步 delta 奖励，每点为最近约 1.3 万个波束决策的平均）', y=1.02)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(figs / 'R1_每组训练奖励.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # R2：分类对比，只画三种子平均
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, (title, keys) in zip(axes.flat, CATEGORIES):
        pal = plt.cm.tab10(np.arange(len(keys)))
        for i, key in enumerate(keys):
            m = mean_curve(data[key], grid)
            ax.plot(grid, m, color='k' if key == 'r3-r0' else pal[i], lw=2.2 if key in ('r3-r0', 'r3-f-all', 'r3-xcombo') else 1.3,
                    ls='--' if key == 'r3-r0' else '-', label=LABEL[key])
        ax.axvline(200000, color='#e67e22', lw=.8, ls=':')
        ax.set_title(title)
        ax.set_xlabel('环境步')
        ax.set_ylabel('每步奖励（三种子平均）')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        ax.set_ylim(lo - pad, hi + pad)
    # 第六格：最终 2 万步平均奖励（20 万步时）排序
    ax = axes.flat[5]
    final = {k: np.mean([y[(x > 180000) & (x <= 200000)].mean() for x, y in data[k]]) for k, _ in GROUPS}
    order = sorted(final, key=final.get)
    ax.barh(np.arange(len(order)), [final[k] for k in order], color=['#2980b9' if k != 'r3-r0' else '#e67e22' for k in order])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LABEL[k] for k in order], fontsize=7)
    ax.set_title('18—20 万步平均每步奖励')
    ax.grid(axis='x', alpha=.3)
    ax.set_xlim(min(final.values()) - .05, max(final.values()) + .02)
    fig.suptitle('训练奖励分类对比（橙点线＝20 万步，之后只有续训组）')
    fig.tight_layout()
    fig.savefig(figs / 'R2_训练奖励分类对比.png', dpi=130)
    plt.close(fig)
    print({k: round(float(v), 3) for k, v in sorted(final.items(), key=lambda kv: -kv[1])})


if __name__ == '__main__':
    main()
