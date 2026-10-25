"""第一轮报告的图与数字：全部从 fetch 下来的原始数据（eval/<split>/*.csv、runs/*/curve.jsonl、dev.jsonl）生成。

  python scripts/s2lab/report_figures.py [--root D:/paper/outputs/experiments/s2lab_20261024]
输出 <root>/report/round1/{figures/*.png, numbers.json}
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401

METHODS = {  # 名称: (文件前缀列表)
    '容量感知贪心': ['rule-greedy'], '模拟退火320次': ['meta-sa320'], '遗传320次': ['meta-ga320'],
    '模拟退火32次': ['meta-sa32'], '遗传32次': ['meta-ga32'], '固定分配': ['rule-fixed'], '随机分配': ['rule-random'],
    '历史版T-SAC＋delta奖励': [f'extra-legacydelta-s{s}' for s in (1, 2, 3)],
    '历史版T-SAC': [f'main-tsaclegacy-s{s}' for s in (1, 2, 3)],
    '新版T-SAC': [f'main-tsac-s{s}' for s in (1, 2, 3)],
    '新版T-SAC＋delta奖励': [f'reward-delta-s{s}' for s in (1, 2, 3)],
    'PPO（Transformer）': [f'main-ppo-s{s}' for s in (1, 2, 3)],
    'DQN（Transformer）': [f'main-dqn-s{s}' for s in (1, 2, 3)],
    'CNN-SAC': [f'main-cnnsac-s{s}' for s in (1, 2, 3)],
    'MLP-SAC': [f'main-mlpsac-s{s}' for s in (1, 2, 3)],
    'PPO（MLP）': [f'extra-ppomlp-s{s}' for s in (1, 2, 3)],
    'DQN（MLP）': [f'extra-dqnmlp-s{s}' for s in (1, 2, 3)],
    '新版＋%8定组': [f'group-mod8-s{s}' for s in (1, 2, 3)],
    '新版＋贪心定组': [f'group-greedy-s{s}' for s in (1, 2, 3)],
    '新版＋每束一令牌': [f'token-beams-s{s}' for s in (1, 2, 3)],
    '新版＋块指针头': [f'token-pointer-s{s}' for s in (1, 2, 3)],
}
NOMASK = {'T-SAC': 'nomask-tsac-s1', 'CNN-SAC': 'nomask-cnnsac-s1', 'MLP-SAC': 'nomask-mlpsac-s1', 'PPO': 'nomask-ppo-s1', 'DQN': 'nomask-dqn-s1'}
SIZES = [('d32L1', 'size-d32L1-s1', dict(d=32, layers=1, heads=4, ff=128)), ('d64L2', 'size-d64L2-s1', dict(d=64, layers=2, heads=4, ff=256)),
         ('d128L2', 'main-tsac-s1', dict(d=128, layers=2, heads=8, ff=512)), ('d128L4', 'size-d128L4-s1', dict(d=128, layers=4, heads=8, ff=512)),
         ('d128L8', 'size-d128L8-s1', dict(d=128, layers=8, heads=8, ff=512)), ('d256L2', 'size-d256L2-s1', dict(d=256, layers=2, heads=8, ff=1024)),
         ('d256L4', 'size-d256L4-s1', dict(d=256, layers=4, heads=8, ff=1024)), ('d512L4', 'size-d512L4-s1', dict(d=512, layers=4, heads=8, ff=2048))]
COLS = ('W', 'SI', 'SGM', 'worst10', 'power_w', 'slots', 'copol_overlap', 'interference_loss', 'mean_in_db', 'unserved')
STRATA = ('ample', 'marginal', 'short')
STRATA_ZH = dict(ample='宽裕', marginal='勉强', short='不够')


def read_csv(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for r in rows:
        d = dict(key=r['key'], stratum=r['stratum'])
        for c in COLS + ('adj_overlap', 'adj_expected'):
            d[c] = float(r[c]) if r[c] not in ('', 'nan') else np.nan
        out.append(d)
    return out


def per_scene(files, eval_dir, col, stratum=None):
    """多种子：每景先对种子平均。返回 {key: value}。"""
    acc = {}
    for f in files:
        for r in read_csv(eval_dir / f'{f}.csv'):
            if stratum and r['stratum'] != stratum:
                continue
            acc.setdefault(r['key'], []).append(r[col])
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def summary(files, eval_dir, stratum=None):
    out = {}
    for c in COLS:
        seeds = []
        for f in files:
            vals = [r[c] for r in read_csv(eval_dir / f'{f}.csv') if not stratum or r['stratum'] == stratum]
            seeds.append(np.nanmean(vals))
        out[c] = float(np.mean(seeds))
        out[c + '_std'] = float(np.std(seeds)) if len(seeds) > 1 else 0.
    rows = [r for f in files for r in read_csv(eval_dir / f'{f}.csv') if not stratum or r['stratum'] == stratum]
    out['adj_ratio'] = float(sum(r['adj_overlap'] for r in rows) / max(1e-9, sum(r['adj_expected'] for r in rows)))
    out['seeds'] = len(files)
    out['seed_SGM'] = [float(np.nanmean([r['SGM'] for r in read_csv(eval_dir / f'{f}.csv') if not stratum or r['stratum'] == stratum])) for f in files]
    return out


def paired(a_files, b_files, eval_dir, col, stratum=None, n=4000, seed=0):
    a, b = per_scene(a_files, eval_dir, col, stratum), per_scene(b_files, eval_dir, col, stratum)
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                win=float((d > 0).mean()), n=len(keys))


def read_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] if path.exists() else []


def smooth(x, w=5):
    x = np.asarray(x, float)
    return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/paper/outputs/experiments/s2lab_20261024')
    args = ap.parse_args()
    root = Path(args.root)
    conf, dev = root / 'eval' / 'confirm', root / 'eval' / 'dev'
    out = root / 'report' / 'round1'
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    numbers = dict(confirm={}, dev={}, strata={}, paired={}, nomask={}, size={}, timing={})
    for name, files in METHODS.items():
        numbers['confirm'][name] = summary(files, conf)
        if all((dev / f'{f}.csv').exists() for f in files):      # 文献方法只评了确认集
            numbers['dev'][name] = summary(files, dev)
        numbers['strata'][name] = {s: summary(files, conf, s) for s in STRATA}
    for name, f in NOMASK.items():
        numbers['nomask'][name] = summary([f], conf)
    G = METHODS['容量感知贪心']
    pairs = {
        '历史版＋delta − 贪心': ('历史版T-SAC＋delta奖励', '容量感知贪心'),
        '历史版 − 贪心': ('历史版T-SAC', '容量感知贪心'),
        '新版 − 贪心': ('新版T-SAC', '容量感知贪心'),
        '遗传320次 − 贪心': ('遗传320次', '容量感知贪心'),
        '模拟退火320次 − 贪心': ('模拟退火320次', '容量感知贪心'),
        '历史版＋delta − 历史版（奖励）': ('历史版T-SAC＋delta奖励', '历史版T-SAC'),
        '新版＋delta − 新版（奖励）': ('新版T-SAC＋delta奖励', '新版T-SAC'),
        '历史版 − 新版（令牌与定组）': ('历史版T-SAC', '新版T-SAC'),
        '新版T-SAC − CNN-SAC': ('新版T-SAC', 'CNN-SAC'),
        '新版T-SAC − MLP-SAC': ('新版T-SAC', 'MLP-SAC'),
        'PPO Transformer − MLP': ('PPO（Transformer）', 'PPO（MLP）'),
        'DQN Transformer − MLP': ('DQN（Transformer）', 'DQN（MLP）'),
        '新版T-SAC − PPO': ('新版T-SAC', 'PPO（Transformer）'),
        '新版T-SAC − DQN': ('新版T-SAC', 'DQN（Transformer）'),
        '历史版T-SAC − PPO': ('历史版T-SAC', 'PPO（Transformer）'),
        '新版＋%8定组 − 新版': ('新版＋%8定组', '新版T-SAC'),
        '新版＋贪心定组 − 新版': ('新版＋贪心定组', '新版T-SAC'),
        '新版＋每束一令牌 − 新版': ('新版＋每束一令牌', '新版T-SAC'),
        '新版＋块指针头 − 新版': ('新版＋块指针头', '新版T-SAC'),
    }
    for label, (a, b) in pairs.items():
        numbers['paired'][label] = {c: paired(METHODS[a], METHODS[b], conf, c) for c in ('SGM', 'W', 'worst10')}
        numbers['paired'][label]['by_stratum'] = {s: paired(METHODS[a], METHODS[b], conf, 'SGM', s) for s in STRATA}
    from satellite_rl.s2lab.agents import make_agent
    for label, run, size in SIZES:
        agent = make_agent('sac', dict(encoder='tf_blocks', size=dict(size, activation='relu', norm_first=False)), 13, 'cpu', 20)
        numbers['size'][label] = dict(params=agent.parameter_count(), **summary([run], conf))
    timing_path = root / 'eval_timing' / 'confirm' / 'timing.json'
    if timing_path.exists():
        numbers['timing'] = {k: v['timing_s'][0] for k, v in json.loads(timing_path.read_text(encoding='utf-8')).items()}
    (out / 'numbers.json').write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding='utf-8')

    # 图1：主表 SGM 与 W（确认集）
    order = ['容量感知贪心', '遗传320次', '模拟退火320次', '历史版T-SAC＋delta奖励', '历史版T-SAC', '新版T-SAC＋delta奖励', '新版T-SAC',
             'PPO（Transformer）', 'DQN（Transformer）', 'CNN-SAC', 'MLP-SAC', '固定分配', '随机分配']
    colors = ['#555555', '#888888', '#aaaaaa', '#c0392b', '#e67e22', '#f1c40f', '#f5b041', '#2980b9', '#16a085', '#8e44ad', '#9b59b6', '#bdc3c7', '#d5dbdb']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, col, title in zip(axes, ('SGM', 'W', 'worst10'), ('SGM（主指标）', '满足率 W', '最差10%飞机满足率')):
        vals = [numbers['confirm'][m][col] for m in order]
        errs = [numbers['confirm'][m][col + '_std'] for m in order]
        ax.barh(y, vals, xerr=errs, color=colors, capsize=3)
        for yi, v in zip(y, vals):
            ax.text(v + .01, yi, f'{v:.3f}', va='center', fontsize=8)
        ax.axvline(numbers['confirm']['容量感知贪心'][col], color='k', ls='--', lw=1)
        ax.set_title(title); ax.set_xlim(0, 1.05); ax.grid(axis='x', alpha=.3)
    axes[0].set_yticks(y); axes[0].set_yticklabels(order)
    fig.suptitle('确认集 287 景直出结果（学习方法为 3 种子均值，误差棒为种子标准差；虚线为贪心）')
    fig.tight_layout(); fig.savefig(figs / 'F1_主表.png', dpi=150); plt.close(fig)

    # 图2：三档
    sel = ['容量感知贪心', '历史版T-SAC＋delta奖励', '历史版T-SAC', '新版T-SAC', 'PPO（Transformer）', 'DQN（Transformer）', 'CNN-SAC', 'MLP-SAC', '固定分配']
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    x = np.arange(len(STRATA))
    w = .8 / len(sel)
    pal = plt.cm.tab10(np.arange(len(sel)))
    for ax, col, title in zip(axes, ('SGM', 'W'), ('SGM', '满足率 W')):
        for i, m in enumerate(sel):
            ax.bar(x + (i - len(sel) / 2 + .5) * w, [numbers['strata'][m][s][col] for s in STRATA], w, label=m, color=pal[i])
        ax.set_xticks(x); ax.set_xticklabels([STRATA_ZH[s] for s in STRATA]); ax.set_title(f'确认集分档 {title}'); ax.grid(axis='y', alpha=.3); ax.set_ylim(0, 1.05)
    axes[1].legend(fontsize=8, ncol=1, bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout(); fig.savefig(figs / 'F2_分档.png', dpi=150); plt.close(fig)

    # 图3：开发集曲线（种子均值，阴影为最小—最大）
    curve_sets = {'历史版T-SAC＋delta奖励': 'extra-legacydelta', '历史版T-SAC': 'main-tsaclegacy', '新版T-SAC': 'main-tsac',
                  'PPO（Transformer）': 'main-ppo', 'DQN（Transformer）': 'main-dqn', 'CNN-SAC': 'main-cnnsac', 'MLP-SAC': 'main-mlpsac'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    pal = plt.cm.tab10(np.arange(len(curve_sets)))
    for i, (name, prefix) in enumerate(curve_sets.items()):
        recs = [read_jsonl(root / 'runs' / f'{prefix}-s{s}' / 'dev.jsonl') for s in (1, 2, 3)]
        steps = sorted(set.intersection(*[set(r['step'] // 10000 for r in rec) for rec in recs]))
        for ax, col in zip(axes, ('SGM', 'W')):
            m = np.array([[next(r[col] for r in rec if r['step'] // 10000 == st) for st in steps] for rec in recs])
            xs = np.array(steps) * 1e4
            ax.plot(xs, m.mean(0), color=pal[i], lw=2, label=name)
            ax.fill_between(xs, m.min(0), m.max(0), color=pal[i], alpha=.12)
    for ax, col in zip(axes, ('SGM', 'W')):
        ax.axhline(numbers['dev']['容量感知贪心'][col], color='k', ls='--', lw=1, label='容量感知贪心')
        ax.set_xlabel('环境步'); ax.set_title(f'开发集 {col}（60 景直出，每 1 万步评一次）'); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / 'F3_开发集曲线.png', dpi=150); plt.close(fig)

    # 图4：最好方法的训练过程（奖励、损失、SGM、SI，滑动平均）
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    pal = plt.cm.Set1(np.arange(3))
    for i, s in enumerate((1, 2, 3)):
        rows = read_jsonl(root / 'runs' / f'extra-legacydelta-s{s}' / 'curve.jsonl')
        st = np.array([r['step'] for r in rows])
        axes[0, 0].plot(st, smooth([r['reward'] for r in rows], 10), color=pal[i], label=f'种子 {s}')
        axes[0, 1].plot(st, smooth([r.get('loss_critic_loss', np.nan) for r in rows], 10), color=pal[i], label=f'评论家 种子{s}')
        axes[0, 1].plot(st, smooth([r.get('loss_actor_loss', np.nan) for r in rows], 10), color=pal[i], ls='--', label=f'演员 种子{s}')
        axes[1, 0].plot(st, smooth([r.get('SGM', np.nan) for r in rows], 10), color=pal[i], label=f'训练场景 种子{s}')
        axes[1, 1].plot(st, smooth([r.get('SI', np.nan) for r in rows], 10), color=pal[i], label=f'训练场景 种子{s}')
        dv = read_jsonl(root / 'runs' / f'extra-legacydelta-s{s}' / 'dev.jsonl')
        axes[1, 0].plot([d['step'] for d in dv], [d['SGM'] for d in dv], 'o', ms=3, color=pal[i], alpha=.6)
        axes[1, 1].plot([d['step'] for d in dv], [d['SI'] for d in dv], 'o', ms=3, color=pal[i], alpha=.6)
    for ax, t in zip(axes.flat, ('每步奖励（delta 奖励）', '损失', 'SGM（线：训练场景最近200回合；点：开发集）', '束平均满足率 SI（同左）')):
        ax.set_title(t); ax.set_xlabel('环境步'); ax.grid(alpha=.3)
    axes[0, 1].set_yscale('symlog'); axes[0, 1].legend(fontsize=7, ncol=2); axes[0, 0].legend(fontsize=8)
    fig.suptitle('历史版T-SAC＋delta 奖励的训练过程（滑动平均 10 点 = 2 万步）')
    fig.tight_layout(); fig.savefig(figs / 'F4_最好方法训练过程.png', dpi=150); plt.close(fig)

    # 图5：网络结构对照（同一算法换编码器）
    groups = [('SAC', ['新版T-SAC', 'CNN-SAC', 'MLP-SAC']), ('PPO', ['PPO（Transformer）', 'PPO（MLP）']), ('DQN', ['DQN（Transformer）', 'DQN（MLP）'])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xpos, labels, vals, errs, cols = [], [], [], [], []
    p = 0
    enc_color = {'T': '#2980b9', 'C': '#8e44ad', 'M': '#95a5a6'}
    for g, ms in groups:
        for m in ms:
            xpos.append(p); labels.append(m); vals.append(numbers['confirm'][m]['SGM']); errs.append(numbers['confirm'][m]['SGM_std'])
            cols.append(enc_color['C'] if 'CNN' in m else enc_color['M'] if 'MLP' in m else enc_color['T']); p += 1
        p += .8
    ax.bar(xpos, vals, yerr=errs, color=cols, capsize=3)
    for xi, v in zip(xpos, vals):
        ax.text(xi, v + .02, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=20, fontsize=9); ax.set_ylabel('SGM'); ax.set_ylim(0, .8)
    ax.set_title('同一算法换编码器（确认集 SGM，蓝=Transformer，紫=CNN，灰=MLP）'); ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(figs / 'F5_网络对照.png', dpi=150); plt.close(fig)

    # 图6：消融（参数量、令牌/定组/奖励）
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    ax = axes[0]
    pr = [numbers['size'][l]['params'] for l, _, _ in SIZES]
    for col, mk in (('SGM', 'o-'), ('W', 's--')):
        ax.plot(pr, [numbers['size'][l][col] for l, _, _ in SIZES], mk, label=col)
    for l, _, _ in SIZES:
        ax.annotate(l, (numbers['size'][l]['params'], numbers['size'][l]['SGM']), fontsize=8, xytext=(3, 4), textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel('策略网络参数量'); ax.set_title('参数量（种子1，确认集）'); ax.legend(); ax.grid(alpha=.3)
    ax = axes[1]
    abl = ['新版T-SAC', '新版＋%8定组', '新版＋贪心定组', '新版＋每束一令牌', '新版＋块指针头', '新版T-SAC＋delta奖励', '历史版T-SAC', '历史版T-SAC＋delta奖励']
    v = [numbers['confirm'][m]['SGM'] for m in abl]
    e = [numbers['confirm'][m]['SGM_std'] for m in abl]
    ax.barh(np.arange(len(abl))[::-1], v, xerr=e, capsize=3, color=['#f5b041'] + ['#5dade2'] * 4 + ['#f1c40f', '#e67e22', '#c0392b'])
    for yi, vi in zip(np.arange(len(abl))[::-1], v):
        ax.text(vi + .01, yi, f'{vi:.3f}', va='center', fontsize=8)
    ax.set_yticks(np.arange(len(abl))[::-1]); ax.set_yticklabels(abl); ax.set_xlim(0, .9)
    ax.axvline(numbers['confirm']['容量感知贪心']['SGM'], color='k', ls='--', lw=1)
    ax.set_title('令牌、选组、奖励（3 种子，确认集 SGM；虚线贪心）'); ax.grid(axis='x', alpha=.3)
    fig.tight_layout(); fig.savefig(figs / 'F6_消融.png', dpi=150); plt.close(fig)

    # 图7：躲干扰
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pts = order + ['新版＋%8定组', '新版＋贪心定组', '新版＋每束一令牌', '新版＋块指针头', 'PPO（MLP）', 'DQN（MLP）']
    for m in pts:
        c = numbers['confirm'][m]
        rule = m in ('容量感知贪心', '遗传320次', '模拟退火320次', '固定分配', '随机分配')
        ax.scatter(c['adj_ratio'], c['interference_loss'], s=60, marker='s' if rule else 'o', color='k' if rule else '#c0392b')
        label = {'容量感知贪心': '贪心（与退火、遗传重合）', '遗传320次': None, '模拟退火320次': None}.get(m, m)
        if label and (rule or m in ('历史版T-SAC＋delta奖励', '历史版T-SAC', '新版T-SAC', 'CNN-SAC', 'PPO（Transformer）')):
            ax.annotate(label, (c['adj_ratio'], c['interference_loss']), fontsize=8, xytext=(6, 2), textcoords='offset points')
    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('相邻同频共槽 ÷ 随机预期（<1 才是在避让）'); ax.set_ylabel('被干扰吃掉的业务量占比')
    ax.set_title('躲干扰能力（确认集；方块=规则/文献，圆点=学习方法）'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(figs / 'F7_躲干扰.png', dpi=150); plt.close(fig)
    print(json.dumps({k: len(v) for k, v in numbers.items()}, ensure_ascii=False))
    print(out)


if __name__ == '__main__':
    main()
