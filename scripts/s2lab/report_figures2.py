"""第二轮报告的数字与图：全部从 remote2.py fetch 下来的原始文件生成。

  python scripts/s2lab/report_figures2.py [--root D:/paper/outputs/experiments/s2lab_round2_20261025] [--which best_sgm]
读：account_{A,B}/runs2/<run>/{eval_<split>_<which>.csv, dev.jsonl, curve.jsonl}、account_A/eval2/<split>/rule-*.csv、
    account_A/eval2_meta/confirm/meta-*.csv、account_*/eval2/confirm/<run>~<诊断>.csv；第一轮对照读 s2lab_20261024/eval/confirm。
写：<root>/report/{numbers.json, figures/*.png}
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401

COLS = ('W', 'SI', 'SGM', 'worst10', 'power_w', 'slots', 'copol_overlap', 'interference_loss', 'mean_in_db', 'unserved')
STRATA = ('ample', 'marginal', 'short')
STRATA_ZH = dict(ample='宽裕', marginal='勉强', short='不够')
ROUND1 = Path('D:/paper/outputs/experiments/s2lab_20261024/eval/confirm')

S = (1, 2, 3)
METHODS = {
    '新版T-SAC': [f'main-tsac-s{s}' for s in S],
    '历史版12令牌T-SAC': [f'main-tsaclegacy-s{s}' for s in S],
    '历史版12令牌＋delta': [f'extra-legacydelta-s{s}' for s in S],
    '新版＋delta': [f'reward-delta-s{s}' for s in S],
    '14令牌 I/N＋paper': [f't14-in-paper-s{s}' for s in S],
    '14令牌 I/N＋delta': [f't14-in-delta-s{s}' for s in S],
    '14令牌 dBW＋paper': [f't14-dbw-paper-s{s}' for s in S],
    'PPO（Transformer）': [f'main-ppo-s{s}' for s in S],
    'DQN（Transformer）': [f'main-dqn-s{s}' for s in S],
    'CNN-SAC': [f'main-cnnsac-s{s}' for s in S],
    'MLP-SAC': [f'main-mlpsac-s{s}' for s in S],
    'PPO（MLP）': [f'extra-ppomlp-s{s}' for s in S],
    'DQN（MLP）': [f'extra-dqnmlp-s{s}' for s in S],
    '躲干扰A 规则落位': [f'avoid-rule-s{s}' for s in S],
    '躲干扰B 前8候选': [f'avoid-topk-s{s}' for s in S],
    '躲干扰C 干扰罚项': [f'avoid-intf-s{s}' for s in S],
    '躲干扰D 40万步': [f'avoid-long-s{s}' for s in S],
    '功率100档': [f'power100-s{s}' for s in S],
    '新版＋%8定组': [f'group-mod8-s{s}' for s in S],
    '新版＋贪心定组': [f'group-greedy-s{s}' for s in S],
    '新版＋每束一令牌': [f'token-beams-s{s}' for s in S],
    '新版＋块指针头': [f'token-pointer-s{s}' for s in S],
    'CNN-SAC＋LayerNorm': ['fix-cnn-ln-s1'], 'MLP-SAC＋LayerNorm': ['fix-mlp-ln-s1'],
    'CNN-SAC＋自动温度': ['fix-cnn-auto-s1'], 'MLP-SAC＋自动温度': ['fix-mlp-auto-s1'],
    '新版（第一轮双Q取法）': ['dq-perhead-tsac-s1'], '历史版12令牌（第一轮双Q取法）': ['dq-perhead-legacy-s1'],
}
for name in ('tsac', 'cnnsac', 'mlpsac', 'ppo', 'dqn'):
    METHODS[f'无掩码 {name}'] = [f'nomask-{name}-s1']
SIZES = [('d32L1', 32, 1), ('d64L2', 64, 2), ('d128L2', 128, 2), ('d128L4', 128, 4), ('d128L8', 128, 8), ('d256L2', 256, 2),
         ('d256L4', 256, 4), ('d512L4', 512, 4)]
RULES = {'容量感知贪心': 'rule-greedy', '贪心（不看干扰）': 'rule-greedy_noint', '贪心（功率100档）': 'rule-greedy-p0.5',
         '分散摆放规则': 'rule-dispersed', '原论文贪心': 'rule-greedy_paper', '旧贪心（固定5槽）': 'rule-greedy_old',
         '固定分配': 'rule-fixed', '随机分配': 'rule-random'}
META = {'模拟退火32次': 'meta-sa32', '模拟退火320次': 'meta-sa320', '遗传32次': 'meta-ga32', '遗传320次': 'meta-ga320'}
ROUND1_MAP = {'新版T-SAC': 'main-tsac', '历史版12令牌T-SAC': 'main-tsaclegacy', '历史版12令牌＋delta': 'extra-legacydelta',
              '新版＋delta': 'reward-delta', 'PPO（Transformer）': 'main-ppo', 'DQN（Transformer）': 'main-dqn', 'CNN-SAC': 'main-cnnsac',
              'MLP-SAC': 'main-mlpsac', 'PPO（MLP）': 'extra-ppomlp', 'DQN（MLP）': 'extra-dqnmlp', '新版＋%8定组': 'group-mod8',
              '新版＋贪心定组': 'group-greedy', '新版＋每束一令牌': 'token-beams', '新版＋块指针头': 'token-pointer'}


def read_csv(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for r in rows:
        d = dict(key=r['key'], stratum=r['stratum'])
        for c in COLS + ('adj_overlap', 'adj_expected'):
            d[c] = float(r[c]) if r.get(c) not in (None, '', 'nan') else np.nan
        out.append(d)
    return out


class Data:
    def __init__(self, root, which):
        self.root, self.which = root, which
        self.run_dirs = {}
        for acc in ('A', 'B'):
            base = root / f'account_{acc}' / 'runs2'
            if base.exists():
                for d in base.iterdir():
                    if d.is_dir() and not d.name.startswith('_'):
                        self.run_dirs[d.name] = d
        self.eval_dirs = [root / f'account_{acc}' / sub for acc in ('A', 'B') for sub in ('eval2', 'eval2_meta')]

    def path(self, name, split='confirm'):
        """运行名 → 训完自动评价的 CSV；规则/文献/诊断 → eval2 目录下的 CSV。"""
        if name in self.run_dirs:
            p = self.run_dirs[name] / f'eval_{split}_{self.which}.csv'
            return p if p.exists() else None
        for d in self.eval_dirs:
            p = d / split / f'{name}.csv'
            if p.exists():
                return p
        return None

    def rows(self, name, split='confirm', stratum=None):
        p = self.path(name, split)
        if p is None:
            return None
        return [r for r in read_csv(p) if not stratum or r['stratum'] == stratum]

    def available(self, names, split='confirm'):
        return [n for n in names if self.path(n, split) is not None]


def summary(rows_by_seed):
    out = {}
    for c in COLS:
        seeds = [np.nanmean([r[c] for r in rows]) for rows in rows_by_seed]
        out[c], out[c + '_std'] = float(np.mean(seeds)), float(np.std(seeds)) if len(seeds) > 1 else 0.
    allr = [r for rows in rows_by_seed for r in rows]
    expected = sum(r['adj_expected'] for r in allr)
    out['adj_ratio'] = float(sum(r['adj_overlap'] for r in allr) / expected) if expected > 0 else float('nan')   # 无相邻同极对时不适用
    out['seeds'] = len(rows_by_seed)
    out['seed_SGM'] = [float(np.nanmean([r['SGM'] for r in rows])) for rows in rows_by_seed]
    out['seed_W'] = [float(np.nanmean([r['W'] for r in rows])) for rows in rows_by_seed]
    return out


def per_scene(rows_by_seed, col):
    acc = {}
    for rows in rows_by_seed:
        for r in rows:
            acc.setdefault(r['key'], []).append(r[col])
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def paired(a_rows, b_rows, col, n=4000, seed=0):
    a, b = per_scene(a_rows, col), per_scene(b_rows, col)
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return dict(diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), win=float((d > 0).mean()), n=len(keys))


def read_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] if path.exists() else []


def smooth(x, w=10):
    x = np.asarray(x, float)
    return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/paper/outputs/experiments/s2lab_round2_20261025')
    ap.add_argument('--which', default='best_sgm')
    args = ap.parse_args()
    root = Path(args.root)
    data = Data(root, args.which)
    out = root / 'report'
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)

    groups = {}                                                   # 名称 → (成员, 每种子行)
    for name, runs in METHODS.items():
        have = data.available(runs)
        if have:
            groups[name] = have
    for name, f in {**RULES, **META}.items():
        if data.path(f):
            groups[name] = [f]

    def rows_of(name, split='confirm', stratum=None):
        return [data.rows(r, split, stratum) for r in groups[name]]

    numbers = dict(which=args.which, members={k: v for k, v in groups.items()}, confirm={}, dev={}, strata={}, paired={}, size={},
                   diag={}, round1={}, checkpoint={}, timing={})
    for name in groups:
        numbers['confirm'][name] = summary(rows_of(name))
        if all(data.path(r, 'dev') for r in groups[name]):
            numbers['dev'][name] = summary(rows_of(name, 'dev'))
        numbers['strata'][name] = {s: summary(rows_of(name, 'confirm', s)) for s in STRATA}
    pairs = [('历史版12令牌＋delta', '容量感知贪心'), ('历史版12令牌T-SAC', '容量感知贪心'), ('新版T-SAC', '容量感知贪心'),
             ('14令牌 I/N＋delta', '容量感知贪心'), ('躲干扰A 规则落位', '容量感知贪心'), ('躲干扰B 前8候选', '容量感知贪心'),
             ('分散摆放规则', '容量感知贪心'), ('遗传320次', '容量感知贪心'), ('模拟退火320次', '容量感知贪心'),
             ('贪心（不看干扰）', '容量感知贪心'), ('贪心（功率100档）', '容量感知贪心'),
             ('历史版12令牌＋delta', '历史版12令牌T-SAC'), ('新版＋delta', '新版T-SAC'), ('14令牌 I/N＋delta', '14令牌 I/N＋paper'),
             ('历史版12令牌T-SAC', '新版T-SAC'), ('14令牌 I/N＋paper', '历史版12令牌T-SAC'), ('14令牌 dBW＋paper', '14令牌 I/N＋paper'),
             ('新版T-SAC', 'CNN-SAC'), ('新版T-SAC', 'MLP-SAC'), ('PPO（Transformer）', 'PPO（MLP）'), ('DQN（Transformer）', 'DQN（MLP）'),
             ('新版T-SAC', 'PPO（Transformer）'), ('新版T-SAC', 'DQN（Transformer）'), ('历史版12令牌T-SAC', 'PPO（Transformer）'),
             ('新版＋%8定组', '新版T-SAC'), ('新版＋贪心定组', '新版T-SAC'), ('新版＋每束一令牌', '新版T-SAC'), ('新版＋块指针头', '新版T-SAC'),
             ('躲干扰A 规则落位', '历史版12令牌＋delta'), ('躲干扰B 前8候选', '历史版12令牌＋delta'), ('躲干扰C 干扰罚项', '历史版12令牌＋delta'),
             ('躲干扰D 40万步', '历史版12令牌＋delta'), ('功率100档', '历史版12令牌＋delta'), ('功率100档', '贪心（功率100档）'),
             ('CNN-SAC＋LayerNorm', 'CNN-SAC'), ('CNN-SAC＋自动温度', 'CNN-SAC'), ('MLP-SAC＋LayerNorm', 'MLP-SAC'), ('MLP-SAC＋自动温度', 'MLP-SAC'),
             ('新版T-SAC', '新版（第一轮双Q取法）'), ('历史版12令牌T-SAC', '历史版12令牌（第一轮双Q取法）')]
    for a, b in pairs:
        if a in groups and b in groups:
            ra, rb = rows_of(a), rows_of(b)
            numbers['paired'][f'{a} − {b}'] = {c: paired(ra, rb, c) for c in ('SGM', 'W', 'worst10')}
            numbers['paired'][f'{a} − {b}']['by_stratum'] = {s: paired(rows_of(a, 'confirm', s), rows_of(b, 'confirm', s), 'SGM') for s in STRATA}
    # 参数量（d128L2 用主表新版 T-SAC 三种子）
    from satellite_rl.s2lab.agents import make_agent
    from satellite_rl.s2lab.matrix import MAIN_SIZE
    for label, d, layers in SIZES:
        runs = [f'main-tsac-s{s}' for s in S] if label == 'd128L2' else [f'size-{label}-s{s}' for s in S]
        have = data.available(runs)
        if not have:
            continue
        size = dict(MAIN_SIZE, d=d, layers=layers, heads=4 if d <= 64 else 8, ff=4 * d)
        params = make_agent('sac', dict(encoder='tf_blocks', size=size), 13, 'cpu', 20).parameter_count()
        numbers['size'][label] = dict(params=params, members=have, **summary([data.rows(r) for r in have]))
    # 诊断：<run>~<mode>
    for name in ('历史版12令牌＋delta', '历史版12令牌T-SAC', '新版T-SAC', '14令牌 I/N＋delta', 'PPO（Transformer）', '躲干扰B 前8候选', '躲干扰D 40万步'):
        if name not in groups:
            continue
        for mode in ('random_start', 'intf_zero', 'intf_shuffle'):
            runs = data.available([f'{r}~{mode}' for r in groups[name]])
            if runs:
                base = [data.rows(r.split('~')[0]) for r in runs]
                diag = [data.rows(r) for r in runs]
                numbers['diag'].setdefault(name, {})[mode] = dict(summary=summary(diag), base=summary(base),
                                                                  paired={c: paired(diag, base, c) for c in ('SGM', 'W')})
    # 第一轮对照（同配置，第一轮 policy_best）
    for name, prefix in ROUND1_MAP.items():
        files = [ROUND1 / f'{prefix}-s{s}.csv' for s in S]
        if name in groups and all(f.exists() for f in files):
            r1 = [read_csv(f) for f in files]
            numbers['round1'][name] = dict(round1=summary(r1), round2=numbers['confirm'][name], paired={c: paired(rows_of(name), r1, c) for c in ('SGM', 'W')})
    # 两种检查点（best_sgm vs best=SGM+W）
    for name, runs in groups.items():
        if runs[0] in data.run_dirs:
            other = [read_csv(data.run_dirs[r] / 'eval_confirm_best.csv') for r in runs if (data.run_dirs[r] / 'eval_confirm_best.csv').exists()]
            if len(other) == len(runs):
                numbers['checkpoint'][name] = dict(best_sgm=numbers['confirm'][name], best=summary(other))
    timing = root / 'account_A' / 'eval2_timing' / 'confirm' / 'summary.json'
    if timing.exists():
        numbers['timing'] = {k: v['timing_s'][0] for k, v in json.loads(timing.read_text(encoding='utf-8')).items() if v.get('timing_s')}
    meta = root / 'account_A' / 'eval2_meta' / 'confirm' / 'summary.json'
    if meta.exists():
        numbers['timing'].update({k: v['timing_s'][0] for k, v in json.loads(meta.read_text(encoding='utf-8')).items() if v.get('timing_s')})
    (out / 'numbers.json').write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(numbers, data, groups, figs)
    print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in numbers.items() if k != 'members'}, ensure_ascii=False))


def plot(n, data, groups, figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    C = n['confirm']
    greedy = C.get('容量感知贪心', {})

    def bars(ax, names, col, title, colors=None, xlim=1.05):
        names = [m for m in names if m in C]
        y = np.arange(len(names))[::-1]
        vals = [C[m][col] for m in names]
        ax.barh(y, vals, xerr=[C[m][col + '_std'] for m in names], color=colors or '#5dade2', capsize=3)
        for yi, v in zip(y, vals):
            ax.text(v + .01, yi, f'{v:.3f}', va='center', fontsize=8)
        if greedy:
            ax.axvline(greedy[col], color='k', ls='--', lw=1)
        ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, xlim); ax.set_title(title); ax.grid(axis='x', alpha=.3)

    # F1 主表
    main = ['容量感知贪心', '遗传320次', '模拟退火320次', '分散摆放规则', '历史版12令牌＋delta', '14令牌 I/N＋delta', '历史版12令牌T-SAC',
            '14令牌 I/N＋paper', '新版＋delta', '新版T-SAC', 'PPO（Transformer）', 'DQN（Transformer）', 'CNN-SAC', 'MLP-SAC', '原论文贪心', '固定分配', '随机分配']
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True)
    for ax, col, t in zip(axes, ('SGM', 'W', 'worst10'), ('SGM（主指标）', '满足率 W', '最差10%飞机满足率')):
        bars(ax, main, col, t)
    fig.suptitle('第二轮确认集 287 景直出（学习方法 3 种子均值±种子标准差，开发集 SGM 最好的检查点；虚线＝容量感知贪心）')
    fig.tight_layout(); fig.savefig(figs / 'F1_主表.png', dpi=150); plt.close(fig)

    # F2 分档
    sel = [m for m in ['容量感知贪心', '分散摆放规则', '历史版12令牌＋delta', '14令牌 I/N＋delta', '历史版12令牌T-SAC', '新版T-SAC', 'PPO（Transformer）',
                       'CNN-SAC', '固定分配'] if m in C]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(3); w = .8 / max(1, len(sel)); pal = plt.cm.tab10(np.arange(len(sel)))
    for ax, col in zip(axes, ('SGM', 'W')):
        for i, m in enumerate(sel):
            ax.bar(x + (i - len(sel) / 2 + .5) * w, [n['strata'][m][s][col] for s in STRATA], w, label=m, color=pal[i])
        ax.set_xticks(x); ax.set_xticklabels([STRATA_ZH[s] for s in STRATA]); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=.3)
        ax.set_title(f'确认集分档 {col}')
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout(); fig.savefig(figs / 'F2_分档.png', dpi=150); plt.close(fig)

    # F3 开发集曲线
    curve = [m for m in ['历史版12令牌＋delta', '14令牌 I/N＋delta', '历史版12令牌T-SAC', '新版T-SAC', 'PPO（Transformer）', 'DQN（Transformer）',
                         'CNN-SAC', 'MLP-SAC', '躲干扰B 前8候选'] if m in groups and groups[m][0] in data.run_dirs]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5)); pal = plt.cm.tab10(np.arange(len(curve)))
    for i, m in enumerate(curve):
        recs = [read_jsonl(data.run_dirs[r] / 'dev.jsonl') for r in groups[m]]
        steps = sorted(set.intersection(*[set(r['step'] // 10000 for r in rec) for rec in recs]))
        if not steps:
            continue
        for ax, col in zip(axes, ('SGM', 'W')):
            v = np.array([[next(r[col] for r in rec if r['step'] // 10000 == st) for st in steps] for rec in recs])
            xs = np.array(steps) * 1e4
            ax.plot(xs, v.mean(0), color=pal[i], lw=2, label=m); ax.fill_between(xs, v.min(0), v.max(0), color=pal[i], alpha=.12)
    for ax, col in zip(axes, ('SGM', 'W')):
        if n['dev'].get('容量感知贪心'):
            ax.axhline(n['dev']['容量感知贪心'][col], color='k', ls='--', lw=1, label='容量感知贪心')
        ax.set_xlabel('环境步'); ax.set_title(f'开发集 {col}（60 景，每 1 万步一次；线＝种子均值，阴影＝最小—最大）'); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / 'F3_开发集曲线.png', dpi=150); plt.close(fig)

    # F4 最好学习方法训练过程
    learned = [m for m in C if groups[m][0] in data.run_dirs and not m.startswith('无掩码') and C[m]['seeds'] == 3]
    if learned:
        best = max(learned, key=lambda m: C[m]['SGM'])
        fig, axes = plt.subplots(2, 2, figsize=(13, 8)); pal = plt.cm.Set1(np.arange(3))
        for i, r in enumerate(groups[best]):
            rows = read_jsonl(data.run_dirs[r] / 'curve.jsonl'); st = [x['step'] for x in rows]
            axes[0, 0].plot(st, smooth([x['reward'] for x in rows]), color=pal[i], label=r)
            axes[0, 1].plot(st, smooth([x.get('loss_critic_loss', np.nan) for x in rows]), color=pal[i], label=f'评论家 {r[-2:]}')
            axes[0, 1].plot(st, smooth([x.get('loss_actor_loss', np.nan) for x in rows]), color=pal[i], ls='--', label=f'演员 {r[-2:]}')
            dv = read_jsonl(data.run_dirs[r] / 'dev.jsonl')
            for ax, col in ((axes[1, 0], 'SGM'), (axes[1, 1], 'SI')):
                ax.plot(st, smooth([x.get(col, np.nan) for x in rows]), color=pal[i])
                ax.plot([d['step'] for d in dv], [d[col] for d in dv], 'o', ms=3, color=pal[i], alpha=.6)
        for ax, t in zip(axes.flat, ('每步奖励', '损失', 'SGM（线：训练场景近200回合；点：开发集）', '束平均满足率 SI（同左）')):
            ax.set_title(t); ax.set_xlabel('环境步'); ax.grid(alpha=.3)
        axes[0, 1].set_yscale('symlog'); axes[0, 1].legend(fontsize=7, ncol=2); axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{best} 的训练过程（滑动平均 10 点＝2 万步）')
        fig.tight_layout(); fig.savefig(figs / 'F4_最好方法训练过程.png', dpi=150); plt.close(fig)

    # F5 网络对照
    grp = [('SAC', ['新版T-SAC', 'CNN-SAC', 'MLP-SAC', 'CNN-SAC＋LayerNorm', 'MLP-SAC＋LayerNorm', 'CNN-SAC＋自动温度', 'MLP-SAC＋自动温度']),
           ('PPO', ['PPO（Transformer）', 'PPO（MLP）']), ('DQN', ['DQN（Transformer）', 'DQN（MLP）'])]
    fig, ax = plt.subplots(figsize=(12, 4.8)); xs, labels, vals, errs, cols = [], [], [], [], []; p = 0
    for _, ms in grp:
        for m in ms:
            if m in C:
                xs.append(p); labels.append(m); vals.append(C[m]['SGM']); errs.append(C[m]['SGM_std'])
                cols.append('#8e44ad' if 'CNN' in m else '#95a5a6' if 'MLP' in m else '#2980b9'); p += 1
        p += .8
    ax.bar(xs, vals, yerr=errs, color=cols, capsize=3)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + .02, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, fontsize=8, ha='right'); ax.set_ylabel('SGM'); ax.set_ylim(0, 1)
    ax.set_title('同一算法换编码器（确认集 SGM；蓝 Transformer、紫 CNN、灰 MLP）'); ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(figs / 'F5_网络对照.png', dpi=150); plt.close(fig)

    # F6 消融：参数量、令牌/选组/奖励、躲干扰/功率档
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    ax = axes[0]; sz = n['size']
    if sz:
        labs = list(sz)
        ax.errorbar([sz[l]['params'] for l in labs], [sz[l]['SGM'] for l in labs], yerr=[sz[l]['SGM_std'] for l in labs], fmt='o-', capsize=3, label='SGM')
        ax.errorbar([sz[l]['params'] for l in labs], [sz[l]['W'] for l in labs], yerr=[sz[l]['W_std'] for l in labs], fmt='s--', capsize=3, label='W')
        for l in labs:
            ax.annotate(f"{l}（{sz[l]['seeds']}种子）", (sz[l]['params'], sz[l]['SGM']), fontsize=7, xytext=(3, 4), textcoords='offset points')
        ax.set_xscale('log'); ax.set_xlabel('策略网络参数量'); ax.legend(); ax.grid(alpha=.3); ax.set_title('参数量（确认集，误差棒＝种子标准差）')
    bars(axes[1], ['新版T-SAC', '新版＋%8定组', '新版＋贪心定组', '新版＋每束一令牌', '新版＋块指针头', '新版＋delta', '历史版12令牌T-SAC',
                   '14令牌 I/N＋paper', '14令牌 dBW＋paper', '历史版12令牌＋delta', '14令牌 I/N＋delta'], 'SGM', '令牌、选组、奖励（确认集 SGM）', xlim=1)
    bars(axes[2], ['历史版12令牌＋delta', '躲干扰A 规则落位', '躲干扰B 前8候选', '躲干扰C 干扰罚项', '躲干扰D 40万步', '功率100档'], 'SGM',
         '躲干扰与功率档（都基于历史版12令牌＋delta）', xlim=1)
    fig.tight_layout(); fig.savefig(figs / 'F6_消融.png', dpi=150); plt.close(fig)

    # F7 躲干扰散点
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, c in C.items():
        if m.startswith('无掩码'):
            continue
        rule = m in RULES or m in META
        ax.scatter(c['adj_ratio'], c['interference_loss'], s=55, marker='s' if rule else 'o', color='k' if rule else '#c0392b')
        if m in ('容量感知贪心', '贪心（不看干扰）', '分散摆放规则', '固定分配', '历史版12令牌＋delta', '躲干扰A 规则落位', '躲干扰B 前8候选',
                 '躲干扰C 干扰罚项', '新版T-SAC', 'CNN-SAC', '14令牌 I/N＋delta'):
            ax.annotate(m, (c['adj_ratio'], c['interference_loss']), fontsize=8, xytext=(5, 2), textcoords='offset points')
    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('相邻同频共槽 ÷ 随机预期（<1 是在错开）'); ax.set_ylabel('被干扰吃掉的业务量占比'); ax.grid(alpha=.3)
    ax.set_title('躲干扰（确认集；方块＝规则/文献，圆点＝学习方法）')
    fig.tight_layout(); fig.savefig(figs / 'F7_躲干扰.png', dpi=150); plt.close(fig)

    # F8 第一轮 vs 第二轮
    r = n['round1']
    if r:
        names = list(r); y = np.arange(len(names))[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(y + .2, [r[m]['round1']['SGM'] for m in names], .4, color='#bdc3c7', label='第一轮（有缺陷，SGM+W 选模）')
        ax.barh(y - .2, [r[m]['round2']['SGM'] for m in names], .4, color='#2980b9', label='第二轮（修复后，SGM 选模）')
        ax.set_yticks(y); ax.set_yticklabels(names); ax.legend(); ax.grid(axis='x', alpha=.3); ax.set_xlim(0, 1)
        ax.set_title('同配置第一轮与第二轮确认集 SGM')
        fig.tight_layout(); fig.savefig(figs / 'F8_两轮对照.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
