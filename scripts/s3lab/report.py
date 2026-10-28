"""S3-LAB 报告：数字（numbers.json）与 SVG 图。

  python scripts/s3lab/report.py [--runs D:/paper/outputs/experiments/s3lab_20261029/runs]
主口径：接入输出（保 SGM），场景等权；两段式先平均 3 个求解器种子；随机/块指针头 3 套实现取平均。
区间：日期固定，日内按 1 小时时段整块自举 2000 次，同次抽样保持方法/档间配对；三套实现各自按整体重抽。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('D:/paper/outputs/experiments/s3lab_20261029')
TIERS = ['random', 'fixed', 'greedy', 'ptr']
TIER_ZH = {'random': '随机', 'fixed': '固定规则', 'greedy': '容量感知贪心', 'ptr': '块指针头 T-SAC'}
METHODS = ['none', 'sca', 'aravanis', 'wmmse', 'slsqp', 'choi_chan', 'target_cap', 'proportional', 'equal', 'maxmin']
M_ZH = {'none': '不细调', 'sca': 'SCA（本研究）', 'aravanis': 'Aravanis 两段式', 'choi_chan': 'Choi-Chan 注水',
        'proportional': '按需求比例', 'equal': '等功率', 'target_cap': '目标容量迭代', 'maxmin': '最大最小公平',
        'wmmse': 'WMMSE', 'slsqp': 'SLSQP'}
SET_ZH = {'old': '旧确认集', 'new': '新确认集'}
TL = {'random': '随机', 'fixed': '固定规则', 'greedy': '容量感知\n贪心', 'ptr': '块指针头\nT-SAC'}
STRATA = ['ample', 'marginal', 'short']
ST_ZH = {'ample': '宽裕', 'marginal': '勉强', 'short': '不够'}
B = 2000


def load_rows(path):
    return pd.DataFrame([json.loads(l) for l in open(path, encoding='utf-8')])


class Boot:
    """一个集合的自举器：场景按 (日期, 小时) 分块，日期固定、日内整块重抽。"""

    def __init__(self, keys, seed=0):
        self.keys = list(keys)
        dates = np.array([k[:8] for k in self.keys])
        hours = np.array([k[:11] for k in self.keys])
        self.groups = []
        for dt in sorted(set(dates)):
            blocks = sorted(set(hours[dates == dt]))
            self.groups.append([np.nonzero(hours == h)[0] for h in blocks])
        rng = np.random.default_rng(seed)
        self.draws = []
        for _ in range(B):
            idx = []
            for blocks in self.groups:
                pick = rng.integers(0, len(blocks), len(blocks))
                idx.append(np.concatenate([blocks[i] for i in pick]))
            self.draws.append(np.concatenate(idx))
        self.impl = rng.integers(0, 3, (B, 2, 3))            # 两类三套实现（随机、块指针头）各自整体重抽

    def stat(self, arrs, fn):
        """arrs: {名字: [实现数, 场景数]}；fn(dict 名字→[场景]) → 标量。返回 (点估计, 下限, 上限)。"""
        point = fn({k: v.mean(0) for k, v in arrs.items()})
        vals = []
        for b, idx in enumerate(self.draws):
            sub = {}
            for k, v in arrs.items():
                if v.shape[0] == 3:
                    which = self.impl[b, 0 if k.startswith('random') else 1]
                    sub[k] = v[which][:, idx].mean(0)
                else:
                    sub[k] = v[0, idx]
            vals.append(fn(sub))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(point), float(lo), float(hi)


def tier_array(d, s, tier, method, col, keys, delta=0., tag=''):
    """[实现数, 场景数] 的数组（场景顺序同 keys；两段式先平均求解器种子）。"""
    x = d[(d.set == s) & (d.tier == tier) & (d.method == method) & (np.isclose(d.delta, delta)) & (d.tag == tag)]
    x = x.groupby(['plan', 'key'])[col].mean().unstack('key')[keys]
    return x.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(ROOT / 'runs'))
    a = ap.parse_args()
    runs = Path(a.runs)
    out = ROOT / 'report'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    d = load_rows(runs / 'main' / 'rows.jsonl')
    bounds = d[d.method == 'bounds'].copy()
    d = d[d.method != 'bounds'].copy()
    d['tag'] = d['tag'].fillna('')
    for c in ('W', 'SGM', 'worst10', 'SI', 'power_w'):
        d[f'd_{c}'] = d[f'a_{c}'] - d[{'W': 'W0', 'SGM': 'F0', 'worst10': 'worst10_0', 'SI': 'SI0', 'power_w': 'P0'}[c]]
    n = dict(main={}, gaps={}, strata={}, time={}, bounds={}, status={}, tradeoff={}, panel={}, bigbeam={})
    boots = {}
    for s in ('old', 'new'):
        keys = sorted(d[(d.set == s) & (d.plan == 'greedy')].key.unique())
        smap = d[(d.set == s) & (d.plan == 'greedy')].groupby('key').stratum.first().to_dict()
        boots[s] = bt = Boot(keys, seed=1 if s == 'old' else 2)
        for t in TIERS:
            for m in METHODS:
                rec = {}
                for col in ('a_W', 'd_W', 'a_SGM', 'd_SGM', 'a_worst10', 'd_worst10', 'a_power_w', 'n_W', 'n_SGM'):
                    arr = tier_array(d, s, t, m, col, keys)
                    if col in ('d_W', 'd_worst10'):
                        rec[col] = bt.stat({'x': arr}, lambda z: z['x'].mean())
                    else:
                        rec[col] = (float(arr.mean()),) * 3
                x = d[(d.set == s) & (d.tier == t) & (d.method == m) & (d.tag == '')]
                rec['t_median'] = float(x.t_native.median())
                rec['t_p90'] = float(x.t_native.quantile(.9))
                rec['t_access'] = float(x.t_access.median())
                rec['fail'] = float((x.status == 'fail').mean())
                rec['limit'] = float((x.status == 'limit').mean())
                rec['tau1'] = float((x.tau >= .999).mean())
                rec['tau0'] = float((x.tau <= 1e-9).mean())
                n['main'][f'{s}|{t}|{m}'] = rec
                for st in STRATA:
                    ks = [k for k in keys if smap[k] == st]
                    arrs = {c: tier_array(d, s, t, m, c, ks) for c in ('a_W', 'd_W', 'a_SGM', 'a_worst10', 'd_worst10', 'a_power_w')}
                    n['strata'][f'{s}|{t}|{m}|{st}'] = {c: float(v.mean()) for c, v in arrs.items()}
        # 档间差距（同方法）：块指针头−贪心、贪心−固定、固定−随机
        for m in METHODS:
            for hi_t, lo_t in (('ptr', 'greedy'), ('greedy', 'fixed'), ('fixed', 'random')):
                rec = {}
                for col in ('a_W', 'a_worst10', 'a_SGM'):
                    arrs = {hi_t: tier_array(d, s, hi_t, m, col, keys), lo_t + '_': tier_array(d, s, lo_t, m, col, keys)}
                    if lo_t == 'random':
                        arrs = {hi_t: arrs[hi_t], 'random': arrs['random_']}
                        fn = lambda z, h=hi_t: z[h].mean() - z['random'].mean()
                    else:
                        fn = lambda z, h=hi_t, l=lo_t + '_': z[h].mean() - z[l].mean()
                    rec[col] = bt.stat(arrs, fn)
                    for st in STRATA:
                        ks = [i for i, k in enumerate(keys) if smap[k] == st]
                        rec[f'{col}|{st}'] = float(np.mean([v.mean(0)[ks].mean() * (1 if nm == hi_t else -1) for nm, v in arrs.items()]) * 2)
                n['gaps'][f'{s}|{m}|{hi_t}-{lo_t}'] = rec
        # 方法对 SCA 的配对差（同档）
        for t in TIERS:
            for m in METHODS:
                if m == 'sca':
                    continue
                arrs = {'m': tier_array(d, s, t, m, 'a_W', keys), 'sca': tier_array(d, s, t, 'sca', 'a_W', keys)}
                n['main'][f'{s}|{t}|{m}']['vs_sca_W'] = bt.stat(arrs, lambda z: (z['m'] - z['sca']).mean())
        # 上界
        bb = bounds[bounds.set == s]
        for t in TIERS:
            x = bb[bb.tier == t]
            n['bounds'][f'{s}|{t}'] = {c: float(x[c].mean()) for c in ('W_H', 'W_U', 'SGM_U', 'slot_short_beams', 'slot_short_aircraft',
                                                                     'slot_short_demand', 'link_short_beams', 'link_short_demand')}
            for st in STRATA:
                y = x[x.stratum == st]
                n['bounds'][f'{s}|{t}|{st}'] = {c: float(y[c].mean()) for c in ('W_H', 'W_U', 'slot_short_demand', 'link_short_demand')}
        # 时间点
        for t in TIERS:
            for m in METHODS:
                x = d[(d.set == s) & (d.tier == t) & (d.method == m) & (d.tag == '')]
                curve = {}
                for mk in ('0.1', '0.3', '1', '3', '10'):
                    v = np.array([sn[mk] for sn in x.snaps])
                    curve[mk] = dict(nW=float((v[:, 0] - x.W0.to_numpy()).mean()), aW=float((v[:, 2] - x.W0.to_numpy()).mean()))
                n['time'][f'{s}|{t}|{m}'] = curve
    # 取舍曲线
    tp = runs / 'tradeoff' / 'rows.jsonl'
    if tp.exists():
        tr = load_rows(tp)
        tr['tag'] = tr['tag'].fillna('')
        for s in ('old', 'new'):
            keys = boots[s].keys
            for t in TIERS:
                for delta in (0., .005, .01, .02):
                    src = d if delta == 0 else tr
                    rec = {}
                    for col, base in (('a_W', 'W0'), ('a_worst10', 'worst10_0'), ('a_SGM', 'F0')):
                        x = src[(src.set == s) & (src.tier == t) & (src.method == 'sca') & np.isclose(src.delta, delta) & (src.tag == '')]
                        rec[col] = float((x[col] - x[base]).mean())
                    n['tradeoff'][f'{s}|{t}|{delta:g}'] = rec
    pp = runs / 'panel' / 'rows.jsonl'
    if pp.exists():
        pn = load_rows(pp)
        pn['tag'] = pn['tag'].fillna('')
        for t in TIERS:
            for m in ('sca', 'slsqp', 'aravanis', 'wmmse'):
                x = pn[(pn.tier == t) & (pn.method == m) & (pn.tag == 't30')]
                if len(x):
                    n['panel'][f't30|{t}|{m}'] = {mk: float(np.mean([sn[mk][2] for sn in x.snaps]) - x.W0.mean()) for mk in ('1', '3', '10', '30')}
            for m in ('slsqp', 'aravanis'):
                for delta in (.005, .01, .02):
                    x = pn[(pn.tier == t) & (pn.method == m) & np.isclose(pn.delta, delta) & (pn.tag == '')]
                    if len(x):
                        n['panel'][f'tradeoff|{t}|{m}|{delta:g}'] = dict(dW=float((x.a_W - x.W0).mean()), d10=float((x.a_worst10 - x.worst10_0).mean()))
            # 同一面板上 SCA 的取舍（来自全量）
            keyset = set(zip(pn.set, pn.key))
            for delta in (0., .005, .01, .02):
                src = d if delta == 0 else load_rows(tp) if tp.exists() else None
                if src is None:
                    continue
                if 'tag' in src:
                    src = src.fillna({'tag': ''})
                x = src[(src.tier == t) & (src.method == 'sca') & np.isclose(src.delta, delta)]
                x = x[[(a, b) in keyset for a, b in zip(x.set, x.key)]]
                n['panel'][f'tradeoff|{t}|sca|{delta:g}'] = dict(dW=float((x.a_W - x.W0).mean()), d10=float((x.a_worst10 - x.worst10_0).mean()))
    bigbeam(n, d, runs)
    (out / 'numbers.json').write_text(json.dumps(n, ensure_ascii=False, indent=1), encoding='utf-8')
    figures(n, out / 'figures')
    print('done', {k: len(v) for k, v in n.items()})


def bigbeam(n, d, runs):
    """不够档：每景按飞机数最大的 6 束，细调前后的满足率、功率（SCA 接入；块指针头与贪心）。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from satellite_rl.s2lab.scenes import load_scene, read_config
    from satellite_rl.s3lab.plans import load
    from satellite_rl.s3lab.problem import FixedPlan
    ds = {'old': 'D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1', 'new': 'D:/paper/outputs/experiments/s2lab_round4_20261028/data/c2'}
    for s in ('old', 'new'):
        cfg = read_config(ds[s])
        ph = cfg.physics_obj()
        for pid in ('ptr-s1', 'ptr-s2', 'ptr-s3', 'greedy'):
            obj = load(ROOT / 'plans' / s / f'{pid}.pkl')
            acc = []
            for j, r in enumerate(obj['records']):
                if r['stratum'] != 'short':
                    continue
                z = np.load(runs / 'main' / 'powers' / f'{s}_{pid}_{j}.npz')
                fp = FixedPlan(load_scene(ds[s], r['key']), ph, r['demand'], r['groups'], r['starts'], r['counts'])
                top = np.argsort(-fp.members)[:6]
                for tag, p in (('before', z['p0'].astype(np.float64)), ('after', z['a_sca_0_0_'].astype(np.float64))):
                    e = fp.evaluate(p)
                    si = np.minimum(e.C / fp.demand, 1.)
                    full = fp.full(p)
                    acc.append(dict(tag=tag, si_top=float(si[top].mean()), p_top=float(full[top].mean()), slots_top=float(r['counts'][top].mean()),
                                    members_top=float(fp.members[top].mean()), si_rest=float(np.delete(si, top).mean())))
            x = pd.DataFrame(acc).groupby('tag').mean()
            n['bigbeam'][f'{s}|{pid}'] = x.to_dict('index')


def figures(n, figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'], 'font.size': 10.5, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'axes.grid': True, 'grid.alpha': .3, 'grid.linestyle': '--'})
    colors = {'none': '#7f7f7f', 'sca': '#d62728', 'aravanis': '#1f77b4', 'wmmse': '#2ca02c', 'slsqp': '#9467bd', 'choi_chan': '#8c564b',
              'target_cap': '#e377c2', 'proportional': '#bcbd22', 'equal': '#17becf', 'maxmin': '#ff7f0e'}
    markers = {'none': 'o', 'sca': 's', 'aravanis': '^', 'wmmse': 'v', 'slsqp': 'D', 'choi_chan': 'P', 'target_cap': 'X',
               'proportional': '*', 'equal': 'h', 'maxmin': '<'}
    # 图1：各档细调后的 W（两集）
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 7.5 / 2.54), sharey=True)
    for ax, s in zip(axes, ('old', 'new')):
        x = np.arange(len(TIERS))
        w = .8 / len(METHODS)
        for i, m in enumerate(METHODS):
            v = [n['main'][f'{s}|{t}|{m}']['a_W'][0] for t in TIERS]
            ax.bar(x + (i - len(METHODS) / 2 + .5) * w, v, w, color=colors[m], label=M_ZH[m], edgecolor='k', linewidth=.3)
        ax.set_xticks(x)
        ax.set_xticklabels([TL[t] for t in TIERS], fontsize=9)
        ax.set_title(SET_ZH[s], fontsize=10.5)
        ax.set_ylim(.4, 1.)
    axes[0].set_ylabel('需求满足率 W（SGM 不降）')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(.5, 1.12))
    fig.tight_layout()
    fig.savefig(figs / 'S1_各档细调后满足率.svg', bbox_inches='tight'); fig.savefig(figs / 'S1_各档细调后满足率.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    # 图2：W 提升—时间曲线（块指针头、贪心；旧集）
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 6.5 / 2.54), sharey=False)
    marks = ['0.1', '0.3', '1', '3', '10']
    for ax, t in zip(axes, ('ptr', 'greedy')):
        for m in METHODS:
            if m == 'none':
                continue
            c = n['time'][f'old|{t}|{m}']
            ax.plot([float(k) for k in marks], [c[k]['aW'] for k in marks], marker=markers[m], color=colors[m], label=M_ZH[m], ms=4, lw=1.2)
        ax.set_xscale('log')
        ax.set_xticks([.1, .3, 1, 3, 10])
        ax.set_xticklabels(['0.1', '0.3', '1', '3', '10'])
        ax.minorticks_off()
        ax.set_xlabel('求解时间（秒）')
        ax.set_title(TIER_ZH[t] + '方案（旧确认集）', fontsize=10.5)
    axes[0].set_ylabel('W 提升（SGM 不降）')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(.5, 1.13))
    fig.tight_layout()
    fig.savefig(figs / 'S2_满足率提升随时间.svg', bbox_inches='tight'); fig.savefig(figs / 'S2_满足率提升随时间.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    # 图3：块指针头−贪心的差距，细调前后（W、最差10%、SGM）
    fig, axes = plt.subplots(1, 3, figsize=(16 / 2.54, 6 / 2.54))
    ms = ['none', 'sca', 'aravanis', 'wmmse', 'slsqp']
    for ax, col, ttl in zip(axes, ('a_W', 'a_worst10', 'a_SGM'), ('需求满足率 W', '最差 10% 飞机', 'SGM')):
        for k, s in enumerate(('old', 'new')):
            vals = [n['gaps'][f'{s}|{m}|ptr-greedy'][col] for m in ms]
            y = np.arange(len(ms)) + (k - .5) * .35
            ax.errorbar([v[0] for v in vals], y, xerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt='o' if k == 0 else 's',
                        color='#1f77b4' if k == 0 else '#d62728', ms=4, capsize=2, label=SET_ZH[s])
        ax.axvline(0, color='k', lw=.8)
        ax.set_yticks(np.arange(len(ms)))
        ax.set_yticklabels([M_ZH[m] for m in ms] if col == 'a_W' else [], fontsize=8.5)
        ax.set_title(ttl, fontsize=10)
        ax.invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(.5, 1.08))
    fig.supxlabel('块指针头 − 贪心（同一细调方法）', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(figs / 'S3_块指针头与贪心差距.svg', bbox_inches='tight'); fig.savefig(figs / 'S3_块指针头与贪心差距.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    # 图4：SCA 细调后 W 与两种上界（按负载档）
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 6.5 / 2.54), sharey=True)
    for ax, s in zip(axes, ('old', 'new')):
        x = np.arange(len(TIERS))
        for k, st in enumerate(STRATA):
            off = (k - 1) * .27
            before = [n['strata'][f'{s}|{t}|none|{st}']['a_W'] for t in TIERS]
            after = [n['strata'][f'{s}|{t}|sca|{st}']['a_W'] for t in TIERS]
            WU = [n['bounds'][f'{s}|{t}|{st}']['W_U'] for t in TIERS]
            ax.bar(x + off, after, .25, color=['#c6dbef', '#6baed6', '#2171b5'][k], edgecolor='k', linewidth=.3, label=f'{ST_ZH[st]}档 SCA 后')
            ax.scatter(x + off, before, marker='_', color='k', s=60, zorder=3, label='细调前' if k == 0 else None)
            ax.scatter(x + off, WU, marker='v', color='#d62728', s=14, zorder=3, label='无干扰 50 W 上界' if k == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([TL[t] for t in TIERS], fontsize=9)
        ax.set_title(SET_ZH[s], fontsize=10.5)
    axes[0].set_ylabel('需求满足率 W')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(.5, 1.1))
    fig.tight_layout()
    fig.savefig(figs / 'S4_满足率与上界.svg', bbox_inches='tight'); fig.savefig(figs / 'S4_满足率与上界.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    # 图5：取舍曲线（SCA；允许 SGM 下降 δ）
    if n['tradeoff']:
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 6.5 / 2.54))
        ds = [0., .005, .01, .02]
        for ax, col, ttl in zip(axes, ('a_W', 'a_worst10'), ('W 提升', '最差 10% 飞机变化')):
            for t, mk, cc in zip(TIERS, 'os^D', ('#7f7f7f', '#1f77b4', '#2ca02c', '#d62728')):
                ax.plot(ds, [n['tradeoff'][f'old|{t}|{dd:g}'][col] for dd in ds], marker=mk, color=cc, label=TIER_ZH[t] + '（旧）', lw=1.2, ms=4)
                ax.plot(ds, [n['tradeoff'][f'new|{t}|{dd:g}'][col] for dd in ds], marker=mk, color=cc, mfc='white', ls='--', label=TIER_ZH[t] + '（新）', lw=1, ms=4)
            ax.set_xlabel('允许 SGM 下降 δ')
            ax.set_title(ttl, fontsize=10.5)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(.5, 1.15))
        fig.tight_layout()
        fig.savefig(figs / 'S5_取舍曲线.svg', bbox_inches='tight'); fig.savefig(figs / 'S5_取舍曲线.png', bbox_inches='tight', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    main()

