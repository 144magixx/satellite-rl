"""逐束诊断分析：读一个或多个 beams.csv，按策略组（去掉种子后缀）汇总。
  python scripts/s2lab/beam_analyze.py <beams.csv> [...] [--out summary.json]
看三件事：
1. 宽裕档过供：每束容量/需求、功率、槽数；容量超出需求 20% 以上的波束占比和多给的功率。
2. 资源按波束大小（飞机数）与处理先后（局内名次三等分）怎么分：槽数、功率、满足率。
3. 最差飞机在哪：不够档里按飞机加权、满足率最低 10% 的飞机所在波束的特征（大小、名次、槽数、功率、干扰损失）。
"""
import argparse
import json

import numpy as np
import pandas as pd


def group_of(p):
    return p if p.startswith('rule-') else p.rsplit('-s', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out')
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in a.files], ignore_index=True)
    d['grp'] = d.policy.map(group_of)
    d['ratio'] = d.cap_mbps / d.demand_mbps
    d['intf_loss'] = 1 - d.cap_mbps / d.cap_noint_mbps.clip(lower=1e-9)
    d['size'] = pd.cut(d.members, [0, 2, 5, 10, 100], labels=['1—2架', '3—5架', '6—10架', '11架以上'])
    d['third'] = pd.cut(d['rank'] / d.n, [-1e-9, 1 / 3, 2 / 3, 1.0001], labels=['前三分之一', '中间', '后三分之一'])
    out = {}
    nrun = d.groupby('grp').policy.nunique()
    # 1 宽裕档过供（每景平均，策略组内各种子再平均）
    am = d[d.stratum == 'ample']
    per = am.groupby(['grp', 'policy', 'key']).agg(power=('power', 'sum'), slots=('slots', 'sum'),
                                                    over=('ratio', lambda r: float((r > 1.2).mean())), ratio=('ratio', 'median'),
                                                    extra_power=('power', lambda p: 0.)).reset_index()
    over_power = am[am.ratio > 1.2].groupby(['grp', 'policy', 'key']).power.sum().rename('over_power')
    per = per.merge(over_power, on=['grp', 'policy', 'key'], how='left').fillna({'over_power': 0.})
    out['ample'] = per.groupby('grp')[['power', 'slots', 'over', 'ratio', 'over_power']].mean().round(3).to_dict('index')
    # 2 资源按大小、名次
    for col, name in (('size', 'by_size'), ('third', 'by_rank')):
        t = d.groupby(['grp', 'stratum', col], observed=True).agg(slots=('slots', 'mean'), power=('power', 'mean'), si=('si', 'mean'),
                                                                 intf=('intf_loss', 'mean')).round(3)
        out[name] = {f'{g}|{s}|{c}': v for (g, s, c), v in t.to_dict('index').items()}
    # 3 不够档最差 10% 飞机所在波束
    sh = d[d.stratum == 'short']
    rows = {}
    for (g, pol, key), x in sh.groupby(['grp', 'policy', 'key']):
        planes = x.loc[x.index.repeat(x.members)]
        k = max(1, int(np.ceil(.1 * len(planes))))
        worst = planes.nsmallest(k, 'si')
        beams = x.loc[worst.index.unique()]
        rows.setdefault(g, []).append(dict(worst10=float(worst.si.mean()), beams=len(beams), size=float(beams.members.mean()),
                                           rank_frac=float((beams['rank'] / beams.n).mean()), slots=float(beams.slots.mean()),
                                           power=float(beams.power.mean()), intf=float(beams.intf_loss.mean()),
                                           zero_si=float((beams.si < .05).mean())))
    out['worst'] = {g: pd.DataFrame(v).mean().round(3).to_dict() for g, v in rows.items()}
    out['runs'] = nrun.to_dict()
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    pd.set_option('display.width', 200)
    print('== 宽裕档每景：功率、槽数、容量超需求 20% 以上的波束占比、容量/需求中位、这些波束用的功率')
    print(pd.DataFrame(out['ample']).T)
    print('== 不够档最差 10% 飞机所在波束：飞机数、局内名次（0＝最先放）、槽数、功率、干扰损失、几乎没服务的占比')
    print(pd.DataFrame(out['worst']).T)
    t = d[d.stratum == 'short'].groupby(['grp', 'size'], observed=True).agg(slots=('slots', 'mean'), power=('power', 'mean'), si=('si', 'mean')).round(3)
    print('== 不够档按波束大小'); print(t.unstack('grp'))
    t = d[d.stratum == 'short'].groupby(['grp', 'third'], observed=True).agg(slots=('slots', 'mean'), power=('power', 'mean'), si=('si', 'mean')).round(3)
    print('== 不够档按处理先后'); print(t.unstack('grp'))


if __name__ == '__main__':
    main()
