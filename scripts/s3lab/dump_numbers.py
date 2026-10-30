"""把 numbers.json 打成写报告用的文本。"""
import json

n = json.load(open('D:/paper/outputs/experiments/s3lab_20261029/report/numbers.json', encoding='utf-8'))
T = ['random', 'fixed', 'greedy', 'ptr']
M = ['none', 'sca', 'aravanis', 'wmmse', 'slsqp', 'choi_chan', 'target_cap', 'proportional', 'equal', 'maxmin']
for s in ('old', 'new'):
    print(f'================ {s}')
    for t in T:
        print(f'--- {t}  上界 {n["bounds"][f"{s}|{t}"]}')
        for m in M:
            r = n['main'][f'{s}|{t}|{m}']
            vs = r.get('vs_sca_W')
            print(f"{m:12s} W {r['a_W'][0]:.4f} dW {r['d_W'][0]:+.4f}[{r['d_W'][1]:+.4f},{r['d_W'][2]:+.4f}] 原生W {r['n_W'][0]:.4f} 原生SGM {r['n_SGM'][0]:.4f} "
                  f"SGM {r['a_SGM'][0]:.4f} dSGM {r['d_SGM'][0]:+.4f} 10% {r['a_worst10'][0]:.3f} d10 {r['d_worst10'][0]:+.3f}[{r['d_worst10'][1]:+.3f},{r['d_worst10'][2]:+.3f}] "
                  f"P {r['a_power_w'][0]:.0f} t中位 {r['t_median']:.2f} t90 {r['t_p90']:.2f} 限时 {r['limit']:.2f} 失败 {r['fail']:.3f} τ1 {r['tau1']:.2f} τ0 {r['tau0']:.2f}"
                  + (f" 对SCA {vs[0]:+.4f}[{vs[1]:+.4f},{vs[2]:+.4f}]" if vs else ''))
    print('--- 分档（SCA 前后 W / 最差10% / 功率）')
    for t in T:
        for st in ('ample', 'marginal', 'short'):
            a, b = n['strata'][f'{s}|{t}|none|{st}'], n['strata'][f'{s}|{t}|sca|{st}']
            bd = n['bounds'][f'{s}|{t}|{st}']
            print(f"{t:7s} {st:8s} W {a['a_W']:.3f}→{b['a_W']:.3f} 10% {a['a_worst10']:.3f}→{b['a_worst10']:.3f} SGM {a['a_SGM']:.3f}→{b['a_SGM']:.3f} "
                  f"P {a['a_power_w']:.0f}→{b['a_power_w']:.0f} | W_U {bd['W_U']:.3f} W_H {bd['W_H']:.3f} 缺槽需求 {bd['slot_short_demand']:.3f} 链路需求 {bd['link_short_demand']:.3f}")
    print('--- 档间差距')
    for m in M:
        for pair in ('ptr-greedy', 'greedy-fixed', 'fixed-random'):
            g = n['gaps'][f'{s}|{m}|{pair}']
            print(f"{m:12s} {pair:13s} W {g['a_W'][0]:+.4f}[{g['a_W'][1]:+.4f},{g['a_W'][2]:+.4f}] 10% {g['a_worst10'][0]:+.3f}[{g['a_worst10'][1]:+.3f},{g['a_worst10'][2]:+.3f}] "
                  f"SGM {g['a_SGM'][0]:+.4f}[{g['a_SGM'][1]:+.4f},{g['a_SGM'][2]:+.4f}] | 分档W " + ' '.join(f"{st}:{g[f'a_W|{st}']:+.3f}" for st in ('ample', 'marginal', 'short'))
                  + ' 10%: ' + ' '.join(f"{st}:{g[f'a_worst10|{st}']:+.3f}" for st in ('ample', 'marginal', 'short')))
    print('--- 时间曲线（接入 ΔW）')
    for t in ('ptr', 'greedy'):
        for m in M[1:]:
            c = n['time'][f'{s}|{t}|{m}']
            print(f"{t:7s} {m:12s} " + ' '.join(f"{k}s:{c[k]['aW']:+.4f}" for k in ('0.1', '0.3', '1', '3', '10')))
    print('--- 取舍（SCA）')
    for t in T:
        print(t, ' '.join(f"δ{d}: dW {n['tradeoff'][f'{s}|{t}|{d}']['a_W']:+.4f} d10 {n['tradeoff'][f'{s}|{t}|{d}']['a_worst10']:+.3f} dSGM {n['tradeoff'][f'{s}|{t}|{d}']['a_SGM']:+.4f}" for d in ('0', '0.005', '0.01', '0.02')))
print('=== 面板')
for k, v in n['panel'].items():
    print(k, {kk: round(vv, 4) for kk, vv in v.items()})
print('=== 大波束（不够档每景最大 6 束；SCA 接入前后）')
for k, v in n['bigbeam'].items():
    print(k, {tag: {kk: round(vv, 3) for kk, vv in x.items()} for tag, x in v.items()})
