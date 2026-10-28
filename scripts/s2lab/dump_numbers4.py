"""把第四轮 numbers.json 打成写报告用的文本。"""
import json

import numpy as np

n = json.load(open('D:/paper/outputs/experiments/s2lab_round4_20261028/report/numbers.json', encoding='utf-8'))
G = n['groups']
print('== 旧确认集（3 种子均值）')
for k, c in sorted(G.items(), key=lambda kv: -kv[1]['SGM']):
    print(f"{k:12s} {c['label']:20s} SGM {c['SGM']:.3f}±{c['SGM_std']:.3f} {[round(x, 3) for x in c['seed_SGM']]} W {c['W']:.3f} 10% {c['worst10']:.3f} "
          f"P {c['power_w']:.0f} 槽 {c['slots']:.0f} 干扰 {c['interference_loss']:.3f} 相邻 {c['adj_ratio']:.2f} 截断 {c['truncated']:.1f} 时长 {np.mean(n['timing'][k]):.2f}h"
          + (f" 补种子 {c.get('extra_seeds')} {[round(x, 3) for x in c.get('extra_SGM', [])]} dev {[round(x, 3) for x in c.get('extra_dev_SGM', [])]}" if c.get('extra_seeds') else ''))
for k, c in n['rules'].items():
    print(f"{k:12s} SGM {c['SGM']:.3f} W {c['W']:.3f} 10% {c['worst10']:.3f} P {c['power_w']:.0f} 干扰 {c['interference_loss']:.3f} 相邻 {c['adj_ratio']:.2f}")
for key in ('paired_parent', 'paired_base', 'paired_greedy'):
    print(f'== {key}')
    for k, v in n[key].items():
        s = v['SGM']
        print(f"{k:12s} SGM {s['diff']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}] 胜 {s['win']:.2f} W {v['W']['diff']:+.3f} 10% {v['worst10']['diff']:+.3f} "
              f"种子 {[round(x, 3) for x in v.get('seed_diff', [])]} 开发集 {[round(x, 3) for x in v.get('dev_seed_diff', [])]} 分档 "
              + ' '.join(f"{x}:{v['by_stratum'][x]['diff']:+.3f}" for x in v['by_stratum']))
print('== 分档 SGM/W/最差10%/功率')
for k in ['r4-base', 'r4-ptr', 'r4-ptrA', 'r4-AB', 'r4-A', 'r4-Aqx', '容量感知贪心', '分散摆放规则']:
    if k in n['strata']:
        st = n['strata'][k]
        print(f'{k:10s}', ' | '.join(f"{s}: {st[s]['SGM']:.3f}/{st[s]['W']:.3f}/{st[s]['worst10']:.3f}/{st[s]['power_w']:.0f}W" for s in st))
print('== 复现（本轮探索组合 vs 第三轮）')
r = n['repro']
print(f"r3 {r['r3']['SGM']:.3f}/{r['r3']['W']:.3f}  r4 {r['r4']['SGM']:.3f}/{r['r4']['W']:.3f}  配对 SGM {r['diff']['SGM']['diff']:+.4f} [{r['diff']['SGM']['lo']:+.4f},{r['diff']['SGM']['hi']:+.4f}]")
print('== 收敛（开发集末段变化）')
for k, v in n['convergence'].items():
    print(k, [round(x, 3) for x in v])
print('== 行为克隆')
for k, v in n['bc'].items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print('== 新确认集（每个运行/规则）')
NC = n['newconf']
groups = {}
for k, v in NC.items():
    groups.setdefault(k if k.startswith('rule') else k.rsplit('-s', 1)[0], []).append((k, v))
for g, items in sorted(groups.items(), key=lambda kv: -np.mean([v['SGM'] for _, v in kv[1]])):
    main3 = [v for k, v in items if k.startswith('rule') or int(k.rsplit('-s', 1)[1]) <= 3]
    print(f"{g:16s} n{len(main3)} SGM {np.mean([v['SGM'] for v in main3]):.3f} {[round(v['SGM'], 3) for v in main3]} W {np.mean([v['W'] for v in main3]):.3f} "
          f"10% {np.mean([v['worst10'] for v in main3]):.3f} P {np.mean([v['power_w'] for v in main3]):.0f} 干扰 {np.mean([v['interference_loss'] for v in main3]):.3f} | 分档SGM "
          + ' '.join(f"{s}:{np.mean([v['strata'][s]['SGM'] for v in main3]):.3f}" for s in ('ample', 'marginal', 'short'))
          + (f" | 种子4-6 {[round(v['SGM'], 3) for k, v in items if not k.startswith('rule') and int(k.rsplit('-s', 1)[1]) > 3]}" if any(not k.startswith('rule') and int(k.rsplit('-s', 1)[1]) > 3 for k, _ in items) else ''))
