"""把第三轮 numbers.json 打成写报告用的文本。"""
import json
import sys

n = json.load(open('D:/paper/outputs/experiments/s2lab_round3_20261027/report/numbers.json', encoding='utf-8'))
sec = sys.argv[1:] or ['groups', 'parent', 'r0', 'greedy', 'strata', 'conv']
G = n['groups']
if 'groups' in sec:
    print('== 各组（确认集，3 种子均值）')
    for k, c in sorted(G.items(), key=lambda kv: -kv[1]['SGM']):
        print(f"{k:22s} {c['label']:18s} n{len(c['seeds'])} SGM {c['SGM']:.3f}±{c['SGM_std']:.3f} {[round(x, 3) for x in c['seed_SGM']]} W {c['W']:.3f} "
              f"10% {c['worst10']:.3f} P {c['power_w']:.0f} 槽 {c['slots']:.0f} 干扰 {c['interference_loss']:.3f} 相邻 {c['adj_ratio']:.2f} 截断 {c['truncated']:.1f}")
    for k, c in n['rules'].items():
        print(f"{k:22s} SGM {c['SGM']:.3f} W {c['W']:.3f} 10% {c['worst10']:.3f} P {c['power_w']:.0f} 干扰 {c['interference_loss']:.3f} 相邻 {c['adj_ratio']:.2f}")
for key, title in (('paired_parent', 'parent'), ('paired_r0', 'r0'), ('paired_greedy', 'greedy')):
    if title in sec:
        print(f'== 配对差 {key}')
        for k, v in n[key].items():
            s = v['SGM']
            print(f"{k:22s} SGM {s['diff']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}] 胜 {s['win']:.2f} W {v['W']['diff']:+.3f} 10% {v['worst10']['diff']:+.3f} "
                  f"种子 {[round(x, 3) for x in v.get('seed_diff', [])]} 分档 " + ' '.join(f"{x}:{v['by_stratum'][x]['diff']:+.3f}" for x in v['by_stratum']))
if 'strata' in sec:
    print('== 分档 SGM/W/功率')
    for k in ['r3-r0', 'r3-f-all', 'r3-f-pneed', 'r3-replay', 'r3-condtop8', 'r3-p3', 'r3-xcombo', '容量感知贪心', '分散摆放规则']:
        if k in n['strata']:
            st = n['strata'][k]
            print(k, ' | '.join(f"{s}: {st[s]['SGM']:.3f}/{st[s]['W']:.3f}/{st[s]['worst10']:.3f}/{st[s]['power_w']:.0f}W" for s in st))
if 'conv' in sec:
    print('== 收敛（开发集末段变化，各种子）')
    for k, v in n['convergence'].items():
        print(k, [round(x, 3) for x in v])
if 'ckpt' in sec:
    for k, v in n['checkpoint'].items():
        print(k, f"best_sgm {v['best_sgm']:.3f} best {v['best']:.3f}")
