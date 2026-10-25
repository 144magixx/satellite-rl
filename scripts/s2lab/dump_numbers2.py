"""把第二轮 numbers.json 打成便于写报告的文本。"""
import json
import sys

n = json.load(open('D:/paper/outputs/experiments/s2lab_round2_20261025/report/numbers.json', encoding='utf-8'))
sections = sys.argv[1:] or ['confirm', 'paired', 'strata', 'size', 'diag', 'round1', 'checkpoint']
if 'confirm' in sections:
    print('== confirm')
    for m, c in sorted(n['confirm'].items(), key=lambda kv: -kv[1]['SGM']):
        print(f"{m}: SGM {c['SGM']:.3f}±{c['SGM_std']:.3f} {[round(x, 3) for x in c['seed_SGM']]} W {c['W']:.3f} 10% {c['worst10']:.3f} "
              f"P {c['power_w']:.0f} slots {c['slots']:.0f} ov {c['copol_overlap']:.0f} il {c['interference_loss']:.3f} IN {c['mean_in_db']:.1f} "
              f"adj {c['adj_ratio']:.2f} uns {c['unserved']:.1f} n{c['seeds']}")
if 'paired' in sections:
    print('== paired')
    for k, v in n['paired'].items():
        s, w = v['SGM'], v['W']
        print(f"{k}: SGM {s['diff']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}] win {s['win']:.2f} | W {w['diff']:+.3f} [{w['lo']:+.3f},{w['hi']:+.3f}] | "
              f"10% {v['worst10']['diff']:+.3f} | " + ' '.join(f"{x}:{v['by_stratum'][x]['diff']:+.3f}" for x in v['by_stratum']))
if 'strata' in sections:
    print('== strata')
    for m, st in n['strata'].items():
        print(m, ' | '.join(f"{s}: {st[s]['SGM']:.3f}/{st[s]['W']:.3f}/{st[s]['power_w']:.0f}W" for s in st))
if 'size' in sections:
    print('== size')
    for m, c in n['size'].items():
        print(m, c['params'], f"SGM {c['SGM']:.3f}±{c['SGM_std']:.3f} W {c['W']:.3f}±{c['W_std']:.3f}", c['members'])
if 'diag' in sections:
    print('== diag')
    for m, d in n['diag'].items():
        for mode, v in d.items():
            print(m, mode, f"SGM {v['summary']['SGM']:.3f} vs {v['base']['SGM']:.3f} ({v['paired']['SGM']['diff']:+.3f} [{v['paired']['SGM']['lo']:+.3f},{v['paired']['SGM']['hi']:+.3f}]) "
                           f"W {v['summary']['W']:.3f} vs {v['base']['W']:.3f} ({v['paired']['W']['diff']:+.3f}) slots {v['summary']['slots']:.0f} vs {v['base']['slots']:.0f} "
                           f"il {v['summary']['interference_loss']:.3f} vs {v['base']['interference_loss']:.3f}")
if 'round1' in sections:
    print('== round1')
    for m, v in n['round1'].items():
        print(m, f"R1 SGM {v['round1']['SGM']:.3f} W {v['round1']['W']:.3f} -> R2 SGM {v['round2']['SGM']:.3f} W {v['round2']['W']:.3f} "
                 f"(dSGM {v['paired']['SGM']['diff']:+.3f} [{v['paired']['SGM']['lo']:+.3f},{v['paired']['SGM']['hi']:+.3f}])")
if 'checkpoint' in sections:
    print('== checkpoint (best_sgm vs best)')
    for m, v in n['checkpoint'].items():
        print(m, f"SGM {v['best_sgm']['SGM']:.3f} vs {v['best']['SGM']:.3f} | W {v['best_sgm']['W']:.3f} vs {v['best']['W']:.3f}")
