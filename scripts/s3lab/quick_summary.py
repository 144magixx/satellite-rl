"""快速汇总：各档 × 方法的接入后 ΔW、SGM、最差 10%、功率、耗时（场景等权；两段式先平均 3 个求解器种子）。"""
import json
import sys

import pandas as pd

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding='utf-8')]
d = pd.DataFrame(rows)
d = d[d.method != 'bounds']
d['dW'] = d.a_W - d.W0
d['dSGM'] = d.a_SGM - d.F0
d['d10'] = d.a_worst10 - d.worst10_0
d['nW'] = d.n_W - d.W0
g = d.groupby(['set', 'tier', 'plan', 'idx', 'method']).agg(dW=('dW', 'mean'), dSGM=('dSGM', 'mean'), d10=('d10', 'mean'), nW=('nW', 'mean'),
                                                          P=('a_power_w', 'mean'), t=('t_native', 'mean'), W0=('W0', 'first'),
                                                          fail=('status', lambda s: (s == 'fail').mean()), tau=('tau', 'mean')).reset_index()
g = g.groupby(['set', 'tier', 'idx', 'method']).mean(numeric_only=True).reset_index()     # 随机/块指针头 3 套求平均
pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)
out = g.groupby(['set', 'tier', 'method'])[['W0', 'dW', 'nW', 'dSGM', 'd10', 'P', 't', 'tau', 'fail']].mean()
print(out.round(4))
