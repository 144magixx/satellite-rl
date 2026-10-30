"""开发集定参数：两段式按自身原生目标（欠供＋功率代价，越低越好）、目标容量迭代按自身原生误差选；同时列 W 供参考。"""
import json

import pandas as pd

rows = [json.loads(l) for l in open('D:/paper/outputs/experiments/s3lab_20261029/runs/calib/rows.jsonl', encoding='utf-8')]
d = pd.DataFrame(rows)
d['n_cost'] = 1 - d.n_W + .001 * d.n_power_w / 2000
pd.set_option('display.width', 200)
ar = d[d.method == 'aravanis'].groupby('tag')[['n_cost', 'n_W', 'n_SGM', 'a_W', 't_native']].mean().sort_values('n_cost')
print('两段式（原生代价越低越好）'); print(ar.round(5))
tc = d[d.method == 'target_cap'].copy()
print('目标容量迭代'); print(tc.groupby('tag')[['n_W', 'n_SGM', 'a_W', 'iters', 't_native']].mean().round(5))
