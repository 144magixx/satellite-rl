"""Summarize saved paired results; bootstrap dates, then scenes within dates."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(root):
    rows=[]
    for path in root.glob('*/results.csv'):
        rows.extend(csv.DictReader(path.open(encoding='utf-8')))
    groups={}
    for row in rows:groups.setdefault((row['set'],row['method'],row['seed']),[]).append(row)
    grouped=[]
    fields=[k for k in rows[0] if k.startswith(('before_','after_')) or k.endswith('_seconds')]
    for (dataset,method,seed),values in sorted(groups.items()):
        item=dict(set=dataset,method=method,seed=int(seed),n=len(values),
                  constraints_failed=sum(v['constraints_ok']!='True' for v in values))
        for field in fields:
            a=np.array([float(v[field]) for v in values])
            item[field]=float(a.mean())
            if field.endswith('_seconds'):
                item[field+'_median']=float(np.median(a));item[field+'_p95']=float(np.percentile(a,95))
        item['probes_mean']=float(np.mean([int(v['probes']) for v in values]))
        grouped.append(item)
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(grouped[0]));writer.writeheader();writer.writerows(grouped)
    comparisons=[('PTS32','GTS32'),('PTS96','GTS96'),('PTS320','GTS320'),
                 ('PTS32','G1'),('PTS96','G1'),('PTS320','G1'),('Random0','G1'),
                 ('PTS32','Random0'),('PTS96','Random0'),('PTS320','Random0'),
                 ('PTS320','PTS32'),('GTS320','GTS32')]
    rng=np.random.default_rng(20261021)
    paired=[]
    for dataset in sorted({r['set'] for r in rows}):
        for lhs,rhs in comparisons:
            left={r['scene_id']:r for r in groups[(dataset,lhs,'51001')]}
            right={r['scene_id']:r for r in groups[(dataset,rhs,'51001')]}
            ids=sorted(left.keys()&right.keys());dates=sorted({left[i]['date'] for i in ids})
            date_indices=[np.array([k for k,i in enumerate(ids) if left[i]['date']==date]) for date in dates]
            indices=[]
            for _ in range(2500):
                sampled=rng.integers(len(dates),size=len(dates))
                indices.append(np.concatenate([rng.choice(date_indices[d],size=len(date_indices[d])) for d in sampled]))
            item=dict(set=dataset,lhs=lhs,rhs=rhs,n=len(ids),bootstrap='date then scene; 2500 draws')
            for field in ('before_W','after_W','before_SGM','after_SGM','before_aircraft_worst10',
                          'after_aircraft_worst10','after_total_power_w','total_seconds'):
                delta=np.array([float(left[i][field])-float(right[i][field]) for i in ids])
                estimates=np.array([delta[idx].mean() for idx in indices])
                item[field]=dict(mean=float(delta.mean()),ci95=np.percentile(estimates,[2.5,97.5]).tolist(),
                                 better=int((delta>1e-9).sum()),same=int((abs(delta)<=1e-9).sum()),
                                 worse=int((delta < -1e-9).sum()))
            paired.append(item)
    payload=dict(rows=len(rows),groups=grouped,paired=paired,
                 constraints_failed=sum(r['constraints_ok']!='True' for r in rows),
                 units='rates are fractions; seconds; watts; equal weight per scene; seeds kept separate')
    (root/'analysis.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();result=summarize(Path(a.root))
    print(json.dumps(dict(rows=result['rows'],constraints_failed=result['constraints_failed'])))
