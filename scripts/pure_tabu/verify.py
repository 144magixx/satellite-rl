"""Independently check saved beam arrays, scalar metrics, and final constraints."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from satellite_rl.power_refinement.physics import _sgm


def verify(root):
    errors={};failed=[];count=0
    for path in root.glob('*/results.csv'):
        for row in csv.DictReader(path.open(encoding='utf-8')):
            with np.load(root/row['beams']) as saved:
                demand=saved['demand_bps'];aircraft=saved['aircraft_count'];count+=1
                for prefix in ('before','after'):
                    c=saved[f'capacity_{prefix}'];si=np.minimum(c/demand,1.)
                    service=np.repeat(si,aircraft);worst=max(1,int(np.ceil(.1*len(service))))
                    independently=dict(W=np.minimum(c,demand).sum()/demand.sum(),SI=si.mean(),
                        SGM=_sgm(c,demand)[0].mean(),aircraft_worst10=np.sort(service)[:worst].mean(),
                        total_power_w=saved[f'power_{prefix}'].sum(),total_slots=saved['counts'].sum())
                    for field,value in independently.items():
                        key=f'{prefix}_{field}';error=abs(value-float(row[key]))
                        errors[key]=max(errors.get(key,0.),float(error))
                        if error>1e-8:failed.append(dict(scene_id=row['scene_id'],method=row['method'],field=key,error=float(error)))
                    for field,actual in [('served',np.minimum(c,demand)),('ratio',c/demand),('beam_si',si),
                                         ('beam_sgm',_sgm(c,demand)[0])]:
                        np.testing.assert_allclose(saved[f'{field}_{prefix}'],actual,rtol=0,atol=1e-8)
                group,start,slots=saved['groups'],saved['starts'],saved['counts']
                legal=bool(np.all((group>=0)&(group<8)) and np.all((slots>=1)&(slots<=10))
                           and np.all(start>=0) and np.all(start+slots<=100))
                for g in range(8):
                    members=np.flatnonzero(group==g);ordered=members[np.argsort(start[members])]
                    legal=legal and bool(np.all(start[ordered][1:]>= (start[ordered]+slots[ordered])[:-1]))
                for prefix in ('before','after'):
                    power=saved[f'power_{prefix}']
                    legal=legal and bool(np.all(power>=-1e-7) and power.max()<=50+1e-7 and power.sum()<=3000+1e-7)
                if not legal or row['constraints_ok']!='True':
                    failed.append(dict(scene_id=row['scene_id'],method=row['method'],field='constraints'))
    payload=dict(rows=count,max_abs_error=errors,failures=failed,
                 service_definition='beam capacity clipped by beam demand; replicated beam SI by aircraft count for worst10')
    (root/'verification.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=count,failures=len(failed),max_abs_error=max(errors.values()))))
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();verify(Path(a.root))
