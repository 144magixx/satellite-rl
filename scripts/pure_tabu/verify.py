"""Independently check saved beam arrays, scalar metrics, and final constraints."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from satellite_rl.power_refinement.physics import _sgm


def verify(root):
    errors={};failed=[];count=0
    inputs={};input_scenes=0;missing_inputs=[];input_over_cap={}
    for path in root.glob('*/results.csv'):
        dataset=path.parent.name;source=root.parent/'data'/dataset
        if source.exists():
            manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
            over=0
            for record in manifest['rows']:
                case=json.loads((source/record['case_path']).read_text(encoding='utf-8'))
                layout=json.loads((source/record['layout_path']).read_text(encoding='utf-8'))
                ids=case['aircraft_ids'];fixed=case['fixed_request_bps']
                assert len(ids)==len(set(ids))
                assert all(r==fixed for r in case['request_bps_int64'])
                assert sum(case['request_bps_int64'])==case['total_request_bps_int64']
                owners=layout['owner_aircraft_to_beam_id'];beams=layout['beam_ids']
                assert len(owners)==len(ids) and {a for a,b in owners}==set(ids)
                counts=np.array([sum(b==beam for a,b in owners) for beam in beams])
                np.testing.assert_array_equal(counts,layout['aircraft_count_by_beam'])
                np.testing.assert_array_equal(counts*fixed,layout['request_bps_by_beam_int64'])
                assert counts.sum()==len(ids) and (counts*fixed).sum()==case['total_request_bps_int64']
                inputs[(dataset,record['case_id'])]=layout;input_scenes+=1
                over+=int((counts>int(1125000000//fixed)).any())
            input_over_cap[dataset]=over
        else:missing_inputs.append(dataset)
        for row in csv.DictReader(path.open(encoding='utf-8')):
            with np.load(root/row['beams']) as saved:
                demand=saved['demand_bps'];aircraft=saved['aircraft_count'];count+=1
                if dataset not in missing_inputs:
                    layout=inputs[(dataset,row['scene_id'])]
                    np.testing.assert_array_equal(saved['beam_ids'],layout['beam_ids'])
                    np.testing.assert_array_equal(demand,layout['request_bps_by_beam_int64'])
                    np.testing.assert_array_equal(aircraft,layout['aircraft_count_by_beam'])
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
                 input_scenes_checked=input_scenes,input_missing_sets=missing_inputs,
                 input_over_cap_scenes=input_over_cap,
                 service_definition='beam capacity clipped by beam demand; replicated beam SI by aircraft count for worst10')
    (root/'verification.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=count,failures=len(failed),max_abs_error=max(errors.values()))))
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();result=verify(Path(a.root))
    if result['failures']:raise ValueError('saved metrics or allocation constraints failed verification')
