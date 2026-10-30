"""Generate FLOW-AO layouts for the exact W26/W30 aircraft records."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from satellite_rl.coverage.s1fix.data import Frame
from satellite_rl.coverage.s1fix.geom import rays
from satellite_rl.coverage.s1fix.skeleton import initial_cover, solve
from satellite_rl.power30.simple_data import authority_for
from satellite_rl.shared_interface.content import read_wire, wire_bytes
from satellite_rl.shared_interface.geometry import directions_and_distance, cone_omega
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
from satellite_rl.shared_interface.stages import make_layout, UniformStage2Input

SETS = ('train_n100', 'dev_n100', 'dev_n60', 'dev_n150', 'dev_n200', 'conf_n100')


def generate(job):
    index, dataset, source, target, destination = job
    started = time.perf_counter()
    check = dict(dataset=dataset, index=index, case_id=source['case_id'], requested_budget=target,
                 n=0, N0=0, used_budget=0, N=0, unique_ownership=False,
                 integer_demand_conserved=False, widths_valid=False, coverage_valid=False,
                 nonempty_N_valid=False, UniformStage2Input_ok=False, widened_beams=0,
                 max_widening_deg=0., error='')
    stat = None
    record = None
    try:
        raw = Path(source['case_path']).read_bytes()
        case = read_wire(raw)
        n = len(case['aircraft_ids'])
        check['n'] = n
        frame = Frame(case['UTC'][:10], 0., index, np.asarray(case['aircraft_ids']),
                      case['lat_lon_deg'], np.zeros(n), np.zeros(n))
        N0 = len(initial_cover(rays(frame.positions)))
        check['N0'] = N0
        if N0 > 200:
            raise ValueError('initial cover requires more than 200 beams')
        budget = max(target, N0)
        check['used_budget'] = budget
        result = solve(frame, N_budget=budget, spacing=True, effort=1)
        if not result.feasible or result.layout is None:
            raise ValueError('frozen FLOW-AO returned infeasible')
        output = result.layout
        owner, centers = output.owner, output.centers_latlon
        widths = output.widths.copy()
        N = len(widths)
        check['N'] = N
        aircraft_rays, _ = directions_and_distance(case['lat_lon_deg'])
        center_rays, _ = directions_and_distance(centers)
        distances = np.rad2deg(2*np.arcsin(np.minimum(
            np.linalg.norm(aircraft_rays - center_rays[owner], axis=1)/2, 1)))
        needed = np.zeros(N)
        np.maximum.at(needed, owner, 2*distances)
        widen = needed > widths
        widths[widen] = needed[widen] + 1e-6
        check['widened_beams'] = int(widen.sum())
        check['max_widening_deg'] = float((widths-output.widths).max())
        ids = [f'flow:{int(bid):04d}' for bid in output.beam_ids]
        anchors = []
        for j in range(N):
            members = np.flatnonzero(owner == j)
            anchors.append(case['aircraft_ids'][min(members,
                key=lambda i: (distances[i], case['aircraft_ids'][i]))])
        layout = make_layout(case, beam_ids=ids, source_slot_ids=ids,
            anchor_aircraft_ids=anchors, centres_lat_lon_deg=centers, full_width_deg=widths,
            owner_aircraft_to_beam_id=[[aid, ids[j]] for aid, j in zip(case['aircraft_ids'], owner)],
            Ncap=min(200,n), geometry_Omega_cap_sr=cone_omega(np.ones(min(200,n))),
            geometry_Omega_sr=cone_omega(widths),
            geometry_audit=dict(method='FLOW-AO', spacing=True),
            source_and_cost=dict(method='FLOW-AO', source_commit='19f6c45', effort=1,
                requested_budget=target, used_budget=budget, N0=N0, seconds=result.t_total,
                widened_beams=check['widened_beams'], max_widening_deg=check['max_widening_deg']))
        counts = np.bincount(owner, minlength=N)
        check['unique_ownership'] = bool(len(owner)==n and len(layout['owner_aircraft_to_beam_id'])==n)
        check['integer_demand_conserved'] = bool(int(layout['request_bps_by_beam_int64'].sum())==n*21600000
            and np.array_equal(layout['request_bps_by_beam_int64'], counts*21600000))
        check['widths_valid'] = bool(((widths>=.3)&(widths<=1.)).all())
        check['coverage_valid'] = bool((distances<=widths[owner]/2).all())
        check['nonempty_N_valid'] = bool(N<=200 and counts.all())
        UniformStage2Input(case, layout, authority_for(case, layout), PHYSICS)
        check['UniformStage2Input_ok'] = True
        if not all(check[k] for k in ('unique_ownership','integer_demand_conserved','widths_valid',
                                     'coverage_valid','nonempty_N_valid')):
            raise ValueError('explicit data checks failed')
        directory = Path(destination) / dataset / 'objects'
        directory.mkdir(parents=True, exist_ok=True)
        case_path, layout_path = directory/f'{index:04d}_case.wire', directory/f'{index:04d}_layout.wire'
        case_path.write_bytes(raw)
        layout_path.write_bytes(wire_bytes(layout))
        record = dict(case_id=case['case_id'], role=case['actual_role'],
                      case_path=str(case_path), layout_path=str(layout_path))
        stat = dict(dataset=dataset,index=index,case_id=case['case_id'],n=n,N=N,
                    peak_aircraft=int(counts.max()),mean_width_deg=float(widths.mean()),
                    max_width_deg=float(widths.max()),N0=N0,used_budget=budget)
    except Exception as exc:
        check['error'] = f'{type(exc).__name__}: {exc}'
    check['seconds'] = time.perf_counter()-started
    return check, stat, record


def write_csv(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    if args.merge:
        outputs=[]
        for shard in range(args.shards):
            outputs.extend(json.loads((root/f'part_{shard}.json').read_text(encoding='utf-8')))
        outputs.sort(key=lambda out:(SETS.index(out[0]['dataset']),out[0]['index']))
        write_csv(root/'data_checks.csv',[out[0] for out in outputs])
        write_csv(root/'layout_stats.csv',[out[1] for out in outputs if out[1] is not None])
        for dataset in SETS:
            records=[out[2] for out in outputs if out[0]['dataset']==dataset and out[2] is not None]
            directory=root/dataset;directory.mkdir(exist_ok=True)
            filename='train_runtime.json' if dataset=='train_n100' else 'development_runtime.json'
            (directory/filename).write_text(json.dumps(dict(rows=records)),encoding='utf-8')
            print(dataset,'passed',len(records),'failed',sum(out[0]['dataset']==dataset and out[2] is None for out in outputs),flush=True)
        shutil.copyfile(root/'dev_n100/development_runtime.json',root/'train_n100/development_runtime.json')
        return
    inputs=json.loads(Path(args.inputs).read_text(encoding='utf-8'))
    jobs=[]
    for dataset in SETS:
        role='train' if dataset.startswith('train') else 'test' if dataset.startswith('conf') else 'development'
        target=int(dataset.rsplit('n',1)[1])
        for index,source in enumerate(inputs[role]):
            jobs.append((index,dataset,source,target,str(root)))
    jobs=jobs[args.shard::args.shards]
    outputs=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for output in pool.map(generate,jobs,chunksize=1):
            outputs.append(output)
            check=output[0]
            print(check['dataset'],check['index'],check['N'],check['error'] or 'OK',flush=True)
    (root/f'part_{args.shard}.json').write_text(json.dumps(outputs),encoding='utf-8')


if __name__=='__main__':
    main()
