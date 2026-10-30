"""Serial paired evaluation, with independent initialization and stage timing."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import numpy as np
import scipy
import torch
from algorithm import initialize, tabu_search
from satellite_rl.s2fix.data import read_records, load_input
from satellite_rl.s2fix.evaluate import metrics, run_sca

SETS = ['conf_d432_n150','dev_d432_n150','dev_d216_n150','dev_d864_n150','dev_d432_n200']


def scalar_metrics(stage, allocation):
    return metrics(stage,*allocation)


def evaluate(record,index,method,seed,directory):
    stage = load_input(record)
    random = method.startswith('PTS') or method == 'Random0'
    budget = int(method[3:]) if method.startswith(('PTS','GTS')) else 0
    init_seed = seed+index*100000
    started = perf_counter()
    env, initial = initialize(stage,'random' if random else 'greedy',init_seed,budget)
    initial_seconds = perf_counter()-started
    started = perf_counter()
    allocation, trace = tabu_search(env,budget) if budget else (initial,[])
    search_seconds = perf_counter()-started
    started = perf_counter()
    before = scalar_metrics(stage,allocation)
    verify_seconds = perf_counter()-started
    started = perf_counter()
    optimized = run_sca(stage,*allocation)
    sca_seconds = perf_counter()-started
    after_allocation = (*allocation[:3],np.asarray(optimized['power_w']))
    started = perf_counter()
    after = scalar_metrics(stage,after_allocation)
    verify_seconds += perf_counter()-started
    groups,starts,counts,power = allocation
    power_after = after_allocation[3]
    constraints = bool(env._legal(starts,counts,groups) and np.all((groups>=0)&(groups<8))
        and power.max()<=50+1e-7 and power.sum()<=3000+1e-7
        and power_after.max()<=50+1e-7 and power_after.sum()<=3000+1e-7)
    basename = f'{index:04d}_{method}_s{seed}'
    demand = np.asarray(stage.layout['request_bps_by_beam_int64'])
    arrays = dict(beam_ids=np.asarray(stage.layout['beam_ids']),groups=groups,starts=starts,
        counts=counts,power_before=power,power_after=power_after,demand_bps=demand,
        aircraft_count=np.asarray(stage.layout['aircraft_count_by_beam']),
        initial_groups=initial[0],initial_starts=initial[1],initial_counts=initial[2],initial_power=initial[3])
    for prefix,value in [('before',before),('after',after)]:
        arrays.update({f'capacity_{prefix}':value['capacity_bps'],
            f'served_{prefix}':np.minimum(value['capacity_bps'],demand),
            f'ratio_{prefix}':value['capacity_bps']/demand,
            f'beam_si_{prefix}':value['beam_si'],f'beam_sgm_{prefix}':value['beam_sgm']})
    np.savez_compressed(directory/'beams'/f'{basename}.npz',**arrays)
    with gzip.open(directory/'trace'/f'{basename}.jsonl.gz','wt',encoding='utf-8') as stream:
        for step in trace: stream.write(json.dumps(step)+'\n')
    row = dict(set=record['set_name'],scene_id=record['case_id'],date=stage.case['UTC'][:10],
        method=method,seed=seed,init_seed=init_seed,N=len(groups),probes=env.probe_count,
        decisions=env.decision_count,initial_W=env.base_eval.W,initial_SGM=env.base_eval.sgm,
        initial_seconds=initial_seconds,search_seconds=search_seconds,verify_seconds=verify_seconds,
        sca_seconds=sca_seconds,sca_solver_seconds=optimized['sca_seconds'],
        total_seconds=initial_seconds+search_seconds+verify_seconds+sca_seconds,
        constraints_ok=constraints,beams=f"{record['set_name']}/beams/{basename}.npz",
        trace=f"{record['set_name']}/trace/{basename}.jsonl.gz")
    for prefix,value in [('before',before),('after',after)]:
        row.update({f'{prefix}_{k}':v for k,v in value.items() if np.isscalar(v)})
    return row


def summary(output):
    rows = []
    for path in output.glob('*/results.csv'):
        rows.extend(csv.DictReader(path.open(encoding='utf-8')))
    groups = {}
    for row in rows: groups.setdefault((row['set'],row['method'],row['seed']),[]).append(row)
    stats = []
    fields = [k for k in rows[0] if k.startswith(('before_','after_')) or k.endswith('_seconds')]
    for (dataset,method,seed),values in groups.items():
        item = dict(set=dataset,method=method,seed=int(seed),n=len(values),
                    constraints_failed=sum(v['constraints_ok']!='True' for v in values))
        for field in fields:
            a=np.array([float(v[field]) for v in values])
            item[field]=dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)))
        stats.append(item)
    (output/'summary.json').write_text(json.dumps(dict(groups=stats,rows=len(rows)),indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--sets',nargs='+',default=SETS)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--sensitivity',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    manifest=dict(args=vars(args),hostname=platform.node(),python=sys.executable,
        numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,threads=1,
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        source_commit=(Path(__file__).parents[2]/'COMMIT').read_text().strip() if (Path(__file__).parents[2]/'COMMIT').exists() else None,
        process_id=os.getpid(),method_order='cyclic rotation by scene index',timing='CPU one thread; all arms serial; load and output IO excluded')
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    started=perf_counter(); failures=[]; completed=0
    for dataset in args.sets:
        directory=output/dataset
        (directory/'beams').mkdir(parents=True,exist_ok=True); (directory/'trace').mkdir(exist_ok=True)
        records=read_records(args.data_dir,dataset)[:args.limit]
        jobs=[('G1',51001),('GTS32',51001),('GTS96',51001),('GTS320',51001),
              ('Random0',51001),('PTS32',51001),('PTS96',51001),('PTS320',51001)]
        if args.sensitivity and dataset=='dev_d432_n150':
            jobs += [(method,seed) for seed in (51002,51003) for method in ('Random0','PTS32','PTS96','PTS320')]
        with (directory/'results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=None
            for index,record in enumerate(records):
                rotation=index%len(jobs)
                for method,seed in jobs[rotation:]+jobs[:rotation]:
                    try:
                        if args.check: check(record,index,seed)
                        row=evaluate(record,index,method,seed,directory)
                        if writer is None:
                            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);stream.flush();completed+=1
                        print(json.dumps(dict(completed=completed,set=dataset,index=index,method=method,seed=seed,
                            W=row['after_W'],seconds=row['total_seconds'],elapsed=perf_counter()-started)),flush=True)
                    except Exception:
                        failure=dict(set=dataset,index=index,method=method,seed=seed,error=traceback.format_exc())
                        failures.append(failure); print(json.dumps(failure),flush=True)
                    (output/'status.json').write_text(json.dumps(dict(completed=completed,failures=len(failures),
                        set=dataset,index=index,method=method,elapsed=perf_counter()-started)),encoding='utf-8')
    (output/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    summary(output)
    (output/'completed.json').write_text(json.dumps(dict(completed=completed,failures=len(failures),seconds=perf_counter()-started)),encoding='utf-8')


def check(record,index,seed):
    """Real-data equivalence with original search and no G1 call from pure path."""
    from unittest.mock import patch
    from satellite_rl.s2fix.search_env import run_search
    stage=load_input(record)
    with patch('algorithm.greedy_allocation',side_effect=RuntimeError('pure constructor called G1')):
        env,allocation=initialize(stage,'random',seed+index*100000,32)
        assert env._legal(allocation[1],allocation[2],allocation[0])
        result,_=tabu_search(env,32)
        assert env._legal(result[1],result[2],result[0])
        assert env.best_eval.W>=env.base_eval.W
    stage=load_input(record)
    expected,info=run_search(stage,'S1',max_probes=32,return_info=True)
    stage=load_input(record)
    env,_=initialize(stage,'greedy',seed,32)
    actual,_=tabu_search(env,32)
    for a,b in zip(actual,expected): np.testing.assert_array_equal(a,b)
    assert env.probe_count==info['probes']


if __name__=='__main__':main()
