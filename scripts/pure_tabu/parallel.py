"""Four pinned scene workers; all arms of a scene remain serial on one core."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter,sleep
import traceback
import numpy as np
import scipy
import torch
from evaluate import SETS,evaluate
from satellite_rl.s2fix.data import read_records

CORES=[106,110,114,118]


def jobs(dataset):
    result=[('G1',51001),('GTS32',51001),('GTS96',51001),('GTS320',51001),
            ('Random0',51001),('PTS32',51001),('PTS96',51001),('PTS320',51001)]
    if dataset=='dev_d432_n150':
        result += [(method,seed) for seed in (51002,51003) for method in ('Random0','PTS32','PTS96','PTS320')]
    return result


def write_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def worker(args):
    index=args.worker;core=CORES[index]
    os.sched_setaffinity(0,{core});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(args.output_dir);directory=output/'shards'/str(index);directory.mkdir(parents=True,exist_ok=True)
    count=0;failures=[];started=perf_counter()
    with (directory/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=None
        for dataset in SETS:
            destination=output/dataset
            (destination/'beams').mkdir(parents=True,exist_ok=True);(destination/'trace').mkdir(exist_ok=True)
            records=read_records(args.data_dir,dataset)
            for scene,record in enumerate(records):
                if scene%4!=index:continue
                methods=jobs(dataset);rotation=scene%len(methods)
                for method,seed in methods[rotation:]+methods[:rotation]:
                    try:
                        row=evaluate(record,scene,method,seed,destination)
                        row.update(worker=index,core=core,scene_index=scene)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);stream.flush();count+=1
                        print(json.dumps(dict(completed=count,set=dataset,index=scene,method=method,seed=seed,
                            W=row['after_W'],seconds=row['total_seconds'],elapsed=perf_counter()-started)),flush=True)
                    except Exception:
                        failure=dict(worker=index,set=dataset,index=scene,method=method,seed=seed,error=traceback.format_exc())
                        failures.append(failure);print(json.dumps(failure),flush=True)
                    write_json(directory/'status.json',dict(completed=count,failures=len(failures),
                        worker=index,core=core,set=dataset,index=scene,method=method,elapsed=perf_counter()-started))
            write_json(directory/f'done_{dataset}.json',dict(worker=index,count=count,failures=len(failures)))
    write_json(directory/'failures.json',failures)
    write_json(directory/'completed.json',dict(completed=count,failures=len(failures),seconds=perf_counter()-started))


def merge(output,dataset):
    rows=[]
    for index in range(4):
        with (output/'shards'/str(index)/'results.csv').open(encoding='utf-8') as stream:
            rows += [row for row in csv.DictReader(stream) if row['set']==dataset]
    rows.sort(key=lambda row:(int(row['scene_index']),row['method'],int(row['seed'])))
    with (output/dataset/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return len(rows)


def main(args):
    os.sched_setaffinity(0,{120})
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    commit=(Path(__file__).parents[2]/'COMMIT').read_text().strip()
    manifest=dict(algorithm_commit='3aa19288adcaa8437f83c6e8beb7db67fb2e67b5',execution_commit=commit,
        hostname=platform.node(),python=sys.executable,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
        threads=1,workers=4,cores=CORES,mapping='scene_index modulo 4; each scene all arms serial on one pinned physical core',
        method_order='cyclic rotation by original scene index',data_dir=args.data_dir,sets=SETS,
        timing='uniform four-worker concurrency; load/output IO excluded; original single-worker runs are serial_diagnostic')
    write_json(output/'run_manifest.json',manifest)
    processes=[];streams=[];started=perf_counter()
    for index in range(4):
        directory=output/'shards'/str(index);directory.mkdir(parents=True,exist_ok=True)
        stream=(directory/'run.log').open('w');streams.append(stream)
        processes.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
            '--data-dir',args.data_dir,'--output-dir',args.output_dir,'--worker',str(index)],
            stdout=stream,stderr=subprocess.STDOUT))
    merged={}
    while True:
        states=[]
        for index in range(4):
            path=output/'shards'/str(index)/'status.json'
            if path.exists():
                try:states.append(json.loads(path.read_text()))
                except json.JSONDecodeError:pass
        for dataset in SETS:
            if dataset not in merged and all((output/'shards'/str(i)/f'done_{dataset}.json').exists() for i in range(4)):
                merged[dataset]=merge(output,dataset);print('MERGED',dataset,merged[dataset],flush=True)
        status=dict(completed=sum(s['completed'] for s in states),failures=sum(s['failures'] for s in states),
            elapsed=perf_counter()-started,workers=states,merged=merged)
        write_json(output/'status.json',status)
        if all(p.poll() is not None for p in processes):break
        sleep(5)
    failures=[]
    for index,p in enumerate(processes):
        path=output/'shards'/str(index)/'failures.json'
        if path.exists():failures.extend(json.loads(path.read_text()))
        if p.returncode:failures.append(dict(worker=index,returncode=p.returncode,reason='worker exited with error; see run.log'))
    write_json(output/'failures.json',failures)
    write_json(output/'completed.json',dict(completed=status['completed'],failures=len(failures),
        seconds=perf_counter()-started,merged=merged))
    for stream in streams:stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--worker',type=int);args=p.parse_args()
    if args.worker is None:main(args)
    else:worker(args)
