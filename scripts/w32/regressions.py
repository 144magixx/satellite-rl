"""The two explicitly requested continuous-center interface regressions."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from satellite_rl.power30.agent import Power30Agent
from satellite_rl.power30.lean_runner import evaluate
from satellite_rl.power30.simple_data import load_input


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--old-data',required=True)
    parser.add_argument('--old-run',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(Path(args.inputs).read_text(encoding='utf-8'))
    passed=0;total=sum(len(rows) for rows in inputs.values())
    with (root/'compatibility.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=['case_id','role','layout_sha_unchanged','valid','error']);w.writeheader()
        for role,rows in inputs.items():
            for row in rows:
                error=''
                try:
                    value=load_input(row)
                    same=value.layout['content_sha256']==row['layout_content_SHA']
                    if not same:raise ValueError('original layout SHA differs')
                    passed+=1
                except Exception as exc:error=str(exc);same=False
                w.writerow(dict(case_id=row['case_id'],role=role,layout_sha_unchanged=same,valid=not error,error=error))
    print(json.dumps(dict(regression='old_data_compatibility',passed=passed,total=total)),flush=True)
    if passed!=total:raise RuntimeError('old object regression failed')
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    checkpoint=torch.load(Path(args.old_run)/'policy_150000.pt',map_location='cpu',weights_only=False)
    config=checkpoint['config'];config['training']['device']='cuda'
    agent=Power30Agent(config)
    for name in ('actor','critic1','critic2'):getattr(agent,name).load_state_dict(checkpoint[name])
    dev=json.loads((Path(args.old_data)/'development_runtime.json').read_text(encoding='utf-8'))['rows']
    evaluate(agent,dev,'P00','M0',root,150000,repeats=3)
    shutil.copyfile(Path(args.old_run)/'eval_150000.csv',root/'original_eval_150000.csv')
    old=list(csv.DictReader((root/'original_eval_150000.csv').open()))
    new=list(csv.DictReader((root/'eval_150000.csv').open()))
    if len(old)!=60 or len(new)!=60:raise RuntimeError('expected 60 evaluation rows')
    diffs=[]
    for a,b in zip(old,new):
        if any(a[k]!=b[k] for k in ('mother','case_id','repeat','action_seed','env_step')):
            raise RuntimeError('evaluation row identity differs')
        diffs.append(dict(case_id=a['case_id'],repeat=a['repeat'],**{
            k+'_abs_diff':abs(float(a[k])-float(b[k])) for k in ('SI','SGM','power_w')}))
    with (root/'differences.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(diffs[0]));w.writeheader();w.writerows(diffs)
    maximum={k:max(row[k+'_abs_diff'] for row in diffs) for k in ('SI','SGM','power_w')}
    print(json.dumps(dict(regression='old_results',rows=60,max_abs_diff=maximum,passed=max(maximum.values())<=1e-6)),flush=True)
    if max(maximum.values())>1e-6:raise RuntimeError('old result reproducibility regression failed')


if __name__=='__main__':main()
