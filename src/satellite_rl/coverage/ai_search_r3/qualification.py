"""Explicit once-only synthetic qualification; ordinary tests never spend it."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .solver import atomic_json,run_native
from .pool import prepare_case
from .integer import build_problem,decode
from .model import CandidateCountNetwork,features_graph
from ..ai_search.core import Scene

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vendor',required=True);parser.add_argument('--ledger',required=True)
    parser.add_argument('--solver-calls',type=int,default=3);parser.add_argument('--cuda-updates',type=int,default=4)
    parser.add_argument('--pipeline',action='store_true')
    parser.add_argument('--pipeline-methods',default='REF20,WITNESS,RULE,RANDOM,FULL,ORACLE_RESTRICT')
    args=parser.parse_args();path=Path(args.ledger)
    import json
    ledger=json.loads(path.read_text()) if path.exists() else dict(schema='r3_unique_core_qualification_v2',calls=[])
    spent_solver=sum(c['kind']=='toy_MILP' for c in ledger['calls']);spent_cuda=sum(c['kind']=='synthetic_CUDA_update' for c in ledger['calls'])
    methods=args.pipeline_methods.split(',');pipeline_calls=sum(2 if m=='REF20' else 0 if m=='WITNESS' else 1 for m in methods)
    if spent_solver+args.solver_calls+(pipeline_calls if args.pipeline else 0)>12 or spent_cuda+args.cuda_updates>4:raise RuntimeError('engineering_budget_guard')
    if any(c['state']=='started_unknown_spend' for c in ledger['calls']):raise RuntimeError('unknown_engineering_spend_requires_manual_audit')
    scene=Scene([[45,10],[45.05,10.05],[50,15],[50.01,15.01]],[3e6,0,8e6,1e6],('d','a','c','b'))
    p=prepare_case(scene,'synthetic_qualification');p.save(path.parent/'synthetic_case.npz');problem=build_problem(p,p.available)
    stage1=None
    def begin(kind):
        entry=dict(kind=kind,state='started_unknown_spend',started=time.time());ledger['calls'].append(entry);atomic_json(path,ledger);return entry
    for i in range(args.solver_calls):
        if i%3==2:
            if stage1 is None:raise RuntimeError('stage2_qualification_requires_valid_stage1')
            current=build_problem(p,p.available,2,stage1);seconds,nodes=8,10000
        else:current=problem;seconds,nodes=(2,2000) if i%3==0 else (12,20000)
        entry=begin('toy_MILP');result=run_native(current,path.parent/'native',args.vendor,seconds,nodes,30)
        if not result['value_valid']:raise RuntimeError('toy_native_no_result')
        layout,evidence=decode(p,current,result['vector']);stage1=np.asarray(result['vector'])[:problem.zindex+1]
        entry.update(state='completed',result=result,independent=evidence,stage=current.stage);atomic_json(path,ledger)
    if args.cuda_updates:
        if not torch.cuda.is_available():raise RuntimeError('CUDA_required_no_CPU_fallback')
        torch.manual_seed(51001);net=CandidateCountNetwork().cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
        f,g=features_graph(p);f=torch.tensor(f,device='cuda');g=torch.tensor(g,device='cuda');target=torch.zeros(len(f),3,dtype=torch.long,device='cuda')
        for _ in range(args.cuda_updates):
            entry=begin('synthetic_CUDA_update');before=[x.detach().clone() for x in net.parameters()];optimizer.zero_grad()
            logits=net(f,g);loss=torch.nn.functional.cross_entropy(logits.reshape(-1,3),target.reshape(-1),weight=torch.tensor([.1,1,1],device='cuda'))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();torch.cuda.synchronize()
            changed=any(not torch.equal(a,b) for a,b in zip(before,net.parameters()))
            if not changed:raise RuntimeError('CUDA_parameters_not_updated')
            entry.update(state='completed',loss=float(loss),parameters_changed=changed,device=torch.cuda.get_device_name(0));atomic_json(path,ledger)
    if args.pipeline:
        from . import evaluate as evaluate_module
        from .teacher import build_teacher_label
        actual_native=evaluate_module.run_native
        def counted_native(*call_args,**kwargs):
            if sum(c['kind']=='toy_MILP' for c in ledger['calls'])>=12:raise RuntimeError('engineering_budget_guard')
            entry=begin('toy_MILP');result=actual_native(*call_args,**kwargs)
            entry.update(state='completed',purpose='six_method_pipeline',stage=call_args[0].stage,result=result);atomic_json(path,ledger)
            return result
        evaluate_module.run_native=counted_native
        ref_path=path.parent/'pipeline'/'REF20'/'result.json'
        label=build_teacher_label(p,json.loads(ref_path.read_text())) if ref_path.exists() else None
        rows=ledger.get('pipeline_summary',{})
        for method in methods:
            row=evaluate_module.evaluate_method(p,method,path.parent/'pipeline'/method,vendor_dir=args.vendor,oracle_label=label)
            rows[method]=dict(status=row['status'],metrics=row.get('metrics'),comparable_seconds=row['comparable_seconds'])
            if method=='REF20':ref=row;label=build_teacher_label(p,row)
            if label and label['status']!='label_available':label=None
        ledger['pipeline_summary']=rows;atomic_json(path,ledger)
    ledger['finished']=time.time();atomic_json(path,ledger)

if __name__=='__main__':main()
