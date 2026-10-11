"""Build the finite authorized API DAG, leaving release proof to live audit."""
import argparse
import json
from pathlib import Path
from . import ops
from .controller import validate_plan
from .tunnel import atomic


def ref(identity,*keys):return {'$task':{'id':identity,'path':list(keys)}}


def task(identity,port,kind,body,requires):
    return {'id':identity,'port':port,'kind':kind,'body':{'request_id':'s2f20-'+identity,**body},'requires':requires}


def build(commit,sourceprep):
    ports=(20474,);ts=[task('sourceprep',ports[0],'sourceprep',sourceprep,[])]
    for port in ports:
        ts.append(task('prepare_'+str(port),port,'prepare',{'sourceprep_run_id':ref('sourceprep','run_id')},['sourceprep']))
    tuning={};formal={}
    for a,port in ((a,20474) for a in ('sac','ppo','dqn')):
        prep='prepare_'+str(port);tuning[a]=[]
        for candidate in ('A','B'):
            identity='tune_'+a+'_'+candidate
            ts.append(task(identity,port,'shorttune',{'algorithm':a,'candidate':candidate,'prepare_run_id':ref(prep,'run_id')},[prep]))
            tuning[a].append(identity)
        train='train_'+a;formal[a]=train
        ts.append(task(train,port,'train',{'algorithm':a,'prepare_run_id':ref(prep,'run_id'),'selection_run_ids':[ref(e,'run_id') for e in tuning[a]]},tuning[a]))
    # Trainer emits the four direct+B validation nodes inline (1728 B).
    # Each scene rotates all five methods on the same leased physical GPU.
    # Direct and full A+B are computed once; no unbudgeted timing replay is added.
    prerequisites=list(formal.values())
    prep='prepare_20474'
    ts.append(task('test_matrix',20474,'matrix',{'prepare_run_id':ref(prep,'run_id'),'models_by_algorithm':{a:ref(train,'run_id') for a,train in formal.items()}},prerequisites+[prep]))
    plan={'experiment_id':'S2-F20','scientific_commit':commit,'allowed_ports':[20474],'tasks':ts,'budgets':{'tuning_interactions':196608,'formal_interactions':1200000,'engineering_max':6144,'complete_B_calls':8328},'delivery_gate':'report_and_dual_archive_verified','timing_claim':'same_GPU_interleaved_matrix_with_actual_resource_evidence'}
    return validate_plan(plan)


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--sourceprep',type=Path,required=True);p.add_argument('--output',type=Path,default=ops.LOCAL/'run_plan.json');a=p.parse_args()
    if len(a.commit)!=40:raise ValueError('explicit frozen commit required')
    if a.output.exists():raise ValueError('refuse overwriting frozen plan')
    atomic(a.output,build(a.commit,json.loads(a.sourceprep.read_text(encoding='utf-8-sig'))))

if __name__=='__main__':main()
