"""Bounded pure-geometry teacher; late completed trials never publish."""
import json
from itertools import islice
from pathlib import Path
from time import perf_counter
import numpy as np
from .config import SearchConfig
from .core import Scene, Layout, initial_layout, validate_layout, metrics, objective_key, layout_sha256
from .classical import candidates
from .evidence import pool_features, write_sample

def better(a,b,tol=1e-9):
    for x,y in zip(a,b):
        if x < y-tol: return True
        if x > y+tol: return False
    return False

def layout_delta(prior,candidate):
    users=np.flatnonzero(prior.assignment!=candidate.assignment)
    beams=np.flatnonzero(np.any(prior.centers!=candidate.centers,axis=1))
    return dict(users=users.tolist(),owners=candidate.assignment[users].tolist(),beams=beams.tolist(),centers=candidate.centers[beams].tolist())

def teacher_case(case, output, config=None):
    cfg=SearchConfig(**(config or {'seconds':10.,'max_attempts':20480}))
    scene=Scene(case['positions'],case['demands'],case.get('profile','europe_v1'))
    current=Layout.from_record(case['initial']) if 'initial' in case else initial_layout(scene)
    initial=current
    if not validate_layout(scene,current)['valid']: raise ValueError('invalid initial')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    start=perf_counter();deadline=start+cfg.seconds;attempted=0;samples=[];journal=[];pools=[];changed=0;stage=0;stream=None;retained_unchanged=False
    while len(scene.demands) and perf_counter()<deadline and attempted<cfg.max_attempts:
        if stream is None:stream=candidates(scene,current,'JOINT',cfg)
        generated_at=perf_counter()
        pool=list(islice(stream,min(64,cfg.max_attempts-attempted)))
        generation_end=perf_counter()
        pool_record=dict(stage=stage,current_sha256=layout_sha256(current),generation_started=generated_at-start,generation_ended=generation_end-start,pool_size=len(pool),committed=False)
        pools.append(pool_record)
        if not pool:break
        if generation_end>=deadline:pool_record['stop_reason']='generation_deadline';break
        features,mapping=pool_features(scene,current,pool)
        feature_end=perf_counter();pool_record['features_ended']=feature_end-start
        if feature_end>=deadline:pool_record['stop_reason']='features_deadline';break
        before=objective_key(metrics(scene,current));best=current;bestkey=before;before_sha=layout_sha256(current);best_trial=None
        for index,(kind,candidate,detail) in enumerate(pool):
            trial_start=perf_counter()
            if trial_start>=deadline:break
            attempted+=1;evidence=validate_layout(scene,candidate)
            values=metrics(scene,candidate,evidence=evidence) if evidence['valid'] else None
            key=objective_key(values) if values is not None else None
            trial_end=perf_counter();late=trial_end>=deadline
            same_n=len(candidate.centers)==len(current.centers)
            improved=key is not None and same_n and better(key,bestkey)
            temporary_best=improved and not late
            trial=dict(stage=stage,index=index,kind=kind,details=detail,prior_sha256=before_sha,current_sha256=before_sha,best_before_sha256=layout_sha256(best),candidate_sha256=layout_sha256(candidate),delta=layout_delta(current,candidate) if same_n else None,layout=candidate.to_record() if not same_n else None,geometry=evidence,metrics=values,objective=key,started=trial_start-start,ended=trial_end-start,elapsed=trial_end-trial_start,late=late,temporary_best=temporary_best,pool_committed=False,rejection='deadline' if late else ('invalid_geometry' if not evidence['valid'] else ('N_changed' if not same_n else ('no_lex_improvement' if not improved else None))))
            journal.append(trial)
            if temporary_best:best,bestkey,best_trial=candidate,key,len(journal)-1
        improved=layout_sha256(best)!=before_sha
        # A candidate completed on time is publishable; late candidates cannot replace it.
        if improved:
            pool_record.update(committed=True,committed_trial=best_trial,published_sha256=layout_sha256(best));journal[best_trial]['pool_committed']=True
        labels=mapping(best);sample=dict(features,labels=labels,improved=np.asarray(improved),stage=np.asarray(stage));sample.pop('centers')
        record=write_sample(root/f'sample_{stage:05d}.npz',sample) if improved or not retained_unchanged else None
        if record is not None:
            record.update(mother_id=case['mother_id'],profile=case.get('demand_profile','base'),improved=improved,changed_aircraft=int(np.sum(labels!=features['initial'])),pool_sha256=__import__('hashlib').sha256(json.dumps([layout_sha256(x[1]) for x in pool]).encode()).hexdigest());samples.append(record)
        retained_unchanged=retained_unchanged or not improved;changed+=int(improved);stage+=1
        if improved:current=best;stream=None
    actual=perf_counter()-start
    result=dict(status='completed',samples=samples,journal=journal,pools=pools,input=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),profile=scene.profile,initial=initial.to_record()),layout=current.to_record(),metrics=metrics(scene,current),budget=dict(seconds=actual,limit_seconds=cfg.seconds,overrun_seconds=max(0.,actual-cfg.seconds),attempts=attempted,unknown=0,late_completed_candidates=sum(x['late'] for x in journal)),teacher_success=changed>0,changed_states=changed,states=stage)
    (root/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
