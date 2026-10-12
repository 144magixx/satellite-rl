"""Bounded teacher trajectory with features frozen before scoring candidates."""
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

def teacher_case(case, output, config=None):
    cfg=SearchConfig(**(config or {'seconds':10.,'max_attempts':20480}))
    scene=Scene(case['positions'],case['demands'],case.get('profile','europe_v1'))
    current=Layout.from_record(case['initial']) if 'initial' in case else initial_layout(scene)
    if not validate_layout(scene,current)['valid']: raise ValueError('invalid initial')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    start=perf_counter(); attempted=0; samples=[]; journal=[]; changed=0; stage=0; stream=None; retained_unchanged=False
    while len(scene.demands) and perf_counter()-start<cfg.seconds and attempted<cfg.max_attempts:
        if stream is None: stream=candidates(scene,current,'JOINT',cfg)
        pool=list(islice(stream,min(64,cfg.max_attempts-attempted)))
        if not pool: break
        features,mapping=pool_features(scene,current,pool)
        before=objective_key(metrics(scene,current)); best=current; bestkey=before
        before_sha=layout_sha256(current)
        for kind,candidate,detail in pool:
            if perf_counter()-start>=cfg.seconds: break
            attempted+=1; evidence=validate_layout(scene,candidate)
            key=objective_key(metrics(scene,candidate,evidence=evidence)) if evidence['valid'] else None
            accepted=key is not None and len(candidate.centers)==len(current.centers) and better(key,bestkey)
            journal.append(dict(stage=stage,kind=kind,details=detail,valid=evidence['valid'],reasons=evidence['reasons'],objective=key,accepted=accepted,sha256=layout_sha256(candidate)))
            if accepted: best,bestkey=candidate,key
        improved=layout_sha256(best)!=before_sha
        labels=mapping(best)
        sample=dict(features,labels=labels,improved=np.asarray(improved),stage=np.asarray(stage))
        sample.pop('centers')
        record=write_sample(root/f'sample_{stage:05d}.npz',sample) if improved or not retained_unchanged else None
        if record is not None: record.update(mother_id=case['mother_id'],profile=case.get('demand_profile','base'),improved=improved,changed_aircraft=int(np.sum(labels!=features['initial'])),pool_sha256=__import__('hashlib').sha256(json.dumps([layout_sha256(x[1]) for x in pool]).encode()).hexdigest())
        if record is not None: samples.append(record)
        retained_unchanged=retained_unchanged or not improved
        changed+=int(improved);stage+=1
        if improved: current=best;stream=None
    result=dict(status='completed',samples=samples,journal=journal,layout=current.to_record(),metrics=metrics(scene,current),budget=dict(seconds=perf_counter()-start,attempts=attempted,unknown=0),teacher_success=changed>0,changed_states=changed,states=len(samples))
    (root/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

