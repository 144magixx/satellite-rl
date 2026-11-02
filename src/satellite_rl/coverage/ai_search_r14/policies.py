"""Eight prospectively fixed causal stopping rules and strict teacher safety."""
import copy
import math
import json
import hashlib

ACTIONS=('cap0','cap4','cap8','cap16','cap32','window2','window4','window8')

class StopRule:
    def __init__(self,action):
        if action not in ACTIONS:raise ValueError('fixed eight-rule family')
        self.action=action;self.completed=0;self.nonacceptance=0
    @property
    def stop_before_next(self):
        if self.completed>=32:return True
        return self.completed>=int(self.action[3:]) if self.action.startswith('cap') else self.nonacceptance>=int(self.action[6:])
    def after(self,status):
        if self.stop_before_next:raise ValueError('cannot enter another kernel after selected rule stopped')
        if status not in ('admissible','rejected','no_improvement'):raise ValueError('unknown native result never completes a policy step')
        self.completed+=1;self.nonacceptance=0 if status=='admissible' else self.nonacceptance+1
        return self.stop_before_next

def stop_index(action,statuses):
    rule=StopRule(action)
    if rule.stop_before_next:return 0
    for i,status in enumerate(statuses,1):
        if rule.after(status):return i
    return len(statuses)

def quality_preserving(candidate,teacher):
    keys=('maxload','top10_mean','CV','N','actual_Omega')
    for m in (candidate,teacher):
        if any(type(m.get(k)) not in (int,float) or not math.isfinite(m[k]) or m[k]<0 for k in keys):raise ValueError('complete finite saved metrics required')
        if type(m['N']) is not int or not 1<=m['N']<=200:raise ValueError('actual beam count')
    return (all(candidate[k]<=teacher[k]+(1e-9 if k=='CV' else 1e-6)+1e-12*abs(teacher[k]) for k in ('maxload','top10_mean','CV'))
        and candidate['N']<=teacher['N'] and candidate['actual_Omega']<=teacher['actual_Omega']+1e-12)

def apply_calibration_veto(tree,baseline,records,calibration_mother_ids):
    """Exactly one union over original24x2 frozen routes, never a refit/replay."""
    from .tree import route
    if tree.get('policy_type')!='depth3_tree' or baseline.get('policy_type')!='strong_single_split':raise ValueError('original AI and strong baseline policy types')
    def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if len(calibration_mother_ids)!=24 or len(set(calibration_mother_ids))!=24:raise ValueError('fixed24 distinct calibration mothers')
    if tree.get('calibration_applied') or baseline.get('calibration_applied'):raise ValueError('union veto may run once only')
    if len(records)!=48:raise ValueError('all24 frozen AI/baseline routes required')
    seen=set();veto={'AI':set(),'baseline':set()};mother_features={}
    for record in records:
        mother=record['mother_id'];kind=record['policy'];key=(mother,kind)
        if mother not in calibration_mother_ids or kind not in veto or key in seen or type(record.get('safe')) is not bool:raise ValueError('unknown/missing/duplicate calibration evidence')
        features=list(record['features'])
        if mother in mother_features and features!=mother_features[mother]:raise ValueError('same mother must use identical initial-B features in both frozen policies')
        mother_features[mother]=features
        seen.add(key);policy=tree if kind=='AI' else baseline
        actual=route(policy,record['features'])
        if actual['node_id']!=record['route_id'] or actual['action']!=record['action']:raise ValueError('route/action must belong to original frozen policy')
        if record['safe'] is False:veto[kind].add(actual['node_id'])
    if seen!={(m,k) for m in calibration_mother_ids for k in veto}:raise ValueError('complete fixed calibration denominator')
    def updated(policy,kind):
        result=copy.deepcopy(policy)
        def visit(node):
            if node['kind']=='leaf':
                if node['id'] in veto[kind]:node['action']='cap32'
            else:visit(node['left']);visit(node['right'])
        visit(result['root']);result['calibration_applied']=True
        result['calibration_veto_node_ids']=sorted(veto[kind]);return result
    return updated(tree,'AI'),updated(baseline,'baseline'),dict(schema='R14UnionCalibrationVetoV1',mothers=24,records=48,calibration_same_B_features_verified=True,original_tree_SHA=fingerprint(tree),original_baseline_SHA=fingerprint(baseline),veto={k:sorted(v) for k,v in veto.items()},refit=False,new_thresholds=False)
