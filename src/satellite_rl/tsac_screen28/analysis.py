"""Completed-matrix arithmetic only; never loads a policy, scene or PHY."""
import json
import math
from pathlib import Path
import pickle
from satellite_rl.t0_width.logging import content_sha
from .workbench import episode_identity
from .plan import development_gate

def summarize_completed(plan,root):
    root=Path(root);cursor=pickle.loads((root/'matrix_cursor.pkl').read_bytes());jobs=plan['matrix']['jobs']
    if cursor['plan_SHA']!=content_sha(plan) or cursor['cursor']!=len(jobs) or cursor['active'] is not None or cursor['unknown']:raise ValueError('exact fully closed matrix required')
    cells={};qualified=True
    for index,job in enumerate(jobs):
        row=json.loads((root/'episodes'/('%04d'%index)/'terminal.json').read_text(encoding='utf8'))
        if row['identity']!=episode_identity(plan,index):raise ValueError('terminal plan/source/policy/scene identity differs')
        metrics=row['metrics'];beams=row['beam_results']
        if (metrics['N']!=job['N'] or metrics['processed']!=job['N'] or metrics['complete'] is not True
                or len(beams)!=job['N'] or len({b['beam_id'] for b in beams})!=job['N']):raise ValueError('missing actual N or complete all-beam terminal')
        values=dict(SI=metrics['SI_mean'],SGM=metrics['SGM'],power_w=metrics['total_power_w'])
        if not all(math.isfinite(x) for x in values.values()) or values['power_w']>3000:raise ValueError('nonfinite or resource-illegal terminal')
        seen_slots=set();beam_power=0.
        for beam in beams:
            start,count,power,group=(beam[k] for k in ('start','count','power_w','group'))
            if not (type(start) is int and type(count) is int and 0<=start<100 and 1<=count<=10 and start+count<=100
                    and 5<=power<=50 and power%5==0 and type(group) is int and 0<=group<8):raise ValueError('illegal full beam terminal allocation')
            slots={(group,slot) for slot in range(start,start+count)}
            if seen_slots & slots:raise ValueError('same-group slot overlap')
            seen_slots.update(slots);beam_power+=power
        if beam_power!=values['power_w']:raise ValueError('beam/metric power conservation differs')
        observed_beams=set();observed_queries=0
        with (root/'episodes'/('%04d'%index)/'steps.jsonl').open(encoding='utf8') as stream:
            for raw in stream:
                step=json.loads(raw);context=step['context'];info=step['info']
                if (context['case_id']!=job['case_id'] or context['method']!=job['method'] or context['repeat']!=job['repeat']
                        or context['step']!=len(observed_beams) or info['beam_id'] in observed_beams):raise ValueError('step identity/order/beam duplicate differs')
                observed_beams.add(info['beam_id']);f,l,p=info['requested_action'];actual=info['actual_action']
                if actual!=dict(start=f,count=l+1,power_w=(p+1)*5):raise ValueError('execution clipping detected')
                if job['method'] in ('P4','R4'):observed_queries+=step['candidates']['unique_candidates']
        if observed_beams!={b['beam_id'] for b in beams}:raise ValueError('step evidence omits actual beams')
        key=(job['mother'],job['method'],job['repeat'])
        if key in cells:raise ValueError('duplicate terminal matrix cell')
        cells[key]=values
        cost=row['costs']
        if (cost['commit']!=job['N'] or cost['query']>(4*job['N'] if job['method'] in ('P4','R4') else 0)
                or cost['query']!=observed_queries or cost['forward']!=(job['N'] if job['method'].startswith('P') else 0)):raise ValueError('terminal method-specific finite cost differs')
    methods=sorted({j['method'] for j in jobs});mothers=sorted({j['mother'] for j in jobs});means={}
    for method in methods:
        repeats=[0] if method in ('fixed','greedy') else range(3)
        means[method]={name:sum(sum(cells[(mother,method,repeat)][name] for repeat in repeats)/len(repeats) for mother in mothers)/len(mothers) for name in ('SI','SGM','power_w')}
    result=dict(schema='W28MatrixSummaryV1',root_approved=False,phase=plan['phase'],means=means,
        episodes=len(cells),costs=cursor['costs'],not_independent=True,
        timing_unknown=cursor.get('timing_unknown',[]),complete_wall_comparison_available=not cursor.get('timing_unknown',[]))
    if plan['phase']=='development':
        gate=development_gate(means,qualified)
        gate.update(development_plan_SHA=content_sha(plan),completed_development_episodes=len(cells),
            policy_SHA=plan['policy_ref']['sha256'],source_commit=plan['source_commit'],development_manifest_SHA=plan['data_manifest_ref']['sha256'])
        result['decision_candidate']=gate
    return result
