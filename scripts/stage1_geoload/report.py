"""Offline G16 audit/report from frozen plan, controller snapshot and job ZIPs.

Compact CSVs only; raw layouts and journals remain in immutable ZIP members.
Run explicitly after computing/archiving; never schedules or resumes science.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .strict_audit import JobZip, AuditError, digest, audit_search, paired_bootstrap


def write_csv(path, rows):
    rows=list(rows)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader();w.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class Dataset:
    def __init__(self, archive, member, expected_count):
        # NPZ is read from the original ZIP; never creates another raw-data copy.
        with archive.zip.open(member) as f, np.load(f,allow_pickle=False) as data:
            # np.load already returns owned arrays; copying doubles peak buffers.
            self.positions=data['positions']; self.demands=data['demands']
            self.offsets=data['offsets']; self.ids=data['case_ids'].astype(str)
            self.mothers=data['mother_ids'].astype(str); self.profiles=data['demand_profiles'].astype(str)
        n=len(self.ids)
        if n != expected_count or len(set(self.ids)) != n or self.offsets.shape!=(n+1,) or self.offsets[0]!=0 or self.offsets[-1]!=len(self.demands) or (np.diff(self.offsets)<0).any():
            raise AuditError('packed input index invalid')
        self.index={key:i for i,key in enumerate(self.ids)}
        self.ref=archive.ref(member)

    def get(self,key):
        i=self.index[key];a,b=map(int,self.offsets[i:i+2])
        return dict(positions=self.positions[a:b],demands=self.demands[a:b],
                    mother_id=self.mothers[i],demand_profile=self.profiles[i],profile='europe_v1')


def _date(mother):
    hit=re.search(r'20\d{2}[-_]?\d{2}[-_]?\d{2}',mother)
    if not hit: raise AuditError('mother date absent; date strata cannot be invented')
    value=hit.group().replace('_','-')
    return value if '-' in value else value[:4]+'-'+value[4:6]+'-'+value[6:]


def _model_metadata(archive):
    """CPU metadata/state inspection only; this is never a GPU inference receipt."""
    import torch
    with archive.zip.open('model/checkpoint.pt') as f:
        checkpoint=torch.load(f,map_location='cpu',weights_only=False)
    with archive.zip.open('model/final.pt') as f:
        final=torch.load(f,map_location='cpu',weights_only=False)
    if set(checkpoint['model'])!=set(final['model']) or any(
            not torch.equal(value,final['model'][key]) for key,value in checkpoint['model'].items()):
        raise AuditError('final/checkpoint parameter state differs')
    return checkpoint,final


def validation_tables(archive, dataset, initials, metadata_reader=_model_metadata):
    """Read the four actual _train validation directories; never calls an evaluator.

    node_done records have no per-node checkpoint SHA in science495. Association
    therefore follows the final checkpoint's completed-node list, protocol and
    immutable job/member hashes; do not invent intermediate checkpoint files.
    """
    cases=[];curves=[];issues=[]
    try:
        cp,final=metadata_reader(archive)
        protocol=cp['protocol']
        protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        for state in (cp,final):
            if state.get('format')!='s1_g16_geometry_mlm_v1':raise AuditError('validation model format differs from frozen worker')
            if state.get('update')!=10000 or state.get('seed')!=51001 or state.get('from_scratch') is not True:
                raise AuditError('validation source endpoint/seed/from_scratch differs')
            if state.get('protocol_sha256')!=protocol_sha or state.get('dataset_sha256')!=cp['dataset_sha256']:
                raise AuditError('validation checkpoint/final protocol/teacher data differs')
        if protocol.get('scientific_commit')!=archive.job['scientific_commit'] or protocol.get('validation_sha256')!=dataset.ref['sha256']:
            raise AuditError('validation model/source input SHA differs')
        teacher=archive.job.get('references',{}).get('teacher',{})
        if protocol.get('teacher_source_sha256')!=teacher.get('sha256'):
            raise AuditError('validation teacher source SHA differs')
        if cp.get('nodes')!=[2500,5000,7500,10000]:raise AuditError('checkpoint completed validation nodes differ')
        cp_sha=archive.manifest['members']['model/checkpoint.pt']['sha256']
        final_sha=archive.manifest['members']['model/final.pt']['sha256']
    except Exception as e:
        return cases,curves,['MLM validation source: '+str(e)]
    for node in (2500,5000,7500,10000):
        node_cases=[]
        prefix='model/validation_'+str(node).zfill(5)+'/'
        try:
            done_member=prefix+'node_done.json';done=archive.read(done_member)
            if done.get('update')!=node or done.get('count')!=80 or done.get('validation_sha256')!=dataset.ref['sha256']:
                raise AuditError('node_done update/count/validation source differs')
            index=done.get('index',[])
            if len(index)!=80 or {i['case_id'] for i in index}!=set(dataset.ids):
                raise AuditError('node_done cases missing/duplicated')
            actual={n for n in archive.manifest['members'] if n.startswith(prefix) and n.endswith('.json') and n!=done_member}
            wanted={prefix+hashlib.sha256((i['case_id']+'\0MLM').encode()).hexdigest()+'.json' for i in index}
            if actual!=wanted:raise AuditError('validation node JSON membership differs')
            for item in index:
                member=prefix+hashlib.sha256((item['case_id']+'\0MLM').encode()).hexdigest()+'.json'
                ref=archive.manifest['members'][member]
                if ref['sha256']!=item['sha256']:raise AuditError('node case SHA differs')
                raw=archive.read_case(member) if hasattr(archive,'read_case') else archive.read(member);scene=dataset.get(item['case_id'])
                if raw.get('case_id')!=item['case_id'] or raw.get('mother_id')!=scene['mother_id'] or raw.get('demand_profile')!=scene['demand_profile']:
                    raise AuditError('node case/mother/profile identity differs')
                metric,events=audit_search(scene,{**raw,'method':'MLM','initial_metrics':initials[item['case_id']]},compact=True)
                budget=raw.get('budgets',{})
                required=('seconds','attempts','geometry_calls','late_candidates')
                if any(k not in budget for k in required):raise AuditError('validation actual budget unknown')
                if any(not np.isfinite(budget[k]) or budget[k]<0 for k in required):raise AuditError('validation actual budget invalid')
                node_cases.append(dict(node=node,case_id=item['case_id'],mother_id=scene['mother_id'],date=_date(scene['mother_id']),
                    demand_profile=scene['demand_profile'],**metric,coverage_100=True,unique_assignment=True,demand_conserved=True,
                    search_seconds=budget['seconds'],attempts=budget['attempts'],geometry_calls=budget['geometry_calls'],
                    late_candidates=budget['late_candidates'],accepted=events['accepted'],
                    source_zip=str(archive.path),source_member=member,source_sha256=ref['sha256'],
                    node_done_sha256=archive.manifest['members'][done_member]['sha256'],checkpoint_sha256=cp_sha,
                    endpoint_sha256=final_sha,validation_input_sha256=dataset.ref['sha256'],protocol_sha256=protocol_sha,
                    model_association='final_checkpoint_completed_nodes; intermediate_checkpoint_not_retained',
                    scope='training_diagnostic_not_common_timing'))
            mothers=defaultdict(list)
            for row in node_cases:mothers[row['mother_id']].append(row)
            if len(mothers)!=20 or any(len(rows)!=4 or {r['demand_profile'] for r in rows}!=
                    {'base','one_hotspot','two_hotspots','edge_heavy'} for rows in mothers.values()):
                raise AuditError('validation 20 mothers x 4 profiles incomplete')
            # Each mother contributes its four-profile mean once, regardless of aircraft count.
            means={k:[float(np.mean([r[k] for r in rows])) for rows in mothers.values()] for k in ('P','T','Q','CV','N')}
            curves.append(dict(node=node,mothers=20,cases=80,
                **{k:float(np.mean(values)) for k,values in means.items()},
                hard_coverage_cases=sum(r['coverage_100'] for r in node_cases),hard_violations=0,
                search_seconds_total=sum(r['search_seconds'] for r in node_cases),
                attempts_total=sum(r['attempts'] for r in node_cases),geometry_calls_total=sum(r['geometry_calls'] for r in node_cases),
                late_candidates_total=sum(r['late_candidates'] for r in node_cases),
                node_done_sha256=archive.manifest['members'][done_member]['sha256'],checkpoint_sha256=cp_sha,
                endpoint_sha256=final_sha,validation_input_sha256=dataset.ref['sha256'],
                aggregation='mother_mean_of_four_profiles',scope='training_diagnostic_not_common_timing'))
            cases.extend(node_cases)
        except Exception as e:issues.append('MLM validation node '+str(node)+': '+str(e))
    return cases,curves,issues


def _expected_keys(plan, gate):
    keys={'prepare_'+k for k in plan['preparations']}
    for role in ('train','development','validation','test'):
        if sum(x['role']==role for x in plan['preparations'].values())>1:keys.add('merge_'+role)
        keys.update('initial_'+role+'_'+str(p) for p in (20424,20473))
    keys.update(prefix+'_'+str(p) for prefix in ('dev','reference','variable','common_validation','common_test') for p in (20424,20473))
    if gate:keys.update(('teacher_20424','teacher_20473','teacher_merge','train_mlm'))
    return keys


def build(plan_path,state_path,destination,closure=None,*,progress=None):
    plan=read_json(plan_path);state=read_json(state_path);out=Path(destination)
    out.mkdir(parents=True,exist_ok=True)
    issues=[]; archives={};datasets={};case_rows=[];timing=[];candidate_rows=[];learning=[];initials={};validation_cases=[];validation_curves=[]
    tasks=state.get('tasks',{});science=plan['scientific_commit']
    began=time.monotonic();last_progress=0.
    def notify(event):
        nonlocal last_progress
        current=time.monotonic()
        if progress is not None and current-last_progress>=5:
            last_progress=current;progress(dict(elapsed_seconds=current-began,**event))
    if state.get('plan_sha256')!=digest(plan_path):issues.append('controller snapshot plan SHA differs')
    if plan.get('group')!='S1-G16' or state.get('group')!='S1-G16':raise AuditError('wrong experiment')
    gate=state.get('gate',{}).get('passed')
    if type(gate) is not bool:issues.append('development gate unknown')
    expected=_expected_keys(plan,gate is True)
    if set(tasks)!=expected:issues.append('completed task DAG differs from frozen gate branch')
    owners=[(t.get('port'),t.get('run_id')) for t in tasks.values()]
    if len(set(owners))!=len(owners):issues.append('duplicate task owner')
    try:
        for key,task in tasks.items():
            notify(dict(phase='ZIP_integrity',task=key))
            try:
                if task.get('stage')!='completed':raise AuditError('task not completed')
                receipt=state.get('archives',{}).get(key)
                if receipt is None:raise AuditError('ZIP receipt absent')
                if receipt['port']!=task['port'] or receipt['run_id']!=task['run_id']:raise AuditError('receipt task owner differs')
                archive=JobZip(receipt,science);archives[key]=archive
                if archive.status['stage']!='completed':raise AuditError('archived status not completed')
                if archive.job.get('request_id')!=task['request_id']:raise AuditError('request identity differs')
            except Exception as e:issues.append(key+': '+str(e))
        # Only canonical role inputs referenced by controller, hash located in verified ZIP.
        for role,reference in state.get('datasets',{}).items():
            matches=[(a,n) for a in archives.values() for n,r in a.manifest['members'].items()
                     if n.endswith('.npz') and r['sha256']==reference['sha256']]
            try:
                if not matches:raise AuditError('canonical packed input absent')
                a,n=matches[0];datasets[role]=Dataset(a,n,reference['count'])
                if len(set(datasets[role].mothers))!=plan['mother_counts'][role]:raise AuditError('mother count differs')
                groups=defaultdict(list)
                for i,m in enumerate(datasets[role].mothers):groups[m].append(i)
                for ids in groups.values():
                    if {datasets[role].profiles[i] for i in ids}!={'base','one_hotspot','two_hotspots','edge_heavy'} or len(ids)!=4:
                        raise AuditError('four profiles per mother required')
                    totals=[datasets[role].get(datasets[role].ids[i])['demands'].sum() for i in ids]
                    if not np.allclose(totals,totals[0],rtol=1e-12,atol=1e-6):raise AuditError('profile demand not conserved')
                    positions=[datasets[role].get(datasets[role].ids[i])['positions'] for i in ids]
                    if any(not np.array_equal(p,positions[0]) for p in positions[1:]):raise AuditError('four profiles have different aircraft geometry')
            except Exception as e:issues.append(role+': '+str(e))
        for key,a in archives.items():
            role=a.job.get('dataset_role');data=datasets.get(role)
            if 'job/case_index.json' not in a.manifest['members']:continue
            index=a.read('job/case_index.json')
            for stem,ref in index.items():
                notify(dict(phase='case_audit',task=key,case=stem,completed_compact_rows=len(case_rows)))
                member='job/cases/'+stem+'.json'
                try:
                    if a.manifest['members'][member]['sha256']!=ref['sha256']:raise AuditError('case index SHA differs')
                    raw=a.read_case(member)
                    if data is None:raise AuditError('canonical dataset unresolved')
                    scene=data.get(raw['case_id'])
                    if raw['mother_id']!=scene['mother_id'] or raw['demand_profile']!=scene['demand_profile']:
                        raise AuditError('case input identity differs')
                    base=dict(task=key,role=role,case_id=raw['case_id'],mother_id=raw['mother_id'],
                              date=_date(raw['mother_id']),demand_profile=raw['demand_profile'],method=raw['method'],
                              port=tasks[key]['port'],source_zip=str(a.path),source_member=member,
                              source_sha256=ref['sha256'],outcome=raw['outcome'])
                    entries=raw.get('entries',[raw])
                    for entry in entries:
                        small=dict(base)
                        if 'entries' in raw:small.update(stage=entry['stage'],target_N=entry['target_N'],outcome=entry['outcome'])
                        if 'layout' in entry:
                            audited={**entry,'method':raw['method']}
                            if raw['method']!='VARIABLE' and 'initial_metrics' not in audited and raw['case_id'] in initials:
                                audited['initial_metrics']=initials[raw['case_id']]
                            metric,events=audit_search(scene,audited,compact=True,
                                progress=lambda event:notify(dict(phase='journal_replay',task=key,case=raw['case_id'],**event)))
                            if key.startswith('initial_'):initials[raw['case_id']]=metric
                            small.update(metric)
                            if entry.get('certified_infeasible'):raise AuditError('bounded not_found labeled infeasible')
                            budget=entry.get('budgets',entry.get('budget',{}))
                            init=raw.get('initialization_seconds');online=raw.get('online_seconds');cold=raw.get('model_cold_load_seconds',0)
                            reconstructed=init+online+cold if init is not None and online is not None else None
                            timing.append({**{k:small[k] for k in ('task','case_id','method')},
                                'stage':small.get('stage','online'),'initialization_seconds':raw.get('initialization_seconds'),
                                'online_seconds':raw.get('online_seconds'),'end_to_end_seconds':raw.get('end_to_end_seconds'),
                                'model_cold_load_seconds':raw.get('model_cold_load_seconds'),
                                'reconstructed_total_seconds':reconstructed,
                                'recorded_minus_reconstructed_seconds':raw.get('end_to_end_seconds')-reconstructed if raw.get('end_to_end_seconds') is not None and reconstructed is not None else None,
                                **{k:budget.get(k) for k in ('seconds','attempts','accepted','geometry_calls','late_candidates','deadline_overrun_seconds')},
                                'cost_scope':'known_completed; crash_unknown_costs_separate'})
                            candidate_rows.append(dict(task=key,case_id=raw['case_id'],method=raw['method'],stage=small.get('stage','online'),
                                **events,source_zip=str(a.path),source_member=member,
                                source_sha256=ref['sha256'],raw_journal_retained=True))
                        elif entry.get('outcome') not in ('not_found','not_run_constructor_not_found'):
                            raise AuditError('completed case layout absent')
                        case_rows.append(small)
                except Exception as e:issues.append(key+'/'+stem+': '+str(e))
        # Strict method matrix checks retain failed cases instead of excluding them.
        methods=['BH','07','09-CV','09-LEX','09+SWAP','JOINT']+(['MLM','rule','random'] if gate else [])
        for role,prefix in [('development','dev_'),('validation','common_validation_'),('test','common_test_')]:
            selected=[r for r in case_rows if r['task'].startswith(prefix)]
            expected_methods=methods[:6] if role=='development' else methods
            expected_pairs={(c,m) for c in datasets.get(role,object()).ids for m in expected_methods} if role in datasets else set()
            actual=[(r['case_id'],r['method']) for r in selected]
            if len(actual)!=len(set(actual)) or set(actual)!=expected_pairs:issues.append(role+' method matrix incomplete/duplicated')
            same_case=defaultdict(list)
            for r in selected:same_case[r['case_id']].append(r)
            for case_id,rows in same_case.items():
                if len({(r['mother_id'],r['date'],r['demand_profile']) for r in rows})!=1:
                    issues.append(role+' same-case mother/profile/date differs '+case_id)
                completed=[r for r in rows if r.get('outcome')=='completed']
                if any(r.get('N') is None for r in completed) or len({r.get('N') for r in completed})>1:
                    issues.append(role+' same-case method N differs '+case_id)
            hosts=defaultdict(set)
            for r in selected:hosts[r['mother_id']].add(r['port'])
            if any(len(h)!=1 for h in hosts.values()):issues.append(role+' mother split across hosts')
        # Training history is streamed, retaining only loss/update/attempt identity.
        a=archives.get('train_mlm')
        if gate is True:
            if a is None:issues.append('gate passed but new MLM evidence absent')
            else:
                members=a.manifest['members']
                for suffix in ('final.pt','checkpoint.pt'):
                    if not any(n.endswith('/'+suffix) for n in members):issues.append('MLM '+suffix+' absent')
                logs=[n for n in members if n.endswith('/loss.jsonl')]
                if not logs:issues.append('MLM training history absent')
                for name in logs:
                    with a.zip.open(name) as f:
                        for line in f:
                            row=json.loads(line);learning.append({k:row.get(k) for k in ('update','loss','attempt_id')})
                if [r['update'] for r in learning]!=list(range(1,10001)):issues.append('MLM 10000 retained updates not complete')
                if 'validation' not in datasets:issues.append('MLM validation canonical input unresolved')
                else:
                    validation_cases,validation_curves,node_issues=validation_tables(a,datasets['validation'],initials)
                    issues.extend(node_issues)
                artifacts=a.read('job/artifacts.json');ledger=artifacts.get('training',{}).get('ledger')
                if ledger is None:issues.append('MLM actual update/unknown/replay ledger absent')
                else:
                    if ledger.get('retained_updates')!=10000 or ledger.get('unknown_attempts',4)>3:issues.append('MLM retained/unknown budget differs')
                    timing.append(dict(task='train_mlm',stage='learning_updates',**ledger,
                        unknown_cost_interpretation='attempt count known; interrupted FLOPs/time not known, never imputed zero'))
        elif gate is False and any(k.startswith('teacher') or k=='train_mlm' for k in tasks):issues.append('gate failed but teacher/MLM branch executed')
        mother_rows=[]
        grouped=defaultdict(list)
        for r in case_rows:
            if r['task'].startswith('common_') and r.get('outcome')=='completed':grouped[(r['role'],r['mother_id'],r['date'],r['method'])].append(r)
        for (role,mother,date,method),rows in grouped.items():
            mother_rows.append(dict(role=role,mother_id=mother,date=date,method=method,profiles=len(rows),
                **{k:float(np.mean([r[k] for r in rows])) for k in ('N','P','T','Q','CV') if all(r.get(k) is not None for r in rows)}))
        # Reconstruct teacher gate from independent audited 30-second layouts.
        old={r['case_id']:r for r in case_rows if r['task'].startswith('dev_') and r['method']=='09-LEX'}
        reference={}
        for key,a in archives.items():
            if not key.startswith('reference_'):continue
            for stem in a.read('job/case_index.json'):
                raw=a.read_case('job/cases/'+stem+'.json')
                nodes=[n for n in raw.get('reference_nodes',[]) if n.get('kind')=='paired' and n.get('seconds_limit')==30 and n.get('candidate_limit')==61440]
                if len(nodes)!=1:issues.append('reference paired 30s node absent '+raw['case_id']);continue
                reference[raw['case_id']]=nodes[0]['metrics']
        if len(old)!=80 or set(old)!=set(reference):issues.append('independent development gate pairs incomplete')
        else:
            mothers=sorted({r['mother_id'] for r in old.values()});halves=[set(mothers[:10]),set(mothers[10:])]
            wins=[]
            for case_id,b in old.items():
                r=reference[case_id]
                qualified=b.get('P') not in (None,0) and ((b['P']-r['P'])/b['P']>=.01 or (abs(b['P']-r['P'])<=1e-9 and b['T']>0 and (b['T']-r['T'])/b['T']>=.01))
                if qualified:wins.append(b['mother_id'])
            actual_gate=len(wins)/80>=.2 and all(sum(m in h for m in wins)/40>=.1 for h in halves)
            if actual_gate!=gate:issues.append('independent gate differs from dispatched branch')
        comparisons=[]
        if gate:
            test=[r for r in case_rows if r['task'].startswith('common_test_')]
            for baseline in ('JOINT','rule','random'):
                try:
                    value=paired_bootstrap(test,'MLM',baseline)
                    value['pairwise_P_quality_threshold_met']=value['mean']>=.01 and value['lower95']>0
                    value['hierarchical_conclusion']='pairwise_P_quality_criterion_only' if value['pairwise_P_quality_threshold_met'] else 'P_quality_not_established; T_equivalence_or_same_target_speed_not_established'
                    value['speed_success_claim_allowed']=False
                    comparisons.append(value)
                except Exception as e:issues.append('learning '+baseline+': '+str(e))
        if closure is None or closure.get('status')!='accepted':issues.append('external data/source/model reference closure not accepted')
        # GPU model verification remains a separately supplied actual receipt.
        if gate and not (closure or {}).get('model_gpu_validation',{}).get('verified'):issues.append('actual endpoint/checkpoint strict GPU model validation absent')
        status='formal_ready' if not issues and state.get('compute_and_archives_complete') else 'data_unknown'
        if status!='formal_ready' and not issues:issues.append('controller completion not observed')
        summary=dict(group='S1-G16',status=status,formal_ready=status=='formal_ready',issues=issues,
            scientific_commit=science,report_source='operational_postprocessing_separate_from_worker',
            plan_sha256=digest(plan_path),state_sha256=digest(state_path),gate_passed=gate,
            learning_status='science_stopped_teacher_gate_failed_not_trained' if gate is False else 'completed_only_if_audited',
            case_rows=len(case_rows),mother_rows=len(mother_rows),comparisons=comparisons,
            validation_diagnostic_cases=len(validation_cases),validation_diagnostic_nodes=len(validation_curves),
            no_global_optimality_certificate=True,single_training_seed=51001,
            candidate_geometry_limitation='final/reference layouts independently audited; ordinary candidate geometry retained as worker evidence with independent load algebra',
            unknown_replay_budget='must be closed by archive acceptance recovery/attempt ledger; unknown is never zero',
            broad_multiple_comparison_success_claim_allowed=False)
        summary['memory_strategy']='ijson3.3.0 case metadata + one journal event; compact accepted histories and scalar CSV rows; four canonical numeric/index NPZ arrays only'
        summary['stream_parser']=dict(package='ijson',version='3.3.0',journal_reopens='one full event traversal per search; metadata traversal separately; variable entry selection may traverse preceding entries; no sampling')
        grouped_summary=defaultdict(list)
        for r in mother_rows:grouped_summary[(r['role'],r['method'])].append(r)
        descriptive=[]
        for (role,method),rows in grouped_summary.items():
            descriptive.append(dict(role=role,method=method,mothers=len(rows),
                **{k:float(np.mean([r[k] for r in rows])) for k in ('N','P','T','Q','CV') if all(k in r for r in rows)}))
        for name,rows in [('per_case',case_rows),('per_mother',mother_rows),('timing',timing),('candidate',candidate_rows),('learningcurve',learning),('summary',descriptive),('learning_comparisons',comparisons),('validation_cases',validation_cases),('validation_curves',validation_curves)]:write_csv(out/(name+'.csv'),rows)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        return summary
    finally:
        for archive in archives.values():archive.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--state',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--closure')
    parser.add_argument('--progress',help='optional compact progress JSON, throttled at five seconds')
    args=parser.parse_args()
    def persist_progress(event):
        path=Path(args.progress);temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(event,ensure_ascii=False),encoding='utf-8');temporary.replace(path)
    result=build(args.plan,args.state,args.output,read_json(args.closure) if args.closure else None,
        progress=persist_progress if args.progress else None)
    print(json.dumps({'status':result['status'],'issues':result['issues']}))
    return 0 if result['formal_ready'] else 2


if __name__=='__main__':raise SystemExit(main())
