"""Prespecified read-only prefix-margin evidence; zero physical/model calls."""
import ast
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

RULE_SHA='f4ffcfadb1d4ae2a5145493fd4fa86c584d81e81a89a44b3286a387744de61ce'
SEMANTICS_SHA='c13bb5a1e984808befc4fb018e0fef612d71f52c9d90d705355869bde1621d5a'
KERNEL_CLARIFICATION_SHA='69080f186d02189afa63bf5d9fbcf8505ff3c86719a5fb63f2fb00682c842dd4'
EPS=1e-12


def unavailable(reason):return dict(status='unavailable',reason=str(reason))


def unavailable_state(state_id,mother,prefix,reason):
    return dict(state_id=state_id,mother=mother,prefix=prefix,**unavailable(reason),
        groups=[dict(count=count,power_w=power,status='unavailable',reason=str(reason),rho_X=None,rho_U=None)
                for count in range(1,11) for power in range(5,51,5)])


def _sgm(capacity,demand,beta):
    values=np.zeros_like(capacity,dtype=np.float64);positive=capacity>0
    ratio=capacity[positive]/demand[positive]
    real=np.where(ratio>=1,ratio-1,1-1/ratio)
    values[positive]=1-(1-np.exp(-np.abs(real+1j*(capacity[positive]-demand[positive])/beta)))**3
    return values


def features(prefix,demands,indices,groups,kernel,physics):
    """Only arithmetic on saved gain/path and allocations; no physics helpers."""
    try:
        demands=np.asarray(demands);indices=np.asarray(indices);groups=np.asarray(groups);n=len(demands)
        if demands.shape!=(n,) or not np.isfinite(demands).all() or (demands<=0).any():raise ValueError('positive saved demands required')
        k=prefix['beam_idx']
        if type(k)is not int or not 0<=k<n:raise ValueError('decision-before prefix required')
        if indices.shape!=(n,) or len(set(indices.tolist()))!=n or set(indices.tolist())!=set(range(n)):raise ValueError('stable complete processing order')
        gain=np.asarray(kernel['gain']);path=np.asarray(kernel['path'])
        if gain.shape!=(n,n) or path.shape!=(n,100) or not np.isfinite(gain).all() or not np.isfinite(path).all() or (gain<0).any() or (path<0).any():raise ValueError('saved gain/path arrays unavailable')
        if groups.shape!=(n,) or ((groups<0)|(groups>7)).any():raise ValueError('actual group identity')
        noise=float(physics['k_J_per_K'])*float(physics['noise_temperature_K'])*float(physics['slot_Hz'])
        rx=10**(float(physics['rx_gain_dBi'])/10)
        if not math.isfinite(noise) or noise<=0 or not math.isfinite(rx) or rx<=0:raise ValueError('saved physical noise/rx parameters')
        rates=np.asarray(prefix['rates']);starts=np.asarray(prefix['starts']);counts=np.asarray(prefix['counts']);powers=np.asarray(prefix['powers'])
        if any(value.shape!=(n,) or not np.isfinite(value).all() for value in (rates,starts,counts,powers)) or (rates<0).any():raise ValueError('complete prefix numerical arrays')
        processed=indices[:k];current=int(indices[k]);slots=np.zeros((k,100),bool)
        for row,j in enumerate(processed):
            f=int(starts[j]);count=int(counts[j]);p=float(powers[j])
            if starts[j]!=f or counts[j]!=count or not 1<=count<=10 or not 0<=f<=100-count or p not in range(5,51,5):raise ValueError('actual original prefix allocation')
            slots[row,f:f+count]=True
        if any((value[indices[k:]]!=0).any() for value in (starts,counts,powers,rates)):raise ValueError('unprocessed prefix entries not zero')
        denominator=np.full((k,100),noise,dtype=np.float64)
        for tx in processed:
            f=int(starts[tx]);count=int(counts[tx]);eligible=(groups[processed]%2==groups[tx]%2)&(groups[processed]!=groups[tx])
            coupling=gain[processed,tx,None]*rx*path[processed,f:f+count]*eligible[:,None]
            denominator[:,f:f+count]+=coupling*float(powers[tx])/count
        eligible=(groups[processed]%2==groups[current]%2)&(groups[processed]!=groups[current])
        kappa=gain[processed,current,None]*rx*path[processed,:]*eligible[:,None]
        exposure=kappa/denominator*slots
        weights=np.exp(-np.abs(rates[processed]/demands[processed]-1))
        x=np.log1p((weights[:,None]*exposure).sum(axis=0));u=np.log1p(exposure.sum(axis=0))
        if not np.isfinite(x).all() or not np.isfinite(u).all():raise ValueError('nonfinite saved-array feature arithmetic')
        return dict(status='available',X=x,U=u,processed=processed.copy(),noise_W=noise,
            X_minus_U_range=float(np.ptp(x-u)),X_minus_U_norm=float(np.linalg.norm(x-u)),
            X_minus_U_max_abs=float(np.max(np.abs(x-u))),prefix_ratio=rates[processed]/demands[processed])
    except (KeyError,TypeError,ValueError,IndexError,OverflowError) as error:return unavailable(error)


def average_tie_ranks(values):
    values=np.asarray(values,dtype=np.float64);order=np.argsort(values,kind='mergesort');ranks=np.empty(len(values),dtype=np.float64)
    start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        ranks[order[start:end]]=(start+end-1)/2+1;start=end
    return ranks


def spearman(hazard,damage):
    hazard=np.asarray(hazard,dtype=np.float64);damage=np.asarray(damage,dtype=np.float64)
    if len(hazard)!=len(damage) or not np.isfinite(hazard).all() or not np.isfinite(damage).all():return dict(value=None,reason='missing_or_nonfinite')
    if len(hazard)<2:return dict(value=None,reason='fewer_than_two_legal_frequencies')
    if float(np.ptp(hazard))<=EPS:return dict(value=None,reason='constant_hazard')
    if float(np.ptp(damage))<=EPS:return dict(value=None,reason='constant_damage')
    x=average_tie_ranks(hazard);y=average_tie_ranks(damage);x-=x.mean();y-=y.mean()
    return dict(value=float(np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y))),reason=None)


def consequences(prefix,demands,indices,after_capacity):
    demands=np.asarray(demands,dtype=np.float64);rates=np.asarray(prefix['rates']);after=np.asarray(after_capacity)
    n=len(demands);k=prefix['beam_idx'];previous=np.asarray(indices)[:k];current=int(indices[k]);beta=float(demands.mean())
    if after.shape!=(n,) or not np.isfinite(after).all() or (after<0).any():raise ValueError('candidate capacity unavailable')
    before_si=np.minimum(rates[previous]/demands[previous],1);after_si=np.minimum(after[previous]/demands[previous],1)
    damage=float(np.maximum(0,before_si-after_si).sum()/n)
    signed=float((_sgm(after[previous],demands[previous],beta)-_sgm(rates[previous],demands[previous],beta)).sum()/n)
    ratios=rates[previous]/demands[previous];affected=after[previous]!=rates[previous]
    bins=[]
    for name,mask in [('under_1',ratios<1),('from_1_to_1p1',(ratios>=1)&(ratios<1.1)),('at_least_1p1',ratios>=1.1)]:
        selected=mask&affected;mass=float(demands[previous][selected].sum());prefix_mass=float(demands[previous].sum())
        bins.append(dict(bin=name,receiver_count=int(selected.sum()),demand_mass_bps=mass,
            fraction_of_processed_demand=mass/prefix_mass if prefix_mass>0 else None))
    total_mass=float(demands[previous][affected].sum());prior_mass=float(demands[previous].sum())
    return dict(SI_damage=damage,SGM_signed_change=signed,current_SI=float(min(after[current]/demands[current],1)),
        current_SGM=float(_sgm(after[[current]],demands[[current]],beta)[0]),affected_bins=bins,
        affected_all=dict(receiver_count=int(affected.sum()),demand_mass_bps=total_mass,
            fraction_of_processed_demand=total_mass/prior_mass if prior_mass>0 else None))


def analyze_state(state_id,mother,prefix,demands,indices,groups,kernel,physics,queries,selected_points):
    f=features(prefix,demands,indices,groups,kernel,physics)
    if f['status']!='available':return unavailable_state(state_id,mother,prefix.get('beam_idx'),f['reason'])
    groups_out={(count,float(power)):dict(count=count,power_w=power,frequencies=[],HX=[],HU=[],damage=[]) for count in range(1,11) for power in range(5,51,5)}
    selected={};points=set();bins_sum=np.zeros((3,3));valid=0;feature_different=f['X_minus_U_max_abs']>EPS
    n=len(demands);k=prefix['beam_idx'];current=int(indices[k]);occupied=np.zeros((8,100),bool)
    for j in indices[:k]:
        start=int(prefix['starts'][j]);count=int(prefix['counts'][j]);group=int(groups[j])
        if occupied[group,start:start+count].any():return unavailable_state(state_id,mother,k,'prefix slot conflict')
        occupied[group,start:start+count]=True
    reserve=int(np.count_nonzero(np.asarray(groups)[np.asarray(indices)[k:]]==groups[current]))-1
    free=~occupied[int(groups[current])];available=3000-float(np.asarray(prefix['powers']).sum())-5*(n-k-1)
    expected={(start,count,float(power)) for start in range(100) for count in range(1,11)
        if start+count<=100 and free[start:start+count].all() and count<=int(free.sum())-reserve
        for power in range(5,51,5) if power<=available}
    for point,capacity in queries:
        start,count,power=point
        if point in points:raise ValueError('duplicate original legal action')
        points.add(point)
        if type(start)is not int or type(count)is not int or not 0<=start<=100-count or not 1<=count<=10 or power not in range(5,51,5):raise ValueError('non-original action in margin reader')
        change=consequences(prefix,demands,indices,capacity);hx=float(power/count*f['X'][start:start+count].sum());hu=float(power/count*f['U'][start:start+count].sum())
        row=groups_out[(count,float(power))];row['frequencies'].append(start);row['HX'].append(hx);row['HU'].append(hu);row['damage'].append(change['SI_damage']);valid+=1
        bins_sum[:,:2]+=np.asarray([[b['receiver_count'],b['demand_mass_bps']] for b in change['affected_bins']])
        if len(f['processed']):bins_sum[:,2]+=np.asarray([b['fraction_of_processed_demand'] for b in change['affected_bins']])
        for name,selected_point in selected_points.items():
            if tuple(selected_point)==tuple(point):selected[name]=dict(action=dict(start=start,count=count,power_w=float(power)),HX=hx,HU=hu,**change)
    defined_x=[];defined_u=[];paired=[];nontrivial=0;records=[]
    for row in groups_out.values():
        rx=spearman(row['HX'],row['damage']);ru=spearman(row['HU'],row['damage']);row.update(rho_X=rx['value'],rho_U=ru['value'],undefined_X=rx['reason'],undefined_U=ru['reason'],legal_frequency_count=len(row['frequencies']))
        row['degeneracy_flags']=dict(fewer_than_two_frequencies=len(row['frequencies'])<2,
            constant_HX=not row['HX'] or float(np.ptp(row['HX']))<=EPS,
            constant_HU=not row['HU'] or float(np.ptp(row['HU']))<=EPS,
            constant_damage=not row['damage'] or float(np.ptp(row['damage']))<=EPS)
        varying_x=len(row['HX'])>=2 and np.ptp(row['HX'])>EPS
        row['nontrivial_X_vs_U']=bool(varying_x and feature_different)
        nontrivial+=int(row['nontrivial_X_vs_U'])
        if rx['value'] is not None:defined_x.append(rx['value'])
        if ru['value'] is not None:defined_u.append(ru['value'])
        if rx['value'] is not None and ru['value'] is not None:paired.append(rx['value']-ru['value'])
        row['vector_SHA']=hashlib.sha256(b''.join(np.asarray(row[key],dtype='<f8').tobytes() for key in ('HX','HU','damage'))).hexdigest()
        row['ranges']={key:[float(min(row[key])),float(max(row[key]))] if row[key] else None for key in ('HX','HU','damage')}
        for key in ('HX','HU','damage'):row.pop(key)
        records.append(row)
    missing_selected=sorted(set(selected_points)-set(selected));table_complete=points==expected
    bins_average=(bins_sum/valid).tolist() if valid else None
    if bins_average is not None and not len(f['processed']):
        for row in bins_average:row[2]=None
    comparison=None
    if 'original' in selected and 'reward_max' in selected:
        comparison={key:selected['reward_max'][key]-selected['original'][key] for key in ('current_SI','current_SGM','SI_damage','SGM_signed_change')}
    return dict(status='available' if valid and not missing_selected and table_complete else 'unavailable',state_id=state_id,mother=mother,prefix=prefix['beam_idx'],
        reason=None if valid and not missing_selected and table_complete else 'incomplete/misaligned legal table or selected action',groups=records,valid_actions=valid,missing_selected=missing_selected,
        expected_legal_actions=len(expected),legal_table_complete=table_complete,
        rho_X=float(np.mean(defined_x)) if defined_x else None,rho_U=float(np.mean(defined_u)) if defined_u else None,
        paired_rho_X_minus_U=float(np.mean(paired)) if paired else None,defined_X_groups=len(defined_x),defined_U_groups=len(defined_u),paired_defined_groups=len(paired),
        group_coverage=dict(both_defined=len(paired),X_only=len(defined_x)-len(paired),U_only=len(defined_u)-len(paired),
            neither_defined=100-len(defined_x)-len(defined_u)+len(paired),unavailable=0),
        nontrivial_groups=nontrivial,X=f['X'].tolist(),U=f['U'].tolist(),X_minus_U_range=f['X_minus_U_range'],X_minus_U_norm=f['X_minus_U_norm'],
        affected_bins_equal_action_mean=bins_average,
        affected_all_equal_action_mean=[sum(row[c] for row in bins_average) if all(row[c] is not None for row in bins_average) else None for c in range(3)] if bins_average is not None else None,
        affected_fraction_status='defined' if len(f['processed']) else 'undefined_no_prior_demand',
        affected_bin_columns=['receiver_count','demand_mass_bps','fraction_of_processed_demand'],
        affected_definition='exact saved capacity inequality, including floating-point tiny changes; each action/receiver counted once; no added effect threshold',
        selected_actions=selected,reward_max_minus_original=comparison)


def source_audit(environment_path,schema_path,expected_environment_SHA,expected_schema_SHA):
    try:
        env_raw=Path(environment_path).read_bytes();schema_raw=Path(schema_path).read_bytes()
        if hashlib.sha256(env_raw).hexdigest()!=expected_environment_SHA or hashlib.sha256(schema_raw).hexdigest()!=expected_schema_SHA:raise ValueError('frozen S10 source SHA differs')
        tree=ast.parse(env_raw.decode('utf-8-sig'));schema=ast.parse(schema_raw.decode('utf-8-sig'))
        observation=next(node for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='_observation')
        attrs=sorted({node.attr for node in ast.walk(observation) if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self'})
        mapping=next(ast.literal_eval(node.value) for node in schema.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='GROUPS' for target in node.targets))
        explicit_missing='rates' not in attrs and mapping['S10']==(True,False)
        return dict(status='available',environment_SHA=expected_environment_SHA,schema_SHA=expected_schema_SHA,S10_width_relation_flags=list(mapping['S10']),
            observation_attributes=attrs,prior_receiver_margin_not_explicit=explicit_missing,
            structural_support=explicit_missing,claim='S10 exposes current incoming interference, current-group occupancy, request/global context and width; prior receiver C_j/D_j margin-weighted outgoing exposure is not an explicit field.',
            limitation='No claim of mathematical unrecoverability or proved observation aliasing; requires root source review.')
    except (OSError,ValueError,KeyError,StopIteration,SyntaxError) as error:return unavailable(error)


def summarize(states,audit,a1_summary):
    mothers=[]
    for mother in range(20):
        selected=[state for state in states if state['mother']==mother]
        row=dict(mother=mother,represented_positions=len(selected),unavailable_positions=sum(s['status']!='available' for s in selected))
        row['group_coverage']={key:sum(s.get('group_coverage',{}).get(key,100 if key=='unavailable' and s['status']!='available' else 0) for s in selected)
            for key in ('both_defined','X_only','U_only','neither_defined','unavailable')}
        for field in ('rho_X','rho_U','paired_rho_X_minus_U'):
            values=[s[field] for s in selected if s['status']=='available' and s.get(field) is not None]
            row[field]=float(np.mean(values)) if values else None;row[field+'_defined_positions']=len(values)
        bins=[s['affected_bins_equal_action_mean'] for s in selected if s['status']=='available' and s.get('affected_bins_equal_action_mean') is not None]
        row['affected_bins_equal_state_mean']=[[
            float(np.mean([value[b][column] for value in bins if value[b][column] is not None]))
            if any(value[b][column] is not None for value in bins) else None
            for column in range(3)] for b in range(3)] if bins else None
        mass=[s['affected_all_equal_action_mean'] for s in selected if s.get('affected_all_equal_action_mean') is not None]
        row['affected_all_equal_state_mean']=[float(np.mean([v[c] for v in mass if v[c] is not None])) if any(v[c] is not None for v in mass) else None for c in range(3)] if mass else None
        mothers.append(row)
    reports={}
    for field in ('rho_X','rho_U','paired_rho_X_minus_U'):
        values=[m[field] for m in mothers if m[field] is not None]
        reports[field]=dict(mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None,
            positive_mothers=sum(v>0 for v in values),negative_mothers=sum(v<0 for v in values),zero_mothers=sum(v==0 for v in values),undefined_mothers=20-len(values))
    complete=len(states)==60 and all(sum(s['mother']==m for s in states)==3 for m in range(20))
    raw_valid=complete and all(s['status']=='available' for s in states)
    directional=(reports['rho_X']['mean'] is not None and reports['rho_X']['mean']>0 and reports['paired_rho_X_minus_U']['mean'] is not None and reports['paired_rho_X_minus_U']['mean']>0)
    mechanistic=any(abs(action['SI_damage'])>1e-6 or abs(action['SGM_signed_change'])>1e-6 for state in states for action in state.get('selected_actions',{}).values())
    nontrivial=any(state.get('nontrivial_groups',0)>0 for state in states)
    a1_complete=a1_summary.get('status')=='complete_verified';a1_gate=bool(a1_summary.get('reward_max_causal_single_intervention_gate',{}).get('passed') or a1_summary.get('terminal_SGM_selected_oracle_mechanism_gate',{}).get('passed'))
    eligible=bool(a1_complete and a1_gate and raw_valid and audit.get('structural_support') and nontrivial and directional and mechanistic)
    alignment_defined=reports['rho_X']['mean'] is not None and reports['paired_rho_X_minus_U']['mean'] is not None
    paired_states=[s['paired_rho_X_minus_U'] for s in states if s.get('paired_rho_X_minus_U') is not None]
    all_mass=[m['affected_all_equal_state_mean'] for m in mothers if m['affected_all_equal_state_mean'] is not None]
    return dict(schema='W29A1MarginEvidenceV1',status='descriptive_complete' if raw_valid and audit.get('status')=='available' and alignment_defined else 'evidence_unavailable',represented_states=len(states),mothers=mothers,
        paired_equal_state_mean=float(np.mean(paired_states)) if paired_states else None,
        affected_all_equal_mother_mean=[float(np.mean([v[c] for v in all_mass if v[c] is not None])) if any(v[c] is not None for v in all_mass) else None for c in range(3)] if all_mass else None,
        equal_mother_reports=reports,available_states=sum(s['status']=='available' for s in states),nontrivial_groups=sum(s.get('nontrivial_groups',0) for s in states),
        nontrivial_states=sum(s.get('nontrivial_groups',0)>0 for s in states),
        defined_groups_X=sum(s.get('defined_X_groups',0) for s in states),defined_groups_U=sum(s.get('defined_U_groups',0) for s in states),
        undefined_or_constant_groups_X=sum(100-s.get('defined_X_groups',0) for s in states),undefined_or_constant_groups_U=sum(100-s.get('defined_U_groups',0) for s in states),
        source_audit=audit,structural_support=bool(audit.get('structural_support')) and raw_valid,nontriviality=nontrivial,
        directional_support=bool(directional),mechanistic_presence=bool(mechanistic),A1_complete=a1_complete,A1_registered_gate_passed=a1_gate,
        conditional_B_preparation_evidence=eligible,root_independent_admission_still_required=True,training_admitted=False,auto_launch=False,
        Q_status='not_available',
        actual_calls=dict(PHY=0,kernel_build=0,Actor_forward=0,observation=0,parameter_updates=0),
        interpretation='Prespecified descriptive reused-prefix association only; no causal bottleneck identification, Q error, independent confirmation, or predicted B gain.')


def readonly_context(config,facts,mother):
    """Pure wire decoding/identity checks. Deliberately no stage importer/geometry."""
    from types import SimpleNamespace
    from satellite_rl.shared_interface.content import read_wire
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS,PHYSICS_SHA
    base=Path(config['input_root']).resolve()
    def decoded(ref):
        path=(base/ref['member']).resolve()
        if base not in path.parents:raise ValueError('immutable input outside registered root')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('immutable input bytes differ')
        return read_wire(raw)
    case=decoded(mother['case_ref']);layout=decoded(mother['layout_ref'])
    if (case['case_id']!=mother['case_id'] or case['content_sha256']!=mother['case_content_SHA'] or
            layout['content_sha256']!=mother['layout_content_SHA'] or layout['parent_case_SHA']!=case['content_sha256']):raise ValueError('accepted case/layout parent identity')
    ids=tuple(layout['beam_ids']);order=list(mother['processing_index_to_beam_id']);n=len(ids)
    if n!=mother['N'] or not 0<n<=200 or len(set(ids))!=n or len(order)!=n or set(order)!=set(ids):raise ValueError('saved actual beam/order identity')
    indices=np.asarray([ids.index(beam) for beam in order],dtype=np.int64)
    by_id={beam:i%8 for i,beam in enumerate(order)};groups=np.asarray([by_id[beam] for beam in ids],dtype=np.int64)
    demands=np.asarray(layout['request_bps_by_beam_int64'],dtype=np.float64)
    if int(np.asarray(layout['request_bps_by_beam_int64']).sum())!=int(np.asarray(case['request_bps_int64']).sum()):raise ValueError('accepted demand conservation')
    if facts.bundle is not None:kernel=facts._bundle_value(facts.bundle['kernels'][mother['case_id']])
    else:kernel,_=facts._read(facts.operations[mother['case_id']+'/kernel'])
    if kernel['layout_SHA']!=layout['content_sha256'] or kernel['physics_SHA']!=PHYSICS_SHA:raise ValueError('saved kernel physical/layout identity')
    identity=dict(layout_SHA=layout['content_sha256'],physics_SHA=PHYSICS_SHA,order=order,groups=groups.tolist())
    return SimpleNamespace(N=n,ids=ids,indices=indices,groups=groups,demands=demands,kernel=kernel,physics=PHYSICS,identity=identity,layout=layout)


def analyze(config_paths,rule_path,a1_summary_path,a1_states_path,*,source_audit_refs,accepted_analysis_refs,semantics_path,kernel_clarification_path,path_map=None,output_dir=None):
    """Consume only completed SHA-accepted A1 artifacts; no live science reading."""
    from .a1_analysis import _mapped,_state_values,_qid
    from .a1_inputs import A0Facts
    from .a1_stream import StreamStore
    started=time.monotonic();path_map=path_map or {};rule_raw=Path(rule_path).read_bytes()
    if hashlib.sha256(rule_raw).hexdigest()!=RULE_SHA:raise ValueError('prespecified evidence rule SHA differs')
    for path,sha in [(semantics_path,SEMANTICS_SHA),(kernel_clarification_path,KERNEL_CLARIFICATION_SHA)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('prespecified arithmetic/semantics clarification SHA differs')
    a1_raw=Path(a1_summary_path).read_bytes();a1=json.loads(a1_raw)
    state_raw=Path(a1_states_path).read_bytes();state_metrics=json.loads(state_raw)
    if hashlib.sha256(a1_raw).hexdigest()!=accepted_analysis_refs['summary_SHA'] or hashlib.sha256(state_raw).hexdigest()!=accepted_analysis_refs['states_SHA']:raise ValueError('root-accepted A1 analysis output SHAs required')
    if a1.get('status')!='complete_verified' or a1.get('states')!=60:raise ValueError('complete verified A1 required before evidence effects are read')
    metrics={(row['mother'],row['prefix']):row for row in state_metrics}
    if len(metrics)!=60:raise ValueError('fixed60 terminal state summaries required')
    audit=source_audit(**source_audit_refs);states=[];inputs=[]
    for config_path in config_paths:
        raw=Path(config_path).read_bytes();original=json.loads(raw);lane=original['lane']
        if hashlib.sha256(raw).hexdigest()!=a1['config_SHAs'][str(lane)] or original['binding']!=a1['binding']:raise ValueError('original A1 config/binding differs')
        config=_mapped(original,path_map);root=Path(config['root'])/'science'
        expected=next(row for row in a1['lane_costs'] if row['lane']==lane)
        journal=root/'a1.journal'
        if hashlib.sha256(journal.read_bytes()).hexdigest()!=expected['journal_SHA']:raise ValueError('A1 journal differs from accepted reader result')
        store=StreamStore(root,config['binding'],config['science_limits'],storage_limit=config['science_storage_bytes'],read_only=True)
        try:facts=A0Facts(config);facts_error=None
        except (KeyError,ValueError,OSError) as error:facts=None;facts_error=str(error)
        inputs.append(dict(lane=lane,config_SHA=hashlib.sha256(raw).hexdigest(),journal_SHA=expected['journal_SHA']))
        for mother in config['mothers']:
            if mother['mother']%4!=lane:continue
            n=mother['N'];positions=sorted(set(min(n-1,int(fraction*n)) for fraction in (.125,.5,.875)))
            try:
                if facts is None:raise ValueError(facts_error)
                ctx=readonly_context(config,facts,mother)
            except (KeyError,ValueError,OSError) as error:
                states.extend(unavailable_state(mother['case_id']+'/'+str(prefix),mother['mother'],prefix,error) for prefix in positions);continue
            for position in positions:
                sid=mother['case_id']+'/'+str(position)
                try:
                    prefix,snapshot_ref,cache=facts.state(ctx,mother,position);values=_state_values(store,sid)
                    selection=values[sid+'/candidate_slots'];metric=metrics[(mother['mother'],position)]
                    chosen={}
                    for name,slot in [('original',0),('reward_max',3),('oracle',metric['oracle_slot'])]:
                        item=next(row for row in selection['slots'] if row['slot']==slot)
                        if item.get('action') is None:raise ValueError('registered selected action unavailable')
                        f,l,p=item['action'];chosen[name]=(f,l+1,float((p+1)*5))
                    def queries():
                        for start in range(100):
                            for count in range(1,11):
                                if start+count>100:continue
                                for p in range(5,51,5):
                                    point=(start,count,float(p));old=cache.get(point)
                                    if old is not None:
                                        if not old['spec']['valid']:continue
                                        value=old['result']
                                    else:
                                        value=values[_qid(sid,*point)]
                                        if value['status']=='invalid':continue
                                    yield point,value['capacity_bps']
                    result=analyze_state(sid,mother['mother'],prefix,ctx.demands,ctx.indices,ctx.groups,ctx.kernel,ctx.physics,queries(),chosen)
                    result['registered_terminal_deltas']={name:{key:metric[field][key]-metric['reference'][key] for key in ('SI','SGM','power')} for name,field in [('reward_max','causal'),('oracle','oracle')]}
                    result['terminal_outcomes_used_in_feature']=False;states.append(result)
                except (KeyError,ValueError,OSError,IndexError) as error:states.append(unavailable_state(sid,mother['mother'],position,error))
    if len({row['state_id'] for row in states})!=60 or len(states)!=60:raise ValueError('all60 fixed states must remain represented')
    summary=summarize(states,audit,a1);summary.update(rule_SHA=RULE_SHA,A1_summary_SHA=hashlib.sha256(a1_raw).hexdigest(),
        semantics_addendum_SHA=SEMANTICS_SHA,saved_kernel_arithmetic_clarification_SHA=KERNEL_CLARIFICATION_SHA,
        A1_states_SHA=hashlib.sha256(state_raw).hexdigest(),inputs=inputs,arithmetic_IO_seconds=time.monotonic()-started,
        margin_mass_implementation='Exact saved Cprime != C; count unique prior receivers per action and sum their D, report fraction of all processed D; floating-point tiny changes count; all legal actions ->state->mother equal means. Separate from fixed1e-6 SI/SGM mechanism gate.',
        paired_aggregation='Same fixed(l,p) groups where BOTH rhoX and rhoU defined: group differences ->state mean ->mother defined-state mean ->defined-mother mean. Never subtract means from different group sets.')
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        for filename,value in [('margin_summary.json',summary),('margin_states.json',states)]:
            path=out/filename
            if path.exists():raise ValueError('preserve existing analysis output')
            path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return summary,states
