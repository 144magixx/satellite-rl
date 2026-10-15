"""Independent read-only A1 result verification and frozen gates. No PHY/Actor."""
import argparse
import base64
import hashlib
import json
import math
import os
import pickle
import zlib
from pathlib import Path
import numpy as np
from .ledger import Halt,digest
from . import core,a1_core,a1_policy
from .a1_inputs import A0Facts
from .a1_stream import StreamStore,_frames
from satellite_rl.t0_width.environment import sgm_values,legacy_repaired_reward


def require(condition,message):
    if not condition:raise Halt(message)


def same(actual,expected,label):
    if isinstance(expected,np.ndarray):
        require(isinstance(actual,np.ndarray) and actual.dtype==expected.dtype and actual.shape==expected.shape,label+' dtype/shape')
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12,err_msg=label)
    elif isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+' keys')
        for key,value in expected.items():same(actual[key],value,label+'/'+key)
    elif isinstance(expected,(list,tuple)):
        require(len(actual)==len(expected),label+' length')
        for index,value in enumerate(expected):same(actual[index],value,label+'/'+str(index))
    elif isinstance(expected,(float,np.floating)):
        require(float(actual)==float(expected) or math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-12),label+' value')
    else:require(actual==expected,label+' value')


def verify_query(ctx,state,result,action):
    f,count,watts=action;spec=core.candidate_spec(ctx,state,f,count,watts,'original')
    require(spec['valid'],'query outside original legal domain')
    require(result['status']=='completed' and result['state_SHA']==core.content_sha(state),'query status/source')
    same(result['action'],dict(start=f,count=count,power_w=float(watts)),'actual action')
    k=state['beam_idx'];index=int(ctx.indices[k]);rates=result['capacity_bps']
    require(isinstance(rates,np.ndarray) and rates.dtype==np.float64 and rates.shape==(ctx.N,) and np.isfinite(rates).all() and (rates>=0).all(),'full capacity schema')
    require((rates[ctx.indices[k+1:]]==0).all(),'unprocessed capacity must be zero')
    require(result['beam_id']==ctx.ids[index],'current beam identity')
    si=np.minimum(rates/ctx.demands,1);sgm=sgm_values(rates,ctx.demands,float(ctx.demands.mean()));processed=ctx.indices[:k+1]
    total=float(si[processed].sum());increment=total-state['total_satisfaction']
    reward=legacy_repaired_reward(float(rates[index]),float(ctx.demands[index]),float(ctx.demands.mean()),increment,0,count)
    same(result['reward'],float(reward),'reward')
    same(result['prefix_SI_sum'],total,'prefix SI');same(result['prefix_SGM_sum'],float(sgm[processed].sum()),'prefix SGM')
    same(result['current'],dict(capacity_bps=float(rates[index]),demand_bps=float(ctx.demands[index]),SI=float(si[index]),SGM=float(sgm[index])),'current service')
    same(result['reward_parts'],dict(SI_threshold=1. if increment>=1 else -1.,current_SGM=float(sgm[index]),
        deviation_penalty=max(abs(float(rates[index])-float(ctx.demands[index]))-5e7,0.)/1e9,
        occupancy_penalty=0.,satisfaction_increment=increment),'reward components')
    same(result['total_power_w'],float(state['powers'].sum())+watts,'actual total power')
    same(result['remaining_reserved_power_w'],5*(ctx.N-k-1),'future reserve')
    require(result['total_power_w']+result['remaining_reserved_power_w']<=3000,'power constraint')


def verify_terminal(ctx,state,saved):
    require(state['beam_idx']==ctx.N,'incomplete suffix')
    occupied=np.zeros((8,100),bool)
    for i in range(ctx.N):
        f=int(state['starts'][i]);count=int(state['counts'][i]);p=float(state['powers'][i]);g=int(ctx.groups[i])
        require(1<=count<=10 and 0<=f<=100-count and p in range(5,51,5),'terminal original action bounds')
        require(not occupied[g,f:f+count].any(),'terminal slot conflict');occupied[g,f:f+count]=True
    require(np.array_equal(occupied,state['occupied']) and float(state['powers'].sum())<=3000,'terminal resource accounting')
    same(saved,a1_core.terminal_summary(ctx,state),'terminal original full N arrays')
    require(saved['complete'] and saved['zero_slot_count']==0,'terminal full allocation')
    return dict(SI=float(saved['SI_mean']),SGM=float(saved['SGM_mean']),power=float(saved['total_power_w']))


def _state_values(store,sid):
    prefix=hashlib.sha256(sid.encode()).hexdigest()[:24]+'.';values={}
    for name,info in store.chunks.items():
        if not name.startswith(prefix):continue
        require(info['sealed'],'state has unsealed chunk');path=store._chunk_path(name)
        require(path.stat().st_size==info['bytes'] and digest(path)==info['sha'],'chunk bytes/SHA')
        rows=0
        with path.open('rb') as stream:
            for offset,length,sha,raw in _frames(stream):
                record=pickle.loads(zlib.decompress(raw));op_id=record['id'];op=store.operations.get(op_id)
                require(op is not None and op.get('state')==sid and op.get('done')==dict(chunk=name,offset=offset,length=length,sha=sha),'row offset/index/source')
                require(op_id not in values,'duplicate numerical row');values[op_id]=record['value'];rows+=1
        require(rows==info['rows']<=256,'chunk rows')
    return values


def _qid(sid,f,count,p):return sid+'/enum/'+str(f)+'/'+str(count)+'/'+str(int(p))


def verify_state(store,facts,ctx,mother,prefix):
    sid=mother['case_id']+'/'+str(prefix);require(sid in store.closed_states,'state missing close record')
    state,snapshot_ref,cache=facts.state(ctx,mother,prefix);values=_state_values(store,sid);expected=[]
    def alias(op_id,source):
        op=store.operations.get(op_id);require(op is not None and op.get('alias')==source,'A0 alias provenance mismatch');expected.append(op_id)
    def get(op_id,charges=None):
        require(op_id in values,'missing durable row '+op_id)
        if charges is not None:same(store.operations[op_id]['charges'],charges,'row charges')
        expected.append(op_id);return values[op_id]
    alias(sid+'/A0_snapshot',snapshot_ref)
    mask=a1_core.action_mask(ctx,state)
    law=get(sid+'/Actor_law',dict(forward=1,observation=1))
    fl,power=a1_policy.normalize_probabilities(law['frequency_slots'],law['power'],mask)
    require(law['forward_calls']==1,'initial Actor forward provenance')
    best=None;best_reward=None;enumeration={};tally=dict(structural=0,A0_alias=0,new_valid=0,new_invalid=0,valid=0,invalid=0)
    for f in range(100):
        for count in range(1,11):
            if f+count>100:continue
            for watts in range(5,51,5):
                tally['structural']+=1;point=(f,count,float(watts));op_id=_qid(sid,*point)
                spec=core.candidate_spec(ctx,state,*point,'original');old=cache.get(point)
                if old is not None:
                    alias(op_id,old['source_ref']);tally['A0_alias']+=1
                    require(old['spec']['valid']==spec['valid'],'A0 legality disagrees')
                    result=a1_core.result_from_a0(ctx,state,old['result']) if spec['valid'] else dict(status='invalid',spec=spec,physical_calls=0)
                else:
                    result=get(op_id);cost=store.operations[op_id]['charges']
                    if spec['valid']:
                        require(cost.get('phy')==1 and cost.get('enumeration')==1 and cost.get('coefficient') in (0,1) and set(cost)=={'phy','enumeration','coefficient'},'enumeration cost')
                        tally['new_valid']+=1
                    else:require(cost=={},'invalid row charged');tally['new_invalid']+=1
                if not spec['valid']:
                    require(result['status']=='invalid' and result['physical_calls']==0,'illegal row not missing/invalid')
                    if old is None:same(result['spec'],spec,'illegal reasons')
                    tally['invalid']+=1;continue
                verify_query(ctx,state,result,point);tally['valid']+=1;enumeration[point]=result
                action=(f,count-1,watts//5-1);reward=float(result['reward'])
                if best is None or reward>best_reward or reward==best_reward and action<best:best=action;best_reward=reward
    require(tally['structural']==9550,'structural row count')
    selection=get(sid+'/candidate_slots',{})
    original=mother['actions_indices_by_processing_order'][prefix]
    expected_selection=a1_policy.select_slots(mother['case_id'],prefix,original,best,fl,power,mask)
    same(selection,expected_selection,'candidate slots/proposals/RNG')
    require(not selection['Q_available'] and selection['slots'][2]['status']=='not_available' and len(selection['unique_actions'])<=5,'missing Q boundary')
    action_results={};terminals=[]
    for candidate in selection['unique_actions']:
        owner=candidate['slot'];f,l,p=candidate['action'];point=(f,l+1,float((p+1)*5));first=enumeration[point]
        first_probability=a1_policy.action_probability(fl,power,(f,l,p));branch_results=[]
        for repeat in (0,1):
            branch=sid+'/suffix/slot'+str(owner)+'/repeat'+str(repeat);current=a1_core.advance_state(ctx,state,first)
            rewards=[first['reward']];logs=[first_probability['log_probability']]
            while current['beam_idx']<ctx.N:
                absolute=current['beam_idx'];step_id=branch+'/step/'+str(absolute)
                result=get(step_id,dict(phy=1,suffix=1,forward=1,coefficient=1,observation=1))
                current_mask=a1_core.action_mask(ctx,current);af,al,ap=result['action_indices'];action=(af,al+1,float((ap+1)*5))
                require(current_mask['frequency_slots'][af,al] and current_mask['power'][ap],'suffix action outside original mask')
                verify_query(ctx,current,result,action)
                same(result['rng'],a1_policy.uniforms(mother['case_id'],prefix,'suffix',repeat=repeat,absolute_beam_index=absolute),'common-uniform suffix key')
                probability=result['probability'];logp=float(probability['log_probability']);prob=float(probability['probability'])
                require(math.isfinite(logp) and logp<=1e-12 and 0<prob<=1 and not probability['log_probability_is_negative_infinity'],'sampled action probability')
                require(math.isclose(math.exp(logp),prob,rel_tol=1e-12,abs_tol=1e-300),'sampled log probability')
                rewards.append(result['reward']);logs.append(logp);current=a1_core.advance_state(ctx,current,result)
            terminal=get(branch+'/terminal',{})
            same(terminal['slot'],owner,'terminal owner');same(terminal['repeat'],repeat,'terminal repeat')
            same(terminal['first_query_ref'],_qid(sid,*point),'cached initial action reference');same(terminal['first_probability'],first_probability,'initial probability')
            ret=terminal['returns'];ordinary=sum(.99**k*float(r) for k,r in enumerate(rewards))
            soft=ordinary-.01*sum(.99**k*float(logs[k]) for k in range(1,len(rewards)))
            same(ret['ordinary_discounted'],ordinary,'discounted return');same(ret['Q_semantic_soft_sample'],soft,'first-entropy-excluded return')
            require(ret['count']==len(rewards) and ret['gamma']==.99 and ret['alpha']==.01,'return step count/constants')
            metric=verify_terminal(ctx,current,terminal['terminal']);branch_results.append(metric)
            terminals.append(dict(mother=mother['mother'],prefix=prefix,slot=owner,repeat=repeat,**metric))
        action_results[owner]={name:float(np.mean([r[name] for r in branch_results])) for name in ('SI','SGM','power')}
    slot_owner={s['slot']:(s['slot'] if s['status']=='active' else s.get('alias_slot')) for s in selection['slots']}
    reference=action_results[slot_owner[0]];causal=action_results[slot_owner[3]]
    oracle_owner=max(action_results,key=lambda slot:(action_results[slot]['SGM'],action_results[slot]['SI'],-slot))
    oracle=action_results[oracle_owner]
    # Verify close bitmap and exact registered row identity sequence, not just its existence.
    if not hasattr(store,'_analysis_closed'):
        store._analysis_closed={};store._analysis_owned={}
        with store.journal_path.open('rb') as stream:
            for _,_,_,raw in _frames(stream):
                event=json.loads(raw)
                if event['type']=='state_closed':store._analysis_closed[event['state']]=event
                elif event['type'] in ('intent','alias'):store._analysis_owned.setdefault(event['state'],set()).add(event['id'])
    close=store._analysis_closed.get(sid)
    require(close is not None and close.get('expected_rows')==len(expected),'state completion expected rows')
    same(close['ordered_ids_SHA'],hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest(),'completion identity order')
    bitmap=base64.b64decode(close['completed_bitmap_base64'],validate=True)
    require(len(bitmap)==(len(expected)+7)//8 and all(bitmap[i//8]&(1<<(i%8)) for i in range(len(expected))),'completion bitmap')
    owned=store._analysis_owned.get(sid,set())
    require(owned==set(expected),'extra or missing registered scientific row')
    return dict(mother=mother['mother'],prefix=prefix,reference=reference,causal=causal,oracle=oracle,oracle_slot=oracle_owner,tally=tally),terminals


def gate(mothers,kind):
    rows=[]
    for mother,states in sorted(mothers.items()):
        reference={key:float(np.mean([s['reference'][key] for s in states])) for key in ('SI','SGM','power')}
        candidate={key:float(np.mean([s[kind][key] for s in states])) for key in ('SI','SGM','power')}
        rows.append(dict(mother=mother,reference=reference,candidate=candidate,delta_SI=candidate['SI']-reference['SI'],delta_SGM=candidate['SGM']-reference['SGM']))
    require(len(rows)==20 and all(len(states)==3 for states in mothers.values()),'gate needs20 mothers and3 positions')
    dsi=float(np.mean([r['delta_SI'] for r in rows]));dsgm=float(np.mean([r['delta_SGM'] for r in rows]))
    si_count=sum(r['delta_SI']>0 for r in rows);sgm_count=sum(r['delta_SGM']>0 for r in rows)
    ref_power=float(np.mean([r['reference']['power'] for r in rows]));candidate_power=float(np.mean([r['candidate']['power'] for r in rows]))
    improvement=(dsi>=.003 and si_count>=12) or (dsgm>=.01 and sgm_count>=12)
    protection=dsi>=-.002 and dsgm>=-.005;power_ok=candidate_power<=1.10*ref_power
    return dict(passed=bool(improvement and protection and power_ok),delta_SI=dsi,delta_SGM=dsgm,
        SI_improved_mothers=si_count,SGM_improved_mothers=sgm_count,improvement_gate=bool(improvement),
        protection_gate=bool(protection),power_gate=bool(power_ok),hard_constraints_passed=True,
        reference_power=ref_power,candidate_power=candidate_power,mothers=rows,
        interpretation='causal single intervention on reused development states with exhaustive search cost' if kind=='causal' else 'hindsight terminal-SGM mechanism oracle; no deployed policy improvement')


def _mapped(value,path_map):
    if isinstance(value,dict):return {key:_mapped(item,path_map) for key,item in value.items()}
    if isinstance(value,list):return [_mapped(item,path_map) for item in value]
    if isinstance(value,str):
        for old,new in sorted(path_map.items(),key=lambda pair:-len(pair[0])):
            if value==old or value.startswith(old.rstrip('/')+'/'):return str(new)+value[len(old):]
    return value


def analyze(config_paths,contract_path,*,path_map=None,expected_binding=None,output_dir=None):
    """Read accepted original configs plus copied result trees; no runtime writes.

    path_map redirects remote canonical roots to local copies without editing the
    frozen config bytes. Config/receipt binding always checks ORIGINAL SHA bytes.
    """
    require(len(config_paths)==4,'exact four lane configs required');path_map=path_map or {}
    contract_raw=Path(contract_path).read_bytes();contract=json.loads(contract_raw);contract_sha=hashlib.sha256(contract_raw).hexdigest()
    require(contract['schema']=='W29A1ScientificContractV1' and not contract['comparisons_and_gates']['training_admitted_by_this_contract'],'A1 finite contract only')
    all_states=[];terminal_rows=[];costs=[];bound=expected_binding;seen_lanes=set();config_shas={};source_shas={}
    global_expected_states=set();loaded_configs=[]
    for config_path in config_paths:
        original_raw=Path(config_path).read_bytes();original=json.loads(original_raw);config_sha=hashlib.sha256(original_raw).hexdigest()
        lane=original['lane'];require(lane not in seen_lanes and lane in range(4),'unique lane ownership');seen_lanes.add(lane)
        config=_mapped(original,path_map);loaded_configs.append(config);config_shas[str(lane)]=config_sha
        if bound is None:bound=original['binding']
        require(original['binding']==bound and bound['contract_SHA']==contract_sha,'source/Actor/A0/data/authority/contract binding differs')
        require(bound['Q_status']=='unavailable_same_node_150000' and len(bound['actor_SHA'])==64,'full Actor provenance and missing Q')
        require([m['mother'] for m in original['mothers']]==list(range(20)) and len({m['case_id'] for m in original['mothers']})==20,'original ordered twenty mothers')
        receipt_path=Path(config['science_admission']);receipt=json.loads(receipt_path.read_bytes())
        require(receipt['root_approved'] and receipt['phase']=='A1' and receipt['mode']=='science' and receipt['config_SHA']==config_sha and receipt['binding']==bound and receipt['lane']==lane,'root science admission/config proof')
        require(receipt['limits']==config['science_limits'] and receipt['engineering_accepted'],'root finite engineering admission')
        for original_path,sha in original['pinned_files'].items():
            local=Path(_mapped(original_path,path_map));require(digest(local)==sha,'pinned original source/data/Actor bytes changed')
            source_shas[original_path]=sha
        root=Path(config['root']);dispatch_path=root/'science_dispatch.pkl'
        with dispatch_path.open('rb') as stream:dispatch=pickle.load(stream)
        require(dispatch['status']=='completed' and dispatch['phase']=='A1' and dispatch['mode']=='science' and dispatch['config_SHA']==config_sha,'actual completed dispatch proof')
        require(dispatch['generation']==receipt['generation'] and dispatch.get('unledgered_seconds',0)==0,'generation/unledgered preparation')
        store=StreamStore(root/'science',config['binding'],config['science_limits'],storage_limit=config['science_storage_bytes'],read_only=True)
        require(not store.unknown() and all('elapsed' in s for s in store.sessions.values()),'unknown operation or unfinished clock')
        require(all(info['sealed'] for info in store.chunks.values()),'unsealed scientific chunks')
        expected_files={store._chunk_path(name).name for name in store.chunks}
        with os.scandir(root/'science'/'chunks') as entries:
            observed={entry.name for entry in entries if entry.is_file(follow_symlinks=False)}
        require(observed==expected_files,'unregistered/orphan scientific chunk')
        for key,value in store.counts.items():require(key in store.limits and value<=store.limits[key],'finite cumulative cost allocation')
        ledger_seconds=sum(float(s['elapsed']) for s in store.sessions.values())+sum(float(d['seconds']) for d in store.debits.values())
        dispatch_seconds=float(dispatch['elapsed_seconds']);history=root/'dispatch_history'
        if history.exists():
            with os.scandir(history) as entries:
                historical=list(entries)
            require(len(historical)<=1000,'bounded dispatch history')
            for entry in historical:
                require(entry.is_file(follow_symlinks=False),'invalid dispatch history entry')
                with open(entry.path,'rb') as stream:old=pickle.load(stream)
                if old.get('mode')!='science':continue
                require(old.get('phase')=='A1' and old.get('config_SHA')==config_sha and 'elapsed_seconds' in old,'unresolved prior science dispatch clock/config')
                dispatch_seconds+=float(old['elapsed_seconds'])
        require(ledger_seconds<=store.limits['seconds'] and dispatch_seconds<=store.limits['seconds'],'actual aggregate lane active seconds exceeded')
        provenance=store.metadata['Actor_provenance']
        require(provenance['policy_SHA']==bound['actor_SHA'] and provenance['env_step']==150000 and provenance['update_step']==148001 and not provenance['Q_constructed'] and not provenance['optimizers_constructed'],'real frozen Actor metadata')
        require(store.metadata['status']=='completed' and store.metadata['scientific_complete']['states']==15,'scientific lane completion metadata')
        facts=A0Facts(config);expected_states=set()
        for mother in original['mothers']:
            if mother['mother']%4!=lane:continue
            # Map only runtime input refs; stable scientific IDs remain original.
            local_mother=config['mothers'][mother['mother']];ctx,actions=facts.mother(local_mother)
            require(actions==mother['actions_indices_by_processing_order'] and ctx.N==mother['N'],'original action and actual-N source')
            for prefix in core.prefix_positions(ctx.N):
                sid=mother['case_id']+'/'+str(prefix);expected_states.add(sid)
                state,terminals=verify_state(store,facts,ctx,local_mother,prefix);all_states.append(state);terminal_rows.extend(terminals)
        require(store.closed_states==expected_states and len(expected_states)==15,'fixed15 state lane ownership')
        require(set(store._analysis_owned)==expected_states and
                set().union(*store._analysis_owned.values())==set(store.operations),'unregistered operation state/ID')
        global_expected_states.update(expected_states)
        costs.append(dict(lane=lane,counts=store.counts,ledger_worker_seconds=ledger_seconds,
            dispatch_worker_seconds=dispatch_seconds,stored_bytes=store.bytes,config_SHA=config_sha,journal_SHA=digest(store.journal_path)))
    require(len(all_states)==60 and len(global_expected_states)==60,'all60 unique states')
    counts={key:sum(cost['counts'].get(key,0) for cost in costs) for key in set().union(*(cost['counts'] for cost in costs))}
    require(counts.get('phy',0)<=635700 and counts.get('enumeration',0)<=515700 and counts.get('suffix',0)<=120000 and counts.get('forward',0)<=120060,'global missing-Q finite cap')
    require(counts.get('kernel',0)==0 and counts.get('prefix',0)==0 and counts.get('update',0)==0,'unregistered kernel/prefix/training work')
    tallies={key:sum(s['tally'][key] for s in all_states) for key in all_states[0]['tally']}
    require(tallies['structural']==573000 and counts.get('enumeration',0)==tallies['new_valid'],'573000 table/accounting agreement')
    require(counts.get('forward',0)==60+counts.get('suffix',0) and counts.get('phy',0)==counts.get('enumeration',0)+counts.get('suffix',0),'actual PHY/Actor transaction count agreement')
    require(sum(c['ledger_worker_seconds'] for c in costs)<=14400 and sum(c['dispatch_worker_seconds'] for c in costs)<=14400,'aggregate worker time cap')
    require(sum(c['stored_bytes'] for c in costs)<=sum(c['science_storage_bytes'] for c in loaded_configs),'registered actual byte accounting')
    mothers={}
    for state in all_states:mothers.setdefault(state['mother'],[]).append(state)
    summary=dict(schema='W29A1ReadonlyAnalysisV1',status='complete_verified',binding=bound,contract_SHA=contract_sha,
        config_SHAs=config_shas,pinned_original_sources=source_shas,states=60,mothers=20,table_rows=tallies,
        counts=counts,lane_costs=costs,completed_branches=len(terminal_rows),Q_status='not_available',
        reward_max_causal_single_intervention_gate=gate(mothers,'causal'),
        terminal_SGM_selected_oracle_mechanism_gate=gate(mothers,'oracle'),
        actual_reader_calls=dict(PHY=0,forward=0,kernel=0,updates=0),
        limitations=['Reused development states; not independent generalization.',
            'Suffix complete Actor CDFs were not stored: no-forward reader verifies recorded common uniforms, chosen action legality and probability/logpi consistency, not an independent Actor replay.',
            'Oracle uses future terminal outcomes and supports mechanism space only; no deployed-policy improvement.',
            'Two paired suffix samples are not exact value truth; Q diagnostics are not available.'],
        training_admitted=False,next_stage_auto_start=False)
    if output_dir is not None:
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        for name,value in [('summary.json',summary),('states.json',all_states),('terminals.json',terminal_rows)]:
            path=output_dir/name
            require(not path.exists(),'analysis output exists; preserve prior result')
            path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--configs',nargs=4,required=True);parser.add_argument('--contract',required=True)
    parser.add_argument('--path-map');parser.add_argument('--expected-binding');parser.add_argument('--output',required=True);args=parser.parse_args()
    path_map=json.loads(Path(args.path_map).read_bytes()) if args.path_map else {}
    expected=json.loads(Path(args.expected_binding).read_bytes()) if args.expected_binding else None
    print(json.dumps(analyze(args.configs,args.contract,path_map=path_map,expected_binding=expected,output_dir=args.output),ensure_ascii=True))


if __name__=='__main__':main()
