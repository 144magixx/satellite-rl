"""Concrete future engineering checks. No callback can supply a pass verdict."""
import copy
import json
import os
from pathlib import Path
import pickle
from time import perf_counter
from unittest.mock import patch
import numpy as np
from satellite_rl.t0_width.data import read_json_ref,file_sha
from satellite_rl.t0_width.logging import content_sha,atomic_json
from .checks import catalog,fixture_manifest,validate_catalog,construct_fixture,REQUIRED_COVERAGE,prefix_recipe
from .binding import pack_environment,unpack_environment
from .transaction import Journal,UnknownCost,snapshot,state_sha,fork,decision,fsync_directory
from .worker import one_decision
from .loading import load_policy
from .proposals import frozen_s10_probabilities,frozen_s10_forward
from .runtime import save_pickle

class PlannedPowerLoss(BaseException):pass
class QualificationStopped(Exception):pass

def run_check(check,folder,plan,agent,stop_requested,*,fresh):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);binding=folder/'binding.pkl';binding_ref=folder/'binding_ref.json'
    spec=next(s for s in fixture_manifest()['rows'] if s['id']==check['fixture_id'])
    if binding_ref.exists():
        ref=json.loads(binding_ref.read_text());env=unpack_environment(pickle.loads(read_json_ref_bytes(ref)))
    elif fresh:
        # A marker precedes constructor/kernel. A missing binding after marker
        # is unknown setup, not permission to reconstruct without accounting.
        marker=folder/'setup.intent'
        if marker.exists():raise UnknownCost('fixture setup intent already exists; no free kernel rebuild')
        with marker.open('xb') as stream:stream.write(content_sha(check).encode());stream.flush();os.fsync(stream.fileno())
        fsync_directory(folder);env=construct_fixture(spec);save_pickle(binding,pack_environment(env))
        atomic_json(binding_ref,dict(path=str(binding),sha256=file_sha(binding)));fsync_directory(folder)
    else:raise UnknownCost('active fixture binding not closed; preserve unknown setup')
    cap=check['worst_case'];limits=dict(query=cap['query'],commit=cap['commit'],forward=cap['forward'],proposal=1)
    identity=dict(plan_SHA=content_sha(plan),check_SHA=content_sha(check),check_id=check['check_id'])
    def reopen():return Journal(folder/'current.pkl',identity,limits)
    journal=reopen();case_id=spec['id'];index=check['check_index'];method=check['method'];kind=check['kind']
    if journal.value['transaction'] and journal.value['transaction'].get('pending'):raise UnknownCost('prior check effect intent is unresolved; no retry')
    observation=env._observation();before=state_sha(snapshot(env));proof={}
    proposals=lambda obs,mask:frozen_s10_probabilities(agent,obs,mask)
    def execute(stop=lambda:False):
        result=one_decision(env,observation,method,case_id,index,journal,proposals if method.startswith('P') else None,lambda:stop_requested() or stop())
        if result is None and stop_requested():raise QualificationStopped()
        return result
    if kind=='method':
        assert execute() is not None
        assert journal.value['attempts']['commit']==1
        assert journal.value['attempts']['query']<=cap['query']
        assert journal.value['attempts']['forward']==cap['forward']
        proof['trial_commit_exact']=True
    elif kind=='dedup_tie':
        result=decision(env,[dict(action=[0,0,0]) for _ in range(4)],journal,dict(case_id=case_id,method=method,repeat=index,step=0))
        tx=journal.value['transaction'];assert tx['selected']==0 and tx['unique_candidates']==1 and tx['cache_hits']==3
        assert journal.value['attempts']['query']==1 and journal.value['attempts']['commit']==1
        proof.update(unique_candidates=1,cache_hits=3,selected=0)
    elif kind=='fork':
        child=fork(env,snapshot(env));assert np.shares_memory(child.kernel['gain'],env.kernel['gain'])
        try:child.kernel['gain'][0,0]=0
        except ValueError:pass
        else:raise AssertionError('kernel must be read only')
        child.starts[0]=77;assert state_sha(snapshot(env))==before
        proof['parent_unchanged']=True
    elif kind=='probability':
        import torch
        from satellite_rl.agents.action_masking import masked_policy,mask_batch
        from satellite_rl.t0_width.agent import WidthAgent
        from satellite_rl.t0_width.runner import isolated_rng
        import inspect
        mask=env.action_mask()
        if journal.value['transaction'] is None:journal.value['transaction']=dict(stage='probability',pending=None)
        tx=journal.value['transaction']
        if tx.get('pending'):raise UnknownCost('probability forward intent unknown; no replay')
        if 'new_probabilities' not in tx:
            journal.intent('forward');tx['new_probabilities']=frozen_s10_forward(agent,observation,mask);journal.finish('forward')
        new_forward=tx['new_probabilities'];new_fl,new_p=new_forward['frequency_slots'],new_forward['power']
        if 'reference_probabilities' not in tx:
            # Build the original WidthAgent, not a second reference to the new
            # actor. All original caller-chain source files are byte-pinned to
            # the real W26 producer job before using its constructor/forward.
            original_config=read_json_ref(plan['producer_config_ref']);producer=read_json_ref(plan['producer_job_ref'])
            source=Path(plan['source_root'])
            chain=('src/satellite_rl/t0_width/agent.py','src/satellite_rl/agents/workbench_algorithms.py',
                'src/satellite_rl/agents/workbench_networks.py','src/satellite_rl/agents/action_masking.py',
                'src/satellite_rl/workbench/config.py','src/satellite_rl/state_layouts.py')
            assert all(file_sha(source/name)==producer['source_manifest'][name] for name in chain)
            with isolated_rng(73028):
                reference=WidthAgent(original_config,agent.device)
                saved=torch.load(plan['policy_ref']['path'],map_location='cpu',weights_only=False)
                reference.actor.load_state_dict(saved['actor'],strict=True);reference.actor.eval()
            journal.intent('forward')
            with torch.no_grad():
                x=torch.as_tensor(observation,dtype=torch.float32,device=agent.device).reshape(1,-1)
                original_heads=reference.actor(x)
                fm,pm=mask_batch([mask],(100,10,10),agent.device);fl,p,*_=masked_policy(original_heads,fm,pm)
                # Original act() samples Categorical(probs=...) on these arrays.
                ref_fl=torch.distributions.Categorical(probs=fl.flatten(1)).probs[0].cpu().numpy().astype(np.float64).reshape(100,10)
                ref_p=torch.distributions.Categorical(probs=p).probs[0].cpu().numpy().astype(np.float64)
            tx['reference_probabilities']=dict(frequency_slots=ref_fl,power=ref_p,logits=[h[0].cpu().numpy().copy() for h in original_heads]);journal.finish('forward')
            del reference
        reference_forward=tx['reference_probabilities'];ref_fl,ref_p=reference_forward['frequency_slots'],reference_forward['power']
        for actual,expected in zip(new_forward['logits'],reference_forward['logits']):np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(new_fl,ref_fl,rtol=1e-6,atol=1e-7);np.testing.assert_allclose(new_p,ref_p,rtol=1e-6,atol=1e-7)
        assert not new_fl[~mask['frequency_slots']].any() and not new_p[~mask['power']].any()
        proof.update(rtol=1e-6,atol=1e-7,model_forwards=2,original_WidthAgent_constructor=True,
            original_caller_chain_SHA_verified=True,logits_and_masked_probabilities_compared=True,
            reference_parameter_updates=0,torch_bitwise_trajectory_claim=False)
    elif kind=='mask_shrink':
        from satellite_rl.shared_interface.link import build_link_coefficients,capacity_bps
        from .transaction import restore_memory
        recipe=prefix_recipe();cached=folder/'prefix_state.pkl'
        if cached.exists():
            if journal.value.get('prefix_state_ref',{}).get('sha256')!=file_sha(cached):raise UnknownCost('prefix state exists without matching committed query reference')
            saved=pickle.loads(cached.read_bytes());restore_memory(env,saved['state']);observation=saved['observation']
            if journal.value['transaction']['stage']=='prefix_capacity':journal.value['transaction']['stage']='closed';journal.save()
        else:
            env.beam_idx=196;env.requested=[]
            for action in recipe['actions']:
                i=int(env.indices[action['processing_index']]);g=int(env.groups[i]);assert g==action['group']
                f,l,p=action['start'],action['count'],action['power_w'];env.starts[i]=f;env.counts[i]=l;env.powers[i]=p
                assert not env.occupied[g,f:f+l].any();env.occupied[g,f:f+l]=True
                env.requested.append(dict(beam_id=env.ids[i],frequency_index=f,count_index=l-1,power_index=p//5-1,start=f,count=l,power_w=p))
            assert env.powers.sum()==2960.;allocation=env.export_allocation()
            journal.value['transaction']=dict(stage='prefix_capacity',pending=None)
            journal.intent('query')
            coeff=build_link_coefficients(env.layout,allocation,env.physics,kernel=env.kernel);env.rates=capacity_bps(coeff)
            processed=env.indices[:196];env.total_satisfaction=float(np.minimum(env.rates[processed]/env.demands[processed],1).sum())
            observation=env._observation()
            save_pickle(cached,dict(state=snapshot(env),observation=observation,allocation=allocation))
            journal.value['prefix_state_ref']=dict(path=str(cached),sha256=file_sha(cached))
            journal.finish('query');journal.value['transaction']['stage']='closed';journal.save()
        mask=env.action_mask();assert mask['frequency_slots'].sum()==10 and mask['power'].sum()==5
        assert execute() is not None;assert journal.value['attempts']['commit']==1
        proof.update(engineered_prefix=True,prefix_real_steps=0,prefix_capacity_calls=1,
            prefix_power_w=2960.,prefix_processed=196,FL_legal=10,P_legal=5,trial_commit_exact=True)
    elif kind.startswith('pause_'):
        if kind=='pause_before':
            assert execute(lambda:True) is None;assert not any(journal.value['attempts'].values())
        elif kind=='pause_query':
            stop=lambda: bool(journal.value['transaction'] and len(journal.value['transaction'].get('results',{}))>=1)
            assert execute(stop) is None;assert journal.value['attempts']['commit']==0
        elif kind=='pause_selection':
            stop=lambda: bool(journal.value['transaction'] and journal.value['transaction']['stage']=='selected')
            assert execute(stop) is None;assert journal.value['attempts']['commit']==0
        elif kind=='pause_commit':
            assert execute() is not None
        spent=dict(journal.value['attempts']);journal=reopen()
        assert execute() is not None;assert journal.value['attempts']['commit']==1
        if kind=='pause_commit':assert journal.value['attempts']==spent
        assert journal.value['attempts']['query']<=cap['query'];assert journal.value['attempts']['forward']==cap['forward']
        proof.update(same_ID_reloaded=True,prior_attempts=spent,after_attempts=dict(journal.value['attempts']),no_extra_commit=True)
    elif kind in ('unknown_query','unknown_commit'):
        original=type(env).step;calls=[];fault_result=[]
        def power_loss(actual,action):
            calls.append(list(action));result=original(actual,action);fault_result.append(dict(result=result,state=snapshot(actual)))
            raise PlannedPowerLoss()
        with patch.object(type(env),'step',power_loss):
            try:execute()
            except PlannedPowerLoss:pass
            else:raise AssertionError('fault point not reached')
        assert len(calls)==1;save_pickle(folder/'fault_actual_result.pkl',fault_result)
        journal=reopen();before_calls=len(calls)
        try:execute()
        except UnknownCost:pass
        else:raise AssertionError('unknown physical intent was replayed')
        assert len(calls)==before_calls;assert journal.value['transaction']['pending']==('query' if kind=='unknown_query' else 'commit')
        proof.update(unknown_preserved=True,physical_attempts=1,unknown_not_zero=True,automatic_replay=False,fault_result_SHA=file_sha(folder/'fault_actual_result.pkl'))
    elif kind=='binding':
        from satellite_rl.t0_width.environment import T0WidthEnv
        with patch.object(T0WidthEnv,'__init__',side_effect=AssertionError('constructor replay forbidden')),patch.object(T0WidthEnv,'step',side_effect=AssertionError('step replay forbidden')),patch.object(T0WidthEnv,'load_state_dict',side_effect=AssertionError('old replay forbidden')):
            restored=unpack_environment(pickle.loads(pickle.dumps(pack_environment(env),protocol=5)))
        assert state_sha(snapshot(restored))==before and not restored.kernel['gain'].flags.writeable
        proof.update(restore_PHY=0,constructor_calls=0,step_replay_calls=0)
    else:raise ValueError('unregistered builtin qualification check')
    costs=journal.value['attempts']
    assert all(costs[key]<=cap[key] for key in ('query','commit','forward'))
    return dict(schema='W28BuiltinCheckEvidenceV1',check_SHA=content_sha(check),check_id=check['check_id'],
        fixture_SHA=content_sha(spec),passed=True,coverage=check['coverage'],attempts=costs,
        completed=journal.value['completed'],proof=proof,actual_N=env.N,fixture_geometry_validated=True,
        uniform_request_bps=21600000,current_journal_SHA=file_sha(folder/'current.pkl'),
        current_journal_ref=dict(path=str(folder/'current.pkl'),sha256=file_sha(folder/'current.pkl')),kernel_preparation_calls=1)

def read_json_ref_bytes(ref):
    from satellite_rl.t0_width.data import read_ref
    return read_ref(ref,limit=64*1024*1024)

def run_checks(plan,root,stop_requested,progress=lambda value:None):
    if plan.get('runtime_approved') is not True or plan.get('root_approved') is not True:raise ValueError('root exact qualification plan required')
    fixtures=read_json_ref(plan['fixture_manifest_ref']);bounds=validate_catalog(plan['checks'],fixtures)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    config_path=root/'engineering_config.json'
    if config_path.exists():
        if content_sha(json.loads(config_path.read_text()))!=content_sha(plan):raise ValueError('immutable engineering config differs')
    else:atomic_json(config_path,plan);fsync_directory(root)
    budget=Journal(root/'engineering_cursor.pkl',dict(plan_SHA=content_sha(plan),catalog_SHA=bounds['catalog_SHA']),dict(query=256,commit=64,forward=64,proposal=64))
    value=budget.value;value.setdefault('cursor',0);value.setdefault('active',None);value.setdefault('unknown',[]);value.setdefault('elapsed_active_seconds',0.)
    started=perf_counter();elapsed=value['elapsed_active_seconds'];agent=None
    try:
        for check in plan['checks'][value['cursor']:]:
            if stop_requested():return dict(stage='safe_stopped',checks=value['cursor'],reserved=value['attempts'])
            index=check['check_index'];fresh=value['active'] is None
            folder=root/'checks'/check['check_id'];evidence=folder/'evidence.json'
            if fresh:
                cap=check['worst_case']
                for key in ('query','commit','forward'):
                    value['attempts'][key]+=cap[key]
                    if value['attempts'][key]>value['limits'][key]:raise ValueError('global engineering reservation exhausted')
                value['active']=dict(check_index=index,check_SHA=content_sha(check),reserved=cap);budget.save()
            elif value['active']['check_SHA']!=content_sha(check):raise ValueError('active engineering check differs')
            if evidence.exists():
                result=json.loads(evidence.read_text())
                if result['check_SHA']!=content_sha(check) or result.get('passed') is not True:raise ValueError('closed check evidence differs')
            else:
                if check['worst_case']['forward'] and agent is None:
                    # Loading creates no forward, but failed setup is preserved.
                    agent,policy_timing=load_policy(plan)
                    atomic_json(root/'policy_loading.json',policy_timing);fsync_directory(root)
                try:result=run_check(check,folder,plan,agent,stop_requested,fresh=fresh)
                except QualificationStopped:return dict(stage='safe_stopped',checks=value['cursor'],reserved=value['attempts'])
                atomic_json(evidence,result);fsync_directory(folder)
            value['cursor']=index+1;value['active']=None;value['elapsed_active_seconds']=elapsed+perf_counter()-started;budget.save()
            progress(dict(stage='engineering',checks=value['cursor'],reserved_worst_case=value['attempts']))
        actual={key:0 for key in ('query','commit','forward')};refs=[]
        for check in plan['checks']:
            path=root/'checks'/check['check_id']/'evidence.json';result=json.loads(path.read_text())
            for key in actual:actual[key]+=result['attempts'][key]
            refs.append(dict(path=str(path),sha256=file_sha(path)))
        candidate=dict(schema='W28QualificationCandidateV1',experiment_id='S2-W28',root_approved=False,passed=False,
            checks_closed=True,coverage=list(REQUIRED_COVERAGE),check_evidence_refs=refs,catalog_SHA=bounds['catalog_SHA'],
            fixture_manifest_ref=plan['fixture_manifest_ref'],engineering_plan_SHA=content_sha(plan),source_commit=plan['source_commit'],policy_SHA=plan['policy_ref']['sha256'],
            engineering_plan_ref=dict(path=str(config_path),sha256=file_sha(config_path)),
            physical_calls_attempted=actual['query']+actual['commit'],real_commit_attempts=actual['commit'],candidate_restore_attempts=actual['query'],model_forward_attempts=actual['forward'],
            parameter_updates=0,restore_PHY_calls=0,kernel_preparation_calls=len(plan['checks']),directory_fsync_available=os.name=='posix',
            root_review_required=True,expected_fault_unknowns_preserved=['unknown_query','unknown_commit'])
        atomic_json(root/'qualification_candidate.json',candidate);fsync_directory(root)
        return dict(candidate,stage='completed')
    finally:value['elapsed_active_seconds']=elapsed+perf_counter()-started;budget.save()
