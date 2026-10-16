"""Root-admitted finite physical fixtures; no agent or optimizer construction.

Preparation only until an owned executor binds a fresh root fixture admission.
One qualified train mother, paired M0/M1 full episodes for all four menus,
plus old W26 S10 on the exact P00 actions: at most 1800 physical calls.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import numpy as np
from .logging import atomic_json, content_sha, EvidenceLogs
from .control import read_ref
from .menus import MENUS, validate_actual_power


def expected_summary(env):
    """Independent slot-first arithmetic on already computed arrays, no PHY."""
    result=np.zeros(100,np.float64)
    if env.beam_idx==env.N or env.last_coefficients is None:
        return result
    c=int(env.indices[env.beam_idx]);coeff=env.last_coefficients
    for slot in range(100):
        terms=[]
        for j in env.indices[:env.beam_idx]:
            if (env.groups[j]==env.groups[c] or env.groups[j]%2!=env.groups[c]%2
                    or not env.starts[j]<=slot<env.starts[j]+env.counts[j]):
                continue
            row=np.flatnonzero((coeff.slot_owner==j)&(coeff.occupied_slots==slot))
            if row.size!=1:
                raise ValueError('Unique actual occupied-slot coefficient row required')
            r=int(row[0])
            denominator=float(coeff.noise[r])+math.fsum(float(a)*float(b) for a,b in zip(coeff.interference[r],env.powers))
            kappa=float(env.kernel['gain'][j,c])*10**(env.physics['rx_gain_dBi']/10)*float(env.kernel['path'][j,slot])
            terms.append(math.exp(-abs(float(env.rates[j]/env.demands[j])-1.))*kappa/denominator)
        result[slot]=math.log1p(math.fsum(terms))
    return result


def assert_summary(state,env):
    future=env.indices[env.beam_idx:]
    if np.any(env.counts[future]) or np.any(env.powers[future]):
        raise ValueError('Future actual allocations must not exist in summary fixture')
    expected=expected_summary(env).astype(np.float32)
    np.testing.assert_allclose(state[208:],expected,rtol=2e-6,atol=1e-7)
    return float(np.max(np.abs(state[208:]-expected)))


@contextmanager
def metered_old_environment(hook):
    """Local harness process only: meter each old actual shared-function entry."""
    import satellite_rl.t0_width.environment as old
    originals={name:getattr(old,name) for name in ('prepare_slot_kernel','build_link_coefficients','capacity_bps')}
    for name,kind in (('prepare_slot_kernel','kernel'),('build_link_coefficients','coeff'),('capacity_bps','phy')):
        original=originals[name]
        def wrapped(*args,_kind=kind,_function=original,**kwargs):
            hook(_kind,reference='W26_S10')
            return _function(*args,**kwargs)
        setattr(old,name,wrapped)
    try:
        yield old.T0WidthEnv
    finally:
        for name,original in originals.items():
            setattr(old,name,original)


def fixture_admission(config):
    receipt=read_ref(config['fixture_admission_ref'])
    if (receipt.get('schema')!='W30PhysicalFixtureAdmissionV1'
            or receipt.get('experiment_id')!='S2-W30' or receipt.get('root_approved') is not True
            or receipt.get('passed') is not True or receipt.get('source_sha')!=config['source_sha']
            or receipt.get('case_record_sha')!=content_sha(config['case_record'])
            or receipt.get('independent_fixture_review_passed') is not True):
        raise ValueError('Root and independent-reviewed exact fixture/case/source admission required')
    if (receipt.get('max_phy')!=1800 or receipt.get('max_kernel')!=9
            or receipt.get('max_coeff')!=1800 or receipt.get('max_forward')!=0
            or receipt.get('max_updates')!=0 or config['case_record'].get('role')!='train'):
        raise ValueError('Exact bounded original physical fixture contract required')
    return receipt


def run_fixtures(config,directory,*,call_hook,stop_requested,owner_check):
    """Owned persistent executor supplies metering, pause and identity controls.

    No CLI or default hook: this cannot be used as an unmetered bare worker.
    All real entries are charged before execution. Failure keeps original
    evidence and cannot rerun automatically. A clean early stop is partial,
    not a fixture pass, and needs root same-ID spend reconciliation to continue.
    """
    receipt=fixture_admission(config)
    owner_check()
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Existing fixture evidence: never replay or overwrite')
    directory.mkdir(parents=True,exist_ok=True)
    logs=EvidenceLogs(directory/'logs')
    counts=dict(phy=0,kernel=0,coeff=0,feature=0)
    def hook(kind,**metadata):
        if kind not in counts:
            raise ValueError('No neural forward/update in physical fixtures')
        bound=dict(phy=1800,kernel=9,coeff=1800,feature=1000)[kind]
        if counts[kind]>=bound:
            raise RuntimeError('Fixture local call cap exhausted')
        if stop_requested():
            raise RuntimeError('Fixture stop boundary; preserve partial evidence and cost')
        call_hook(kind,**metadata)
        counts[kind]+=1
        atomic_json(directory/'entered_calls.json',dict(counts=counts,unknown_completion_possible=True))
    from satellite_rl.t0_width.runner import _default_loader
    from .environment import Power30Env
    rows=[]
    try:
        stage_input=_default_loader(config['case_record'])
        if not 1<=len(stage_input.layout['beam_ids'])<=200:
            raise ValueError('Qualified physical fixture mother must satisfy real1<=N<=200')
        with metered_old_environment(hook) as OldEnv:
            for menu in MENUS:
                a=Power30Env(stage_input,menu,'M0',call_hook=hook)
                b=Power30Env(stage_input,menu,'M1',call_hook=hook)
                old=OldEnv(stage_input,'S10') if menu=='P00' else None
                s0=a.reset();s1=b.reset();old_state=old.reset() if old is not None else None
                trace={k:[] for k in ('before_M0','before_M1','after_M0','after_M1','actions','rewards','service_bps')}
                summary_max=assert_summary(s1,b)
                done=False;step=0
                while not done:
                    if stop_requested():
                        raise RuntimeError('Fixture stopped before next transition')
                    mask=a.action_mask();other=b.action_mask()
                    if not all(np.array_equal(mask[k],other[k]) for k in mask):
                        raise ValueError('Summary changed feasible domain')
                    expected_power=np.asarray(MENUS[menu])<=3000-float(a.powers.sum())-5*(a.N-step-1)
                    if not np.array_equal(mask['power'],expected_power):
                        raise ValueError('Physical menu/future5W reserve mask differs')
                    legal=np.argwhere(mask['frequency_slots'])
                    f,l=map(int,legal[step%len(legal)])
                    legal_p=np.flatnonzero(mask['power'])
                    p=int(legal_p[(0,len(legal_p)-1,len(legal_p)//2)[step%3]])
                    action=(f,l,p)
                    trace['before_M0'].append(s0.copy());trace['before_M1'].append(s1.copy())
                    before_phy=counts['phy'];before_coeff=counts['coeff']
                    atomic_json(directory/'transition_intent.json',dict(menu=menu,group='M0',step=step,
                        requested_action=action,before=s0,actual_next_spend_unknown_until_step_evidence=True))
                    n0,r0,done,i0=a.step(action)
                    logs.append('steps',dict(menu=menu,group='M0',step=step,requested_action=action,
                        actual_action=i0['actual_action'],reward=r0,service=i0['beam_service_step'],
                        before=s0,after=n0,natural_terminal=done))
                    logs.barrier()
                    atomic_json(directory/'transition_intent.json',dict(menu=menu,group='M1',step=step,
                        requested_action=action,before=s1,actual_next_spend_unknown_until_step_evidence=True))
                    n1,r1,done1,i1=b.step(action)
                    logs.append('steps',dict(menu=menu,group='M1',step=step,requested_action=action,
                        actual_action=i1['actual_action'],reward=r1,service=i1['beam_service_step'],
                        before=s1,after=n1,natural_terminal=done1))
                    logs.barrier()
                    if counts['phy']-before_phy!=2 or counts['coeff']-before_coeff!=2:
                        raise ValueError('Summary caused extra physical or coefficient call')
                    if r0!=r1 or done!=done1 or not np.array_equal(n0,n1[:208]) or not np.array_equal(a.rates,b.rates):
                        raise ValueError('M1 treatment altered physical trajectory')
                    summary_max=max(summary_max,assert_summary(n1,b))
                    if old is not None:
                        old_mask=old.action_mask()
                        if not all(np.array_equal(mask[k],old_mask[k]) for k in mask) or not np.array_equal(s0,old_state):
                            raise ValueError('P00M0 differs from original W26 observation/domain')
                        atomic_json(directory/'transition_intent.json',dict(menu='P00',group='old_W26_S10',step=step,
                            requested_action=action,before=old_state,actual_next_spend_unknown_until_step_evidence=True))
                        old_state,old_reward,old_done,old_info=old.step(action)
                        logs.append('steps',dict(menu='P00',group='old_W26_S10',step=step,
                            requested_action=action,actual_action=old_info['actual_action'],reward=old_reward,
                            service=old_info['beam_service_step'],after=old_state,natural_terminal=old_done))
                        logs.barrier()
                        if old_reward!=r0 or old_done!=done or not np.array_equal(n0,old_state) or not np.array_equal(a.rates,old.rates):
                            raise ValueError('P00M0 differs from original W26 transition/reward')
                    trace['after_M0'].append(n0.copy());trace['after_M1'].append(n1.copy())
                    trace['actions'].append(action);trace['rewards'].append(r0);trace['service_bps'].append(i0['rate_bps'])
                    s0,s1=n0,n1;step+=1
                if step!=a.N or not (a.counts>0).all() or not (b.counts>0).all():
                    raise ValueError('All real beams must be served at natural terminal')
                validate_actual_power(menu,a.powers);validate_actual_power(menu,b.powers)
                np.savez_compressed(directory/('trace_'+menu+'.npz'),**{k:np.asarray(v) for k,v in trace.items()},
                    final_capacity_bps=a.rates,demands_bps=a.demands,final_starts=a.starts,
                    final_counts=a.counts,final_power_w=a.powers)
                rows.append(dict(menu=menu,N=a.N,steps=step,all_beams_served=True,
                    matched_M0_M1_physical_arrays=True,matched_W26_S10=old is not None,
                    maximum_summary_float32_error=summary_max,metrics=a.metrics(),calls=dict(counts)))
                atomic_json(directory/'completed_menus.json',rows)
        result=dict(status='passed',source_sha=config['source_sha'],case_record_sha=content_sha(config['case_record']),
                    rows=rows,calls=counts,agent_constructions=0,netforward=0,optimizer_updates=0,
                    fixture_admission_ref=config['fixture_admission_ref'],root_actual_acceptance_required=True,
                    gpu_learning_memory_throughput_qualification=False)
        atomic_json(directory/'result.json',result)
        logs.close()
        return result
    except BaseException as exc:
        logs.close()
        atomic_json(directory/'failure.json',dict(error=repr(exc),calls_entered=counts,
                    unknown_current_call_not_zero=True,no_automatic_retry=True))
        raise
