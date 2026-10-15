"""Explicit SHA substreams and unchanged masked S10 proposal probabilities."""
import hashlib
import json
import math
import numpy as np

SEED_CONTRACT=dict(schema='W28ProposalSeedV1',base_seed=73028,
    encoding='UTF-8 JSON sort_keys=True separators=(comma,colon) ensure_ascii=True',
    digest='SHA256',integer='first 16 digest bytes unsigned big endian',
    generator='numpy.random.PCG64',sampling='inverse CDF float64; FL then independent P',
    normalization='masked_policy output converted to float64; divide each probability array by float64 sum; abs sum tolerance 1e-7 or rel 1e-6 before normalization',
    numerical_claim='same masked categorical law to stated float tolerance; not Torch RNG bitwise trajectories',
    coupling='P1/P4 family P and R1/R4 family R; candidate index zero shared; later state divergence means actions need not match')

def seed_record(case_id,family,repeat,step,candidate):
    if family not in ('P','R') or not isinstance(case_id,str):raise ValueError('explicit case/family required')
    if any(type(x) is not int or x<0 for x in (repeat,step,candidate)):raise ValueError('nonnegative integer cursor required')
    value=dict(schema='W28ProposalSeedV1',base_seed=73028,case_id=case_id,family=family,repeat=repeat,step=step,candidate=candidate)
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf8')
    digest=hashlib.sha256(raw).digest()
    return dict(identity=value,canonical_JSON=raw.decode(),SHA256=digest.hex(),seed=int.from_bytes(digest[:16],'big'))

def validate_probabilities(fl,power,mask):
    fl=np.asarray(fl,dtype=np.float64);power=np.asarray(power,dtype=np.float64)
    fm=np.asarray(mask['frequency_slots']);pm=np.asarray(mask['power'])
    if fm.dtype!=np.bool_ or pm.dtype!=np.bool_ or fl.shape!=(100,10) or power.shape!=(10,) or fm.shape!=fl.shape or pm.shape!=power.shape:raise ValueError('exact joint FL and independent P shapes required')
    for p,m in ((fl,fm),(power,pm)):
        if not np.isfinite(p).all() or (p<0).any() or (p[~m]!=0).any() or not m.any() or not math.isclose(float(p.sum()),1.,rel_tol=1e-6,abs_tol=1e-7):raise ValueError('invalid normalized legal distribution')
    # Categorical(probs=...) in original S10 normalizes supplied probabilities.
    return fl/fl.sum(),power/power.sum()

def uniform_probabilities(mask):
    fl=np.asarray(mask['frequency_slots']);power=np.asarray(mask['power'])
    if fl.dtype!=np.bool_ or power.dtype!=np.bool_ or not fl.any() or not power.any():raise ValueError('nonempty boolean domain required')
    return validate_probabilities(fl/fl.sum(),power/power.sum(),mask)

def draw(fl,power,mask,case_id,family,repeat,step,candidate):
    fl,power=validate_probabilities(fl,power,mask)
    record=seed_record(case_id,family,repeat,step,candidate)
    rng=np.random.Generator(np.random.PCG64(record['seed']))
    before=rng.bit_generator.state
    def choose(p):
        flat=p.reshape(-1);support=np.flatnonzero(flat>0)
        cumulative=np.cumsum(flat[support]);cumulative[-1]=1.
        return int(support[np.searchsorted(cumulative,rng.random(),side='right')])
    index=choose(fl);pi=choose(power);action=(index//10,index%10,pi)
    return dict(action=list(action),probability=float(fl[action[:2]]*power[pi]),
        seed=record,rng_before=before,rng_after=rng.bit_generator.state)

def frozen_s10_probabilities(agent,observation,mask):
    result=frozen_s10_forward(agent,observation,mask)
    return result['frequency_slots'],result['power']

def frozen_s10_forward(agent,observation,mask):
    """One actual actor forward only when called after root runtime admission."""
    import torch
    from satellite_rl.agents.action_masking import masked_policy,mask_batch
    if agent.schema['group']!='S10' or agent.sac_policy_structure!='factorized':raise ValueError('original frozen W26 S10 factorized actor required')
    net=agent.actor;net.eval()
    with torch.no_grad():
        x=torch.as_tensor(observation,dtype=torch.float32,device=agent.device).reshape(1,-1)
        flmask,pmask=mask_batch([mask],(100,10,10),agent.device)
        heads=net(x);fl,p,*_=masked_policy(heads,flmask,pmask)
    fl,p=validate_probabilities(fl[0].cpu().numpy(),p[0].cpu().numpy(),mask)
    return dict(frequency_slots=fl,power=p,logits=[h[0].cpu().numpy().copy() for h in heads])
