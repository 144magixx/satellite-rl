"""Frozen A1 RNG, masked laws, missing-Q slots and sampled soft returns."""
import hashlib,json,math
import numpy as np


def normalize_probabilities(fl,power,mask):
    fl=np.asarray(fl,dtype=np.float64);power=np.asarray(power,dtype=np.float64)
    fm=np.asarray(mask['frequency_slots']);pm=np.asarray(mask['power'])
    if fl.shape!=(100,10) or power.shape!=(10,) or fm.shape!=fl.shape or pm.shape!=power.shape or fm.dtype!=bool or pm.dtype!=bool:raise ValueError('masked law dimensions')
    for value,legal in ((fl,fm),(power,pm)):
        if not np.isfinite(value).all() or (value<0).any() or (value[~legal]!=0).any() or not legal.any() or not math.isclose(float(value.sum()),1.,rel_tol=1e-6,abs_tol=1e-7):raise ValueError('masked probability law')
    return fl/fl.sum(),power/power.sum()


def action_probability(fl,power,action):
    f,l,p=action;probability=float(fl[f,l]*power[p])
    return dict(probability=probability,log_probability=-math.inf if probability==0 else math.log(float(fl[f,l]))+math.log(float(power[p])),
                log_probability_is_negative_infinity=probability==0)


def uniforms(case_id,prefix,role,*,draw_index=None,repeat=None,absolute_beam_index=None):
    if not isinstance(case_id,str) or type(prefix) is not int or prefix<0:raise ValueError('RNG fixed state identity')
    key=dict(schema='W29A1RNGv1',base_seed=73029,case_id=case_id,prefix=prefix,role=role)
    if role in ('actor_proposal','uniform_proposal'):
        if type(draw_index) is not int or not 0<=draw_index<4 or repeat is not None or absolute_beam_index is not None:raise ValueError('proposal RNG key')
        key['draw_index']=draw_index
    elif role=='suffix':
        if type(repeat) is not int or repeat not in (0,1) or type(absolute_beam_index) is not int or absolute_beam_index<=prefix or draw_index is not None:raise ValueError('suffix RNG key')
        key.update(repeat=repeat,absolute_beam_index=absolute_beam_index)
    else:raise ValueError('RNG role')
    raw=json.dumps(key,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf8');digest=hashlib.sha256(raw).digest()
    seed=int.from_bytes(digest[:16],'big');values=np.random.Generator(np.random.PCG64(seed)).random(2)
    return dict(key=key,canonical_JSON=raw.decode(),SHA256=digest.hex(),seed=seed,uniforms=values.tolist())


def draw_action(fl,power,mask,uniform_pair):
    fl,power=normalize_probabilities(fl,power,mask)
    if len(uniform_pair)!=2 or any(not 0<=float(u)<1 for u in uniform_pair):raise ValueError('two uniforms in [0,1) required')
    def draw(values,u):
        flat=values.reshape(-1);support=np.flatnonzero(flat>0);cdf=np.cumsum(flat[support]);cdf[-1]=1.
        return int(support[np.searchsorted(cdf,float(u),side='right')])
    index=draw(fl,uniform_pair[0]);p=draw(power,uniform_pair[1])
    return (index//10,index%10,p)


def select_slots(case_id,prefix,original_action,reward_max_action,fl,power,mask):
    fl,power=normalize_probabilities(fl,power,mask)
    complete=fl[:,:,None]*power[None,None,:];index=int(complete.reshape(-1).argmax())
    mode=(index//100,index//10%10,index%10);slots=[];owners={};proposals=[]
    def add(slot,name,action):
        action=tuple(int(v) for v in action)
        if not mask['frequency_slots'][action[:2]] or not mask['power'][action[2]]:raise ValueError('selected action outside original legal domain')
        alias=owners.get(action);status='active' if alias is None else 'alias'
        if alias is None:owners[action]=slot
        slots.append(dict(slot=slot,name=name,status=status,action=list(action),alias_slot=alias,**action_probability(fl,power,action)))
    add(0,'original',original_action);add(1,'actor_mode',mode)
    slots.append(dict(slot=2,name='Q_max',status='not_available',action=None,reason='same-node150kCriticMissing; no replacement'))
    add(3,'reward_max',reward_max_action)
    fm=mask['frequency_slots'];pm=mask['power'];uf=fm.astype(np.float64)/fm.sum();up=pm.astype(np.float64)/pm.sum()
    for slot,role,proposal_fl,proposal_p in ((4,'actor_proposal',fl,power),(5,'uniform_proposal',uf,up)):
        selected=None
        for draw_index in range(4):
            rng=uniforms(case_id,prefix,role,draw_index=draw_index)
            action=draw_action(proposal_fl,proposal_p,mask,rng['uniforms'])
            proposals.append(dict(slot=slot,draw_index=draw_index,action=list(action),rng=rng,**action_probability(fl,power,action)))
            if selected is None and action not in owners:selected=action
        if selected is None:slots.append(dict(slot=slot,name=role,status='all_proposals_duplicate',action=None))
        else:add(slot,role,selected)
    return dict(slots=slots,proposals=proposals,unique_actions=[dict(slot=slot,action=list(action)) for action,slot in owners.items()],Q_available=False)


def new_return_state():
    return dict(count=0,discount=1.,ordinary_discounted=0.,Q_semantic_soft_sample=0.,gamma=.99,alpha=.01)


def accumulate_return(state,reward,log_probability):
    if state['gamma']!=.99 or state['alpha']!=.01 or not math.isfinite(float(reward)):raise ValueError('frozen return contract')
    k=state['count'];weight=state['discount'];result=dict(state)
    # An original/uniform intervention may have logpi=-inf. First entropy is
    # deliberately excluded, so it cannot contaminate the sampled soft return.
    if k>0 and (not math.isfinite(float(log_probability)) or float(log_probability)>1e-12):raise ValueError('sampled suffix log probability')
    result['ordinary_discounted']+=weight*float(reward)
    result['Q_semantic_soft_sample']+=weight*(float(reward) if k==0 else float(reward)-.01*float(log_probability))
    result['count']=k+1;result['discount']=weight*.99
    return result
