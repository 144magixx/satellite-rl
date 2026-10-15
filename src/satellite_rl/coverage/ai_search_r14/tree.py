"""Bounded cost-sensitive tree and strong train-selected one-split baseline.

Top-level fitting is never called on import or by component tests. Runtime owns
the one formal fit intent and the exact96-mother input qualification.
"""
from .features import nearest_rank,validate_vector,FEATURE_NAMES
from .policies import ACTIONS

def validate_tables(X,safe,paid,mother_ids,*,formal=False):
    n=len(X)
    if not n or len(safe)!=n or len(paid)!=n or len(mother_ids)!=n or len(set(mother_ids))!=n:raise ValueError('distinct complete mother rows')
    if formal and n!=96:raise ValueError('formal fit uses exactly96 training mothers')
    for x,s,c in zip(X,safe,paid):
        validate_vector(x)
        if len(s)!=8 or any(type(v) is not bool for v in s):raise ValueError('unknown safety is not false or safe')
        if len(c)!=8 or any(type(v) is not int or not 0<=v<=32 for v in c):raise ValueError('exact paid attempt proxy per action')
        total=c[4]
        if c[:5]!=[0,min(4,total),min(8,total),min(16,total),total] or any(not min(w,total)<=c[j]<=total for j,w in ((5,2),(6,4),(7,8))):raise ValueError('paid proxies must preserve fixed caps/windows and natural teacher limit')
        if s[4] is not True:raise ValueError('cap32/full teacher safety must be qualified')

def leaf_choice(indices,safe,paid):
    if not indices:raise ValueError('empty leaf')
    feasible=[(sum(paid[i][a] for i in indices),a) for a in range(8) if all(safe[i][a] is True for i in indices)]
    if not feasible:raise ValueError('no fully qualified safe action')
    cost,a=min(feasible)
    return dict(action=ACTIONS[a],action_index=a,cost=cost)

def thresholds(X,indices,feature):
    return sorted({nearest_rank([X[i][feature] for i in indices],q) for q in (.25,.5,.75)})

def baseline_candidate_count(X):
    """Pure enumeration-size component; no policy fitting or labels."""
    indices=list(range(len(X)));count=8
    for feature in range(12):
        for threshold in thresholds(X,indices,feature):
            if any(x[feature]<=threshold for x in X) and any(x[feature]>threshold for x in X):count+=8*7
    return count

def best_split(X,safe,paid,indices,*,min_leaf=8):
    """One node component, not a top-level model fit."""
    parent=leaf_choice(indices,safe,paid);candidates=[]
    for feature in range(12):
        for threshold in thresholds(X,indices,feature):
            left=[i for i in indices if X[i][feature]<=threshold];right=[i for i in indices if X[i][feature]>threshold]
            if min(len(left),len(right))<min_leaf:continue
            l=leaf_choice(left,safe,paid);r=leaf_choice(right,safe,paid);gain=parent['cost']-l['cost']-r['cost']
            if gain<=0:continue
            candidates.append(((-gain,feature,threshold,l['action_index'],r['action_index']),
                dict(feature=feature,threshold=threshold,left_indices=left,right_indices=right,left_choice=l,right_choice=r,gain=gain)))
    return min(candidates,key=lambda x:x[0])[1] if candidates else None

def fit_tree(X,safe,paid,mother_ids,*,seed=51001):
    if seed!=51001:raise ValueError('sole frozen seed')
    validate_tables(X,safe,paid,mother_ids,formal=True)
    splits=[]
    def grow(indices,depth,node_id):
        choice=leaf_choice(indices,safe,paid)
        node=dict(kind='leaf',id=node_id,action=choice['action'],train_mothers=len(indices),train_proxy=choice['cost'])
        split=best_split(X,safe,paid,indices,min_leaf=8) if depth<3 else None
        if split is not None:
            splits.append(dict(node_id=node_id,feature=split['feature'],threshold=split['threshold'],gain=split['gain']))
            node=dict(kind='split',id=node_id,feature=split['feature'],threshold=split['threshold'],train_mothers=len(indices),gain=split['gain'],
                left=grow(split['left_indices'],depth+1,node_id+'L'),right=grow(split['right_indices'],depth+1,node_id+'R'))
        return node
    root=grow(list(range(len(X))),0,'root')
    if len(splits)>7:raise RuntimeError('depth3 split ceiling')
    return dict(schema='R14FrozenCausalPolicyV1',policy_type='depth3_tree',seed=seed,feature_names=list(FEATURE_NAMES),actions=list(ACTIONS),
        root=root,train_mother_ids=list(mother_ids),split_count=len(splits),split_trace=splits,max_depth=3,min_leaf_mothers=8,calibration_applied=False)

def select_baseline(X,safe,paid,mother_ids):
    validate_tables(X,safe,paid,mother_ids,formal=True);all_indices=list(range(len(X)));choices=[];enumerated=8
    for action in range(8):
        if all(safe[i][action] for i in all_indices):
            choices.append(((sum(paid[i][action] for i in all_indices),0,action),dict(kind='leaf',id='root',action=ACTIONS[action])))
    for feature in range(12):
        for threshold in thresholds(X,all_indices,feature):
            left=[i for i in all_indices if X[i][feature]<=threshold];right=[i for i in all_indices if X[i][feature]>threshold]
            if not left or not right:continue
            for l in range(8):
                for r in range(8):
                    if l==r:continue
                    enumerated+=1
                    if not all(safe[i][l] for i in left) or not all(safe[i][r] for i in right):continue
                    cost=sum(paid[i][l] for i in left)+sum(paid[i][r] for i in right)
                    choices.append(((cost,1,feature,threshold,l,r),dict(kind='split',id='root',feature=feature,threshold=threshold,
                        left=dict(kind='leaf',id='rootL',action=ACTIONS[l]),right=dict(kind='leaf',id='rootR',action=ACTIONS[r]))))
    if enumerated>2024:raise RuntimeError('fixed baseline candidate ceiling')
    winner,node=min(choices,key=lambda x:x[0])
    return dict(schema='R14FrozenCausalPolicyV1',policy_type='strong_single_split',feature_names=list(FEATURE_NAMES),actions=list(ACTIONS),root=node,
        train_mother_ids=list(mother_ids),train_proxy=winner[0],candidate_count=enumerated,calibration_applied=False)

def route(policy,x):
    x=validate_vector(x)
    if policy.get('schema')!='R14FrozenCausalPolicyV1' or policy.get('actions')!=list(ACTIONS) or policy.get('feature_names')!=list(FEATURE_NAMES):raise ValueError('exact frozen policy schema/features/actions')
    node=policy['root'];depth=0
    while node['kind']=='split':
        depth+=1
        if depth>3 or type(node.get('feature')) is not int or not 0<=node['feature']<12:raise ValueError('bounded original feature route')
        threshold=node['threshold'];validate_vector([threshold]*12)
        node=node['left'] if x[node['feature']]<=threshold else node['right']
    if node['kind']!='leaf' or node.get('action') not in ACTIONS or not isinstance(node.get('id'),str):raise ValueError('saved actual leaf/action')
    return dict(action=node['action'],node_id=node['id'])
