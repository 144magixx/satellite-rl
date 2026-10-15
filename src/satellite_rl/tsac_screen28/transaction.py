"""Memory forks without PHY replay, and durable per-query intent accounting."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
from time import perf_counter
from types import MappingProxyType
import numpy as np

FIELDS=('beam_idx','starts','counts','powers','occupied','rates','requested','total_satisfaction')

class UnknownCost(RuntimeError):pass

def fsync_directory(path):
    """Linux production requires a successful directory metadata barrier."""
    if os.name=='posix':
        descriptor=os.open(str(path),os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

class FrozenDict(dict):
    """Dictionary-compatible with the existing content/physics validators."""
    def _immutable(self,*args,**kwargs):raise TypeError('shared candidate input is read only')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=_immutable

def rng_snapshot():
    value=dict(python=random.getstate(),numpy=np.random.get_state())
    torch=sys.modules.get('torch')
    if torch is not None:
        value['torch_cpu']=torch.get_rng_state().cpu().numpy().copy()
        if torch.cuda.is_initialized():value['torch_cuda']=[x.cpu().numpy().copy() for x in torch.cuda.get_rng_state_all()]
    return value

def check_rng(before):
    if state_sha(before)==state_sha(rng_snapshot()):return
    random.setstate(before['python']);np.random.set_state(before['numpy'])
    torch=sys.modules.get('torch')
    if torch is not None and 'torch_cpu' in before:
        torch.set_rng_state(torch.from_numpy(before['torch_cpu']))
        if 'torch_cuda' in before:torch.cuda.set_rng_state_all([torch.from_numpy(s) for s in before['torch_cuda']])
    raise ValueError('candidate/commit changed global RNG; restored and stopped')

def snapshot(env):
    return dict(schema=plain_copy(env.schema),layout_SHA=env.layout['content_sha256'],
        physics=plain_copy(dict(getattr(env,'physics',{}))),
        values={key:copy.deepcopy(getattr(env,key)) for key in FIELDS})

def plain_copy(value):
    """Copy a readonly tree without invoking FrozenDict's mutation guard."""
    if isinstance(value,dict):return {key:plain_copy(item) for key,item in value.items()}
    if isinstance(value,list):return [plain_copy(item) for item in value]
    if isinstance(value,tuple):return tuple(plain_copy(item) for item in value)
    return copy.deepcopy(value)

def state_sha(state):
    from satellite_rl.t0_width.logging import content_sha
    return content_sha(state)

def restore_memory(env,state):
    """Direct assignment only; never reset/step/load_state_dict/rebuild the kernel."""
    if (state_sha(state['schema'])!=state_sha(env.schema) or state['layout_SHA']!=env.layout['content_sha256']
            or state_sha(state.get('physics',{}))!=state_sha(dict(getattr(env,'physics',{})))):raise ValueError('snapshot identity mismatch')
    for key,value in state['values'].items():setattr(env,key,copy.deepcopy(value))

def readonly(value):
    if isinstance(value,np.ndarray):
        view=value.view();view.setflags(write=False);return view
    if isinstance(value,dict):return FrozenDict({k:readonly(v) for k,v in value.items()})
    if isinstance(value,list):return tuple(readonly(v) for v in value)
    return value

def fork(env,base):
    child=copy.copy(env)
    # Copy only changing allocation state. Large static input/kernel are views.
    for key in ('kernel','layout','physics','indices','groups','demands'):
        if hasattr(env,key):setattr(child,key,readonly(getattr(env,key)))
    if hasattr(env,'relation_audit'):child.relation_audit=copy.deepcopy(env.relation_audit)
    restore_memory(child,base)
    return child

def result_digest(result,state):
    observation,reward,done,info=result
    if not math.isfinite(reward) or not np.isfinite(observation).all():raise ValueError('nonfinite query result')
    # Timing is nondeterministic and never affects selection/equivalence.
    stable={k:v for k,v in info.items() if k!='timing'}
    return state_sha(dict(state=state,observation=observation,reward=reward,done=done,info=stable))

class Journal:
    def __init__(self,path,identity,limits):
        limits=dict(limits)
        limits.setdefault('forward',limits['commit']);limits.setdefault('proposal',limits['commit'])
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        remnants=[self.path.with_suffix(suffix) for suffix in ('.tmp','.previous.tmp')]
        if any(p.exists() for p in remnants):raise UnknownCost('unconfirmed journal write residue exists; preserve every generation and audit')
        if self.path.exists():
            self.value=pickle.loads(self.path.read_bytes())
            if self.value['identity']!=identity or self.value['limits']!=limits:raise ValueError('same-ID immutable contract differs')
        else:
            remnants=[self.path.with_suffix(suffix) for suffix in ('.previous','.tmp','.previous.tmp','.intent')]
            remnants.extend(self.path.parent.glob('*.jsonl'))
            if any(p.exists() for p in remnants):
                raise UnknownCost('current journal absent beside existing recovery/evidence artifacts; preserve and audit, never reset costs')
            self.value=dict(identity=identity,limits=limits,attempts=dict(query=0,commit=0,forward=0,proposal=0),completed=dict(query=0,commit=0,forward=0,proposal=0),transaction=None)
            marker=self.path.with_suffix('.intent')
            with marker.open('xb') as stream:
                stream.write(pickle.dumps(dict(identity=identity,limits=limits),protocol=5));stream.flush();os.fsync(stream.fileno())
            fsync_directory(self.path.parent)
            self.save()

    def save(self):
        raw=pickle.dumps(self.value,protocol=5);tmp=self.path.with_suffix('.tmp')
        with tmp.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if self.path.exists():
            # Copy old bytes while keeping current present. Only the final
            # atomic replace changes current; no current->previous rename gap.
            previous=self.path.with_suffix('.previous');prior_tmp=self.path.with_suffix('.previous.tmp')
            with prior_tmp.open('wb') as stream:
                stream.write(self.path.read_bytes());stream.flush();os.fsync(stream.fileno())
            os.replace(prior_tmp,previous)
        os.replace(tmp,self.path)
        fsync_directory(self.path.parent)

    def intent(self,kind):
        if self.value['attempts'][kind]>=self.value['limits'][kind]:raise ValueError('finite '+kind+' budget exhausted')
        self.value['attempts'][kind]+=1
        self.value['transaction']['pending']=kind
        self.save()

    def finish(self,kind):
        self.value['completed'][kind]+=1;self.value['transaction']['pending']=None;self.save()

def decision(env,proposals,journal,context,stop_requested=lambda:False):
    """Returns committed step or None at a known safe boundary. No auto retry."""
    value=journal.value;tx=value['transaction']
    if tx is not None and tx.get('pending'):
        raise UnknownCost('interrupted '+tx['pending']+' intent: preserve unknown cost, no replay')
    if tx is None or tx.get('stage')=='closed':
        base=snapshot(env)
        tx=dict(context=context,base=base,base_SHA=state_sha(base),proposals=proposals,
            results={},selected=None,stage='proposed',pending=None,commit_result=None,rng=rng_snapshot())
        value['transaction']=tx;journal.save()
    else:
        if tx['context']!=context or tx['proposals']!=proposals:raise ValueError('same-ID candidate/context differs')
        if tx.get('stage')=='committed':
            restore_memory(env,tx['committed_state']);return tx['commit_result']
        if state_sha(snapshot(env))!=tx['base_SHA']:raise ValueError('actual base changed during candidate transaction')
    actions=[tuple(p['action']) for p in tx['proposals']]
    screening=len(actions)>1
    if screening:
        for index,action in enumerate(actions):
            if stop_requested():return None
            if action in tx['results']:continue
            fork_started=perf_counter();child=fork(env,tx['base']);fork_seconds=perf_counter()-fork_started
            journal.intent('query');started=perf_counter();rng=rng_snapshot()
            try:
                result=child.step(action);next_state=snapshot(child);digest=result_digest(result,next_state);check_rng(rng)
            except Exception as exc:
                try:check_rng(rng)
                except ValueError:pass
                tx['failure']=dict(phase='query',action=list(action),error=type(exc).__name__,message=str(exc),unknown_not_zero=True)
                journal.save();raise
            if state_sha(snapshot(env))!=tx['base_SHA']:raise ValueError('candidate polluted real environment')
            tx['results'][action]=dict(result=result,state=next_state,digest=digest,first_index=index,query_seconds=perf_counter()-started,fork_seconds=fork_seconds)
            journal.finish('query')
        winner=max(range(len(actions)),key=lambda i:tx['results'][actions[i]]['result'][1])
    else:winner=0
    tx['selected']=winner;tx['stage']='selected';journal.save()
    if stop_requested():return None
    journal.intent('commit');started=perf_counter()
    try:
        rng=rng_snapshot();result=env.step(actions[winner]);committed=snapshot(env);digest=result_digest(result,committed);check_rng(rng)
    except Exception as exc:
        try:check_rng(rng)
        except ValueError:pass
        tx['failure']=dict(phase='commit',action=list(actions[winner]),error=type(exc).__name__,message=str(exc),unknown_not_zero=True)
        journal.save();raise
    if screening and digest!=tx['results'][actions[winner]]['digest']:raise ValueError('real commit differs from exact candidate')
    tx.update(stage='committed',commit_result=result,committed_state=committed,
        commit_seconds=perf_counter()-started,unique_candidates=len(set(actions)),cache_hits=len(actions)-len(set(actions)))
    journal.finish('commit')
    return result

def close_transaction(journal):
    tx=journal.value['transaction']
    if tx['stage']!='committed' or tx['pending']:raise ValueError('only externally logged committed transactions may close')
    tx['stage']='closed';journal.save()
