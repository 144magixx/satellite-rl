"""Original SH-C pool/kernel sequence with one initial selected stop rule.

Caller injects exact frozen R10 backend and its original bounded Ledger. No
backend, data, model, geometry or scientific work is imported/executed here.
"""
import copy
from .policies import StopRule

class CooperativeStop(RuntimeError):pass

def _execute(scene,common,action,*,backend,ledger,stop=None,observations=None,engineering_limit=None):
    rule=StopRule(action);current=dict(layout=copy.deepcopy(common['origin']['layout']),metrics=copy.deepcopy(common['origin']['metrics']))
    seen=set();results=[];screenings=[];generations=0;exhausted=False;observations={} if observations is None else observations
    observations.update(results=results,screenings=screenings,selected=dict(terminal=current['layout'],metrics=current['metrics'],status='protected_current'))
    def boundary():
        if stop is not None and stop():raise CooperativeStop('pause or outer budget; not a completed stopping-policy result')
    def result(termination):
        return dict(results=results,screenings=screenings,kernel_attempts=len(results),pool_generations=generations,termination=termination,
            selected_rule=action,engineering_only=engineering_limit is not None,selected=dict(terminal=current['layout'],metrics=current['metrics'],status='protected_current'))
    try:
        boundary()
        if rule.stop_before_next:return result('selected_rule_boundary')
        while len(results)<32:
            boundary();ledger.tick('pool_generations')
            pool=backend._pool(scene,common,current,'SH-C',ledger);generations+=1
            if generations>33:raise RuntimeError('finite original pool-generation ceiling')
            rebuild=False;poolkeys=set()
            for descriptor in pool:
                boundary()
                if len(screenings)>=26400:raise backend.ResourceUnknown('descriptor_screenings_cap',ledger)
                key=descriptor['semantic_SHA']
                reason=('known_unavailable' if not descriptor['structurally_eligible'] else 'duplicate_semantics' if key in poolkeys else 'already_seen' if key in seen else 'kernel_attempted')
                screenings.append(dict(descriptor=descriptor,screening=reason));ledger.tick('screenings_'+reason);poolkeys.add(key)
                if reason!='kernel_attempted':continue
                seen.add(key);native=backend._kernel(scene,common,current,descriptor,ledger);native['attempt']=len(results);results.append(native)
                if native['status']=='admissible':
                    current=dict(layout=native['terminal'],metrics=native['metrics'])
                    observations['selected']=dict(terminal=current['layout'],metrics=current['metrics'],status='protected_current');rebuild=True
                stopped=rule.after(native['status']);boundary()
                if engineering_limit is not None and len(results)>=engineering_limit:return result('engineering_two_kernel_boundary')
                if stopped:
                    return result('kernel_attempt_limit' if len(results)==32 else 'selected_rule_boundary')
                if rebuild:break
                if len(results)>=32:break
            if len(results)>=32:break
            if not rebuild:exhausted=True;break
        return result('no_progress_pool_exhausted' if exhausted else 'kernel_attempt_limit')
    except BaseException as error:
        error.partial_result=copy.deepcopy(observations)
        error.partial_ledger=ledger.record()
        raise


def execute_stopping_trajectory(scene,common,action,*,backend,ledger,stop=None,observations=None):
    return _execute(scene,common,action,backend=backend,ledger=ledger,stop=stop,observations=observations)


def execute_engineering_prefix(scene,common,*,backend,ledger,stop=None,observations=None):
    """Separate frozen8-fixture gate, <=2kernels; never a scientific action."""
    return _execute(scene,common,'cap32',backend=backend,ledger=ledger,stop=stop,observations=observations,engineering_limit=2)
