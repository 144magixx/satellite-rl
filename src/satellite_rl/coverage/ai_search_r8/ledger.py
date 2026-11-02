"""Finite numerical operations and active-clock ledger, including persistence."""
import time
class Limit(RuntimeError):pass
class Ledger:
    def __init__(self,n,clock=time.monotonic,prior=None,verified_seconds=None):
        self.n=n;self.clock=clock;self.started=clock();self.terminal=False
        self.data=dict(prior or {});self.base=float(verified_seconds if verified_seconds is not None else self.data.get('active_seconds',0.))
        if not 0<=self.base<=15:raise ValueError('root verified finite same-budget active clock')
        self.caps=dict(candidate_extractions=128,geometric_attempts=128,scalar_rank_items=65536,
            rank_sort_items=65536,integer_scan_items=200*65536,integer_heap_input_items=200*65536,
            pool_generated=4*n,pool_sort_items=4*n,destination_sort_items=200*n,
            raypairs=512*n,cached_member_distance_scan=128*n,distance_table_elements=200*n,
            softmax_exp=200*n,initial_member_sum_elements=200*n,
            affinity_multiply_add_items=26214400,accepted_member_sum_update_elements=51200,
            proposal_score_elements=200*n,proposal_mask_elements=200*n)
    def elapsed(self):return self.base+self.clock()-self.started
    def check(self):
        if not self.terminal and self.elapsed()>=10:raise Limit('nominal_seconds')
    def charge(self,key,count):
        self.check()
        if type(count)is not int or count<0:raise ValueError('exact nonnegative operation count')
        cap=self.caps.get(key)
        if key=='raypairs' and not self.terminal:cap-=3*self.n
        if cap is not None and self.data.get(key,0)+count>cap:raise Limit(key+'_limit')
        self.data[key]=self.data.get(key,0)+count
    def geometry(self,count,label='geometry'):
        self.charge('raypairs',int(count));self.charge('geometry_calls',1)
    def center_rays(self,centers,profile):
        from ..geometry import satellite_rays_numpy
        self.charge('center_constructions',len(centers));return satellite_rays_numpy(centers,profile)
    def record(self):
        return {**self.data,'active_seconds':self.elapsed(),'nominal_seconds':10.,'hard_seconds':15.,
            'nominal_overrun_seconds':max(0.,self.elapsed()-10),'caps':dict(self.caps),
            'final_audit_reserved_raypairs':3*self.n,'refits':0}
