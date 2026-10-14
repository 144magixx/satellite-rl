"""Frozen future F definitions; M never initializes or calls this model."""
import math
import numpy as np

SEED=51001
FEATURE_NAMES=('n/2000','N/200','Ncap/200','Omega/Omegacap','global_max_count/n','global_top10_mean_count/n','global_CV','source_count/n','partner_count/n','region_count/n','old_region_k/3','target_k/3','source_fullwidth','partner_fullwidth','partner_angle','region_Omega/Omegacap','remaining_Omega/Omegacap','remaining_Ncap/200','outside_max_count/n','ceil_0.1N_rank_count/n','region_count/targetk/n','global_peak_lower_count/n','single_source_split','delta_region_k/2')
ARCHITECTURE=(24,64,32,1)
OPTIMIZER=dict(name='Adam',lr=.001,weight_decay=0,updates=1000,batch_pairs=64)

def initialize_model():
    import torch
    torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    return torch.nn.Sequential(torch.nn.Linear(24,64,dtype=torch.float32),torch.nn.ReLU(),torch.nn.Linear(64,32,dtype=torch.float32),torch.nn.ReLU(),torch.nn.Linear(32,1,dtype=torch.float32))

def pair_loss(better,worse):
    import torch.nn.functional as F
    return F.softplus(-better+worse).mean()

def strict_pairs(terminals):
    ranks=[]
    for value in terminals:
        rank=tuple(value[k] for k in ('maxload','top10_mean','CV','N','actual_Omega'))
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in rank):raise ValueError('finite actual protected terminal rank')
        ranks.append(rank)
    pairs=[]
    for i in range(len(ranks)):
        for j in range(i+1,len(ranks)):
            if ranks[i]<ranks[j]:pairs.append((i,j))
            elif ranks[j]<ranks[i]:pairs.append((j,i))
    return pairs

class PairSampler:
    """Only eligible fit mothers, uniform mother then strict ordered pair."""
    def __init__(self,mother_pairs,fit_ids):
        self.ids=tuple(sorted(aid for aid in fit_ids if mother_pairs.get(aid)));self.pairs={aid:tuple(mother_pairs[aid]) for aid in self.ids}
        if not self.ids:raise ValueError('no eligible fit mother; no fake pair')
        self.rng=np.random.Generator(np.random.PCG64(SEED))
    def sample(self,batch_pairs=64):
        if batch_pairs!=64:raise ValueError('fixed64pair batch')
        result=[]
        for _ in range(64):
            aid=self.ids[int(self.rng.integers(len(self.ids)))];pair=self.pairs[aid][int(self.rng.integers(len(self.pairs[aid])))];result.append((aid,*pair))
        return result
    def state(self):return self.rng.bit_generator.state
    def restore(self,state):self.rng.bit_generator.state=state
