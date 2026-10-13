"""34D input-only candidate graph and count network (170249 parameters)."""
import numpy as np
import torch
from torch import nn
from .pool import WIDTHS,omega
from ..geometry import satellite_rays_numpy,ray_angle_numpy

def features_graph(p):
    s=p.scene; rays=satellite_rays_numpy(p.centers,s.profile); total=float(s.demands.sum())
    share=s.demands/total if total else np.zeros(len(s.demands)); f=[]
    global_features=[np.log1p(len(s.demands)),np.log1p(total/1e6),len(p.witness.centers)/200,p.ncap/200,
        p.omega_cap/(200*omega(1.)),float(share.max()),float(s.demands.std()/s.demands.mean()) if total else 0,
        p.witness.widths[0],len(p.centers)/512]
    for c,ray in enumerate(rays):
        row=list((ray-s.rays.mean(0))/np.deg2rad(1.))+list(p.origins[c])
        angles=ray_angle_numpy(s.rays,ray)
        for w,width in enumerate(WIDTHS):
            mask=p.coverage[c,w]; values=angles[mask]/(width/2)
            row.extend([mask.mean(),share[mask].sum(),share[mask].max() if mask.any() else 0,
                values.mean() if len(values) else 0,values.max() if len(values) else 0,
                (s.demands[mask]>0).mean() if mask.any() else 0])
        f.append(row+global_features)
    neighbors=np.full((len(rays),8),-1,dtype=np.int64)
    for i in range(len(rays)):
        distance=np.sum((rays-rays[i])**2,axis=1)
        order=sorted((j for j in range(len(rays)) if j!=i),key=lambda j:(distance[j],j))[:8]
        neighbors[i,:len(order)]=order
    return np.asarray(f,dtype=np.float32),neighbors

class CandidateCountNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(34,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,128)) for _ in range(2)])
        self.readout=nn.Sequential(nn.Linear(384,128),nn.ReLU(),nn.Linear(128,9))
        self.register_buffer('feature_mean',torch.zeros(34));self.register_buffer('feature_std',torch.ones(34))
        self.register_buffer('constant_columns',torch.zeros(34,dtype=torch.bool))
    def forward(self,features,neighbors):
        normalized=(features-self.feature_mean)/self.feature_std.clamp_min(1e-8)
        normalized=normalized.masked_fill(self.constant_columns,0);h=self.encoder(normalized)
        valid=neighbors>=0
        for layer in self.messages:
            nh=h[neighbors.clamp_min(0)]*valid[...,None]
            mean=nh.sum(1)/valid.sum(1).clamp_min(1)[:,None]
            h=torch.relu(h+layer(torch.cat((h,mean),-1)))
        global_mean=h.mean(0).expand_as(h);global_max=h.max(0).values.expand_as(h)
        logits=self.readout(torch.cat((h,global_mean,global_max),-1)).reshape(-1,3,3)
        zero_cover=features[:,[7,13,19]]<=0
        logits=logits.clone();logits[:,:,1:]=logits[:,:,1:].masked_fill(zero_cover[:,:,None],-1e9)
        return logits

def select_slots(p,method,model=None,oracle_label=None):
    available=p.available;mandatory=p.mandatory;extra=np.setdiff1d(available,mandatory);m=p.extra_budget
    if method in ('FULL','REF20'):return available,dict(no_selection_opportunity=False)
    if method=='ORACLE_RESTRICT' and oracle_label is None:return None,dict(status='unavailable',reason='missing_nonfallback_reference_label')
    rule=np.array([p.scene.demands[p.coverage[b//6,(b//2)%3]].sum()/max(omega(WIDTHS[(b//2)%3])/p.omega_cap,1e-30) for b in extra])
    scores=rule.copy();meta=dict(no_selection_opportunity=len(extra)<=m)
    if method=='RANDOM':
        import hashlib
        seed=int.from_bytes(hashlib.sha256((p.case_id+':51001').encode()).digest()[:8],'little')
        values=np.random.default_rng(seed).random((len(p.centers),3,2));values.sort(axis=2);values=values[:,:,::-1]
        scores=values.reshape(-1)[extra]
    elif method=='AI' and len(extra)>m:
        if model is None:raise ValueError('AI_requires_pinned_model')
        f,g=features_graph(p);device=next(model.parameters()).device
        if device.type=='cuda':torch.cuda.synchronize(device)
        model.eval()
        with torch.no_grad():prob=model(torch.as_tensor(f,device=device),torch.as_tensor(g,device=device)).softmax(-1).cpu().numpy()
        if device.type=='cuda':torch.cuda.synchronize(device)
        scores=np.stack((prob[:,:,1:].sum(-1),prob[:,:,2]),-1).reshape(-1)[extra]
        meta['model_work']=dict(feature_raypairs=len(p.centers)*len(p.scene.demands),graph_raypairs=len(p.centers)**2,
            feature_shape=list(f.shape),graph_shape=list(g.shape),device=str(device),GPU_synchronized=device.type=='cuda')
    elif method=='ORACLE_RESTRICT':
        chosen=set(int(x) for x in oracle_label['active_slot_ids']);scores=rule+np.array([1e100 if b in chosen else 0 for b in extra])
        if len(chosen.intersection(extra))>m:raise ValueError('oracle_label_exceeds_extra_budget')
    elif method not in ('RULE','AI'):raise ValueError('unknown_method')
    chosen=sorted(range(len(extra)),key=lambda i:(-scores[i],int(extra[i])))[:m]
    return np.sort(np.r_[mandatory,extra[chosen]]),meta
