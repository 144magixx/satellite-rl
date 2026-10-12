"""From-scratch MLM, complete update checkpoints, and common proposal ranking."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .models import make_model, MODEL_FORMAT
from .evidence import EvidenceDataset, pool_features

def collate(samples,rng,device):
    b=len(samples);a=max(len(s['aircraft']) for s in samples);c=max(len(s['candidates']) for s in samples)
    values=dict(aircraft=np.zeros((b,a,8),np.float32),candidates=np.zeros((b,c,8),np.float32),aircraft_mask=np.zeros((b,a),bool),candidate_mask=np.zeros((b,c),bool),legal=np.zeros((b,a,c),bool),visible=np.full((b,a),-1,np.int64),labels=np.zeros((b,a),np.int64),hidden=np.zeros((b,a),bool))
    for i,s in enumerate(samples):
        na,nc=len(s['aircraft']),len(s['candidates'])
        values['aircraft'][i,:na]=s['aircraft'];values['candidates'][i,:nc]=s['candidates']
        values['aircraft_mask'][i,:na]=True;values['candidate_mask'][i,:nc]=True
        values['legal'][i,:na,:nc]=s['legal'];values['labels'][i,:na]=s['labels']
        hidden=np.zeros(na,bool);hidden[rng.choice(na,max(1,int(np.ceil(na*.3))),replace=False)]=True
        values['hidden'][i,:na]=hidden
        # Visible context is CURRENT geometry only, never teacher labels.
        values['visible'][i,:na]=np.where(hidden,-1,s['initial'])
    return {k:torch.as_tensor(v,device=device) for k,v in values.items()}

def loss(model,batch):
    logits=model(batch)['logits']
    return torch.stack([F.cross_entropy(logits[i,batch['hidden'][i]],batch['labels'][i,batch['hidden'][i]]) for i in range(len(logits))]).mean()

def atomic_save(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)

def train(dataset_manifest,output,protocol,device='cuda',resume=False,validation=None,progress=None):
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; forbidden CPU fallback')
    if protocol.get('updates',10000)!=10000 or protocol.get('batch',32)!=32 or protocol.get('microbatch',4)!=4: raise ValueError('fixed G16 update protocol')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    cp=root/'checkpoint.pt';log=root/'loss.jsonl';dataset=EvidenceDataset(dataset_manifest)
    sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    random.seed(51001);np.random.seed(51001);torch.manual_seed(51001)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(51001)
    rng=np.random.default_rng(51001); model=make_model(protocol.get('network')).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol.get('learning_rate',3e-4))
    cursor=0;nodes=[]
    if resume:
        state=torch.load(cp,map_location=device,weights_only=False)
        if state['protocol_sha256']!=sha or state['dataset_sha256']!=dataset.sha256:raise ValueError('resume protocol/evidence mismatch')
        prefix=log.read_bytes() if log.exists() else b''
        if hashlib.sha256(prefix[:state['log_bytes']]).hexdigest()!=state['log_sha256']:raise ValueError('loss prefix changed')
        # Uncommitted tail is kept as evidence before truncating to checkpoint boundary.
        if len(prefix)>state['log_bytes']:
            (root/f'uncommitted_loss_{time.time_ns()}.jsonl').write_bytes(prefix[state['log_bytes']:]);log.write_bytes(prefix[:state['log_bytes']])
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);cursor=state['update'];nodes=state['nodes']
        rng.bit_generator.state=state['sampler_rng'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    elif cp.exists() or log.exists():raise ValueError('existing run requires explicit resume')
    def checkpoint():
        prefix=log.read_bytes() if log.exists() else b''
        atomic_save(cp,dict(format=MODEL_FORMAT,model=model.state_dict(),optimizer=optimizer.state_dict(),update=cursor,nodes=nodes,protocol=protocol,protocol_sha256=sha,dataset_sha256=dataset.sha256,sampler_rng=rng.bit_generator.state,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],log_bytes=len(prefix),log_sha256=hashlib.sha256(prefix).hexdigest(),from_scratch=True,seed=51001))
    def validate_pending():
        if cursor in (2500,5000,7500,10000) and cursor not in nodes:
            if validation is None:raise ValueError('fixed geometry validation callback required')
            saved=(rng.bit_generator.state,random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            model.eval();validation(model,cursor,root/f'validation_{cursor:05d}');model.train()
            rng.bit_generator.state=saved[0];random.setstate(saved[1]);np.random.set_state(saved[2]);torch.set_rng_state(saved[3])
            if device.startswith('cuda'):torch.cuda.set_rng_state_all(saved[4])
            nodes.append(cursor);checkpoint()
    checkpoint();validate_pending()
    for update in range(cursor+1,10001):
        model.train();optimizer.zero_grad(set_to_none=True);samples,ids=dataset.draw(rng,32);total=0.
        for start in range(0,32,4):
            batch=collate(samples[start:start+4],rng,device);term=loss(model,batch)/8;term.backward();total+=float(term.detach())
        optimizer.step();cursor=update
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(update=cursor,loss=total,sample_ids=ids))+'\n')
        if cursor%100==0 or cursor in (2500,5000,7500,10000):checkpoint()
        if progress:progress(cursor,total)
        validate_pending()
    atomic_save(root/'final.pt',dict(format=MODEL_FORMAT,model=model.state_dict(),network=protocol.get('network'),update=10000,protocol_sha256=sha,dataset_sha256=dataset.sha256,from_scratch=True,seed=51001))
    return dict(update=cursor,nodes=nodes,checkpoint=str(cp))

def orderer(method,model=None,seed=51001,journal=None,device='cpu'):
    rng=np.random.default_rng(seed)
    def order(scene,current,pool):
        if method=='rule':indices=list(range(len(pool)));scores=None
        elif method=='random':indices=rng.permutation(len(pool)).tolist();scores=None
        elif method=='MLM':
            if model is None:raise ValueError('MLM model missing')
            features,mapping=pool_features(scene,current,pool)
            sample=dict(features,labels=features['initial']);batch=collate([sample],np.random.default_rng(0),device)
            # Full masked region; no random inference masking and no teacher visibility.
            batch['visible'].fill_(-1);batch['hidden']=batch['aircraft_mask'].clone()
            model.eval()
            with torch.inference_mode():logp=model(batch)['logits'][0].log_softmax(-1).cpu().numpy()
            scores=[float(logp[np.arange(len(scene.demands)),mapping(entry[1])].mean()) for entry in pool]
            indices=sorted(range(len(pool)),key=lambda i:(-scores[i],i))
        else:raise ValueError(method)
        if journal is not None:journal.append(dict(method=method,order=indices,scores=scores,pool_size=len(pool)))
        return [pool[i] for i in indices]
    return order
