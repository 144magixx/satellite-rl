"""From-scratch MLM, complete update checkpoints, and common proposal ranking."""
import os
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
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def atomic_json(path,value):
    temp=path.with_suffix('.tmp');
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def require_finite(values,phase):
    if any(not torch.isfinite(value).all().item() for value in values if isinstance(value,torch.Tensor)):
        raise FloatingPointError('nonfinite '+phase)

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
    cursor=0;nodes=[];ledger={'retained_updates':0,'unknown_attempts':0,'known_failed_attempts':0,'replayed_attempts':0};wal=root/'update_wal.json';history=root/'update_attempts.jsonl'
    if resume:
        state=torch.load(cp,map_location=device,weights_only=False)
        if state['protocol_sha256']!=sha or state['dataset_sha256']!=dataset.sha256:raise ValueError('resume protocol/evidence mismatch')
        prefix=log.read_bytes() if log.exists() else b''
        if hashlib.sha256(prefix[:state['log_bytes']]).hexdigest()!=state['log_sha256']:raise ValueError('loss prefix changed')
        # Uncommitted tail is kept as evidence before truncating to checkpoint boundary.
        if len(prefix)>state['log_bytes']:
            (root/f'uncommitted_loss_{time.time_ns()}.jsonl').write_bytes(prefix[state['log_bytes']:]);log.write_bytes(prefix[:state['log_bytes']])
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);cursor=state['update'];nodes=state['nodes'];ledger=state.get('ledger',ledger)
        rng.bit_generator.state=state['sampler_rng'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    elif cp.exists() or log.exists():raise ValueError('existing run requires explicit resume')
    if resume and wal.exists():
        pending=json.loads(wal.read_text())
        if pending['status']=='pending' and pending['update']>cursor:
            ledger['unknown_attempts']+=1;ledger['replayed_attempts']+=1
            with history.open('a') as f:f.write(json.dumps(dict(pending,recovery='unknown_attempt_may_have_executed',retained_cursor=cursor))+'\n')
            atomic_json(wal,dict(pending,status='recovered_unknown'))
        if pending['status']=='failed' and pending['update']>cursor:ledger['known_failed_attempts']+=1;ledger['replayed_attempts']+=1
        if ledger['unknown_attempts']>3:raise RuntimeError('bounded recovery exceeded three unknown attempts')
    initial_prefix=log.read_bytes() if log.exists() else b''
    log_hasher=hashlib.sha256(initial_prefix);log_bytes=len(initial_prefix)
    def checkpoint():
        atomic_save(cp,dict(format=MODEL_FORMAT,model=model.state_dict(),optimizer=optimizer.state_dict(),update=cursor,nodes=nodes,protocol=protocol,protocol_sha256=sha,dataset_sha256=dataset.sha256,sampler_rng=rng.bit_generator.state,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],log_bytes=log_bytes,log_sha256=log_hasher.hexdigest(),from_scratch=True,seed=51001,ledger=ledger))
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
        attempt=dict(update=update,status='pending',attempt_id=time.time_ns(),started_unix=time.time())
        atomic_json(wal,attempt)
        try:
            model.train();optimizer.zero_grad(set_to_none=True);samples,ids=dataset.draw(rng,32);total=0.
            for start in range(0,32,4):
                batch=collate(samples[start:start+4],rng,device);term=loss(model,batch)/8
                require_finite([term],'loss');term.backward();total+=float(term.detach())
            require_finite([p.grad for p in model.parameters() if p.grad is not None],'gradient')
            optimizer.step()
            require_finite(list(model.parameters()),'parameter')
            require_finite([v for state in optimizer.state.values() for v in state.values()],'Adam state')
        except Exception as exc:
            failure=dict(attempt,status='failed',error=repr(exc),retained_cursor=cursor,ended_unix=time.time())
            with history.open('a') as f:f.write(json.dumps(failure)+'\n')
            atomic_json(wal,failure)
            # Last good checkpoint remains byte-for-byte untouched.
            raise
        cursor=update;ledger['retained_updates']=cursor
        row=(json.dumps(dict(update=cursor,loss=total,sample_ids=ids,attempt_id=attempt['attempt_id']))+'\n').encode('utf-8')
        with log.open('ab') as f:f.write(row);f.flush();os.fsync(f.fileno())
        log_hasher.update(row);log_bytes+=len(row)
        checkpoint()
        committed=dict(attempt,status='committed',ended_unix=time.time())
        atomic_json(wal,committed)
        with history.open('a') as f:f.write(json.dumps(committed)+'\n')
        if progress:progress(cursor,total)
        validate_pending()
    atomic_save(root/'final.pt',dict(format=MODEL_FORMAT,model=model.state_dict(),network=protocol.get('network'),update=10000,protocol_sha256=sha,dataset_sha256=dataset.sha256,from_scratch=True,seed=51001))
    return dict(update=cursor,nodes=nodes,checkpoint=str(cp),ledger=ledger)

def orderer(method,model=None,seed=51001,journal=None,device='cpu'):
    rng=np.random.default_rng(seed);deadline=[float('inf')]
    def order(scene,current,pool):
        if method=='rule':indices=list(range(len(pool)));scores=None
        elif method=='random':indices=rng.permutation(len(pool)).tolist();scores=None
        elif method=='MLM':
            if model is None:raise ValueError('MLM model missing')
            if time.perf_counter()>=deadline[0]:return list(pool)
            features,mapping=pool_features(scene,current,pool)
            if time.perf_counter()>=deadline[0]:
                if journal is not None:journal.append(dict(method=method,stop_reason='features_deadline',pool_size=len(pool)))
                return list(pool)
            sample=dict(features,labels=features['initial']);batch=collate([sample],np.random.default_rng(0),device)
            # Full masked region; no random inference masking and no teacher visibility.
            batch['visible'].fill_(-1);batch['hidden']=batch['aircraft_mask'].clone()
            model.eval()
            if time.perf_counter()>=deadline[0]:return list(pool)
            with torch.inference_mode():logp=model(batch)['logits'][0].log_softmax(-1).cpu().numpy()
            scores=[float(logp[np.arange(len(scene.demands)),mapping(entry[1])].mean()) for entry in pool]
            indices=sorted(range(len(pool)),key=lambda i:(-scores[i],i))
        else:raise ValueError(method)
        if journal is not None:journal.append(dict(method=method,order=indices,scores=scores,pool_size=len(pool)))
        return [pool[i] for i in indices]
    order.set_deadline=lambda value:deadline.__setitem__(0,value)
    return order
