"""One R6 learner; global16 loss and resumable independent PCG64 order."""
from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
from .features import collate_features, record_sha


def solver_identity():
    if scipy.__version__ != '1.14.1': raise RuntimeError('R6 requires SciPy==1.14.1')
    return dict(version=scipy.__version__, package_source=scipy.__file__, platform=platform.platform())


@dataclass
class TeacherTarget:
    aircraft_ids: tuple
    owners: np.ndarray
    anchor_ids: tuple
    slot_ids: tuple
    identity: dict = None
    label_ref_sha: str = None

    def ordered(self):
        n = len(self.aircraft_ids); k = len(self.anchor_ids)
        raw=np.asarray(self.owners)
        if raw.shape!=(n,) or raw.dtype.kind not in 'iu' or not np.isfinite(raw).all():raise ValueError('teacher owners must be actual integer vector')
        owners = np.asarray(raw, np.int64)
        if owners.shape != (n,) or not 1 <= k <= min(200,n) or len(self.slot_ids) != k or (owners < 0).any() or (owners >= k).any(): raise ValueError('teacher target invalid')
        if len(set(self.aircraft_ids)) != n or len(set(self.slot_ids)) != k or any(a not in self.aircraft_ids for a in self.anchor_ids): raise ValueError('teacher IDs invalid')
        members = [np.flatnonzero(owners==j) for j in range(k)]
        if any(not len(m) for m in members): raise ValueError('empty teacher beam')
        columns = sorted(range(k), key=lambda j:(tuple(sorted(self.aircraft_ids[i] for i in members[j])),self.anchor_ids[j],self.slot_ids[j]))
        return owners, members, columns


def match_targets(output, targets):
    if len(targets)!=output['enable'].shape[0] or output['valid'].shape[0]!=len(targets):raise ValueError('matching batch cardinality')
    solver = solver_identity(); records = []
    for b,t in enumerate(targets):
        owners,members,columns = t.ordered(); n=len(t.aircraft_ids)
        valid=output['valid'][b]
        if int(valid.sum()) != n or not valid[:n].all() or valid[n:].any(): raise ValueError('target actual-prefix padding mismatch')
        # Detached CPU float64 logits also make the cost arithmetic explicitly float64.
        enable = -F.logsigmoid(output['enable'][b].detach().cpu().double())
        owner = output['owner'][b,:n].detach().cpu().double().log_softmax(dim=-1)
        pointer = output['pointer'][b,:,:n].detach().cpu().double().log_softmax(dim=-1)
        cost = torch.stack([enable-owner[members[j]].mean(dim=0)-pointer[:,t.aircraft_ids.index(t.anchor_ids[j])] for j in columns],dim=1)
        matrix=np.ascontiguousarray(cost.detach().cpu().numpy(),dtype=np.float64)
        if not np.isfinite(matrix).all(): raise ValueError('nonfinite matching cost')
        rows,cols=linear_sum_assignment(matrix)
        mapping={columns[int(c)]:int(r) for r,c in zip(rows,cols)}
        records.append(dict(mapping=mapping,solver=solver,query_rows=rows.tolist(),ordered_column_indices=columns,
                            solver_columns=cols.tolist(),cost_sha=record_sha(matrix.tolist())))
    return records


def compute_loss_sums(output, targets, matches=None):
    if len(targets)!=output['enable'].shape[0]:raise ValueError('loss batch cardinality')
    matches=match_targets(output,targets) if matches is None else matches
    if len(matches)!=len(targets):raise ValueError('matching result cardinality')
    E=output['enable'].sum()*0; O=E; P=E
    for b,(t,match) in enumerate(zip(targets,matches)):
        t.ordered()
        mapping=match['mapping']; n=len(t.aircraft_ids)
        if set(mapping)!=set(range(len(t.anchor_ids))) or len(set(mapping.values()))!=len(mapping) or any(not 0<=q<200 for q in mapping.values()):raise ValueError('incomplete matching map')
        if int(output['valid'][b].sum())!=n or not output['valid'][b,:n].all() or output['valid'][b,n:].any():raise ValueError('loss actual-prefix padding mismatch')
        y=torch.zeros_like(output['enable'][b]); y[list(mapping.values())]=1
        E=E+(F.binary_cross_entropy_with_logits(output['enable'][b],y,reduction='none')*torch.where(y>0,1.,.1)).sum()
        owner_target=torch.tensor([mapping[int(j)] for j in t.owners],device=y.device)
        O=O+F.cross_entropy(output['owner'][b,:n],owner_target,reduction='sum')
        queries=list(mapping.values()); anchors=[t.aircraft_ids.index(t.anchor_ids[j]) for j in mapping]
        P=P+F.cross_entropy(output['pointer'][b,queries,:n],torch.tensor(anchors,device=y.device),reduction='sum')
    return dict(E=E,O=O,P=P,matches=matches)


def global_loss(sums,A,G,B=16):
    if B!=16 or A<=0 or G<=0: raise ValueError('fixed effective batch16 and actual positive denominators')
    return sums['E']/3200+sums['O']/A+sums['P']/G


class CaseSampler:
    def __init__(self, qualified_table):
        self.table=copy.deepcopy(qualified_table)
        indices=[r['case_index'] for r in self.table]
        if not indices or indices!=sorted(set(indices)) or any(not r.get('case_id') for r in self.table): raise ValueError('ascending immutable qualified case table')
        self.table_sha=record_sha(self.table); self.rng=np.random.Generator(np.random.PCG64(51001))
        self.epoch=0; self.permutation=self.rng.permutation(len(self.table)); self.cursor=0; self.pending=[];self.committed_batches=0

    def assemble(self):
        if self.pending: return copy.deepcopy(self.pending)
        while len(self.pending)<16:
            if self.cursor==len(self.table):
                self.epoch+=1; self.permutation=self.rng.permutation(len(self.table)); self.cursor=0
            take=min(16-len(self.pending),len(self.table)-self.cursor)
            self.pending.extend(int(i) for i in self.permutation[self.cursor:self.cursor+take]); self.cursor+=take
        return copy.deepcopy(self.pending)

    def commit(self):
        if len(self.pending)!=16: raise ValueError('no assembled full batch')
        self.pending=[];self.committed_batches+=1

    def state_dict(self):
        return dict(table_sha=self.table_sha,rng=copy.deepcopy(self.rng.bit_generator.state),epoch=self.epoch,
                    permutation=self.permutation.tolist(),cursor=self.cursor,pending=self.pending.copy(),
                    pending_case_ids=[self.table[i]['case_id'] for i in self.pending],committed_batches=self.committed_batches)

    def load_state_dict(self,state):
        if state['table_sha']!=self.table_sha: raise ValueError('qualified table changed')
        committed=state.get('committed_batches')
        if not isinstance(committed,int) or not 0<=committed<=6000:raise ValueError('committed batch count invalid')
        expected=CaseSampler(self.table)
        for _ in range(committed):expected.assemble();expected.commit()
        if state.get('pending'):expected.assemble()
        if expected.state_dict()!=state:raise ValueError('sampler cursor/permutation/RNG/pending inconsistent with fixed seed history')
        permutation=np.asarray(state['permutation'],np.int64)
        if sorted(permutation.tolist())!=list(range(len(self.table))) or not 0<=state['cursor']<=len(self.table): raise ValueError('sampler state invalid')
        self.rng.bit_generator.state=copy.deepcopy(state['rng']); self.epoch=state['epoch'];self.permutation=permutation
        self.cursor=state['cursor']; self.pending=list(state['pending'])
        self.committed_batches=committed
        if len(self.pending) not in (0,16) or any(not isinstance(i,int) or not 0<=i<len(self.table) for i in self.pending):raise ValueError('pending batch invalid')
        if state['pending_case_ids']!=[self.table[i]['case_id'] for i in self.pending]: raise ValueError('pending identity changed')


class TrainingSession:
    """Ops supplies authorized cases; this class never opens dataset files."""
    def __init__(self,model,stats,sampler,provenance,device='cuda'):
        self.solver=solver_identity(); self.model=model.to(device);self.stats=stats;self.sampler=sampler;self.provenance=copy.deepcopy(provenance)
        self.device=device;self.optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,
            betas=(.9,.999),eps=1e-8,amsgrad=False,foreach=False,fused=False)
        self.updates=0;self.active_seconds=0.;self.active_started=None;self.clock_path=None;self.clock_events=[]

    def _clock_event(self,event,**fields):
        if self.clock_path is None:raise ValueError('durable activity clock path required')
        self.clock_events.append(dict(event=event,wall_time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),
            charged_active_seconds=self.elapsed(),update=self.updates,**fields))
        record=dict(schema='R6ActivityClockV1',events=self.clock_events,charged_active_seconds=self.elapsed())
        path=Path(self.clock_path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('w',encoding='utf8') as stream:json.dump(record,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)

    def start_activity(self,*,clock_path=None):
        """Call BEFORE case supply/load/feature/transfer, including after node pause."""
        if clock_path is not None:self.clock_path=str(clock_path)
        if self.clock_path is None:raise ValueError('clock must start before data supply with durable path')
        if self.active_started is None:
            self.active_started=time.monotonic();self._clock_event('segment_start_before_data_supply')

    def elapsed(self):return self.active_seconds+(0 if self.active_started is None else time.monotonic()-self.active_started)

    def pause_activity(self,*,reason):
        if reason not in ('resource_queue','explicit_node_evaluation'):raise ValueError('only contracted excluded intervals')
        self.active_seconds=self.elapsed();self.active_started=None;self._clock_event('segment_pause',reason=reason)

    def validate_batch(self,cases,targets):
        if len(cases)!=16 or len(targets)!=16 or len(self.sampler.pending)!=16:raise ValueError('assemble sampler full16 before supply')
        for index,case,target in zip(self.sampler.pending,cases,targets):
            expected=self.sampler.table[index]
            identity=case.identity
            if not isinstance(identity,dict) or identity.get('role')!='train':raise ValueError('training feature identity missing/role')
            for key in ('case_index','case_id','input_sha'):
                if key not in expected or identity.get(key)!=expected[key]:raise ValueError('supplied batch differs from sampler '+key)
            if case.input_sha!=expected['input_sha'] or tuple(case.aircraft_ids)!=tuple(target.aircraft_ids):raise ValueError('original input/aircraft order mismatch')
            if target.identity!=identity or not target.label_ref_sha or expected.get('label_ref_sha')!=target.label_ref_sha:raise ValueError('teacher original T label/case binding mismatch')

    def step(self,cases,targets,microbatch=4):
        if len(cases)!=16 or len(targets)!=16 or not 1<=microbatch<=16: raise ValueError('full batch16 required')
        if self.updates>=6000 or self.elapsed()>=28800:raise RuntimeError('science cap reached')
        if self.updates==2000 and not self.provenance.get('node2000_gate_pass'):raise RuntimeError('node2000 authorization absent')
        self.validate_batch(cases,targets)
        if self.active_started is None:raise ValueError('start activity before data supply; step cannot start clock')
        A=sum(len(t.aircraft_ids) for t in targets);G=sum(len(t.anchor_ids) for t in targets)
        self.optimizer.zero_grad(set_to_none=True);self.model.train(); totals=dict(E=0.,O=0.,P=0.);matching=[]
        for begin in range(0,16,microbatch):
            end=min(begin+microbatch,16);out=self.model(**collate_features(cases[begin:end],self.stats,self.device))
            sums=compute_loss_sums(out,targets[begin:end]);global_loss(sums,A,G).backward();matching.extend(sums['matches'])
            for name in totals:totals[name]+=float(sums[name].detach())
        if self.elapsed()>=28800:raise RuntimeError('activity cap before clip; no optimizer update')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
        if self.elapsed()>=28800:raise RuntimeError('activity cap before optimizer; no update')
        self.optimizer.step();self.updates+=1;self.sampler.commit()
        return dict(update=self.updates,**totals,loss=totals['E']/3200+totals['O']/A+totals['P']/G,A=A,G=G,gradient_norm=float(norm),matching=matching,active_seconds=self.elapsed())

    def checkpoint(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        state=dict(schema='R6TrainingCheckpointV1',model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            sampler=self.sampler.state_dict(),stats=self.stats.to_record(),provenance=self.provenance,solver=self.solver,
            updates=self.updates,active_seconds=self.elapsed(),checkpoint_completed_optimizer_step=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        checkpoint_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        self._clock_event('checkpoint_IO_completed',checkpoint_file_sha=checkpoint_sha,
            checkpoint_state_active_seconds=state['active_seconds'])
        return dict(checkpoint_file_sha=checkpoint_sha,charged_active_seconds=self.elapsed(),clock_path=self.clock_path)

    def restore(self,path,*,recovery_clock):
        """Root/ops must supply a hashed verified recovery interval accounting file."""
        checkpoint_sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        evidence_path=Path(recovery_clock['evidence_path']);raw=evidence_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=recovery_clock['evidence_sha']:raise ValueError('recovery clock evidence SHA')
        evidence=json.loads(raw)
        if evidence.get('verified') is not True or evidence.get('checkpoint_file_sha')!=checkpoint_sha or evidence.get('unknown_tail_accounted') is not True or not evidence.get('interval_evidence'):raise ValueError('verified clock recovery required')
        state=torch.load(path,map_location=self.device,weights_only=False)
        charged=float(evidence['charged_active_seconds'])
        if not np.isfinite(charged) or charged<state['active_seconds'] or charged<float(evidence['last_completed_checkpoint_IO_active_seconds']):raise ValueError('recovery loses charged activity')
        if state['schema']!='R6TrainingCheckpointV1' or state['provenance']!=self.provenance or state['solver']!=self.solver or state['stats']!=self.stats.to_record():raise ValueError('resume provenance mismatch')
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer'])
        if state['sampler']['committed_batches']!=state['updates']:raise ValueError('checkpoint update/sampler count mismatch')
        self.sampler.load_state_dict(state['sampler']);torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        self.updates=state['updates'];self.active_seconds=charged;self.active_started=None
        self.clock_path=recovery_clock['clock_path'];self.clock_events=copy.deepcopy(evidence.get('prior_clock_events',[]))
        self._clock_event('verified_recovery',checkpoint_file_sha=checkpoint_sha,recovery_evidence_sha=recovery_clock['evidence_sha'])
