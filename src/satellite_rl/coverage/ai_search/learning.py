"""From-scratch Task ranking with balanced sampling and durable update WAL."""
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import GeometryConfig, SearchConfig
from .features import FEATURE_SCHEMA, Normalizer, pool_features, preference_index
from .teacher import TEACHER_FORMAT, digest, atomic_json, strict_lex_better

MODEL_FORMAT="s1_ai17_task_ranker_v2"


class TaskRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(53,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.score_head=nn.Linear(128,1)
        self.improvement_head=nn.Linear(128,1)

    def forward(self,x):
        hidden=self.encoder(x)
        return dict(score=self.score_head(hidden).squeeze(-1),
                    improvement_logit=self.improvement_head(hidden).squeeze(-1))


def make_model(): return TaskRanker()


class EvidenceDataset:
    """Teacher manifests pinned by SHA; mother/profile/parent uniform draws.

    An auxiliary draw includes completed infeasible and non-improving tasks.
    Pair draws are restricted to feasible completed tasks sharing scene, parent,
    and preference. Censored rows contribute normalization inputs only.
    """
    def __init__(self,path):
        self.path=Path(path); manifest=json.loads(self.path.read_text(encoding="utf-8"))
        self.rows=[]; self.pairs={}; self.aux={}; self.sha256=digest(self.path)
        if manifest.get("role","training")!="training": raise ValueError("training evidence role required")
        if manifest.get("format")==TEACHER_FORMAT:
            records=[dict(path=self.path.name,sha256=self.sha256)]
        else:
            if manifest.get("status")!="completed": raise ValueError("incomplete teacher collection")
            records=manifest.get("cases",manifest.get("manifests",[]))
        groups={}; evidence_hashes=[]; comparison=None
        for record in records:
            item=record.get("manifest",record)
            p=Path(item["path"])
            if not p.is_absolute(): p=self.path.parent/p
            if digest(p)!=item["sha256"]: raise ValueError("teacher manifest hash mismatch")
            case=json.loads(p.read_text(encoding="utf-8"))
            case_comparison=case.get("metric_comparison",dict(metric_abs_tolerances=SearchConfig().metric_abs_tolerances,
                                                           metric_relative_tolerance=SearchConfig().metric_relative_tolerance))
            cfg=SearchConfig(**case_comparison)
            pinned_comparison=(cfg.metric_abs_tolerances,cfg.metric_relative_tolerance)
            if comparison is not None and comparison!=pinned_comparison: raise ValueError("mixed teacher metric tolerances")
            comparison=pinned_comparison; self.search_config=cfg
            if (case.get("format")!=TEACHER_FORMAT or case.get("status")!="completed"
                or case.get("role","training")!="training" or case.get("feature_schema")!=FEATURE_SCHEMA):
                raise ValueError("teacher format/status/feature/training source mismatch")
            for evidence in ([case["scene"]]+case["parents"]+
                ([case["candidate_evidence"]] if case["candidate_evidence"] else [])+
                ([case["initialization"]] if case.get("initialization") else [])):
                ep=(p.parent/evidence["path"]).resolve()
                if not ep.is_relative_to(p.parent.resolve()): raise ValueError("teacher evidence escaped case path")
                if digest(ep)!=evidence.get("file_sha256",evidence["sha256"]): raise ValueError("teacher evidence hash mismatch")
            evidence_hashes.append(item["sha256"])
            for row in case["rows"]:
                values=np.asarray(row["task_predecode_features"],np.float32)
                if values.shape!=(53,) or not np.isfinite(values).all(): raise ValueError("bad teacher features")
                index=len(self.rows); self.rows.append(dict(row,features=values))
                key=(str(row["mother_id"]),row["profile"],row["scene_sha256"],row["parent_layout_sha256"],int(row["preference_index"]))
                groups.setdefault(key,[]).append(index)
        if not self.rows: raise ValueError("no teacher inputs")
        self.normalizer=Normalizer.fit(np.stack([r["features"] for r in self.rows]))
        for key,indices in groups.items():
            completed=[i for i in indices if self.rows[i]["completion_status"] in
                ("completed","no_feasible_found_under_pool") and not self.rows[i].get("zero_total_request",False)]
            if completed: self.aux[key]=completed
            feasible=[i for i in completed if self.rows[i]["completion_status"]=="completed" and self.rows[i]["completed_best_metrics"] is not None]
            pairs=[]
            for a_pos,a in enumerate(feasible):
                for b in feasible[a_pos+1:]:
                    av,bv=self.rows[a]["completed_best_metrics"],self.rows[b]["completed_best_metrics"]
                    pref=key[-1]
                    if strict_lex_better(av,bv,pref,self.search_config): pairs.append((a,b))
                    elif strict_lex_better(bv,av,pref,self.search_config): pairs.append((b,a))
            if pairs: self.pairs[key]=pairs
        if not self.pairs or not self.aux: raise ValueError("no completed non-tied teacher pairs; scientific learning gate must stop")
        self._pair_tree=self._tree(self.pairs); self._aux_tree=self._tree(self.aux)
        self.counts=dict(inputs=len(self.rows),censored=sum(r["completion_status"]=="censored" for r in self.rows),
                        pairs=sum(len(v) for v in self.pairs.values()),completed=sum(len(v) for v in self.aux.values()),
                        eligible_pair_parent_groups=len(self.pairs),eligible_auxiliary_parent_groups=len(self.aux),
                        pair_mothers=len({k[0] for k in self.pairs}),auxiliary_mothers=len({k[0] for k in self.aux}),
                        zero_request_rows=sum(r.get("zero_total_request",False) for r in self.rows))
        self.evidence_sha256=hashlib.sha256(json.dumps(evidence_hashes,sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _tree(mapping):
        tree={}
        for key in sorted(mapping): tree.setdefault(key[0],{}).setdefault(key[1],[]).append(key)
        return tree

    @staticmethod
    def _draw_group(tree,rng):
        mothers=sorted(tree); mother=mothers[int(rng.integers(len(mothers)))]
        profiles=sorted(tree[mother]); profile=profiles[int(rng.integers(len(profiles)))]
        parents=tree[mother][profile]; return parents[int(rng.integers(len(parents)))]

    def draw(self,rng,size=64):
        pairs=[]; aux=[]
        for _ in range(size):
            key=self._draw_group(self._pair_tree,rng); choices=self.pairs[key]
            pairs.append(choices[int(rng.integers(len(choices)))])
            key=self._draw_group(self._aux_tree,rng); choices=self.aux[key]
            aux.append(choices[int(rng.integers(len(choices)))])
        a=[i for i,j in pairs]; b=[j for i,j in pairs]
        batch=dict(preferred=self.normalizer.transform(np.stack([self.rows[i]["features"] for i in a])),
            other=self.normalizer.transform(np.stack([self.rows[i]["features"] for i in b])),
            auxiliary=self.normalizer.transform(np.stack([self.rows[i]["features"] for i in aux])),
            auxiliary_target=np.array([bool(self.rows[i]["new_nondominated"]) for i in aux],np.float32))
        return batch,dict(pairs=[list(x) for x in pairs],auxiliary=aux)


def loss(model,batch):
    preferred=model(batch["preferred"])["score"]; other=model(batch["other"])["score"]
    pair=F.softplus(-(preferred-other)).mean()
    auxiliary=F.binary_cross_entropy_with_logits(model(batch["auxiliary"])["improvement_logit"],batch["auxiliary_target"])
    return pair+.1*auxiliary,pair,auxiliary


def atomic_save(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: torch.save(value,f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def require_finite(values,phase):
    if any(not torch.isfinite(v).all().item() for v in values if isinstance(v,torch.Tensor)):
        raise FloatingPointError("nonfinite "+phase)


@contextmanager
def isolated_evaluation_rng(sampler,device):
    saved=(copy.deepcopy(sampler.bit_generator.state),random.getstate(),np.random.get_state(),torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [])
    try: yield
    finally:
        sampler.bit_generator.state=saved[0]; random.setstate(saved[1]); np.random.set_state(saved[2]); torch.set_rng_state(saved[3])
        if str(device).startswith("cuda"): torch.cuda.set_rng_state_all(saved[4])


def train(dataset_manifest,output,protocol=None,device="cuda",resume=False,validation=None,progress=None,stop=None):
    protocol=dict(protocol or {}); engineering=bool(protocol.get("engineering",False))
    updates=int(protocol.get("updates",10000)); batch=int(protocol.get("batch",protocol.get("effective_batch",64)))
    microbatch=int(protocol.get("microbatch",batch))
    validation_updates=tuple(protocol.get("engineering_validation_nodes",())) if engineering else (2500,5000,7500,10000)
    if (len(validation_updates)>4 or len(set(validation_updates))!=len(validation_updates)
        or any(type(n) is not int or not 1<=n<=updates for n in validation_updates)):
        raise ValueError("bounded validation node schedule required")
    if ((not engineering and updates!=10000) or (engineering and not 1<=updates<=10) or batch!=64
        or batch%microbatch or int(protocol.get("seed",51001))!=51001
        or float(protocol.get("learning_rate",.0003))!=.0003 or float(protocol.get("weight_decay",.0001))!=.0001
        or float(protocol.get("gradient_clip_norm",1))!=1):
        raise ValueError("frozen AI17 seed51001/10000 updates/batch64/AdamW protocol required")
    if str(device).startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    if validation_updates and validation is None: raise ValueError("fixed-node validation callback required")
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    checkpoint_path=root/"checkpoint.pt"; log=root/"loss.jsonl"; wal=root/"update_wal.json"; history=root/"update_attempts.jsonl"
    dataset=EvidenceDataset(dataset_manifest); normalizer=dataset.normalizer
    protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    random.seed(51001); np.random.seed(51001); torch.manual_seed(51001)
    if str(device).startswith("cuda"): torch.cuda.manual_seed_all(51001)
    sampler=np.random.default_rng(51001); model=make_model().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    cursor=0; nodes=[]; nodes_requested=[]; node_status={}
    ledger=dict(retained_updates=0,unknown_attempts=0,known_failed_attempts=0,replayed_attempts=0,
                                  sampler_examples=0,committed_attempts=0)
    if resume:
        state=torch.load(checkpoint_path,map_location=device,weights_only=False)
        if state["format"]!=MODEL_FORMAT or state["protocol_sha256"]!=protocol_sha or state["dataset_sha256"]!=dataset.sha256 or state["evidence_sha256"]!=dataset.evidence_sha256:
            raise ValueError("resume source/protocol/evidence mismatch")
        normalizer=Normalizer.from_record(state["normalizer"])
        if normalizer.to_record()!=dataset.normalizer.to_record(): raise ValueError("training normalization changed")
        prefix=log.read_bytes() if log.exists() else b""
        if len(prefix)<state["log_bytes"] or hashlib.sha256(prefix[:state["log_bytes"]]).hexdigest()!=state["log_sha256"]:
            raise ValueError("loss prefix changed")
        if len(prefix)>state["log_bytes"]:
            (root/f"uncommitted_loss_{time.time_ns()}.jsonl").write_bytes(prefix[state["log_bytes"]:]); log.write_bytes(prefix[:state["log_bytes"]])
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        cursor=state["update"]; nodes=state["nodes"]; ledger=state["ledger"]
        if not 0<=cursor<=updates: raise ValueError("checkpoint update exceeds frozen training budget")
        nodes_requested=state.get("nodes_requested",[]); node_status=state.get("node_status",{})
        sampler.bit_generator.state=state["sampler_rng"]; random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"]); torch.set_rng_state(state["torch_rng"].cpu())
        if str(device).startswith("cuda"): torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        if wal.exists():
            pending=json.loads(wal.read_text())
            if pending["update"]>cursor and pending["status"] in ("pending","failed"):
                key="unknown_attempts" if pending["status"]=="pending" else "known_failed_attempts"
                ledger[key]+=1; ledger["replayed_attempts"]+=1
                with history.open("a") as f: f.write(json.dumps(dict(pending,recovery=key,retained_cursor=cursor))+"\n")
                atomic_json(wal,dict(pending,status="recovered_"+key))
            if ledger["unknown_attempts"]>3: raise RuntimeError("bounded unknown-recovery ceiling exceeded")
    elif checkpoint_path.exists() or log.exists(): raise ValueError("existing training requires explicit resume")
    prefix=log.read_bytes() if log.exists() else b""; log_hasher=hashlib.sha256(prefix); log_bytes=len(prefix)
    def checkpoint():
        atomic_save(checkpoint_path,dict(format=MODEL_FORMAT,model=model.state_dict(),optimizer=optimizer.state_dict(),
            normalizer=normalizer.to_record(),feature_schema=FEATURE_SCHEMA,update=cursor,nodes=nodes,
            nodes_requested=nodes_requested,node_status=node_status,protocol=protocol,
            protocol_sha256=protocol_sha,dataset_sha256=dataset.sha256,evidence_sha256=dataset.evidence_sha256,
            sampler_rng=copy.deepcopy(sampler.bit_generator.state),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
            log_bytes=log_bytes,log_sha256=log_hasher.hexdigest(),ledger=ledger,from_scratch=True,seed=51001,
            sample_cursor=cursor*batch,dataset_counts=dataset.counts,
            model_parameters=sum(p.numel() for p in model.parameters()),engineering=engineering))
    def validate_pending():
        if cursor in validation_updates and str(cursor) not in node_status:
            # Preserve the actual fixed-node model even if validation crashes.
            atomic_save(root/f"checkpoint_{cursor:05d}.pt",torch.load(checkpoint_path,map_location="cpu",weights_only=False))
            model.eval()
            try:
                with isolated_evaluation_rng(sampler,device):
                    response=validation(model,normalizer,cursor,root/f"validation_{cursor:05d}")
            finally: model.train()
            status=response.get("status","completed") if isinstance(response,dict) else "completed"
            if status not in ("requested","completed"): raise ValueError("validation callback must report requested or completed")
            node_status[str(cursor)]=dict(status=status,receipt=response)
            if status=="requested": nodes_requested.append(cursor)
            else: nodes.append(cursor)
            checkpoint()
            atomic_save(root/f"checkpoint_{cursor:05d}.pt",torch.load(checkpoint_path,map_location="cpu",weights_only=False))
    checkpoint(); validate_pending()
    for update in range(cursor+1,updates+1):
        if stop is not None and stop(): return dict(status="stopped",update=cursor,nodes=nodes,nodes_requested=nodes_requested,
                                                   node_status=node_status,checkpoint=str(checkpoint_path),ledger=ledger)
        attempt=dict(update=update,status="pending",attempt_id=time.time_ns(),started_unix=time.time())
        atomic_json(wal,attempt)
        try:
            model.train(); optimizer.zero_grad(set_to_none=True); raw,sample_ids=dataset.draw(sampler,batch)
            totals=np.zeros(3)
            for start in range(0,batch,microbatch):
                tensor={k:torch.as_tensor(v[start:start+microbatch],device=device) for k,v in raw.items()}
                terms=loss(model,tensor); term=terms[0]*(microbatch/batch)
                require_finite(terms,"loss"); term.backward()
                totals+=np.array([float(t.detach()) for t in terms])*(microbatch/batch)
            require_finite([p.grad for p in model.parameters() if p.grad is not None],"gradient")
            gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            require_finite([gradnorm],"gradient norm"); optimizer.step()
            require_finite(list(model.parameters()),"parameters")
            require_finite([v for st in optimizer.state.values() for v in st.values()],"AdamW state")
            if str(device).startswith("cuda"): torch.cuda.synchronize(device)
        except Exception as exc:
            failed=dict(attempt,status="failed",error=repr(exc),retained_cursor=cursor,ended_unix=time.time())
            with history.open("a") as f: f.write(json.dumps(failed)+"\n")
            atomic_json(wal,failed); raise
        cursor=update; ledger["retained_updates"]=cursor
        ledger["sampler_examples"]=cursor*batch; ledger["committed_attempts"]=cursor
        row=(json.dumps(dict(update=cursor,loss=float(totals[0]),pair_loss=float(totals[1]),aux_loss=float(totals[2]),
            gradient_norm=float(gradnorm),sample_ids=sample_ids,attempt_id=attempt["attempt_id"]))+"\n").encode()
        with log.open("ab") as f: f.write(row); f.flush(); os.fsync(f.fileno())
        log_hasher.update(row); log_bytes+=len(row); checkpoint()
        committed=dict(attempt,status="committed",ended_unix=time.time()); atomic_json(wal,committed)
        with history.open("a") as f: f.write(json.dumps(committed)+"\n")
        if progress: progress(cursor,float(totals[0]))
        validate_pending()
    atomic_save(root/"final.pt",torch.load(checkpoint_path,map_location="cpu",weights_only=False))
    return dict(status="completed",update=cursor,nodes=nodes,nodes_requested=nodes_requested,node_status=node_status,
                checkpoint=str(checkpoint_path),final=str(root/"final.pt"),ledger=ledger)


def orderer(method,model=None,normalizer=None,seed=51001,journal=None,device="cpu",geometry=None,variant_count=8):
    method=method.upper(); rng=np.random.default_rng(seed); state=dict(deadline=float("inf"),budget=None,preference=0)
    geometry=GeometryConfig(**geometry) if isinstance(geometry,dict) else (geometry or GeometryConfig())
    def check():
        if state["budget"] is not None: state["budget"].check()
        elif time.perf_counter()>=state["deadline"]:
            from .search import BudgetStop
            raise BudgetStop("time_budget")
    def order(scene,current,pool):
        check(); scores=None; start=time.perf_counter()
        if method in ("RULE","INIT"): indices=list(range(len(pool)))
        elif method=="RANDOM": indices=rng.permutation(len(pool)).tolist()
        elif method=="AI":
            if model is None or normalizer is None: raise ValueError("AI checkpoint and training normalizer required")
            pref=pool[0].preference_index if pool and hasattr(pool[0],"preference_index") else state["preference"]
            features=pool_features(scene,current,pool,pref,geometry,variant_count,state["budget"])
            check(); inputs=normalizer.transform(features); feature_end=time.perf_counter()
            if str(device).startswith("cuda"): torch.cuda.synchronize(device)
            model.eval()
            with torch.inference_mode(): scores=model(torch.as_tensor(inputs,device=device))["score"].detach()
            if str(device).startswith("cuda"): torch.cuda.synchronize(device)
            scores=scores.cpu().numpy().tolist(); check()
            if not np.isfinite(scores).all(): raise FloatingPointError("nonfinite task scores")
            indices=sorted(range(len(pool)),key=lambda i:(-scores[i],i))
        else: raise ValueError("unknown orderer method")
        check()
        if journal is not None: journal.append(dict(method=method,order=indices,scores=scores,pool_size=len(pool),
            seconds=time.perf_counter()-start,feature_seconds=feature_end-start if method=="AI" else 0.,
            inference_seconds=time.perf_counter()-feature_end if method=="AI" else 0.))
        return [pool[i] for i in indices]
    order.set_deadline=lambda value:state.__setitem__("deadline",value)
    order.set_budget=lambda value:state.__setitem__("budget",value)
    order.set_preference=lambda value:state.__setitem__("preference",preference_index(value))
    return order


def load_orderer(reference,device="cuda",allow_engineering=False,allow_intermediate=False,geometry=None,variant_count=8):
    started=time.perf_counter(); path=Path(reference["path"] if isinstance(reference,dict) else reference)
    if isinstance(reference,dict) and reference.get("sha256") and digest(path)!=reference["sha256"]:
        raise ValueError("AI checkpoint hash mismatch")
    value=torch.load(path,map_location=device,weights_only=False)
    if value["format"]!=MODEL_FORMAT or value.get("feature_schema")!=FEATURE_SCHEMA or not value.get("from_scratch"):
        raise ValueError("AI checkpoint schema/source mismatch")
    if value.get("engineering") and not allow_engineering:
        raise ValueError("engineering checkpoint forbidden in scientific AI evaluation")
    if not value.get("engineering") and value["update"]!=10000 and not (allow_intermediate and value["update"] in (2500,5000,7500)):
        raise ValueError("primary AI requires fixed 10000-update checkpoint")
    model=make_model().to(device); model.load_state_dict(value["model"]); model.eval()
    normalizer=Normalizer.from_record(value["normalizer"])
    if str(device).startswith("cuda"): torch.cuda.synchronize(device)
    return orderer("AI",model,normalizer,device=device,geometry=geometry,variant_count=variant_count),time.perf_counter()-started
