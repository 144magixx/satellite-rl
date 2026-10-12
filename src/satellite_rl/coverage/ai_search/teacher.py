"""Bounded task teacher and compact parent-linked geometry evidence.

Every task's inputs are frozen before decoding the pool. Incomplete solves are
censored, including solves with an observed feasible incumbent; they never
become negative labels. All candidate deltas remain replayable.
"""
from dataclasses import replace, asdict
from itertools import islice
from functools import cmp_to_key
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from .config import GeometryConfig, SearchConfig
from .core import Scene, Layout, initial_layout, metrics, layout_sha256, validate_layout
from .features import FEATURE_SCHEMA, PREFERENCE_FIELDS, scene_sha256, pool_features
from .search import Budget, BudgetStop, generate_tasks, decode_task, dominates, compare_metrics, run_search, add_archive

TEACHER_FORMAT = "s1_ai17_predecode_teacher_v2"


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8") as f:
        json.dump(value,f,allow_nan=False); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def compact_metrics(value):
    return {k:value[k] for k in PREFERENCE_FIELDS} if value is not None else None


def lex_key(value, preference):
    first=PREFERENCE_FIELDS[int(preference)]
    fields=(first,)+tuple(k for k in PREFERENCE_FIELDS if k!=first)
    # Zero-total-request cases have no scientific load quality; still legal
    # inputs but excluded from loss labels by the explicit zero_total flag.
    return tuple(float(value[k]) if value[k] is not None else 0. for k in fields)


def strict_lex_better(a,b,preference,config=SearchConfig()):
    return compare_metrics(a,b,preference,config)<0


def best_trial(rows,preference,config):
    def compare(a,b):
        decision=compare_metrics(a["metrics"],b["metrics"],preference,config)
        return decision or ((a["sha256"]>b["sha256"])-(a["sha256"]<b["sha256"]))
    return min(rows,key=cmp_to_key(compare))


def layout_delta(parent,result):
    """Sparse ownership plus changed beam rows handles both variable N and IDs."""
    if len(parent.assignment)!=len(result.assignment): raise ValueError("scene changed in delta")
    users=np.flatnonzero(parent.assignment!=result.assignment)
    changed=[]
    for i in range(len(result.centers)):
        if (i>=len(parent.centers) or result.beam_ids[i]!=parent.beam_ids[i]
                or result.widths[i]!=parent.widths[i] or not np.array_equal(result.centers[i],parent.centers[i])):
            changed.append(i)
    return dict(N=len(result.centers),users=users.tolist(),owners=result.assignment[users].tolist(),
        beams=changed,centers=result.centers[changed].tolist(),widths=result.widths[changed].tolist(),
        beam_ids=[result.beam_ids[i] for i in changed])


def replay_delta(parent,delta):
    n=int(delta["N"]); centers=np.zeros((n,2)); widths=np.zeros(n); ids=[None]*n
    keep=min(n,len(parent.centers)); centers[:keep]=parent.centers[:keep]; widths[:keep]=parent.widths[:keep]
    ids[:keep]=parent.beam_ids[:keep]
    for i,c,w,bid in zip(delta["beams"],delta["centers"],delta["widths"],delta["beam_ids"]):
        centers[i]=c; widths[i]=w; ids[i]=bid
    if any(x is None for x in ids): raise ValueError("incomplete beam delta")
    owners=parent.assignment.copy(); owners[np.asarray(delta["users"],int)]=delta["owners"]
    return Layout(centers,widths,owners,tuple(ids))


def teacher_case(case, output, config=None, geometry=None, stop=None):
    entry_time=perf_counter()
    geometry=GeometryConfig(**geometry) if isinstance(geometry,dict) else (geometry or GeometryConfig())
    cfg=SearchConfig(**config) if isinstance(config,dict) else (config or SearchConfig(seconds=15,max_attempts=3072,max_geometry_calls=100000))
    if (cfg.seconds>15 or cfg.max_attempts>3072 or cfg.pool_size>24 or cfg.task_candidate_limit>64
        or cfg.initial_seconds>2 or cfg.initial_max_attempts>2048):
        raise ValueError("teacher case exceeds frozen 15s/2*24*64 task budget")
    if "initial" in case: raise ValueError("teacher requires the common three-width INIT archive, no caller-supplied parent")
    budget=Budget(cfg,stop=stop)
    budget.start=entry_time; budget.deadline=entry_time+cfg.seconds
    scene=Scene(case["positions"],case["demands"],tuple(case.get("aircraft_ids",())),case.get("profile","europe_v1"))
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists(): raise ValueError("teacher case already exists; runner must resume by completed case identity")
    scene_sha=scene_sha256(scene)
    np.savez_compressed(root/"scene.npz",positions=scene.positions,demands=scene.demands,aircraft_ids=np.asarray(scene.aircraft_ids))
    rows=[]; parents=[]; candidates_count=0; current=None; outcome="completed"; error=None
    profile=case.get("demand_profile","base")
    profiles=("base","one_hotspot","two_hotspots","edge_heavy")
    if profile not in profiles: raise ValueError("unknown frozen request profile")
    if "mother_index" not in case: raise ValueError("frozen selected-mother ordinal required for balanced teacher preferences")
    mother_index=int(case["mother_index"])
    if mother_index<0: raise ValueError("nonnegative frozen selected-mother ordinal required")
    reference=[]; actual_archive=[]; initialization=None; initialization_status="censored"
    evidence_path=root/"candidate_deltas.jsonl"
    try:
        before_init=budget.record()
        initial_result=run_search(scene,config=cfg,geometry=geometry,method="INIT",budget=budget)
        after_init=budget.record()
        actual_archive=list(initial_result.pareto)
        initialization_status="completed" if actual_archive else "censored"
        initialization=dict(source="ai_search.search.run_search(method=INIT,shared_master_budget)",
            scene_sha256=scene_sha,feature_schema=FEATURE_SCHEMA,entry_to_init_end_seconds=budget.elapsed,
            search_config=asdict(cfg),geometry_config=asdict(geometry),
            budget_before=before_init,budget_after=after_init,
            budget_delta={k:after_init[k]-before_init[k] for k in ("seconds","attempts","geometry_calls","ray_pair_checks")},
            initialization_deadline_seconds=min(cfg.seconds,cfg.initial_seconds),
            initialization_candidate_limit=min(cfg.max_attempts,cfg.initial_max_attempts),
            status=initialization_status,result=initial_result.to_record())
        atomic_json(root/"initialization.json",initialization)
        reference=[compact_metrics(row["metrics"]) for row in actual_archive]
        with evidence_path.open("w",encoding="utf-8") as evidence:
            for state in range(2):
                if not actual_archive:
                    error="initialization_no_feasible_found_under_pool"; break
                budget.check()
                preference=((mother_index*4+profiles.index(profile))*2+state)%5
                selected=best_trial(actual_archive,preference,cfg)
                current=selected["layout"]
                parent_sha=layout_sha256(current)
                parent_values=compact_metrics(selected["metrics"])
                before_inputs=budget.record()
                tasks=[replace(t,preference_index=preference) for t in
                    islice(generate_tasks(scene,current,cfg,geometry,budget=budget),min(24,cfg.pool_size))]
                features=pool_features(scene,current,tasks,preference,geometry,cfg.variants,budget)
                after_inputs=budget.record()
                parent_path=root/f"parent_{state}.json"
                atomic_json(parent_path,current.to_record())
                parents.append(dict(state=state,sha256=parent_sha,path=parent_path.name,file_sha256=digest(parent_path),
                    preference_index=preference,metrics=parent_values,
                    archive_member_sha256s_before_state=[row["sha256"] for row in actual_archive],
                    input_preparation_budget_delta={k:after_inputs[k]-before_inputs[k]
                        for k in ("seconds","attempts","geometry_calls","ray_pair_checks")}))
                state_rows=[]; reached=[]
                for index,task in enumerate(tasks):
                    row=dict(scene_sha256=scene_sha,parent_layout_sha256=parent_sha,state=state,
                        mother_id=str(case["mother_id"]),profile=profile,preference_index=preference,
                        task_id=task.task_id,task=task.to_record(),task_predecode_features=features[index].tolist(),
                        completion_status="censored",completed_best_metrics=None,parent_metrics=parent_values,
                        new_nondominated=None,zero_total_request=bool(scene.demands.sum()==0),
                        budget_delta=None,replay_delta_reference=[])
                    before=budget.record()
                    try:
                        budget.check()
                        decoded=decode_task(scene,current,task,budget,geometry,cfg)
                        status=getattr(decoded,"status",None)
                        is_censored=(status=="censored" or decoded.reason in
                            ("time_budget","candidate_budget","geometry_budget","stop_requested","region_user_limit"))
                        feasible=[]
                        for trial in decoded.candidates:
                            delta=layout_delta(current,trial["layout"])
                            if layout_sha256(replay_delta(current,delta))!=trial["sha256"]:
                                raise ValueError("candidate delta replay hash mismatch")
                            item=dict(task_id=task.task_id,parent_sha256=parent_sha,result_sha256=trial["sha256"],
                                delta=delta,metrics=compact_metrics(trial["metrics"]),late=trial["late"],
                                completed_seconds=trial["completed_seconds"],candidate_index=candidates_count)
                            evidence.write(json.dumps(item,allow_nan=False)+"\n")
                            row["replay_delta_reference"].append(candidates_count); candidates_count+=1
                            if trial["metrics"] is not None and not trial["late"]: feasible.append(trial)
                        best=best_trial(feasible,preference,cfg) if feasible else None
                        if not is_censored:
                            row["completion_status"]="completed" if best else "no_feasible_found_under_pool"
                            row["new_nondominated"]=bool(best and not any(
                                compare_metrics(best["metrics"],v,preference,cfg)==0
                                or dominates(v,best["metrics"],PREFERENCE_FIELDS,config=cfg) for v in reference))
                        if best:
                            row["completed_best_metrics"]=compact_metrics(best["metrics"])
                            reached.extend(feasible)
                        row["reason"]=decoded.reason
                    except BudgetStop as exc:
                        row["reason"]=str(exc)
                    after=budget.record()
                    row["budget_delta"]={k:after[k]-before[k] for k in ("seconds","attempts","geometry_calls","ray_pair_checks")}
                    state_rows.append(row)
                rows.extend(state_rows)
                evidence.flush(); os.fsync(evidence.fileno())
                # Actual RULE-reached expanded archive; next state's different
                # frozen preference chooses its own legal parent from it.
                for candidate in reached:
                    actual_archive,_=add_archive(actual_archive,candidate,cfg)
                reference=[compact_metrics(row["metrics"]) for row in actual_archive]
    except BudgetStop as exc:
        error=str(exc)
        if str(exc)=="stop_requested": outcome="stopped"
    if budget.reason=="stop_requested": outcome="stopped"
    result=dict(format=TEACHER_FORMAT,status=outcome,role=case.get("role","training"),feature_schema=FEATURE_SCHEMA,
        mother_id=str(case["mother_id"]),profile=profile,scene_sha256=scene_sha,
        scene=dict(path="scene.npz",sha256=digest(root/"scene.npz"),profile=scene.profile),parents=parents,rows=rows,
        candidate_evidence=dict(path=evidence_path.name,sha256=digest(evidence_path)) if evidence_path.exists() else None,
        candidates=candidates_count,states=len(parents),budget=budget.record(),stop_reason=error,
        initialization_status=initialization_status,
        initialization=dict(path="initialization.json",sha256=digest(root/"initialization.json")) if initialization is not None else None,
        case_completion_status="completed" if actual_archive else "censored_no_feasible_initialization",
        censored_rows=sum(x["completion_status"]=="censored" for x in rows),
        final_layout_sha256=layout_sha256(current) if current is not None else None,
        mother_index=mother_index,preference_assignment="((selected_mother_ordinal*4+profile_index)*2+state)%5",
        metric_comparison=dict(metric_abs_tolerances=cfg.metric_abs_tolerances,metric_relative_tolerance=cfg.metric_relative_tolerance),
        label_definition="same-parent-five-preference-lex-completed-only; BCE-nondominated-against-prior-state-archive")
    atomic_json(root/"manifest.json",result)
    result["manifest"]=dict(path=str((root/"manifest.json").resolve()),sha256=digest(root/"manifest.json"))
    return result
