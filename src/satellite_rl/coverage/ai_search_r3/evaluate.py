"""Method-level results with actual execution and comparable cost kept separate."""
from pathlib import Path
import time
import numpy as np
from .integer import build_problem,decode,ScaleGuard
from .model import select_slots
from .solver import run_native,atomic_json
from .postprocess import metric_key,lex_better,polish_envelope
from ..ai_search.core import metrics,layout_sha256
from ..ai_search_r2.validation import validate_owned

def reference_target_evidence(p,chosen,chosen_source,calls):
    """Bind reference eligibility to the actual final source, never another label."""
    if chosen_source in ('native_stage1','native_stage2'):
        index=0 if chosen_source=='native_stage1' else 1
        if index>=len(calls):return dict(valid=False,reason='missing_selected_native_call')
        call=calls[index]
        valid=bool(call.get('value_valid') and call.get('native_start_feasibility_accepted') and call.get('decode')
            and call.get('reference_type') in ('optimized_native','certified_start'))
        return dict(valid=valid,reason='selected_native_reference' if valid else 'selected_native_reference_unqualified',
            source_call_ref=call.get('call_wal_ref'))
    if chosen_source!='external_W' or not calls:return dict(valid=False,reason='uncertified_external_W')
    first=calls[0];bound=first.get('mip_dual_bound');total=float(p.scene.demands.sum())
    peak=metric_key(p.scene,chosen)[0]/total if total>0 else None
    valid_native=bool(first.get('value_valid') and first.get('native_start_feasibility_accepted') and first.get('decode')
        and first.get('status') in ('HighsModelStatus.kOptimal','HighsModelStatus.kTimeLimit','HighsModelStatus.kIterationLimit','HighsModelStatus.kSolutionLimit'))
    usable=valid_native and bound is not None and np.isfinite(bound) and peak is not None and np.isfinite(peak) and peak>0
    gap=max(0.,(peak-bound)/abs(peak)) if usable and bound<=peak+1e-8 else None
    valid=gap is not None and gap<=.001
    return dict(valid=valid,reason='external_W_peak_certified_by_stage1_bound' if valid else 'uncertified_external_W',
        raw_W_peak_share=peak,stage1_native_dual_bound=bound,gap_against_raw_W=gap,
        source_call_ref=first.get('call_wal_ref'))

def evaluate_method(p,method,output_dir,model=None,oracle_label=None,vendor_dir=None,stop=None,public_input_seconds=0.):
    started=time.monotonic();directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    record=dict(schema='r3_method_result_v2',case_id=p.case_id,method=method,scene_sha=p.scene.input_sha256,
        pool_sha=p.content_sha,status='started_unknown_spend',solver_calls=[],public_input_seconds=public_input_seconds)
    atomic_json(directory/'result.wal.json',record)
    shared=p.ledger['witness']['seconds'] if method=='WITNESS' else p.ledger['seconds']
    def billed():return public_input_seconds+shared+time.monotonic()-started
    def exhausted():return method!='REF20' and billed()>=30
    def fail(reason,candidate=None):
        for raw in record['solver_calls']:raw.pop('vector',None);raw.pop('seed_vector',None)
        record.update(status=reason,layout=None,metrics=None,quality_defined=False,candidate_layout=candidate,
            actual_method_seconds=time.monotonic()-started,actual_shared_seconds=shared,comparable_seconds=billed(),
            publication_seconds=billed(),restricted_cost_seconds=max(30,billed()))
        atomic_json(directory/'result.json',record);return record
    if exhausted():return fail('pipeline_time_guard')
    solver_layouts=[];reference_labels=[];chosen=p.witness;chosen_source='external_W';selection_time=0;build_time=0
    if method!='WITNESS':
        t=time.monotonic();slots,selection=select_slots(p,method,model,oracle_label);selection_time=time.monotonic()-t;record['selection']=selection
        if exhausted():return fail('pipeline_time_guard')
        if slots is None:
            record['reason']=selection['reason'];return fail('label_unavailable')
        record['selected_slot_ids']=slots.tolist()
        if vendor_dir is None:raise ValueError('pinned_highspy_vendor_required')
        reference_started=time.monotonic()
        t=time.monotonic()
        try:problem=build_problem(p,slots)
        except ScaleGuard as exc:problem=None;record['scale_guard']=str(exc)
        except (ValueError,IndexError,KeyError) as exc:
            record['invalid_integer_problem_reason']=str(exc);return fail('invalid_integer_problem')
        build_time+=time.monotonic()-t
        if exhausted():return fail('pipeline_time_guard')
        if problem is not None:
            stages=[(1,12,20000,30)] if method=='REF20' else [(1,2,2000,5)]
            stage1_vector=None
            for stage,seconds,nodes,watchdog in stages:
                remaining_watchdog=30-(time.monotonic()-reference_started) if method=='REF20' else min(watchdog,30-billed())
                if remaining_watchdog<=0:
                    record['reference_time_guard']=True;break
                result=run_native(problem,directory/'native',vendor_dir,seconds,nodes,remaining_watchdog,stop);record['solver_calls'].append(result)
                if result['value_valid'] and result['vector'] is not None:
                    try:
                        layout,evidence=decode(p,problem,result['vector']);result['decode']=evidence
                        same=layout_sha256(layout)==layout_sha256(p.witness)
                        # IDs differ; compare full physical groups instead of solver beam numbering.
                        same=physical_key(layout)==physical_key(p.witness)
                        gap=result.get('mip_gap');bound=result.get('mip_dual_bound');objective=result.get('fun')
                        certified=(gap is not None and np.isfinite(gap) and 0<=gap<=.001 and bound is not None and np.isfinite(bound)
                            and objective is not None and np.isfinite(objective) and bound<=objective+1e-8
                            and result.get('status') in ('HighsModelStatus.kOptimal','HighsModelStatus.kTimeLimit','HighsModelStatus.kIterationLimit','HighsModelStatus.kSolutionLimit')
                            and bool(result.get('native_start_feasibility_accepted')))
                        result.update(solution_same_as_W=same,native_bound_certificate=certified,
                            reference_type='certified_start' if same and certified else ('optimized_native' if not same else 'seed_only'))
                        solver_layouts.append((layout,'native_stage1',evidence))
                        stage1_vector=np.asarray(result['vector'])
                        if result.get('native_start_feasibility_accepted') and (not same or certified):reference_labels.append(dict(layout=layout.to_record(),**evidence,reference_type=result['reference_type']))
                    except ValueError as exc:result['rejection_reason']=str(exc)
            if method=='REF20' and stage1_vector is not None and time.monotonic()-reference_started<30:
                t=time.monotonic();problem2=build_problem(p,slots,2,stage1_vector);build_time+=time.monotonic()-t
                remaining_ref=30-(time.monotonic()-reference_started)
                if remaining_ref<=0:
                    record['reference_time_guard']=True
                    problem2=None
                result=run_native(problem2,directory/'native',vendor_dir,8,10000,remaining_ref,stop) if problem2 is not None else dict(value_valid=False,vector=None,status='reference_time_guard')
                if problem2 is not None:record['solver_calls'].append(result)
                result['stage1_seed_source_call_ref']=record['solver_calls'][0].get('call_wal_ref')
                if result['value_valid'] and result['vector'] is not None:
                    try:
                        layout,evidence=decode(p,problem2,result['vector']);result['decode']=evidence
                        same=physical_key(layout)==physical_key(p.witness)
                        # A stage2 L1 bound cannot certify the stage1 peak objective.
                        certified=bool(record['solver_calls'][0].get('native_bound_certificate'))
                        result.update(solution_same_as_W=same,native_bound_certificate=certified,
                            reference_type='certified_start' if same and certified else ('optimized_native' if not same else 'seed_only'))
                        solver_layouts.append((layout,'native_stage2',evidence))
                        if result.get('native_start_feasibility_accepted') and (not same or certified):reference_labels.append(dict(layout=layout.to_record(),**evidence,reference_type=result['reference_type']))
                    except ValueError as exc:result['rejection_reason']=str(exc)
            for layout,source,evidence in solver_layouts:
                if lex_better(metric_key(p.scene,layout),metric_key(p.scene,chosen)):chosen=layout;chosen_source=source
    if exhausted():return fail('pipeline_time_guard',chosen.to_record())
    target_evidence=reference_target_evidence(p,chosen,chosen_source,record['solver_calls']) if method=='REF20' else None
    final,post=polish_envelope(p,chosen);m=metrics(p.scene,final,evidence=post['audit'])
    for raw in record['solver_calls']:
        # Canonical original vectors live once in native output JSON; compact summaries reference it.
        raw.pop('vector',None);raw.pop('seed_vector',None)
    method_seconds=time.monotonic()-started
    if exhausted():return fail('pipeline_time_guard',final.to_record())
    record.update(status='published',layout=final.to_record(),metrics=m,chosen_source=chosen_source,
        fallback=chosen_source=='external_W' and method!='WITNESS',postprocessing=post,reference_labels=reference_labels,
        valid_reference=target_evidence['valid'] if target_evidence is not None else None,
        valid_reference_target=target_evidence['valid'] if target_evidence is not None else None,
        valid_reference_labels=bool(reference_labels) if method=='REF20' else None,
        reference_target_evidence=target_evidence,selection_seconds=selection_time,
        matrix_build_seconds=build_time,actual_method_seconds=method_seconds,actual_shared_seconds=shared,
        comparable_seconds=public_input_seconds+shared+method_seconds,
        original_solver_outputs_preserved=True,quality_defined=bool(p.scene.demands.sum()))
    atomic_json(directory/'result.json',record)
    record['publication_IO_seconds']=time.monotonic()-started-method_seconds
    record['comparable_seconds']+=record['publication_IO_seconds']
    record['publication_seconds']=record['comparable_seconds'];atomic_json(directory/'result.json',record)
    if exhausted():return fail('pipeline_time_guard',final.to_record())
    return record

def physical_key(layout):
    return tuple(sorted((tuple(c),float(w),tuple(np.flatnonzero(layout.assignment==b))) for b,(c,w) in enumerate(zip(layout.centers,layout.widths))))
