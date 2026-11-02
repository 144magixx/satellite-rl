"""Explicit V2 original-S full-owner source adapter; no inference or solver.

Consumes only the root-accepted9537 audited-counts asset. Never opens old
source gzip or recursively follows historical provenance paths; never inverts CV.
"""
from .persistence import content_sha,identity
from fractions import Fraction

def require(ok,message):
    if not ok:raise ValueError(message)

def audited_table(value,prior_seconds,expected_control_ref=None):
    from .admission import INDICES
    require(value.get('schema')=='R12OldStrongSAuditedCountsV1' and value.get('complete') is True,'accepted complete original-S audited counts asset')
    require(value.get('old_control_ref',{}).get('sha256')=='804dd2e1cbaa264ceda230d153bb2e5b29fb10ab0703c7745aeef1cac1ec49e5','same original804 strong controls')
    if expected_control_ref is not None:
        require(identity(value['old_control_ref'])==identity(expected_control_ref),'old804 same bytes at explicit current transport alias')
    expected=dict(original_seconds=120,IO_subcap_seconds=10,conservative_IO_debit_seconds=10,remaining_reader_hard_seconds=110)
    require(value.get('reader_budget')==expected and prior_seconds==10,'historical S extraction consumes original reader120, remaining110 not reset')
    require(all(value.get(k)==0 for k in ('new_geometry_calls','new_flow_calls','new_model_calls')),'old counts qualifier made no new geometry/solver/model calls')
    require([r['original_index'] for r in value['rows']]==list(INDICES) and sum(r['aircraft_count'] for r in value['rows'])==4715,'complete original12 audited integer counts')
    return {r['original_index']:r for r in value['rows']}

def audited_selected(row,record):
    origin=record['origin'];old=record['saved_controls'];n=len(record['scene']['aircraft_ids'])
    objective=lambda m:tuple(m[k] for k in ('maxload','top10_mean','CV','N','actual_Omega'))
    method=min(('S2','SH-C'),key=lambda m:objective(old[m]['metrics'])+(('S2','SH-C').index(m),))
    chosen=old[method]
    require(row['selected_method']==method and row['original_index']==record['original_index'] and row['aircraft_count']==n,'unchanged old raw-objective S method/index/fulln')
    require(row['identity']==origin['identity'] and row['B_layout_SHA']==origin['layout_SHA'],'same original S input and B')
    require(identity(row['scene_ref'])==identity(record['source_original_input_ref']),'same original scene content at explicit current alias')
    require(row['metrics']==chosen['metrics'] and row['layout_SHA']==row['terminal_SHA']==chosen['layout_SHA'],'same audited old S terminal/display metrics')
    require(row.get('full_ID_order_and_fixed_requests_verified') is True and row['aircraft_ids_sha256']==content_sha(record['scene']['aircraft_ids']),'full original aircraft ID order and uniform requests')
    counts=row['counts'];widths=row['widths_deg'];N=len(counts)
    require(N==chosen['metrics']['N'] and len(widths)==N and all(type(c) is int and c>0 for c in counts) and sum(counts)==n,'original full positive S counts, never inverseCV')
    require(max(counts)*21600000==chosen['metrics']['maxload'] and n*21600000==chosen['metrics']['total_request'],'integer S request/maxload identity')
    t=(N+9)//10;T=sum(sorted(counts,reverse=True)[:t]);U=sum(c*c for c in counts);sufficient=row['exact_count_key']
    require(sufficient['max_count']==max(counts) and sufficient['N']==N and Fraction(*sufficient['top10_mean_count'])==Fraction(T,t)
        and Fraction(*sufficient['CV_squared'])==Fraction(N*U-n*n,n*n),'audited exact integer sufficient statistics')
    require(row['counts_source_ref']['sha256']==row['typed_native_ref']['sha256'],'original compressed S source identity')
    return dict(chosen,method=method,counts=list(counts),widths_deg=list(widths),audited_original_S=row)
