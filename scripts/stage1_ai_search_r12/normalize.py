"""Explicit field-only original12+804 normalization. Never constructs geometry.

Actual payload parsing requires its separate root IO-only admission. This module
imports no numpy, satellite, geometry, solver, model or production algorithm.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from datetime import datetime,timezone
from .admission import INDICES,STOP_NEW
from .persistence import Journal,content_sha,identity

CONTROL_SHA='804dd2e1cbaa264ceda230d153bb2e5b29fb10ab0703c7745aeef1cac1ec49e5'

def require(ok,message):
    if not ok:raise ValueError(message)

def load(ref):
    require(type(ref['bytes']) is int and 0<ref['bytes']<=4000000,'bounded plain source bytes')
    with Path(ref['path']).open('rb') as f:raw=f.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'exact original plain source content')
    return json.loads(raw)

def normalize_one(original,row,spec,control_ref,*,physical_parameters,geometry_profile):
    """Pure copies/arithmetic/hash checks only, no ray or layout generation."""
    require(original.get('schema')=='R10ExistingSceneAndBestOriginInputV1','exact self-contained original R10 input schema')
    index=spec['original_index'];require(original['original_index']==row['original_index']==index,'same original mother index')
    source=original['scene'];origin=original['origin'];layout=origin['layout'];ids=source['aircraft_ids'];n=len(ids)
    require(source.get('profile')==geometry_profile,'original geographic profile distinct from request profile')
    require(n==spec['aircraft_count']==row['original_aircraft_count'] and n>0 and len(set(ids))==n and all(type(a) is str and a for a in ids),'all original stable aircraft')
    require(origin['identity']==spec['identity'] and source['input_sha256']==origin['identity']['input_sha256'],'full original identity')
    require(all(type(origin['identity'].get(k)) is str and origin['identity'][k] for k in
        ('case_id','mother_id','profile','role','input_sha256','identity_sha256','shared_case_content_sha256','source_asset_SHA')),'all full original identity fields')
    require(origin['identity']['role']=='train' and origin['identity']['profile']=='uniform21600000','train diagnostic fixed-request identity')
    positions=source['positions'];demands=source['demands']
    require(len(positions)==len(demands)==n and all(isinstance(p,list) and len(p)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in p) for p in positions),'full finite original positions')
    require(all(type(v) in (int,float) and v==21600000 for v in demands),'all original fixed21600000 requests')
    require(origin['layout_SHA']==spec['B_layout_sha256']==content_sha(layout),'exact original B without reselection')
    b=len(layout['beam_ids']);owners=layout['assignment'];anchors=layout['anchor_aircraft_ids']
    require(1<=b<=min(200,n,origin['Ncap']) and len(set(layout['beam_ids']))==b,'original actual N/Ncap')
    require(origin['Ncap']==spec['Ncap']==row['caps']['Ncap'] and origin['Omega_cap']==spec['Omega_cap']==row['caps']['Omega_cap'],'original physical caps')
    require(len(owners)==n and all(type(o) is int and 0<=o<b for o in owners) and set(owners)==set(range(b)),'full unique integer nonempty ownership')
    require(len(anchors)==len(layout['centers'])==len(layout['widths_deg'])==b and all(a in ids for a in anchors),'full actual anchor arrays')
    require(all(layout['centers'][j]==positions[ids.index(a)] for j,a in enumerate(anchors)),'unchanged original anchor positions')
    require(all(type(w) in (int,float) and math.isfinite(w) and .3<=w<=1 for w in layout['widths_deg']),'original fullwidth domain')
    require(origin['metrics']==row['origin_metrics'] and origin['metrics']['total_request']==n*21600000,'same audited original B/request scale')
    require(row.get('common_state')=='complete','complete old common qualification')
    saved={}
    for method in ('S2','SH-C'):
        arm=row['arms'][method]
        require(arm.get('state')=='complete' and arm.get('whole_method_unknown') is False and arm['metrics']['total_request']==n*21600000,'old strong control complete qualification')
        saved[method]=dict(metrics=copy.deepcopy(arm['metrics']),state=arm['state'],layout_SHA=arm['layout_SHA'],
            source_control_ref=control_ref,original_index=index,original_method=method,
            historical_cost_ref=dict(source=control_ref,json_path=['rows',spec['control_row_index'],'arms',method,'ledger']),
            old_paid_costs_retained_not_reexecuted=True)
    return dict(schema='R12OriginalInputAdapterV1',original_index=index,aircraft_count=n,
        expected_origin_identity=copy.deepcopy(origin['identity']),expected_B_layout_SHA=origin['layout_SHA'],
        expected_Ncap=origin['Ncap'],expected_Omega_cap=origin['Omega_cap'],
        physical_parameters=copy.deepcopy(physical_parameters),geometry_profile=geometry_profile,
        saved_controls=saved,source_original_input_ref=spec['input_ref'],source_control_ref=control_ref,
        historical_rays_source=dict(source_original_input_ref=spec['input_ref'],json_path=['scene','rays'],
                                    used_as_new_production_rays=False),normalization_new_GEO_calls=0)

def run(plan,review):
    began=time.monotonic()
    require(plan.get('schema')=='R12IOOnlyNormalizationPlanV2' and plan.get('group')=='S1-AI17-R12','IO-only normalization plan')
    require(review.get('schema')=='RootR12IOOnlyNormalizationAdmissionV2' and review.get('IO_normalization_authorized') is True,'separate root IO admission')
    require(all(review.get(k) is False for k in ('GEO_authorized','flow_authorized','engineering_execution_authorized','scientific_execution_authorized')),'no mathematical execution')
    require(review.get('physical_parameters_accepted') is True and review.get('plan_content_SHA')==plan.get('content_SHA')==content_sha({k:v for k,v in plan.items() if k!='content_SHA'}),'root exact original physical/source binding')
    require(datetime.now(timezone.utc)<STOP_NEW,'pause prevents new data preparation')
    specs=plan['cases'];require([c['original_index'] for c in specs]==list(INDICES),'all original12 ordered cases')
    refs=[c['input_ref'] for c in specs]+[plan['historical_control_ref']]
    require(len(refs)==plan['source_payload_files_max']==13 and sum(r['bytes'] for r in refs)==plan['source_payload_bytes_max'],'exact13 original-byte bound')
    require(plan['historical_control_ref']['sha256']==CONTROL_SHA,'sole complete old804 control source')
    require(plan['IO_seconds_max']==30,'fixed30s small IO envelope')
    require(sum(c['aircraft_count'] for c in specs)==4715,'original4715 full records')
    from .worker import _storage,source_check,audited_S_rows
    from .historical_s import audited_selected
    source_check(plan)
    audited=audited_S_rows(plan)
    root=Path(plan['archive_root']);storage_root=Path(plan['storage_root'])
    require(root.is_absolute() and root.resolve().is_relative_to(storage_root.resolve()) and not root.exists(),'one-shot owned normalization output; no replay')
    _storage(storage_root)
    control=load(plan['historical_control_ref']);require([r['original_index'] for r in control['rows']]==list(INDICES),'exact old control12 denominator')
    journal=Journal(root);outputs=[]
    try:
        for position,spec in enumerate(specs):
            require(time.monotonic()-began<30 and datetime.now(timezone.utc)<STOP_NEW,'bounded IO/pause; no next source')
            require(spec['control_row_index']==position,'exact original804 row position')
            original=load(spec['input_ref'])
            value=normalize_one(original,control['rows'][position],spec,plan['historical_control_ref'],physical_parameters=plan['physical_parameters'],geometry_profile=plan['geometry_profile'])
            audited_selected(audited[spec['original_index']],dict(value,origin=original['origin'],scene={k:original['scene'][k] for k in ('aircraft_ids','positions','demands','input_sha256')}))
            key='input_'+str(spec['original_index']);journal.write(key,value)
            path=root/(key+'.json');raw=path.read_bytes()
            outputs.append(dict(original_index=spec['original_index'],aircraft_count=spec['aircraft_count'],input_adapter_ref=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))))
            _storage(storage_root)
        complete=time.monotonic()-began<=30 and datetime.now(timezone.utc)<STOP_NEW
        journal.write('normalization_manifest',dict(schema='R12IOOnlyNormalizationReceiptV1',complete=complete,source_payload_files=13,
            source_payload_bytes=sum(r['bytes'] for r in refs),cases=outputs,historical_control_ref=plan['historical_control_ref'],
            observed_seconds=time.monotonic()-began,new_GEO_calls=0,new_flow_calls=0,new_scientific_calls=0,automatic_retry=False))
        require(complete and time.monotonic()-began<=30,'complete IO/save within30s; otherwise incomplete')
    finally:journal.close()

def main():
    parser=argparse.ArgumentParser()
    for name in ('plan','plan-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    def ref(path,sha):return dict(path=path,sha256=sha,bytes=Path(path).stat().st_size)
    run(load(ref(args.plan,args.plan_sha256)),load(ref(args.review,args.review_sha256)))

if __name__=='__main__':main()
