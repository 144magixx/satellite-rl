"""One-shot, eight fixed synthetic cases. Never called by ordinary unit tests."""
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch
from .core import Scene, prepare_case, build_teacher, build_proposal, decode_groups, InterfaceError


FIXTURES = (
    ('clustered', [[48+i*.12,8+(i%3)*.19] for i in range(12)], [20,8,1,12,0,5,4,3,2,1,1,1]),
    ('coincident', [[48,8],[48,8],[48.3,8],[48.3,8],[48.6,8],[48.6,8]], [9,4,3,2,1,0]),
    ('isolated', [[38,-8],[45,0],[50,20],[60,25]], [8,4,2,0]),
    ('zero_request', [[48,8],[48.2,8],[48.4,8]], [0,0,0]),
    ('typed_interfaces', [[48,8],[48.1,8]], [2,1]),
    ('masked', [[48,8],[48.2,8],[48.4,8],[48.6,8]], [4,3,2,1]),
    ('count_and_identity', [[48,8],[48.1,8],[48.2,8]], [3,2,1]),
    ('time_and_invisible', [[48,8],[48.1,8],[48.2,8]], [3,2,1]),
)


def run_once(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'core_geometry_qualification_ledger.json'
    if path.exists():raise RuntimeError('qualification_already_started_no_automatic_replay')
    ledger=dict(schema='R4_geometry_qualification_v1',synthetic_case_cap=8,
        cases=[],calls=[],native_calls=0,CUDA_updates=0,real_data_calls=0,status='running')
    def save():path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    save()
    def call(case,stage,fn,*args,parameters=None,**kwargs):
        row=dict(case=case,stage=stage,parameters=parameters or {},state='intent',started=time.time())
        ledger['calls'].append(row);save()
        try:
            result=fn(*args,**kwargs)
            record=result.to_dict() if hasattr(result,'to_dict') else result
            file=output/f'core_case{case:02d}_call{len(ledger["calls"]):03d}_{stage}.json'
            file.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
            row.update(state='completed',result_path=str(file),status=record.get('status'),
                       cost=record.get('cost',record.get('ledger')),ended=time.time())
            save();return result
        except InterfaceError as exc:
            row.update(state='expected_hard_error',error=str(exc),ended=time.time());save();return None
        except BaseException as exc:
            row.update(state='failed',error=repr(exc),ended=time.time());ledger['status']='failed';save();raise
    for ordinal,(name,positions,demands) in enumerate(FIXTURES):
        scene=Scene(positions,demands,tuple(f'{name}:{i}' for i in range(len(demands))))
        identity=dict(case_id=f'synthetic:{name}',mother_id=f'synthetic:{ordinal}',profile='base',
            input_sha256=scene.input_sha256,identity_sha256='synthetic-fixed-v1',role='development')
        ledger['cases'].append(dict(ordinal=ordinal,name=name,positions=positions,demands=demands,
                                  aircraft_ids=list(scene.aircraft_ids),identity=identity));save()
        prepared=call(ordinal,'prepare',prepare_case,scene,identity,'synthetic-core-source')
        teacher=call(ordinal,'teacher',build_teacher,prepared)
        for method in ('RULE','RANDOM','TEACHER_EXACT','TEACHER_PERTURBED'):
            proposal=call(ordinal,f'proposal_{method}',build_proposal,prepared,method,
                          teacher=teacher if method.startswith('TEACHER') else None)
            if proposal['status']=='ready':call(ordinal,f'decode_{method}',decode_groups,prepared,proposal)
            else:
                ledger['calls'].append(dict(case=ordinal,stage=f'decode_{method}',state='not_called',
                    reason=proposal['reason'],actual_decode_operations=0));save()
        if ordinal==4:
            call(ordinal,'decode_None',decode_groups,prepared,None)
            call(ordinal,'decode_unavailable',decode_groups,prepared,dict(status='unavailable',reason='unavailable_teacher'))
            call(ordinal,'decode_missing_tag',decode_groups,prepared,{})
            call(ordinal,'teacher_missing_tag',build_proposal,prepared,'TEACHER_EXACT',teacher={})
        if ordinal==5:
            proposal=call(ordinal,'proposal_RULE_mask_setup',build_proposal,prepared,'RULE')
            proposal['score_mask_nxK_bool']=[[False]*len(proposal['slot_ids']) for _ in demands]
            call(ordinal,'decode_all_masked',decode_groups,prepared,proposal)
        if ordinal==6:
            proposal=call(ordinal,'proposal_RULE_overcap_setup',build_proposal,prepared,'RULE')
            proposal=copy.deepcopy(proposal);k=prepared.Ncap+1
            proposal.update(slot_ids=[f'slot:{i}' for i in range(k)],
                anchor_aircraft_ids=[scene.aircraft_ids[0]]*k,direction_offset_Kx3=[[0.,0.,0.]]*k,
                scores_nxK_float32_finite=[[0.]*k for _ in demands],score_mask_nxK_bool=[[True]*k for _ in demands])
            call(ordinal,'decode_overcap',decode_groups,prepared,proposal)
            proposal['role']='test'
            call(ordinal,'decode_role_mismatch',decode_groups,prepared,proposal)
        if ordinal==7:
            proposal=call(ordinal,'proposal_RULE_time_setup',build_proposal,prepared,'RULE')
            call(ordinal,'decode_nominal_zero',decode_groups,prepared,proposal,time_limit=0.,parameters={'time_limit':0.})
            perturb=call(ordinal,'proposal_PERTURBED_invisible_setup',build_proposal,prepared,'TEACHER_PERTURBED',teacher=teacher)
            with patch('satellite_rl.coverage.ai_search_r4.core.ray_location',return_value=None):
                call(ordinal,'decode_invisible_mock',decode_groups,prepared,perturb,parameters={'ray_location':'mock_None_no_extra_scene'})
    ledger['status']='completed';save();return ledger


if __name__=='__main__':
    import sys
    result=run_once(sys.argv[1]);print(json.dumps(dict(status=result['status'],cases=len(result['cases']),calls=len(result['calls']))))
