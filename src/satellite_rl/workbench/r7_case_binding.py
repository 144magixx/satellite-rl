"""Pure immutable-case role/index/row binding, before any algorithm call."""
def validate_binding(case,role,index,row=None,recipe_row=None):
    if role not in ('train','development') or type(index)is not int or not 0<=index<(576 if role=='train' else 20):raise ValueError('original R7 role/index')
    source_role='train' if role=='train' else 'validation'
    if case['actual_role']!=role or case['source_role']!=source_role or case['source']['original_case_index']!=index:raise ValueError('immutable case actual/source role or original index differs')
    if case['original_mother_id']!=source_role+':'+case['UTC'] or case['case_id']!=f"r7:{role}:{case['original_mother_id']}:uniform21600000":raise ValueError('immutable case role/mother/case ID differs')
    expected=dict(role=role,case_index=index,case_id=case['case_id'],original_mother_id=case['original_mother_id'],
                  case_content_sha256=case['content_sha256'],aircraft_count=len(case['aircraft_ids']),total_request_bps_int64=case['total_request_bps_int64'])
    if case['total_request_bps_int64']!=len(case['aircraft_ids'])*21600000:raise ValueError('exact original all-plane total')
    if row is not None and any(row.get(k)!=v for k,v in expected.items()):raise ValueError('producer row role/index/count/identity differs from immutable case')
    if recipe_row is not None:
        if recipe_row['source_phase_index']!=index or any(case[k]!=recipe_row[k] for k in ('actual_role','source_role','case_id','original_mother_id','UTC','source_asset_SHA')) or len(case['aircraft_ids'])!=recipe_row['source_aircraft_count']:raise ValueError('immutable case differs from frozen root recipe')
    return expected
