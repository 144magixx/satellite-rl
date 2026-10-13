"""R5 immutable lineage and stage output validation; no geometry operations."""
from copy import deepcopy
from .coverage_ai_search import pin
from .storage import read_json
from .r4_io import check_record
IDENTITY_FIELDS=('case_id','mother_id','profile','input_sha256','identity_sha256','role')
def identity(case):return {k:case[k] for k in IDENTITY_FIELDS}
def original_records(case,teacher_required=True):
 ident=identity(case)
 prep=read_json(pin(case['prepared']['stage_output']),{})['result']
 check_record(prep,'prepared',ident,case['prepared']['source_sha256'])
 if prep['status']!='available':raise ValueError('usable original prepared only')
 teacher=None
 if teacher_required:
  teacher=read_json(pin(case['teacher']['stage_output']),{})['result'];check_record(teacher,'teacher',ident,case['teacher']['source_sha256'])
  if teacher['status']!='available' or case['teacher']['guard_status']!='completed' or case['teacher']['unknown_algorithm_spend']:raise ValueError('usable original teacher only')
 return deepcopy(prep),deepcopy(teacher)

def validate_output(child,request):
 value=child['result'];method=request['method'];decoder=method.split('_',1)[0];ident=identity(request['registry_case'])
 if child.get('identity')!=ident or child.get('source_sha256')!=request['source_sha256'] or child.get('method')!=method:raise ValueError('stage identity/source/method')
 if value.get('status') not in ('success','fallback','unavailable','unpublished'):raise ValueError('decoder tagged union')
 if child.get('original_prepared_ref')!=request['registry_case']['prepared']['stage_output'] or child.get('original_teacher_ref')!=request['registry_case']['teacher']['stage_output']:raise ValueError('immutable original lineage')
 if decoder=='D5' and value.get('schema')!='DecoderResultV2':raise ValueError('D5 schema')
 if decoder=='D4' and not child.get('derived_envelope_ref'):raise ValueError('D4 derived semantics envelope required')
 if value['status'] in ('success','fallback') and not value.get('layout'):raise ValueError('published layout required, including true fallback')
 return value
