"""Exact finite extension admission. Scientific decoder bytes stay unchanged."""
from pathlib import Path
import hashlib,json
from .storage import read_json,digest
from .coverage_ai_search import pin

PILOT_COMMIT='f02c0dcbf76561c325b5a70a1a5b9bf98882b35d'
DESIGN_SHA='b5c0ddb5646ce92bda62806424598bc11cdf07ba30985e04cba6b14eb50c8ba4'
INTERPRETATION_SHA='1ba9128999aa33934af7fe2c7169bd17b698ba549f4dbee394f6a662f5b6f828'
PILOT_AUDIT_SHA='3d2c87c041ec0783eb054e1fafe1e0ca51869f9488a8e341aa14f23a1553c1f3'
PILOT_GATE_SHA='f223c3e8dd109f86e83a92e3a8c90399bb5f827fdf99dfc192eb4a44872ed2d4'
PILOT_FREEZE_SHA='9a385ebe3cb9b23f0367095c6e815a20fdefee47fee1fd0a878aa63850e6b2d1'
PILOT_QUALIFICATION_SHA={'validator_equivalence':'c2bfe22c03c09cb27948842227538b5e799bc5d95c5546aff7a75401a750d542',
    'charge_before_budget':'79439cd7e5500c6e3649086f41afa1e1585c5c394167168e7bc7cc7ad5fca87d'}
SCIENTIFIC_DEPENDENCY_MAP_SHA='f060475453570507e5e32746d396eba9f72eb19f3a1cfa823aafc9ccde520fb8'


def actual_dependency_sha256(expected):
    root=Path(__file__).resolve().parents[3]
    return {relative:digest(root/relative) for relative in expected}


def actual_core_sha256():
    root=Path(__file__).resolve().parents[1]/'coverage/ai_search_r2'
    return {'src/satellite_rl/coverage/ai_search_r2/'+p.name:digest(p) for p in sorted(root.glob('*.py'))}


def mapped(ref,bundle):
    value=bundle.get('aliases',{}).get(ref['path'],ref)
    if value['sha256']!=ref['sha256']:raise ValueError('extension evidence alias changes original SHA')
    return value


def plan_references(plan):
    if plan.get('phase')!='extension':return {}
    bundle=plan.get('extension_admission',{})
    return {'extension_'+key:bundle[key] for key in ('root_gate','audit','gate','source_freeze','interpretation')}


def validate(plan):
    bundle=plan.get('extension_admission')
    if not isinstance(bundle,dict):raise ValueError('extension root gate evidence required')
    if (bundle['audit']['sha256']!=PILOT_AUDIT_SHA or bundle['gate']['sha256']!=PILOT_GATE_SHA
            or bundle['source_freeze']['sha256']!=PILOT_FREEZE_SHA):
        raise ValueError('extension requires the exact root-accepted pilot audit/gate SHA')
    root=read_json(pin(bundle['root_gate']),{});audit=read_json(pin(bundle['audit']),{});gate=read_json(pin(bundle['gate']),{})
    freeze=read_json(pin(bundle['source_freeze']),{});pin(bundle['interpretation'])
    for key,ref in [('audit',root.get('audit_ref',{})),('gate',root.get('gate_ref',{})),('interpretation',root.get('G3_pre_result_interpretation_ref',{}))]:
        if mapped(ref,bundle)!=bundle[key]:raise ValueError('original root gate/audit/protocol chain differs')
    if (root.get('group')!='S1-AI17-R2' or root.get('passed')is not True or root.get('extension_admitted')is not True
            or root.get('pilot_scientific_commit')!=PILOT_COMMIT or root.get('design_sha256')!=DESIGN_SHA
            or root.get('allowed_remaining_mothers')!=14 or root.get('allowed_remaining_cases')!=56
            or root.get('allowed_new_decoder_task_slots')!=896 or root.get('pilot_cases_to_reuse_without_new_solves')!=24
            or root.get('training_admitted')is not False or root.get('validation_test_admitted')is not False):
        raise ValueError('root finite remaining56 admission differs')
    if any(root.get(name,{}).get('passed')is not True or gate.get(name,{}).get('passed')is not True for name in ('G0','G1','G2')):
        raise ValueError('extension requires actual G0 G1 G2 passed; G3 cannot substitute')
    if (gate.get('group')!='S1-AI17-R2' or gate.get('phase')!='pilot' or gate.get('extension_admitted')is not True
            or gate.get('training_admitted')is not False or mapped(gate['audit_source'],bundle)!=bundle['audit']
            or gate['design_ref']['sha256']!=DESIGN_SHA or gate['G3_pre_result_interpretation_ref']['sha256']!=INTERPRETATION_SHA):
        raise ValueError('accepted pilot gate source/design/protocol differs')
    if (audit.get('group')!='S1-AI17-R2' or audit.get('phase')!='pilot' or audit.get('passed')is not True
            or audit.get('formal_ready')is not True or audit.get('issues') or audit.get('actual_cases')!=24
            or audit.get('expected_cases')!=24 or audit.get('actual_decoder_rows')!=48
            or audit.get('design_sha256')!=DESIGN_SHA or len(audit.get('case_audit_index',[]))!=24
            or any(r.get('passed')is not True for r in audit['case_audit_index'])):
        raise ValueError('accepted actual24 pilot audit is incomplete or failed')
    core=freeze.get('core_git_blob_sha256',{})
    if (freeze.get('passed')is not True or freeze.get('scientific_commit')!=PILOT_COMMIT or freeze.get('design_sha256')!=DESIGN_SHA
            or len(core)!=9 or plan.get('scientific_core_sha256')!=core or actual_core_sha256()!=core
            or plan['design']['sha256']!=DESIGN_SHA or bundle['interpretation']['sha256']!=INTERPRETATION_SHA):
        raise ValueError('extension actual core bytes/design differ from original f02')
    dependencies=plan.get('scientific_dependency_sha256',{})
    if (hashlib.sha256(json.dumps(dependencies,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=SCIENTIFIC_DEPENDENCY_MAP_SHA
            or actual_dependency_sha256(dependencies)!=dependencies):
        raise ValueError('extension scientific dependency bytes changed')
    return core
