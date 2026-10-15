"""Strict arithmetic/file-only engineering receipt admission; no model or PHY."""
import pickle
from satellite_rl.t0_width.data import read_json_ref,read_ref
from satellite_rl.t0_width.logging import content_sha
from .checks import catalog,fixture_manifest,validate_catalog,REQUIRED_COVERAGE

def validate_qualification(receipt,plan):
    for key,cap in [('physical_calls_attempted',320),('real_commit_attempts',64),('candidate_restore_attempts',256),('model_forward_attempts',64)]:
        value=receipt.get(key)
        if type(value) is not int or not 0<=value<=cap:raise ValueError('invalid finite engineering receipt count: '+key)
    if (receipt['physical_calls_attempted']!=receipt['real_commit_attempts']+receipt['candidate_restore_attempts']
            or receipt['real_commit_attempts']==0 or receipt['model_forward_attempts']==0):raise ValueError('empty/inconsistent engineering receipt cannot qualify')
    if (receipt.get('schema')!='W28EngineeringQualificationV1' or receipt.get('experiment_id')!='S2-W28'
            or receipt.get('root_approved') is not True or receipt.get('passed') is not True
            or receipt.get('source_commit')!=plan['source_commit'] or receipt.get('policy_SHA')!=plan['policy_ref']['sha256']
            or receipt.get('directory_fsync_available') is not True or receipt.get('parameter_updates')!=0
            or set(receipt.get('coverage',[]))!=set(REQUIRED_COVERAGE)):
        raise ValueError('exact root W28 engineering/source/policy/coverage receipt required')
    engineering=read_json_ref(receipt['engineering_plan_ref'])
    if (engineering.get('check_contract_version')!='W28QualificationChecksV1'
            or engineering.get('source_commit')!=plan['source_commit'] or engineering.get('policy_ref')!=plan['policy_ref']
            or receipt.get('engineering_plan_SHA')!=content_sha(engineering)):
        raise ValueError('actual fixed engineering plan reference differs')
    bounds=validate_catalog(engineering['checks'],read_json_ref(engineering['fixture_manifest_ref']))
    if receipt.get('catalog_SHA')!=bounds['catalog_SHA'] or len(receipt.get('check_evidence_refs',[]))!=len(catalog()):raise ValueError('complete exact builtin checks required')
    totals=dict(query=0,commit=0,forward=0);seen=set()
    specs={s['id']:s for s in fixture_manifest()['rows']}
    for ref in receipt['check_evidence_refs']:
        evidence=read_json_ref(ref);ident=evidence['check_id']
        rows=[c for c in catalog() if c['check_id']==ident]
        if len(rows)!=1 or ident in seen:raise ValueError('missing/duplicate engineering check identity')
        seen.add(ident);check=rows[0];spec=specs[check['fixture_id']]
        if (evidence.get('schema')!='W28BuiltinCheckEvidenceV1' or evidence.get('passed') is not True
                or evidence.get('check_SHA')!=content_sha(check) or evidence.get('fixture_SHA')!=content_sha(spec)
                or evidence.get('coverage')!=check['coverage'] or evidence.get('actual_N')!=spec['N']
                or evidence.get('fixture_geometry_validated') is not True or evidence.get('uniform_request_bps')!=21600000):raise ValueError('concrete fixture/check evidence differs')
        journal=pickle.loads(read_ref(evidence['current_journal_ref']))
        if (journal['identity']['check_SHA']!=content_sha(check) or journal['identity']['plan_SHA']!=content_sha(engineering)
                or journal['attempts']!=evidence['attempts'] or journal['completed']!=evidence['completed']):raise ValueError('actual effect journal differs from evidence')
        for key in totals:
            value=evidence['attempts'][key];completed=evidence['completed'][key]
            if type(value) is not int or type(completed) is not int or not 0<=completed<=value<=check['worst_case'][key]:raise ValueError('noninteger/negative/nonfinite engineering cost')
            if check['kind'] not in ('unknown_query','unknown_commit') and completed!=value:raise ValueError('unexpected unresolved engineering effect')
            totals[key]+=value
        if check['kind'] in ('method','dedup_tie','mask_shrink','pause_before','pause_query','pause_selection','pause_commit') and evidence['attempts']['commit']!=1:raise ValueError('required real commit not performed')
        if check['kind']=='probability' and evidence['attempts']['forward']!=2:raise ValueError('two actual probability comparison forwards required')
        if check['kind']=='probability' and any(evidence['proof'].get(key) is not True for key in ('original_WidthAgent_constructor','original_caller_chain_SHA_verified','logits_and_masked_probabilities_compared')):
            raise ValueError('original WidthAgent and new actor-only forward equivalence proof required')
        if check['kind'] in ('unknown_query','unknown_commit'):
            kind='query' if check['kind']=='unknown_query' else 'commit'
            if journal['transaction'].get('pending')!=kind or evidence['attempts'][kind]!=1 or evidence['proof'].get('unknown_preserved') is not True:raise ValueError('expected fault intent must remain nonzero and blocked')
    actual=dict(physical_calls_attempted=totals['query']+totals['commit'],real_commit_attempts=totals['commit'],
        candidate_restore_attempts=totals['query'],model_forward_attempts=totals['forward'])
    if any(receipt.get(key)!=value for key,value in actual.items()):raise ValueError('receipt total does not equal builtin effect journals')
    return dict(passed=True,actual=actual,expected_fault_unknowns_preserved=True)
