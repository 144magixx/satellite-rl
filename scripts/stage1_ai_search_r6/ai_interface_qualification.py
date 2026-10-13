"""Two preregistered synthetic AI-to-D5 interface fixtures, zero optimizer steps.

No preparation or teacher search. Uses hand declared W fixtures and records
construction geometry separately. Never runs at import or from ordinary tests.
"""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch

DESIGN_SHA='74246c2f5baa2f40cc88fff248add335a2c7151028dd5193873256f55e7fd892'
SPECS=((6,'train'),(7,'development'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    from satellite_rl.training.run_monitor import atomic_json
    atomic_json(p,v)
def sources():
    root=Path(__file__).resolve().parents[2]
    paths=['scripts/stage1_ai_search_r6/ai_interface_qualification.py']
    paths += ['src/satellite_rl/coverage/ai_search_r6/'+f for f in ('model.py','features.py','ai_adapter.py')]
    paths += ['src/satellite_rl/coverage/ai_search_r5/'+f for f in ('core.py','qualification.py')]
    paths += ['src/satellite_rl/coverage/ai_search/core.py','scripts/stage1_ai_search_r4/audit.py',
              'scripts/stage1_ai_search_r5/audit.py']
    return {p:sha(root/p) for p in paths}

def freeze(directory,commit):
    from satellite_rl.coverage.ai_search_r5.qualification import FIXTURES
    if not commit or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('exact source commit required')
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    p=d/'fixture_manifest.json'
    if p.exists():raise ValueError('already frozen')
    save(p,{'schema':'R6TwoAIInterfaceFixturesV1','scientific_commit':commit,'design_sha256':DESIGN_SHA,
        'sources':sources(),'fixtures':[{'original_spec_index':i,'actual_new_role':role,'spec':FIXTURES[i]} for i,role in SPECS],
        'planned_D5_calls':2,'planned_optimizer_updates':0,'prepare_T_calls':0,'real_data_reads':0,
        'fixed_model':'CPU initialize seed51001; untrained synthetic qualification only',
        'statistics':'synthetic identity normalization mean0/scale1, not usable for science'})

def run(directory):
    from satellite_rl.coverage.ai_search_r5.qualification import make_fixture,FIXTURES
    from satellite_rl.coverage.ai_search_r5.core import decode_groups
    from satellite_rl.coverage.ai_search_r6.features import build_features,FeatureStats,collate_features
    from satellite_rl.coverage.ai_search_r6.model import initialize_model,model_tensor_sha,tensor_manifest
    from satellite_rl.coverage.ai_search_r6.ai_adapter import build_ai_proposal,audit_ai_proposal
    from scripts.stage1_ai_search_r5.audit import audit_layout,audit_decoder_ledger
    d=Path(directory);m=json.loads((d/'fixture_manifest.json').read_text(encoding='utf-8'))
    if m['sources']!=sources():raise ValueError('source changed since fixture freeze')
    expected=[{'original_spec_index':i,'actual_new_role':role,'spec':FIXTURES[i]} for i,role in SPECS]
    if (m['fixtures']!=expected or m['design_sha256']!=DESIGN_SHA or m['planned_D5_calls']!=2
        or m['planned_optimizer_updates']!=0 or m['prepare_T_calls']!=0 or m['real_data_reads']!=0
        or len(m.get('scientific_commit',''))!=40):raise ValueError('fixed fixture contract differs')
    receipt=d/'qualification.json'
    if receipt.exists():raise ValueError('qualification WAL exists; no replay')
    ledger={'schema':'R6AIInterfaceQualificationV1','scientific_commit':m['scientific_commit'],'design_sha256':DESIGN_SHA,
        'manifest_sha256':sha(d/'fixture_manifest.json'),'sources':sources(),'passed':False,'status':'running',
        'prepare_calls':0,'T_calls':0,'optimizer_updates':0,'real_dataset_reads':0,'D5_calls':[],'fixtures':[]}
    save(receipt,ledger);torch.set_num_threads(1)
    model=initialize_model().eval();torch.save(model.state_dict(),d/'synthetic_initial_state.pt')
    save(d/'initial_tensor_manifest.json',tensor_manifest(model))
    stats=FeatureStats(np.zeros(10),np.ones(10),'synthetic_only_not_scientific_statistics')
    for ordinal,spec in enumerate(m['fixtures']):
        p,discarded,construction=make_fixture(spec['original_spec_index'],spec['spec'])
        # New synthetic object identity; no prior prepared record is relabelled.
        p.identity={**p.identity,'case_id':f'synthetic:r6:interface:{ordinal}',
            'mother_id':f'synthetic:r6:mother:{ordinal}','identity_sha256':sha(d/'fixture_manifest.json'),
            'role':spec['actual_new_role']}
        construction['prepared']=p.to_dict();construction.pop('proposal',None)
        save(d/f'fixture_{ordinal}.json',construction)
        t=time.monotonic();f=build_features(p.scene)
        with torch.no_grad():out=model(**collate_features([f],stats))
        inference={'seconds':time.monotonic()-t,'model_forward_calls':1,'engineering_only':True,'features':f.cost}
        prov={'checkpoint_file_SHA':sha(d/'synthetic_initial_state.pt'),'model_tensor_SHA':model_tensor_sha(model),
            'model_graph_source_SHA':m['sources']['src/satellite_rl/coverage/ai_search_r6/model.py'],
            'feature_graph_source_SHA':m['sources']['src/satellite_rl/coverage/ai_search_r6/features.py'],
            'normalization_SHA':stats.sha,'model_initial_state_SHA':sha(d/'synthetic_initial_state.pt'),
            'adapter_source_SHA':m['sources']['src/satellite_rl/coverage/ai_search_r6/ai_adapter.py'],
            'b7_decoder_source_SHA':m['sources']['src/satellite_rl/coverage/ai_search_r5/core.py'],
            'source_epoch':m['scientific_commit']}
        envelope=build_ai_proposal(p,out,prov,inference_cost=inference)
        save(d/f'fixture_{ordinal}_AI.json',envelope)
        corrupted=copy.deepcopy(envelope);corrupted['role_SHA']='wrong_role'
        try:audit_ai_proposal(p,corrupted)
        except ValueError:pass
        else:raise AssertionError('mismatched role accepted')
        row={'ordinal':ordinal,'state':'reserved','started':time.time()};ledger['D5_calls'].append(row);save(receipt,ledger)
        result=decode_groups(p,envelope['inner']);save(d/f'fixture_{ordinal}_D5.json',result)
        checked=audit_layout(construction['scene'],result['layout'],p.Ncap,p.Omega_cap)
        budget=audit_decoder_ledger(result['cost'],len(p.scene.demands),'D5')
        if not checked['valid'] or not budget['valid']:raise AssertionError((checked,budget))
        row.update(state='completed',status=result['status'],cost=result['cost'],independent_geometry=checked,independent_budget=budget)
        ledger['fixtures'].append({'ordinal':ordinal,'construction_geometry_pairs':construction['fixture_construction_pairs'],
            'actual_role':p.identity['role'],'actual_model_forward_calls':1,'mismatched_role_rejected_before_D5':True})
        save(receipt,ledger)
    ledger.update(passed=True,status='completed');save(receipt,ledger)
    print(json.dumps({'passed':True,'D5_calls':len(ledger['D5_calls']),'optimizer_updates':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','run'));p.add_argument('--directory',required=True);p.add_argument('--commit');a=p.parse_args()
    if a.action=='freeze':freeze(a.directory,a.commit)
    else:run(a.directory)
