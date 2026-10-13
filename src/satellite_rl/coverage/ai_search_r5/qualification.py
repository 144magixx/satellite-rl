"""Eight fixed synthetic fixtures, explicit WAL; never runs from unittest.

Construction uses original arrays and hand declared legal W groups. It performs
no witness search, teacher optimization, MILP, model, or CUDA operation.
"""
from pathlib import Path
import hashlib,json,time,copy
import numpy as np
from . import core as c

FIXTURES=(
 dict(name='geometry_first',longitudes=[5,6,20,21],demands=[1,1,1,1],w_anchors=[0,2],w_owners=[0,0,1,1],w_widths=[.5,.5],anchors=[0,2],scores=[[0,100],[0,100],[100,0],[100,0]]),
 dict(name='evict_insert',longitudes=[8,10,12,14],demands=[1,1,1,1],w_anchors=[0,2],w_owners=[0,0,1,1],w_widths=[.5,.5],anchors=[0,3],scores=[[1,0],[1,0],[1,0],[0,1]]),
 dict(name='singleton_delete_multiplicity',longitudes=[12]*5,angular_offsets_deg=[0,.10,.35,.35,.90],demands=[1,1,1,1,1],w_anchors=[1,4],w_owners=[0,0,0,0,1],w_widths=[.5,.5],anchors=[0,1,4],scores=[[1,0,0],[0,1,0],[1,0,0],[1,0,0],[0,0,1]]),
 dict(name='no_completion_fallback',longitudes=[12]*3,angular_offsets_deg=[-.23,0,.23],demands=[1,1,1],w_anchors=[1],w_owners=[0,0,0],w_widths=[.5],anchors=[0],scores=[[1],[1],[1]]),
 dict(name='duplicate_anchor_slots',longitudes=[8,8.1,8.2,8.3],demands=[1,1,1,1],w_anchors=[1],w_owners=[0,0,0,0],w_widths=[.5],anchors=[0,0],scores=[[1,0],[1,0],[0,1],[0,1]]),
 dict(name='Q0_CV_guard',longitudes=[8,9,20],demands=[5,5,9],w_anchors=[0,2],w_owners=[0,0,1],w_widths=[.5,.5],anchors=[0,2],scores=[[1,0],[1,0],[0,1]]),
 dict(name='variable_N_nonfloor_widths',longitudes=[8,10,12,14],demands=[1,1,1,1],w_anchors=[2],w_owners=[0,0,0,0],w_widths=[1.],anchors=[2],scores=[[1],[1],[1],[1]]),
 dict(name='zero_one_degree_typed_truncation',longitudes=[8,12],demands=[0,0],w_anchors=[0],w_owners=[0,0],w_widths=[1.],anchors=[0],scores=[[1],[1]],boundary=True),
)


def _save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def source_manifest():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('core.py','adapter.py','qualification.py')}


def make_fixture(ordinal,spec):
    positions=[[48.,x] for x in spec['longitudes']]
    if spec.get('angular_offsets_deg') is not None:
        seed=c.Scene([[48.,12.]],[0.],('origin',));r=seed.rays[0]
        axis=np.eye(3)[int(np.argmin(np.abs(r)))];u=axis-float(axis@r)*r;u/=np.linalg.norm(u)
        positions=[c.ray_location(seed,np.cos(np.deg2rad(a))*r+np.sin(np.deg2rad(a))*u).tolist()
                   for a in spec['angular_offsets_deg']]
    if spec.get('boundary'):
        seed=c.Scene([[48.,8.]],[0.],('origin',));r=seed.rays[0]
        axis=np.eye(3)[int(np.argmin(np.abs(r)))];u=axis-float(axis@r)*r;u/=np.linalg.norm(u)
        positions[1]=c.ray_location(seed,np.cos(np.deg2rad(.5))*r+np.sin(np.deg2rad(.5))*u).tolist()
    scene=c.Scene(positions,spec['demands'],tuple(f'{spec["name"]}:{i}' for i in range(len(positions))))
    identity=dict(case_id='synthetic:'+spec['name'],mother_id=f'synthetic:{ordinal}',profile='base',input_sha256=scene.input_sha256,identity_sha256='fixed_eight_R5_v2',role='development')
    raw=c.Layout(scene.positions[spec['w_anchors']],spec['w_widths'],spec['w_owners'],tuple(f'W:b{i}' for i in range(len(spec['w_anchors']))))
    widths=[];pairs=0
    for b,ai in enumerate(spec['w_anchors']):
        members=np.flatnonzero(raw.assignment==b);pairs+=len(members)
        angles,chords=c._angles_and_chord(scene.rays[members],scene.rays[ai]);required=2*float(angles.max())
        width=max(.3,float(np.nextafter(required,np.inf)))
        if width>1. and required<=1.:width=1.
        # Boundary fixture explicitly retains the independently valid 1deg
        # chord predicate even if inverse-angle roundoff lies on either side.
        if spec.get('boundary') and width>1.:
            limit=4*np.sin(np.deg2rad(.5)/2)**2
            if np.all(chords<=limit+1e-14):width=1.
        widths.append(width)
    wenv=c.Layout(raw.centers,widths,raw.assignment,raw.beam_ids)
    p=c.PreparedCase(scene,identity,'synthetic_declared_fixture_W',raw,wenv,
        tuple(scene.aircraft_ids[i] for i in spec['w_anchors']),min(200,len(positions),int(np.ceil(1.25*len(raw.centers)))),c.omega(raw.widths),
        dict(fixture_construction_pairs=pairs,no_prepare_or_teacher_search=True))
    # Independent fixture validation is construction accounting, not hidden
    # decode or optimizer work; it is retained explicitly outside D5 budgets.
    raw_audit=c.validate_owned(scene,raw);wenv_audit=c.validate_owned(scene,wenv);pairs+=2*len(scene.demands)
    if not raw_audit['valid'] or not wenv_audit['valid']:raise ValueError('hand_fixture_invalid:'+spec['name'])
    slots=[f'proposal:{i}' for i in range(len(spec['anchors']))]
    proposal=dict(schema='ProposalV2',status='ready',case_identity=identity,input_sha=scene.input_sha256,role_sha=identity['identity_sha256'],
        provenance=c._provenance(p,'EXACT',source_manifest()['core.py'],[]),slot_ids=slots,
        anchor_aircraft_ids=[scene.aircraft_ids[i] for i in spec['anchors']],scores_n_by_K=spec['scores'],
        mask_n_by_K=np.ones((len(positions),len(slots)),bool).tolist(),construction_cost=dict(seconds=0.,raypairs=0,hash_calls=0,sort_input_count=0,scan_input_count=0))
    return p,proposal,dict(spec=spec,scene=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),aircraft_ids=list(scene.aircraft_ids),profile=scene.profile),prepared=p.to_dict(),proposal=proposal,fixture_construction_pairs=pairs,raw_audit=raw_audit,Wenv_audit=wenv_audit)


def freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=output/'fixture_manifest.json'
    if manifest.exists():raise RuntimeError('fixture_manifest_already_exists_no_overwrite')
    _save(manifest,dict(schema='R5EightFixturePreExecutionV1',contract_sha256='3555fdc7c64142f7af32fb6b32177db97e931aad52877270a97d1fce6ed74e90',
        fixtures=list(FIXTURES),fixture_cap=8,calls_per_fixture_cap=3,whole_D5_cap=24,planned_calls=10,
        source_sha256=source_manifest(),zero_prepare_teacher_search=True,zero_real_Jan12=True,zero_CUDA_MILP=True,
        call_schedule=[{'fixture':i,'calls':['normal','nominal_cutoff_mock','identity_hard_error'] if i==7 else ['normal']} for i in range(8)]))
    return manifest


def run_once(output):
    output=Path(output);manifest=json.loads((output/'fixture_manifest.json').read_text(encoding='utf8'))
    if manifest['source_sha256']!=source_manifest():raise RuntimeError('source_changed_after_fixture_freeze')
    ledger_path=output/'core_geometry_qualification_ledger.json'
    if ledger_path.exists():raise RuntimeError('qualification_already_started_no_replay')
    ledger=dict(schema='R5GeometryQualificationV1',status='running',source_sha256=source_manifest(),fixture_manifest_sha256=hashlib.sha256((output/'fixture_manifest.json').read_bytes()).hexdigest(),
        fixtures=[],calls=[],new_prepare_search=0,new_teacher_search=0,real_Jan12=0,CUDA=0,MILP=0)
    _save(ledger_path,ledger)
    for ordinal,spec in enumerate(FIXTURES):
        ledger['fixtures'].append(dict(ordinal=ordinal,name=spec['name'],state='construction_intent'));_save(ledger_path,ledger)
        p,prop,record=make_fixture(ordinal,spec)
        fixture_path=output/f'fixture_{ordinal:02d}.json';_save(fixture_path,record)
        ledger['fixtures'][-1].update(state='constructed',fixture_ref=dict(path=str(fixture_path),sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest()),construction_raypairs=record['fixture_construction_pairs']);_save(ledger_path,ledger)
        schedule=['normal','nominal_cutoff_mock','identity_hard_error'] if ordinal==7 else ['normal']
        for label in schedule:
            proposal=copy.deepcopy(prop)
            if label=='identity_hard_error':proposal['case_identity']={**p.identity,'role':'test'}
            row=dict(ordinal=ordinal,label=label,state='intent',source_sha256=source_manifest()['core.py'],started_utc_seconds=time.time(),nominal_limits=dict(seconds=0. if label=='nominal_cutoff_mock' else 5.,proposals=512,refits=8,raypairs=2048*len(p.scene.demands)))
            ledger['calls'].append(row);_save(ledger_path,ledger)
            def persist(event):
                accepted=output/f'fixture_{ordinal:02d}_{label}_accepted';accepted.mkdir(exist_ok=True)
                _save(accepted/f'{len(list(accepted.glob("*.json"))):04d}.json',event)
            try:
                if label=='nominal_cutoff_mock':
                    # Search stops immediately; the real terminal reserve and
                    # independent fallback audit remain executable.
                    result=c.decode_groups(p,proposal,time_limit=0.,on_accepted=persist)
                else:result=c.decode_groups(p,proposal,on_accepted=persist)
                path=output/f'fixture_{ordinal:02d}_{label}_result.json';_save(path,result)
                row.update(state='completed',status=result['status'],cost=result['cost'],result_ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            except c.InterfaceError as exc:
                if label!='identity_hard_error':raise
                row.update(state='expected_hard_error',error=str(exc),known_algorithm_operations=0)
            except BaseException as exc:
                row.update(state='failed',error=repr(exc));ledger['status']='failed';_save(ledger_path,ledger);raise
            row['ended_utc_seconds']=time.time();_save(ledger_path,ledger)
    ledger['status']='completed';ledger['whole_D5_calls']=len(ledger['calls']);_save(ledger_path,ledger)
    return ledger

if __name__=='__main__':
    import sys
    if sys.argv[1]=='freeze':print(freeze(sys.argv[2]))
    elif sys.argv[1]=='run':print(json.dumps({'status':run_once(sys.argv[2])['status']}))
    else:raise ValueError('use freeze or run')
