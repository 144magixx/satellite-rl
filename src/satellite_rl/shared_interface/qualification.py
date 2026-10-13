"""Root-frozen exact12 interface qualification. No execution on import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from . import stages
from . import content
from .content import read_wire,content_sha,seal,wire_bytes
from .contracts import Authority
from .spherical_gaussian import PHYSICS,gain_dbi,pattern_solid_angle
from .fixtures import COUNTS,file_sha,write_plan


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


class BudgetLedger:
    def __init__(self,path,plan_sha,source_commit):
        self.path=Path(path);self.value=dict(schema='R7ExactInterfaceEngineeringLedgerV1',plan_sha=plan_sha,source_commit=source_commit,
            counts={key:0 for key in COUNTS},intents=[],status='consumed_once_no_replay')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('x',encoding='utf8') as stream:json.dump(self.value,stream);stream.flush();os.fsync(stream.fileno())

    def charge(self,name,fixture,amount=1):
        if self.value['counts'][name]+amount>COUNTS[name]:raise RuntimeError('exact fixture quota '+name)
        self.value['counts'][name]+=amount;self.value['intents'].append(dict(kind=name,fixture=fixture,amount=amount,status='reserved_unknown_if_interrupted'))
        atomic_json(self.path,self.value)


def _independent_link_oracle(layout,allocation):
    """Handwritten equations; does not import shared geometry/pattern/link helpers."""
    ids=layout['beam_ids'];N=len(ids);mapping=[allocation['beam_ids'].index(b) for b in ids]
    starts=allocation['actual_start_slot'][mapping];counts=allocation['actual_slot_count'][mapping]
    power=allocation['actual_whole_beam_power_w'][mapping];groups=np.asarray([allocation['group_by_beam_id'][b] for b in ids])
    lat,lon=np.deg2rad(layout['centres_lat_lon_deg']).T
    earth=6371000*np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    satellite=42164000*np.array([np.cos(np.deg2rad(13)),np.sin(np.deg2rad(13)),0.]);vectors=earth-satellite
    distance=np.linalg.norm(vectors,axis=1);rays=vectors/distance[:,None]
    theta=2*np.arcsin(np.minimum(np.linalg.norm(rays[:,None]-rays[None],axis=2)/2,1))
    width=np.deg2rad(layout['full_width_deg']);kap=np.log(2)/(2*np.sin(width/4)**2)
    kapref=np.log(2)/(2*np.sin(np.deg2rad(.5)/4)**2);bref=2*np.pi*(-np.expm1(-2*kapref))/kapref
    B=2*np.pi*(-np.expm1(-2*kap))/kap;peak=100000*bref/B
    gain=peak[None,:]*np.exp(-2*kap[None,:]*np.sin(theta/2)**2)
    owners=[];occupied=[];signal=[];interference=[];noise=[];capacity=np.zeros(N)
    for i in range(N):
        if not counts[i]:continue
        centre=17700000000+(starts[i]+counts[i]/2)*25000000
        signal_i=10**(-.5)*peak[i]*10000*(299792458/(4*np.pi*distance[i]*centre))**2/counts[i]
        for slot in range(int(starts[i]),int(starts[i]+counts[i])):
            row=np.zeros(N);frequency=17700000000+(slot+.5)*25000000
            for j in range(N):
                if counts[j] and groups[i]%2==groups[j]%2 and groups[i]!=groups[j] and starts[j]<=slot<starts[j]+counts[j]:
                    row[j]=gain[i,j]*10000*(299792458/(4*np.pi*distance[i]*frequency))**2/counts[j]
            thermal=1.380649e-23*290*25000000
            capacity[i]+=25000000*np.log1p(signal_i*power[i]/(thermal+row@power))/np.log(2)
            owners.append(i);occupied.append(slot);signal.append(signal_i);interference.append(row);noise.append(thermal)
    return dict(slot_owner=np.asarray(owners,np.int64),occupied_slots=np.asarray(occupied,np.int64),signal=np.asarray(signal),
        interference=np.asarray(interference).reshape(-1,N),noise=np.asarray(noise),capacity_bps=capacity)


def _compare_coeff(first,second):
    if first.beam_ids!=second.beam_ids:raise AssertionError('stable ID mismatch')
    for name in ('slot_owner','occupied_slots','powers_initial','active','demand_bps'):
        np.testing.assert_array_equal(getattr(first,name),getattr(second,name))
    for name in ('signal','interference','noise'):np.testing.assert_allclose(getattr(first,name),getattr(second,name),rtol=1e-10,atol=1e-30)


def execute_plan(directory,budget_path):
    root=Path(directory);plan_path=root/'plan.json';plan=json.loads(plan_path.read_text());budget=json.loads(Path(budget_path).read_text())
    if budget.get('execute_authorized') is not True or budget.get('plan_sha')!=file_sha(plan_path) or budget.get('design_sha')!=plan['design_sha']:raise ValueError('root concrete engineering admission')
    source=Path(__file__).resolve().parent;repo=source.parents[2]
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=budget['source_commit']:raise ValueError('engineering source epoch')
    for name,sha in plan['caller_chain_source_sha256'].items():
        if file_sha(source/name)!=sha:raise ValueError('actual caller source changed')
    for member,sha in plan['fixture_members'].items():
        if file_sha(root/member)!=sha:raise ValueError('fixed fixture bytes changed')
    output=Path(budget['output_path'])
    if not output.is_absolute() or not Path(budget['global_claim_path']).is_absolute():raise ValueError('absolute fixed output/global claim paths required')
    import scipy
    if scipy.__version__!='1.14.1':raise RuntimeError('qualification requires frozen SciPy1.14.1')
    if budget.get('numpy_version')!=np.__version__ or budget.get('scipy_source')!=scipy.__file__:raise ValueError('qualified actual NumPy/SciPy source identity required')
    if output.exists():raise ValueError('qualification output already used')
    ledger=BudgetLedger(budget['global_claim_path'],file_sha(plan_path),budget['source_commit']);output.mkdir(parents=True)
    result=dict(schema='R7ExactInterfaceQualificationV1',passed=False,source_commit=budget['source_commit'],plan_sha=file_sha(plan_path),cases=[],
        numpy_version=np.__version__,numpy_source=np.__file__,scipy_version=scipy.__version__,scipy_source=scipy.__file__)
    originals=(stages.build_link_coefficients,stages.capacity_bps,stages.read_wire,read_wire);current=[None]
    decode_counts=dict(object_reads=0)
    def decode(*args,**kwargs):decode_counts['object_reads']+=1;return originals[3](*args,**kwargs)
    def build(*args,**kwargs):ledger.charge('coefficient_builds',current[0]);return originals[0](*args,**kwargs)
    def evaluate(*args,**kwargs):ledger.charge('capacity_evaluations',current[0]);return originals[1](*args,**kwargs)
    stages.build_link_coefficients=build;stages.capacity_bps=evaluate
    stages.read_wire=decode
    try:
        for index in range(1,13):
            ident=f'F{index:02d}';current[0]=ident;folder=root/ident;expected=json.loads((folder/'expected.json').read_text())
            registry=decode((folder/'authority_registry.json').read_bytes());authority=Authority(registry,expected['authority_registry_sha'])
            outcomes=[];evaluations=[];s2_exported=None;export_chain=None
            for caller in range(3):
                ledger.charge('roundtrips',ident)
                try:
                    case_bytes=(folder/'case.json').read_bytes();layout_bytes=(folder/'layout.json').read_bytes();allocation_bytes=(folder/'allocation.json').read_bytes()
                    case=decode(case_bytes);layout=decode(layout_bytes);allocation=decode(allocation_bytes)
                    if caller==0:stages.s1_publish_import(case,layout,authority)
                    elif caller==1:
                        s2=stages.UniformStage2Input.from_wire(case_bytes,layout_bytes,authority,PHYSICS)
                        rows={bid:dict(group=allocation['group_by_beam_id'][bid],start=int(allocation['actual_start_slot'][i]),
                            count=int(allocation['actual_slot_count'][i]),power_w=float(allocation['actual_whole_beam_power_w'][i])) for i,bid in enumerate(allocation['beam_ids'])}
                        s2_exported=s2.export_actual_allocation(rows,requested_actions_separate=allocation['requested_actions_separate'])
                        if wire_bytes(s2_exported)!=(folder/'allocation_canonical.json').read_bytes():raise AssertionError('actual S2 export differs from preexecution canonical object bytes')
                        evaluations.append(s2.terminal_fixed_evaluation(s2_exported))
                        export_chain=dict(canonical_actual_S2_SHA=s2_exported['content_sha256'],parent_layout_SHA=s2_exported['parent_layout_SHA'])
                    else:
                        actual=s2_exported
                        if actual is not None:
                            import copy
                            actual=copy.deepcopy(actual)
                            permutation=expected['recipe'].get('allocation_row_permutation')
                            if permutation is not None:
                                actual['beam_ids']=[actual['beam_ids'][i] for i in permutation]
                                for key in ('actual_start_slot','actual_slot_count','actual_whole_beam_power_w'):actual[key]=actual[key][permutation].copy()
                                actual=seal(actual)
                            if wire_bytes(actual)!=allocation_bytes:raise AssertionError('actual S2 exported/permuted object differs from frozen S3 input bytes')
                            export_chain['actual_S3_SHA']=actual['content_sha256'];export_chain['permutation']=permutation
                        # For deliberately invalid fixtures S2 cannot export;
                        # S3 independently rejects the same frozen attempted graph.
                        incoming=wire_bytes(actual) if actual is not None else allocation_bytes
                        evaluations.append(stages.UniformStage3FixedInput.from_wire(case_bytes,layout_bytes,incoming,authority,PHYSICS).fixed_initial_evaluation())
                    observed='pass'
                except ValueError as exc:observed=str(exc)
                target=expected['expected'][caller]
                if (target=='pass' and observed!='pass') or (target!='pass' and target not in observed):raise AssertionError(ident+' caller '+str(caller)+' expected '+target+' actual '+observed)
                outcomes.append(observed)
            if evaluations:
                if len(evaluations)!=2:raise AssertionError('both actual callers required')
                _compare_coeff(evaluations[0]['coefficients'],evaluations[1]['coefficients'])
                np.testing.assert_allclose(evaluations[0]['capacity_bps'],evaluations[1]['capacity_bps'],rtol=1e-10,atol=1e-6)
                np.savez_compressed(output/(ident+'_coefficients_capacity.npz'),signal=evaluations[0]['coefficients'].signal,
                    interference=evaluations[0]['coefficients'].interference,noise=evaluations[0]['coefficients'].noise,
                    slot_owner=evaluations[0]['coefficients'].slot_owner,capacity_S2=evaluations[0]['capacity_bps'],capacity_S3=evaluations[1]['capacity_bps'])
                if ident in ('F01','F02','F12'):
                    oracle=_independent_link_oracle(layout,allocation)
                    for name in ('signal','interference','noise'):np.testing.assert_allclose(getattr(evaluations[0]['coefficients'],name),oracle[name],rtol=1e-10,atol=1e-30)
                    np.testing.assert_allclose(evaluations[0]['capacity_bps'],oracle['capacity_bps'],rtol=1e-10,atol=1e-6)
            pattern=[]
            if expected['recipe']['pattern_points']:
                for width in sorted(set(layout['full_width_deg'].tolist())):
                    radians=np.deg2rad(width);angles=np.asarray([0,radians/2,radians,np.pi]);ledger.charge('pattern_points',ident,4)
                    gains=gain_dbi(angles,radians);linear=np.exp(gains*np.log(10)/10)
                    np.testing.assert_allclose(gains[0]-gains[1],10*np.log10(2),rtol=1e-12,atol=1e-12)
                    B=float(pattern_solid_angle(radians));reference=float(pattern_solid_angle(np.deg2rad(.5)))
                    np.testing.assert_allclose(linear[0]*B,100000*reference,rtol=1e-12,atol=1e-12)
                    eta=100000*reference/(4*np.pi)
                    if not 0<eta<=1:raise AssertionError('invalid efficiency not clipped')
                    if width==.5:np.testing.assert_allclose(gains[0],50.,rtol=1e-12,atol=1e-12)
                    record=dict(width_deg=width,angles_rad=angles.tolist(),gain_dBi=gains.tolist(),linear_tail_underflows=int((linear==0).sum()),pattern_B_sr=B,eta=eta)
                    if ident=='F11':
                        import scipy
                        if scipy.__version__!='1.14.1':raise RuntimeError('frozen SciPy1.14.1 quad required')
                        from scipy.integrate import quad
                        concentration=np.log(2)/(2*np.sin(radians/4)**2);rootk=np.sqrt(concentration)
                        def integrand(x):return 2*np.pi*np.exp(-2*concentration*np.sin(x/(2*rootk))**2)*np.sin(x/rootk)/rootk
                        ledger.charge('normalization_integrals',ident)
                        integral,error=quad(integrand,0,20,epsabs=1e-14,epsrel=1e-10,limit=128)
                        tail=4*np.pi*np.exp(-2*concentration*np.sin(20/(2*rootk))**2)
                        if tail>=B*1e-8 or abs(integral-B)>B*1e-8:raise AssertionError('independent normalization integral')
                        record.update(independent_integral=integral,quad_error=error,tail_bound=tail)
                    pattern.append(record)
            result['cases'].append(dict(id=ident,outcomes=outcomes,pattern=pattern,actual_S2_to_S3_export_chain=export_chain))
            atomic_json(output/'partial_result.json',result)
        if ledger.value['counts']!=COUNTS:raise AssertionError('exact planned calls incomplete')
        result['passed']=True
    except BaseException as exc:result['error']=repr(exc);raise
    finally:
        stages.build_link_coefficients,stages.capacity_bps,stages.read_wire=originals[:3];result['actual_counts']=ledger.value['counts']
        result['underlying_object_read_wire_calls']=decode_counts['object_reads']
        result['no_labels_planner_teacher_decoder_model_optimizer_solver']=True;result['ledger_sha']=file_sha(ledger.path)
        atomic_json(output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));parser.add_argument('--design');parser.add_argument('--acceptance');parser.add_argument('--directory',required=True);parser.add_argument('--budget');args=parser.parse_args()
    if args.mode=='prepare':value=write_plan(args.design,args.acceptance,args.directory)
    elif not args.budget:parser.error('--budget required')
    else:value=execute_plan(args.directory,args.budget)
    print(json.dumps(value,sort_keys=True))
