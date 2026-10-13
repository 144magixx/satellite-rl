"""Explicit opt-in numerical qualification; never runs on import."""
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .features import collate_features,build_features,FeatureStats,record_sha
from .learning import compute_loss_sums,global_loss,CaseSampler,TeacherTarget,TrainingSession,match_targets,solver_identity
from .model import initialize_model,tensor_manifest,model_tensor_sha


def _atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _source_binding(source_commit):
    directory=Path(__file__).resolve().parent
    names=('features.py','model.py','learning.py','ai_adapter.py','model_qualification.py')
    return dict(source_commit=source_commit,source_files={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names})


def _consume_budget(budget,spec_sha,output):
    source=_source_binding(budget.get('source_commit'))
    if not source['source_commit'] or budget.get('source_manifest_sha')!=record_sha(source) or budget.get('source_files')!=source['source_files']:raise ValueError('qualification frozen source differs')
    if Path(budget.get('output_path','')).resolve()!=Path(output).resolve():raise ValueError('reservation output path differs')
    claim_path=Path(budget['global_claim_path'])
    if not claim_path.is_absolute() or not budget.get('reservation_id'):raise ValueError('absolute global budget claim and reservation ID required')
    claim_path.parent.mkdir(parents=True,exist_ok=True)
    claim=dict(schema='R6EngineeringBudgetClaimV1',reservation_id=budget['reservation_id'],spec_sha=spec_sha,
        source_manifest_sha=budget['source_manifest_sha'],source_commit=source['source_commit'],output_path=str(Path(output).resolve()),
        status='consumed_once_no_replay',updates=[])
    # An exclusive durable claim precedes any GPU operation. Copies of the same
    # receipt still point to this one global claim and cannot create another run.
    with claim_path.open('x',encoding='utf8') as stream:json.dump(claim,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    return claim_path,claim,source


def _counted_optimizer_step(original_step,opt,args,kwargs,receipt,persist):
    if len(receipt['updates'])>=4:raise RuntimeError('synthetic optimizer update cap4')
    event=dict(ordinal=len(receipt['updates'])+1,actual_optimizer='AdamW',synthetic=True,status='started_unknown_consumed')
    receipt['updates'].append(event);persist()  # write-ahead intent consumes budget
    try:result=original_step(opt,*args,**kwargs)
    except BaseException as exc:
        event['status']='unknown_consumed';event['error']=repr(exc);persist();raise
    event['status']='completed';persist();return result


def compare_accumulation(model,cases,targets,stats,*,device,matching,update_permission=False):
    """Costs TWO synthetic optimizer updates; caller must reserve global R6 budget."""
    if not update_permission or len(cases)!=16:raise ValueError('root must reserve two engineering updates')
    models=[copy.deepcopy(model).to(device) for _ in range(2)];losses=[];gradients=[]
    A=sum(len(t.aircraft_ids) for t in targets);G=sum(len(t.anchor_ids) for t in targets)
    for net,micro in zip(models,(16,4)):
        opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,amsgrad=False,foreach=False,fused=False)
        opt.zero_grad(set_to_none=True);total=0.
        for b in range(0,16,micro):
            output=net(**collate_features(cases[b:b+micro],stats,device))
            loss=global_loss(compute_loss_sums(output,targets[b:b+micro],matching[b:b+micro]),A,G)
            loss.backward();total+=float(loss.detach())
        gradients.append({name:p.grad.detach().clone() for name,p in net.named_parameters()});losses.append(total)
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    torch.testing.assert_close(torch.tensor(losses[0]),torch.tensor(losses[1]),atol=1e-6,rtol=1e-5)
    for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],atol=1e-6,rtol=1e-5)
    for a,b in zip(models[0].parameters(),models[1].parameters()):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    return dict(synthetic_optimizer_updates=2,device=str(device),losses=losses,passed=True)


def fixture_specification():
    return dict(schema='R6SyntheticModelQualificationV1',scientific=False,seed=51001,
        unique_synthetic_cases=[dict(case_index=i,case_id=f'synthetic_{i}',aircraft_n=i+2,
            ray_theta_step_radians=.001,request_bps_start=0,request_bps_step=1e6,teacher_beams=2)
            for i in range(4)],effective_batch=16,fixture_reuse='PCG64_cross_epoch_four_case_table',
        synthetic_CUDA_optimizer_updates_reserved=4,allocation=dict(accumulation_equivalence=2,atomic_resume_equivalence=2),
        no_scientific_reads=True,no_D5_prepare_T=True,matching='CPU_float64_SciPy1.14.1_once_per_sample_fixed_untied_logits',
        graph='R6_frozen_model',tolerances=dict(atol=1e-6,rtol=1e-5))


def _fixtures(spec):
    cases=[];targets=[];table=[]
    for fixture in spec['unique_synthetic_cases']:
        n=fixture['aircraft_n'];theta=np.arange(n)*fixture['ray_theta_step_radians']
        ids=tuple(f'syn{fixture["case_index"]}_aircraft_{i}' for i in range(n))
        rays=np.column_stack((np.sin(theta),np.zeros(n),np.cos(theta)))
        demands=np.arange(n)*fixture['request_bps_step']+fixture['request_bps_start']
        input_sha=record_sha(dict(rays=rays.tolist(),demands=demands.tolist(),ids=ids))
        owners=np.arange(n)%2;anchors=ids[:2];slots=('synthetic_beam0','synthetic_beam1')
        label_sha=record_sha(dict(owners=owners.tolist(),anchors=anchors,slots=slots,input_sha=input_sha))
        row=dict(case_index=fixture['case_index'],case_id=fixture['case_id'],input_sha=input_sha,label_ref_sha=label_sha)
        identity=dict(row,role='train');scene=SimpleNamespace(rays=rays,demands=demands,aircraft_ids=ids,input_sha256=input_sha)
        cases.append(build_features(scene,identity=identity));targets.append(TeacherTarget(ids,owners,anchors,slots,identity,label_sha));table.append(row)
    return cases,targets,table


def _fixed_matching_logits(cases,targets):
    nmax=max(len(c.aircraft_ids) for c in cases);B=len(cases)
    enable=torch.arange(200,dtype=torch.float32)[None,:].expand(B,-1)*-.01
    owner=torch.zeros((B,nmax,200));pointer=torch.zeros((B,200,nmax));valid=torch.zeros((B,nmax),dtype=torch.bool)
    for b,target in enumerate(targets):
        n=len(target.aircraft_ids);valid[b,:n]=True
        for i,j in enumerate(target.owners):owner[b,i,int(j)]=20
        pointer[b,0,0]=20;pointer[b,1,1]=20
        pointer[b,:,n:]=-torch.inf
    return dict(enable=enable,owner=owner,pointer=pointer,valid=valid)


def run_frozen_qualification(spec_path,expected_spec_sha,output_dir,budget_receipt_path):
    """Root executes only after freezing specs/source; exactly four synthetic steps."""
    spec_raw=Path(spec_path).read_bytes()
    if hashlib.sha256(spec_raw).hexdigest()!=expected_spec_sha:raise ValueError('qualification specification SHA')
    spec=json.loads(spec_raw)
    if spec!=fixture_specification():raise ValueError('specification differs from fixed qualification')
    budget=json.loads(Path(budget_receipt_path).read_text(encoding='utf8'))
    if budget.get('reserved_CUDA_updates')!=4 or budget.get('spec_sha')!=expected_spec_sha or budget.get('execute_authorized') is not True:raise ValueError('root engineering reservation required')
    output=Path(output_dir)
    if output.exists():raise ValueError('qualification output must be new; no blind replay')
    claim_path,claim,source=_consume_budget(budget,expected_spec_sha,output)
    if not torch.cuda.is_available():raise RuntimeError('CUDA qualification cannot silently use CPU')
    solver=solver_identity()
    output.mkdir(parents=True);initial=initialize_model();torch.save(initial.state_dict(),output/'initial_state.pt')
    initial_sha=hashlib.sha256((output/'initial_state.pt').read_bytes()).hexdigest()
    cases,targets,table=_fixtures(spec);sampler=CaseSampler(table);indices=sampler.assemble()
    batch=[cases[i] for i in indices];labels=[targets[i] for i in indices]
    stats=FeatureStats.fit(cases,role='train',qualified_table_sha=sampler.table_sha)
    matches=match_targets(_fixed_matching_logits(batch,labels),labels)
    for match,label in zip(matches,labels):
        if match['mapping']!={0:0,1:1}:raise AssertionError('fixed strict matching winners changed')
    receipt=dict(schema='R6SyntheticUpdateReceiptV1',scientific=False,spec_sha=expected_spec_sha,
        unique_fixture_count=4,initial_file_sha=initial_sha,initial_model_tensor_sha=model_tensor_sha(initial),
        initial_tensor_manifest=tensor_manifest(initial),solver=solver,torch_version=torch.__version__,
        torch_source=torch.__file__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0),updates=[],
        frozen_source=source,actual_module_directory=str(Path(__file__).resolve().parent),source_manifest_sha=budget['source_manifest_sha'])
    def save_receipt():
        claim['updates']=copy.deepcopy(receipt['updates']);_atomic_json(claim_path,claim)
        _atomic_json(output/'receipt.json',receipt)
    original_step=torch.optim.AdamW.step
    def counted_step(opt,*args,**kwargs):
        return _counted_optimizer_step(original_step,opt,args,kwargs,receipt,save_receipt)
    torch.optim.AdamW.step=counted_step
    try:
        receipt['accumulation']=compare_accumulation(initial,batch,labels,stats,device='cuda',matching=matches,update_permission=True)
        provenance=dict(synthetic=True,initial_state_sha=initial_sha,spec_sha=expected_spec_sha)
        baseline=TrainingSession(copy.deepcopy(initial),stats,CaseSampler(table),provenance,'cuda')
        baseline.start_activity(clock_path=output/'baseline_clock.json')
        indices=baseline.sampler.assemble();batch=[cases[i] for i in indices];labels=[targets[i] for i in indices]
        checkpoint=output/'before_step.pt';checkpoint_receipt=baseline.checkpoint(checkpoint)
        baseline_result=baseline.step(batch,labels)
        evidence=dict(verified=True,checkpoint_file_sha=checkpoint_receipt['checkpoint_file_sha'],
            charged_active_seconds=checkpoint_receipt['charged_active_seconds'],
            last_completed_checkpoint_IO_active_seconds=checkpoint_receipt['charged_active_seconds'],
            unknown_tail_accounted=True,interval_evidence='controlled synthetic checkpoint branch;no unknown crash interval',
            prior_clock_events=json.loads((output/'baseline_clock.json').read_text())['events'])
        evidence_path=output/'recovery_clock_evidence.json';evidence_path.write_text(json.dumps(evidence,sort_keys=True),encoding='utf8')
        resumed=TrainingSession(copy.deepcopy(initial),stats,CaseSampler(table),provenance,'cuda')
        resumed.restore(checkpoint,recovery_clock=dict(evidence_path=str(evidence_path),evidence_sha=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),clock_path=str(output/'resumed_clock.json')))
        resumed.start_activity();indices=resumed.sampler.assemble()
        resumed_result=resumed.step([cases[i] for i in indices],[targets[i] for i in indices])
        for name,tensor in baseline.model.state_dict().items():torch.testing.assert_close(tensor,resumed.model.state_dict()[name],atol=1e-6,rtol=1e-5)
        for parameter,state in baseline.optimizer.state.items():
            name=next(name for name,p in baseline.model.named_parameters() if p is parameter)
            other=dict(resumed.model.named_parameters())[name]
            for key,value in state.items():torch.testing.assert_close(value,resumed.optimizer.state[other][key],atol=1e-6,rtol=1e-5)
        if baseline.sampler.state_dict()!=resumed.sampler.state_dict():raise AssertionError('resumed sampler differs')
        receipt['resume_equivalence']=dict(passed=True,baseline_update=baseline_result['update'],resumed_update=resumed_result['update'])
        receipt['passed']=True
    except BaseException as exc:
        receipt['passed']=False;receipt['error']=repr(exc);raise
    finally:
        torch.optim.AdamW.step=original_step;save_receipt()
    if len(receipt['updates'])!=4:raise AssertionError('expected exactly4 engineering updates')
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));parser.add_argument('--spec',required=True)
    parser.add_argument('--expected-spec-sha');parser.add_argument('--output');parser.add_argument('--budget-receipt');args=parser.parse_args()
    if args.mode=='freeze':
        path=Path(args.spec)
        if path.exists():raise ValueError('no overwrite frozen specification')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(fixture_specification(),indent=2,sort_keys=True)+'\n',encoding='utf8')
        print(hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        if not args.expected_spec_sha or not args.output or not args.budget_receipt:parser.error('run requires frozen SHA, output and root budget receipt')
        print(json.dumps(run_frozen_qualification(args.spec,args.expected_spec_sha,args.output,args.budget_receipt),sort_keys=True))


if __name__=='__main__':main()
