"""V3 formula and remaining-two-update resume checks; never execute on import.

Original v2 claims/receipts remain immutable. A separate exclusive continuation
claim binds their bytes and retains historical spend in every write-ahead record.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ATOL=1e-6
RTOL=1e-5
V3_DESIGN_SHA='9297be896dc54b4b49d5ebb66f5fed80bc880bfe180692491dd64dce961fc988'
ROOT_ACCEPTANCE_SHA='60389c5477492faec48213dc1439f18dc44e7858ba8278271202a3097b140318'
ORIGINAL_CPU_RESULT_SHA='47be1fc24a0bc12dfd302c8b1e6048eca0d5740f25b38c69e132246efa8d5dfc'
OLD_NAMES=('features.py','model.py','learning.py','ai_adapter.py','model_qualification.py')
NEW_NAMES=('runtime_v3.py','qualification_v3.py')


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def verify_binding(budget_path,*,phase):
    budget=json.loads(Path(budget_path).read_text(encoding='utf8'));directory=Path(__file__).resolve().parent
    if budget.get('v3_design_sha')!=V3_DESIGN_SHA or budget.get('root_acceptance_sha')!=ROOT_ACCEPTANCE_SHA or budget.get('execute_authorized') is not True:raise ValueError('v3 root admission required')
    source_files={name:file_sha(directory/name) for name in OLD_NAMES+NEW_NAMES}
    source=dict(source_commit=budget['source_commit'],source_files=source_files)
    if budget['source_files']!=source_files or budget['source_manifest_sha']!=record_sha(source):raise ValueError('v3 actual frozen sources changed')
    repo=directory.parents[3]
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=budget['source_commit']:raise ValueError('v3 Git epoch changed')
    original_claim_path=Path(budget['original_claim_path']);original_raw=original_claim_path.read_bytes()
    if hashlib.sha256(original_raw).hexdigest()!=budget['original_claim_sha']:raise ValueError('original ledger bytes changed')
    original=json.loads(original_raw)
    if len(original['updates'])!=2 or any(u['status']!='completed' for u in original['updates']):raise ValueError('exactly two historical completed updates required')
    original_receipt_path=Path(budget['original_receipt_path'])
    if file_sha(original_receipt_path)!=budget['original_receipt_sha']:raise ValueError('original receipt changed')
    original_receipt=json.loads(original_receipt_path.read_text())
    if original_receipt['passed'] is not False or any(source_files[n]!=original_receipt['frozen_source']['source_files'][n] for n in OLD_NAMES):raise ValueError('original failure/core source not preserved')
    for path_key,sha_key in [('fixture_path','fixture_sha'),('initial_path','initial_file_sha')]:
        if file_sha(budget[path_key])!=budget[sha_key]:raise ValueError('original fixture/initial bytes changed')
    if budget['fixture_sha']!=original_receipt['spec_sha'] or budget['initial_file_sha']!=original_receipt['initial_file_sha'] or budget['initial_tensor_sha']!=original_receipt['initial_model_tensor_sha']:raise ValueError('original identity changed')
    if file_sha(budget['original_cpu_result_path'])!=ORIGINAL_CPU_RESULT_SHA:raise ValueError('original diagnostic/order evidence changed')
    cpu_result=json.loads(Path(budget['original_cpu_result_path']).read_text())
    if budget['original_batch_table_indices']!=cpu_result['batch_table_indices']:raise ValueError('original frozen16 order changed')
    if budget.get('remaining_optimizer_updates_max')!=2 or budget.get('historical_completed_optimizer_updates')!=2 or budget.get('global_optimizer_updates_max')!=4:raise ValueError('v3 cumulative budget')
    output=Path(budget['output_path'])
    if not output.is_absolute() or phase not in ('A','B'):raise ValueError('exact output/phase')
    return budget,output,original,original_receipt


def original_batch(budget):
    # No random draw for A: use original pending order saved in CPU diagnostic.
    from .model_qualification import _fixtures
    spec=json.loads(Path(budget['fixture_path']).read_text());cases,targets,table=_fixtures(spec)
    pending=budget['original_batch_table_indices']
    if len(cases)!=4 or len(pending)!=16 or sorted(pending)!=[0]*4+[1]*4+[2]*4+[3]*4:raise ValueError('original four-case batch required')
    return cases,targets,table,[cases[i] for i in pending],[targets[i] for i in pending]


def _analytic_reference(arrays,targets):
    """Independent NumPy float64 formulas; no torch/loss/autograd expectation."""
    import numpy as np
    z=np.asarray(arrays['enable'],np.float64);owner=np.asarray(arrays['owner'],np.float64);pointer=np.asarray(arrays['pointer'],np.float64)
    enable_grad=np.empty_like(z);owner_grad=np.zeros_like(owner);pointer_grad=np.zeros_like(pointer)
    E=O=P=0.;A=sum(len(t.aircraft_ids) for t in targets);G=sum(len(t.anchor_ids) for t in targets)
    for b,target in enumerate(targets):
        n=len(target.aircraft_ids);y=np.zeros(200);y[:2]=1;weight=np.where(y==1,1.,.1)
        sigmoid=np.empty(200);positive=z[b]>=0
        sigmoid[positive]=1/(1+np.exp(-z[b,positive]));exp=np.exp(z[b,~positive]);sigmoid[~positive]=exp/(1+exp)
        enable_grad[b]=weight*(sigmoid-y)/3200;E+=float((weight*(np.logaddexp(0,z[b])-y*z[b])).sum())
        for i,label in enumerate(target.owners):
            row=owner[b,i];maximum=row.max();exps=np.exp(row-maximum);prob=exps/exps.sum();grad=prob.copy();grad[int(label)]-=1
            owner_grad[b,i]=grad/A;O+=float(maximum+np.log(exps.sum())-row[int(label)])
        for q,anchor in ((0,0),(1,1)):
            row=pointer[b,q,:n];maximum=row.max();exps=np.exp(row-maximum);prob=exps/exps.sum();grad=prob.copy();grad[anchor]-=1
            pointer_grad[b,q,:n]=grad/G;P+=float(maximum+np.log(exps.sum())-row[anchor])
    return dict(E=E,O=O,P=P,A=A,G=G,loss=E/3200+O/A+P/G),dict(enable=enable_grad,owner=owner_grad,pointer=pointer_grad)


def _array_comparison(a,b,*,exact=False,allow_matching_infinities=False):
    import numpy as np
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype and exact:return dict(passed=False,reason='shape/dtype',shape_a=list(a.shape),shape_b=list(b.shape))
    if a.shape!=b.shape:return dict(passed=False,reason='shape')
    if exact or a.dtype.kind not in 'fc':return dict(passed=bool(np.array_equal(a,b)),bitwise=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),elements=int(a.size))
    finite=np.isfinite(a)&np.isfinite(b)
    allowed=np.array_equal(a[~finite],b[~finite]) if allow_matching_infinities else not (~finite).any()
    aa=a[finite].astype(np.float64);bb=b[finite].astype(np.float64);difference=np.abs(aa-bb)
    failed=int((difference>ATOL+RTOL*np.abs(bb)).sum())
    reference_norm=float(np.linalg.norm(bb));difference_norm=float(np.linalg.norm(aa-bb))
    return dict(passed=bool(allowed and failed==0),finite=bool(finite.all()),permitted_padding_infinities=bool(allow_matching_infinities and allowed),
        bitwise=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),elements=int(a.size),tol_failed_elements=failed,
        max_abs=float(difference.max()) if difference.size else 0.,relative_L2=difference_norm/reference_norm if reference_norm else None,
        zero_reference_norm=reference_norm==0,absolute_L2=difference_norm)


def run_formula_A(budget_path):
    budget,output,_,_=verify_binding(budget_path,phase='A');phase=output/'formula_A'
    phase.mkdir(parents=True,exist_ok=False)
    admission=dict(phase='A',source_manifest_sha=budget['source_manifest_sha'],fixture_sha=budget['fixture_sha'],
        CPU_leaf_backward_max=5,model_forward_backward=0,CUDA=0,optimizer_updates=0,SciPy_calls=0)
    atomic_json(phase/'admission.json',admission)
    import numpy as np
    import torch
    from .model_qualification import _fixed_matching_logits
    from .learning import compute_loss_sums,global_loss
    _,_,_,batch,targets=original_batch(budget)
    base=_fixed_matching_logits(batch,targets);arrays={k:v.numpy().copy() for k,v in base.items()}
    reference,gradients=_analytic_reference(arrays,targets)
    if (reference['A'],reference['G'])!=(56,32):raise ValueError('original denominators')
    matches=[dict(mapping={0:0,1:1}) for _ in targets];results=[];backwards=0
    np.savez_compressed(phase/'reference_arrays.npz',**arrays,**{'gradient_'+k:v for k,v in gradients.items()})
    for micro in (16,4):
        leaves={k:base[k].detach().clone().requires_grad_(True) for k in ('enable','owner','pointer')}
        sums_total=dict(E=0.,O=0.,P=0.);micro_receipts=[]
        for begin in range(0,16,micro):
            end=begin+micro;out={k:v[begin:end] for k,v in leaves.items()};out['valid']=base['valid'][begin:end]
            sums=compute_loss_sums(out,targets[begin:end],matches[begin:end]);loss=global_loss(sums,56,32)
            loss.backward();backwards+=1
            for key in sums_total:sums_total[key]+=float(sums[key].detach())
            micro_receipts.append(dict(begin=begin,end=end,Edenominator=3200,Odenominator=56,Pdenominator=32))
        observed={name:t.grad.numpy().copy() for name,t in leaves.items()}
        np.savez_compressed(phase/f'micro{micro}_leaf_gradients.npz',**observed)
        comparisons={name:_array_comparison(array,gradients[name]) for name,array in observed.items()}
        zero_pass=True
        for b,target in enumerate(targets):
            n=len(target.aircraft_ids)
            zero_pass=zero_pass and bool((observed['owner'][b,n:]==0).all() and (observed['pointer'][b,2:]==0).all() and (observed['pointer'][b,:,:][:,n:]==0).all())
        contributions=dict(enable=sums_total['E']/3200,owner=sums_total['O']/56,pointer=sums_total['P']/32)
        expected_contributions=dict(enable=reference['E']/3200,owner=reference['O']/56,pointer=reference['P']/32)
        contributions['total']=sum(contributions.values());expected_contributions['total']=reference['loss']
        losses={key:_array_comparison(np.asarray(value),np.asarray(expected_contributions[key])) for key,value in contributions.items()}
        results.append(dict(microbatch=micro,loss_sums=sums_total,loss_comparisons=losses,gradients=comparisons,
            normalized_contributions=contributions,raw_sums_descriptive_no_separate_threshold=True,
            excluded_gradients_exact_zero=zero_pass,micro_receipts=micro_receipts,
            passed=all(v['passed'] for v in [*comparisons.values(),*losses.values()]) and zero_pass))
    scalar_reference=640/3200+32/64+16/32
    if scalar_reference!=1.2:raise AssertionError('static original1.2 definition')
    result=dict(schema='R6FormulaQualificationV3',original_v2=False,passed=all(r['passed'] for r in results),
        schedules=results,reference_losses=reference,CPU_leaf_backwards=backwards,optimizer_updates=0,CUDA=0,
        source_manifest_sha=budget['source_manifest_sha'],fixture_sha=budget['fixture_sha'],artifact_sha256={p.name:file_sha(p) for p in phase.glob('*.npz')})
    atomic_json(phase/'result.json',result)
    return result


class ContinuationLedger:
    def __init__(self,budget,original):
        path=Path(budget['continuation_claim_path'])
        if not path.is_absolute():raise ValueError('absolute global continuation ledger required')
        path.parent.mkdir(parents=True,exist_ok=True)
        self.path=path;self.value=dict(schema='R6V3ContinuationLedger',source_commit=budget['source_commit'],
            source_manifest_sha=budget['source_manifest_sha'],original_claim_sha=budget['original_claim_sha'],
            historical_updates=copy.deepcopy(original['updates']),initial_file_sha=budget['initial_file_sha'],
            fixture_sha=budget['fixture_sha'],runtime_profile_sha=budget['runtime_profile_sha'],output_path=budget['output_path'],
            reservation_id=budget['reservation_id'],new_updates=[],historical_spend=2,global_max=4,status='consumed_once')
        with path.open('x',encoding='utf8') as stream:json.dump(self.value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())

    def reserve(self,branch):
        if len(self.value['new_updates'])>=2:raise RuntimeError('remaining two updates exhausted')
        event=dict(global_ordinal=3+len(self.value['new_updates']),branch=branch,status='started_unknown_consumed')
        self.value['new_updates'].append(event);atomic_json(self.path,self.value);return event

    def finish(self,event,*,error=None):
        event['status']='unknown_consumed' if error is not None else 'completed'
        if error is not None:event['error']=repr(error)
        atomic_json(self.path,self.value)


def _cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().contiguous().clone()
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _rng_state():
    import torch
    return dict(cpu=torch.get_rng_state().clone(),cuda=[t.clone() for t in torch.cuda.get_rng_state_all()])


def _named_optimizer_state(session):
    by_parameter={p:name for name,p in session.model.named_parameters()}
    result={by_parameter[p]:_cpu_tree(state) for p,state in session.optimizer.state.items()}
    for name,state in result.items():
        if set(state)!=set(('step','exp_avg','exp_avg_sq')) or float(state['step'])!=1:raise ValueError('nonempty first-step Adam state incomplete '+name)
    if set(result)!=set(dict(session.model.named_parameters())):raise ValueError('optimizer parameter state missing')
    groups=[]
    for group in session.optimizer.param_groups:
        groups.append({key:([by_parameter[p] for p in value] if key=='params' else copy.deepcopy(value)) for key,value in group.items()})
    return dict(state=result,param_groups=groups)


def snapshot_session(session):
    return dict(model=_cpu_tree(session.model.state_dict()),optimizer=_named_optimizer_state(session),
        sampler=session.sampler.state_dict(),rng=_cpu_tree(_rng_state()),provenance=copy.deepcopy(session.provenance),
        updates=session.updates,active_seconds=session.elapsed())


def tensor_tree_manifest(value,path='root'):
    import torch
    result={}
    if isinstance(value,torch.Tensor):
        tensor=value.detach().cpu().contiguous();array=tensor.numpy()
        result[path]=dict(dtype=str(tensor.dtype),shape=list(tensor.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())
    elif isinstance(value,dict):
        for key,item in value.items():result.update(tensor_tree_manifest(item,path+'.'+str(key)))
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):result.update(tensor_tree_manifest(item,path+'.'+str(i)))
    return result


def compare_tree(first,second,*,exact=False,path='root'):
    """All leaves are inspected; no first-tensor early abort."""
    import numpy as np
    import torch
    results={}
    if isinstance(first,dict) and isinstance(second,dict):
        if set(first)!=set(second):results[path]=dict(passed=False,reason='missing keys');return results
        for key in first:results.update(compare_tree(first[key],second[key],exact=exact,path=path+'.'+str(key)))
    elif isinstance(first,(list,tuple)) and isinstance(second,(list,tuple)):
        if len(first)!=len(second):results[path]=dict(passed=False,reason='missing items');return results
        for i,(a,b) in enumerate(zip(first,second)):results.update(compare_tree(a,b,exact=exact,path=path+'.'+str(i)))
    elif isinstance(first,torch.Tensor) and isinstance(second,torch.Tensor):
        if first.dtype!=second.dtype:results[path]=dict(passed=False,reason='dtype');return results
        force_exact=exact or path.endswith('.step') or '.rng.' in path
        result=_array_comparison(first.numpy(),second.numpy(),exact=force_exact,allow_matching_infinities='.pointer' in path)
        if path.endswith('.step') and float(first)!=1:result['passed']=False
        results[path]=result
    elif isinstance(first,(float,np.floating)) and isinstance(second,(float,np.floating)) and not exact:
        results[path]=_array_comparison(np.asarray(first),np.asarray(second))
    else:results[path]=dict(passed=first==second,exact=True)
    return results


def _recovery_evidence(session,checkpoint,checkpoint_receipt,directory):
    path=Path(directory)/'recovery_clock_evidence.json'
    evidence=dict(verified=True,checkpoint_file_sha=file_sha(checkpoint),charged_active_seconds=checkpoint_receipt['charged_active_seconds'],
        last_completed_checkpoint_IO_active_seconds=checkpoint_receipt['charged_active_seconds'],unknown_tail_accounted=True,
        interval_evidence='controlled qualification checkpoint branch; no unknown process-crash interval; all checkpoint IO included',
        prior_clock_events=copy.deepcopy(session.clock_events))
    atomic_json(path,evidence)
    return dict(evidence_path=str(path),evidence_sha=file_sha(path),clock_path=str(Path(directory)/'activity_clock.json'))


class StepTrace:
    """Observe actual a36 step; wrappers call originals exactly once unchanged."""
    def __init__(self,session,ledger,branch,directory):
        self.session=session;self.ledger=ledger;self.branch=branch;self.directory=Path(directory)
        self.trace=dict(branch=branch,micros=[],parameters_before=_cpu_tree(session.model.state_dict()),rng_before=_cpu_tree(_rng_state()))

    def __enter__(self):
        import torch
        from . import learning
        self.original_loss=learning.compute_loss_sums;self.original_clip=torch.nn.utils.clip_grad_norm_;self.original_step=torch.optim.AdamW.step
        def forward_hook(module,args,kwargs,output):
            self.trace['micros'].append(dict(inputs=_cpu_tree(kwargs),outputs=_cpu_tree(output)))
        self.hook=self.session.model.register_forward_hook(forward_hook,with_kwargs=True)
        def loss_wrapper(output,targets,matches=None):
            sums=self.original_loss(output,targets,matches)
            self.trace['micros'][-1]['loss_sums']={name:float(sums[name].detach()) for name in ('E','O','P')}
            self.trace['micros'][-1]['matching']=copy.deepcopy(sums['matches'])
            self.trace['micros'][-1]['global_denominators']=dict(E=3200,O=56,P=32)
            return sums
        def clip_wrapper(parameters,*args,**kwargs):
            self.trace['preclip_gradients']={name:_cpu_tree(p.grad) for name,p in self.session.model.named_parameters()}
            norm=self.original_clip(parameters,*args,**kwargs)
            self.trace['clip_norm']=float(norm);self.trace['clip_coefficient']=min(1.,1/(float(norm)+1e-6))
            self.trace['postclip_gradients']={name:_cpu_tree(p.grad) for name,p in self.session.model.named_parameters()}
            self.persist();return norm
        def step_wrapper(optimizer,*args,**kwargs):
            if optimizer is not self.session.optimizer:raise RuntimeError('unowned optimizer')
            event=self.ledger.reserve(self.branch)
            try:result=self.original_step(optimizer,*args,**kwargs)
            except BaseException as exc:self.ledger.finish(event,error=exc);raise
            self.ledger.finish(event);return result
        learning.compute_loss_sums=loss_wrapper;torch.nn.utils.clip_grad_norm_=clip_wrapper;torch.optim.AdamW.step=step_wrapper
        return self

    def persist(self):
        import torch
        self.directory.mkdir(parents=True,exist_ok=True)
        tree=_cpu_tree(self.trace);path=self.directory/'step_trace.pt';torch.save(tree,path)
        atomic_json(self.directory/'step_trace_manifest.json',dict(file_sha=file_sha(path),tensors=tensor_tree_manifest(tree)))

    def __exit__(self,kind,value,tb):
        import torch
        from . import learning
        learning.compute_loss_sums=self.original_loss;torch.nn.utils.clip_grad_norm_=self.original_clip;torch.optim.AdamW.step=self.original_step;self.hook.remove()
        self.trace['parameters_after']=_cpu_tree(self.session.model.state_dict());self.trace['rng_after']=_cpu_tree(_rng_state())
        if kind is None:self.trace['session_after']=snapshot_session(self.session)
        else:self.trace['error']=repr(value)
        self.persist()


def run_resume_B(budget_path):
    budget,output,original,original_receipt=verify_binding(budget_path,phase='B')
    a_path=output/'formula_A/result.json'
    if file_sha(a_path)!=budget['A_result_sha'] or json.loads(a_path.read_text())['passed'] is not True:raise ValueError('formula A not admitted')
    from .runtime_v3 import configure_v3_runtime,assert_v3_runtime,PROFILE
    if record_sha(PROFILE)!=budget['runtime_profile_sha']:raise ValueError('runtime profile receipt')
    # Exclusive continuation is claimed before any CUDA initialization.
    ledger=ContinuationLedger(budget,original);phase=output/'resume_B';phase.mkdir(parents=True,exist_ok=False)
    result=dict(schema='R6ResumeQualificationV3',original_v2=False,formula_A=True,fixed_micro4_resume_B=False,
        nonempty_checkpoint_readback=False,v3_runtime_admission=False,historical_updates=2,new_updates=0)
    atomic_json(phase/'admission.json',dict(source_manifest_sha=budget['source_manifest_sha'],runtime_profile_sha=budget['runtime_profile_sha'],remaining_updates=2,historical_updates=2))
    try:
        runtime=configure_v3_runtime(expected_identity=budget['expected_runtime_identity']);atomic_json(phase/'runtime_identity.json',runtime)
        import torch
        from .model import initialize_model,model_tensor_sha
        from .features import FeatureStats
        from .learning import TrainingSession,CaseSampler
        cases,targets,table,batch,labels=original_batch(budget)
        initial=initialize_model();initial.load_state_dict(torch.load(budget['initial_path'],map_location='cpu',weights_only=True),strict=True)
        if model_tensor_sha(initial)!=budget['initial_tensor_sha']:raise ValueError('pinned initialized tensors changed')
        stats=FeatureStats.fit(cases,role='train',qualified_table_sha=CaseSampler(table).table_sha)
        provenance=dict(synthetic=True,source_commit=budget['source_commit'],source_manifest_sha=budget['source_manifest_sha'],
            initial_file_sha=budget['initial_file_sha'],fixture_sha=budget['fixture_sha'],runtime=runtime,v3_design_sha=V3_DESIGN_SHA)
        def session():return TrainingSession(copy.deepcopy(initial),stats,CaseSampler(table),provenance,'cuda')
        baseline=session();baseline.start_activity(clock_path=phase/'baseline/activity_clock.json')
        pending=baseline.sampler.assemble()
        if pending!=budget['original_batch_table_indices']:raise ValueError('original sampler order changed')
        before=phase/'before_step0.pt';before_receipt=baseline.checkpoint(before)
        baseline_recovery=_recovery_evidence(baseline,before,before_receipt,phase/'resumed')
        with StepTrace(baseline,ledger,'baseline',phase/'baseline') as baseline_trace:
            baseline_trace.trace['step_result']=baseline.step([cases[i] for i in pending],[targets[i] for i in pending],microbatch=4)
        baseline_after=snapshot_session(baseline)
        after=phase/'after_step1.pt';after_receipt=baseline.checkpoint(after)
        readback_recovery=_recovery_evidence(baseline,after,after_receipt,phase/'readback')
        resumed=session();resumed.restore(before,recovery_clock=baseline_recovery);resumed.start_activity()
        pending=resumed.sampler.assemble();assert_v3_runtime(runtime)
        with StepTrace(resumed,ledger,'resumed',phase/'resumed') as resumed_trace:
            resumed_trace.trace['step_result']=resumed.step([cases[i] for i in pending],[targets[i] for i in pending],microbatch=4)
        resumed_after=snapshot_session(resumed)
        # Clock costs differ between branches and are retained separately; all
        # algorithm data, discrete state and RNG are compared completely.
        first=copy.deepcopy(baseline_trace.trace);second=copy.deepcopy(resumed_trace.trace)
        for trace in (first,second):
            trace.pop('branch');trace['step_result'].pop('active_seconds');trace['session_after'].pop('active_seconds')
            for record in trace['step_result']['matching']:
                if record.get('solver')!=baseline.solver:raise ValueError('matching solver identity changed')
        comparisons=compare_tree(first,second)
        discrete=compare_tree(dict(sampler=baseline_after['sampler'],rng=baseline_after['rng'],provenance=baseline_after['provenance']),
            dict(sampler=resumed_after['sampler'],rng=resumed_after['rng'],provenance=resumed_after['provenance']),exact=True)
        result['resume_comparisons']=comparisons;result['discrete_comparisons']=discrete
        result['fixed_micro4_resume_B']=all(v['passed'] for v in [*comparisons.values(),*discrete.values()])
        # Nonempty step1 readback adds NO forward/backward/optimizer operation.
        saved=torch.load(after,map_location='cpu',weights_only=False)
        readback=session();readback.restore(after,recovery_clock=readback_recovery)
        readback_after=snapshot_session(readback)
        saved_optimizer=copy.deepcopy(baseline_after['optimizer'])
        readback_expected=dict(model=saved['model'],optimizer=saved_optimizer,sampler=saved['sampler'],rng=dict(cpu=saved['cpu_rng'],cuda=saved['cuda_rng']),
            provenance=saved['provenance'],updates=saved['updates'])
        readback_actual={key:readback_after[key] for key in readback_expected}
        result['readback_comparisons']=compare_tree(readback_expected,readback_actual,exact=True)
        result['nonempty_checkpoint_readback']=all(v['passed'] for v in result['readback_comparisons'].values()) and readback_after['active_seconds']>=saved['active_seconds']
        torch.save(_cpu_tree(dict(expected=readback_expected,actual=readback_actual,activity_saved=saved['active_seconds'],activity_readback=readback_after['active_seconds'])),phase/'readback/evidence.pt')
        atomic_json(phase/'readback/evidence_manifest.json',dict(file_sha=file_sha(phase/'readback/evidence.pt'),
            expected_tensors=tensor_tree_manifest(readback_expected),actual_tensors=tensor_tree_manifest(readback_actual)))
        result['v3_runtime_admission']=result['fixed_micro4_resume_B'] and result['nonempty_checkpoint_readback']
        result['passed']=result['v3_runtime_admission'];result['new_updates']=len(ledger.value['new_updates'])
        result['bitwise_tensor_comparisons']=sum(v.get('bitwise',False) for v in comparisons.values())
        if not result['passed']:raise AssertionError('v3 named comparisons/readback failed; no replay')
    except BaseException as exc:
        result['passed']=False;result['error']=repr(exc);result['new_updates']=len(ledger.value['new_updates']);raise
    finally:
        result['ledger_sha']=file_sha(ledger.path);result['artifacts_sha256']={str(p.relative_to(phase)):file_sha(p) for p in phase.rglob('*') if p.is_file()}
        atomic_json(phase/'result.json',result)
    return result


def describe_existing_CPU(path,output):
    """Read existing arrays only; no new model traversal or updates."""
    import numpy as np
    root=Path(path);result={}
    for stage in ('preclip','postclip'):
        with np.load(root/f'full16_gradients_{stage}.npz',allow_pickle=False) as aa,np.load(root/f'micro4_gradients_{stage}.npz',allow_pickle=False) as bb:
            for name in aa.files:
                a=aa[name];b=bb[name];record=_array_comparison(a,b)
                record.update(stage=stage,zero_a=int((a==0).sum()),zero_b=int((b==0).sum()),nearzero_a=int((np.abs(a)<=1e-8).sum()),
                    nearzero_b=int((np.abs(b)<=1e-8).sum()),sign_disagreements=int((np.signbit(a)!=np.signbit(b)).sum()),
                    tensor_a_bytes_sha=hashlib.sha256(a.tobytes()).hexdigest(),tensor_b_bytes_sha=hashlib.sha256(b.tobytes()).hexdigest())
                result[stage+'.'+name]=record
    atomic_json(output,dict(schema='R6ExistingCPUDescriptiveV3',new_model_traversals=0,not_a_gate=True,parameters=result));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('A','B','describe'));parser.add_argument('--budget');parser.add_argument('--existing-cpu');parser.add_argument('--output');args=parser.parse_args()
    if args.phase=='describe':result=describe_existing_CPU(args.existing_cpu,args.output)
    elif not args.budget:parser.error('--budget required')
    elif args.phase=='A':result=run_formula_A(args.budget)
    else:result=run_resume_B(args.budget)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
