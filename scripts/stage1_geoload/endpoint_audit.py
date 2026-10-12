"""One-shot, terminal-only G16 checkpoint/CUDA acceptance and small final export.

Explicit execution only. No training, resume, policy evaluation, optimizer update
or waiting loop. A full checkpoint remains in the original verified task ZIP.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import zipfile

SCIENCE='495bff976c59c40ddb33de6b5b37b20a894a1347'
NODES=[2500,5000,7500,10000]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_readiness(plan,state):
    if plan.get('group')!='S1-G16' or state.get('group')!='S1-G16' or plan.get('scientific_commit')!=SCIENCE or state.get('all_computation_completed') is not True:
        raise ValueError('all scientific computation must finish before endpoint audit')
    if 'train_mlm' not in state.get('tasks',{}):raise ValueError('no trained MLM branch')
    if any(t.get('stage')!='completed' for t in state['tasks'].values()):raise ValueError('scientific tasks incomplete')
    train=state['tasks']['train_mlm']
    if train['port']!=20473:raise ValueError('only73 training owner allowed')
    if 'train_mlm' not in state.get('archives',{}):raise ValueError('original full checkpoint task ZIP required')
    return train


def validate_states(cp,final,protocol,teacher_sha,validation_sha,loss_rows,attempts,wal,
                    *,sample_count,torch_module=None):
    """Independent invariants. Only local RNG objects are constructed/restored."""
    import numpy as np
    torch=torch_module
    if torch is None:import torch
    expected_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if protocol.get('scientific_commit')!=SCIENCE or protocol.get('seed')!=51001:
        raise ValueError('scientific source/seed differs')
    if protocol.get('updates')!=10000 or protocol.get('batch')!=32 or protocol.get('microbatch')!=4:
        raise ValueError('training budget/batch protocol differs')
    if protocol.get('teacher_source_sha256')!=teacher_sha or protocol.get('validation_sha256')!=validation_sha:
        raise ValueError('teacher/validation source differs')
    for obj in (cp,final):
        if obj.get('format')!='s1_g16_geometry_mlm_v1' or obj.get('seed')!=51001 or obj.get('from_scratch') is not True or obj.get('update')!=10000:
            raise ValueError('actual endpoint format/seed/from-scratch/update differs')
        if obj.get('protocol_sha256')!=expected_sha or obj.get('dataset_sha256')!=teacher_sha:
            raise ValueError('endpoint protocol/teacher hash differs')
    if cp.get('protocol')!=protocol or final.get('network')!=protocol.get('network'):
        raise ValueError('saved protocol/network differs')
    if cp.get('nodes')!=NODES:raise ValueError('four validation nodes incomplete')
    if set(cp['model'])!=set(final['model']) or not cp['model']:
        raise ValueError('model parameter keys differ or empty')
    for key,value in cp['model'].items():
        saved=final['model'][key]
        if not isinstance(value,torch.Tensor) or not isinstance(saved,torch.Tensor) or value.dtype!=saved.dtype or value.shape!=saved.shape or not torch.isfinite(value).all() or not torch.equal(value.cpu(),saved.cpu()):
            raise ValueError('model tensor nonfinite/different: '+key)
    if len(loss_rows)!=10000 or [r.get('update') for r in loss_rows]!=list(range(1,10001)):
        raise ValueError('loss updates must be exactly 1..10000')
    for row in loss_rows:
        ids=row.get('sample_ids',[])
        if not math.isfinite(row['loss']) or len(ids)!=32 or any(type(i)is not int or i<0 or i>=sample_count for i in ids):
            raise ValueError('loss/sample sampler record invalid')
    committed=[r for r in attempts if r.get('status')=='committed']
    if len(committed)!=10000 or [r.get('update') for r in committed]!=list(range(1,10001)):
        raise ValueError('committed attempt WAL history incomplete/duplicated')
    if any(a.get('attempt_id')!=b.get('attempt_id') for a,b in zip(loss_rows,committed)):
        raise ValueError('loss/committed attempt identities differ')
    if wal.get('status')!='committed' or wal.get('update')!=10000 or wal.get('attempt_id')!=loss_rows[-1].get('attempt_id'):
        raise ValueError('final WAL is not last retained committed update')
    ledger=cp.get('ledger',{})
    if ledger!=final.get('ledger') or ledger.get('retained_updates')!=10000:
        raise ValueError('endpoint ledgers differ/incomplete')
    keys=('unknown_attempts','known_failed_attempts','replayed_attempts')
    if any(type(ledger.get(k))is not int or ledger[k]<0 for k in keys) or ledger['unknown_attempts']>3:
        raise ValueError('recovery ledger invalid')
    unknown=sum(r.get('recovery')=='unknown_attempt_may_have_executed' for r in attempts)
    failed=sum(r.get('status')=='failed' for r in attempts)
    if ledger['unknown_attempts']!=unknown or ledger['known_failed_attempts']!=failed or ledger['replayed_attempts']!=unknown+failed:
        raise ValueError('recovery attempts/ledger differ')
    optimizer=cp.get('optimizer',{});groups=optimizer.get('param_groups',[]);states=optimizer.get('state',{})
    ids=[i for g in groups for i in g.get('params',[])]
    if not groups or len(ids)!=len(set(ids)) or set(ids)!=set(states):raise ValueError('Adam state coverage incomplete')
    for group in groups:
        if group.get('lr')!=protocol.get('learning_rate',3e-4):raise ValueError('Adam learning rate differs')
    for state in states.values():
        if not {'step','exp_avg','exp_avg_sq'}<=set(state):raise ValueError('Adam moments incomplete')
        if float(state['step'])!=10000:raise ValueError('Adam retained step differs')
        for value in state.values():
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():raise ValueError('Adam nonfinite')
    random.Random().setstate(cp['python_rng'])
    np.random.RandomState().set_state(cp['numpy_rng'])
    local_sampler=np.random.default_rng();local_sampler.bit_generator.state=cp['sampler_rng']
    for value in [cp['torch_rng'],*cp['cuda_rng']]:
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.ndim!=1 or not value.numel():
            raise ValueError('Torch/CUDA RNG state malformed')
    if not cp['cuda_rng']:raise ValueError('CUDA RNG state absent')
    torch.Generator().set_state(cp['torch_rng'].cpu())
    return {'protocol_sha256':expected_sha,'updates':10000,'seed':51001,'from_scratch':True,
            'validation_nodes':NODES,'loss_records':10000,'committed_attempts':10000,'ledger':ledger,
            'model_tensor_count':len(cp['model']),'adam_state_count':len(states),'rng_sampler_verified':True}


def remote_audit(jobroot,manifest,expected_run_id):
    """Called only in remote73 DQN with frozen scientific src on PYTHONPATH."""
    import os,fcntl,subprocess
    import numpy as np
    import torch
    from satellite_rl.coverage.geoload.models import make_model
    root=Path(jobroot);job=read(root/'job.json');status=read(root/'status.json');config=read(root/'config.json')
    if job.get('run_id')!=expected_run_id or job.get('scientific_commit')!=SCIENCE or job.get('ssh_port') not in ('20473',20473):
        raise ValueError('actual training owner/source differs')
    if status.get('stage')!='completed' or status.get('update_step')!=10000:raise ValueError('actual train task nonterminal/incomplete')
    for proc in Path('/proc').glob('[0-9]*'):
        try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if '--job-dir' in cmd and str(root) in cmd:raise ValueError('training worker still alive')
    models=Path(job['model_dir']);lock=Path(__file__).parent/'endpoint_gpu_audit.lock'
    with lock.open('a+b') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
        if active:raise ValueError('GPU has active compute; engineering audit may contend')
        if not torch.cuda.is_available() or not torch.version.cuda:raise ValueError('DQN CUDA unavailable')
        package=Path(sys.modules['satellite_rl'].__file__).resolve()
        if package!=Path(job['source_root'])/'src/satellite_rl/__init__.py':raise ValueError('actual package not frozen science')
        source_manifest=read(Path(job['source_root'])/'source_manifest.json')
        if source_manifest['git_commit']!=SCIENCE:raise ValueError('actual science source manifest differs')
        for name,sha in source_manifest['files'].items():
            if digest(Path(job['source_root'])/name)!=sha:raise ValueError('actual frozen source file differs')
        members={}
        for name,pin in manifest['members'].items():
            path=root/name[4:] if name.startswith('job/') else models/name[6:] if name.startswith('model/') else None
            if path is None or not path.is_file() or digest(path)!=pin['sha256'] or path.stat().st_size!=pin['bytes']:
                raise ValueError('archived/actual endpoint member differs: '+name)
            members[name]=pin
        cp=torch.load(models/'checkpoint.pt',map_location='cpu',weights_only=False)
        final=torch.load(models/'final.pt',map_location='cpu',weights_only=False)
        loss_path=models/'loss.jsonl'
        if cp['log_bytes']!=loss_path.stat().st_size or cp['log_sha256']!=digest(loss_path):raise ValueError('checkpoint loss byte/hash boundary differs')
        loss_rows=[json.loads(row) for row in loss_path.read_text().splitlines()]
        attempts=[json.loads(row) for row in (models/'update_attempts.jsonl').read_text().splitlines()]
        teacher=job['references']['teacher'];validation=config['datasets']['validation']
        if digest(teacher['path'])!=teacher['sha256'] or digest(validation['path'])!=validation['sha256']:raise ValueError('actual teacher/validation pin differs')
        teacher_manifest=read(teacher['path'])
        if teacher_manifest.get('group')!='S1-G16' or teacher_manifest.get('status')!='completed' or teacher_manifest.get('scientific_commit')!=SCIENCE or teacher_manifest.get('dataset_sha256')!=config['datasets']['train']['sha256'] or teacher_manifest.get('count')!=2304:
            raise ValueError('teacher manifest scientific/train input provenance differs')
        teacher_root=Path(teacher['path']).parent.resolve()
        for sample in teacher_manifest['samples']:
            p=(teacher_root/sample['path']).resolve()
            if not p.is_relative_to(teacher_root) or not p.is_relative_to(root.resolve()):raise ValueError('teacher sample escapes retained source')
            name='job/'+p.relative_to(root.resolve()).as_posix()
            if members.get(name,{}).get('sha256')!=sample['sha256']:raise ValueError('teacher sample/archive member SHA differs')
        protocol={**config['training'],'scientific_commit':SCIENCE,'teacher_source_sha256':teacher['sha256'],'validation_sha256':validation['sha256']}
        result=validate_states(cp,final,protocol,teacher['sha256'],validation['sha256'],loss_rows,attempts,read(models/'update_wal.json'),sample_count=len(teacher_manifest['samples']))
        with np.load(validation['path'],allow_pickle=False) as values:case_ids=set(map(str,values['case_ids']))
        if len(case_ids)!=80:raise ValueError('validation must contain80 unique cases')
        node_receipts=[]
        for node in NODES:
            directory=models/('validation_'+str(node).zfill(5));done=read(directory/'node_done.json');index=done.get('index',[])
            if done.get('update')!=node or done.get('count')!=80 or done.get('validation_sha256')!=validation['sha256'] or len(index)!=80 or {i['case_id'] for i in index}!=case_ids:
                raise ValueError('validation node_done incomplete/source differs')
            wanted={hashlib.sha256((i['case_id']+'\0MLM').encode()).hexdigest()+'.json' for i in index}
            if {p.name for p in directory.glob('*.json')}!=wanted|{'node_done.json'}:raise ValueError('validation case member set differs')
            for item in index:
                p=directory/(hashlib.sha256((item['case_id']+'\0MLM').encode()).hexdigest()+'.json')
                if digest(p)!=item['sha256'] or read(p).get('case_id')!=item['case_id']:raise ValueError('validation case SHA/ID differs')
            node_receipts.append({'update':node,'count':80,'node_done_sha256':digest(directory/'node_done.json'),'validation_input_sha256':validation['sha256']})
        # Strict state loading and restoring Adam into an ephemeral object never
        # call optimizer.step/backward or mutate the retained scientific state.
        model=make_model(protocol.get('network')).to('cuda');model.load_state_dict(cp['model'],strict=True)
        if any(value.dtype!=model.state_dict()[key].dtype or value.shape!=model.state_dict()[key].shape for key,value in cp['model'].items()):
            raise ValueError('saved tensor dtype/shape differs from frozen architecture')
        for value in cp['cuda_rng']:
            local_rng=torch.Generator(device='cuda');local_rng.set_state(value.cpu())
        optimizer=torch.optim.Adam(model.parameters(),lr=protocol.get('learning_rate',3e-4));optimizer.load_state_dict(cp['optimizer'])
        if len(optimizer.state)!=len(list(model.parameters())):raise ValueError('Adam state does not cover frozen architecture')
        for param,state in optimizer.state.items():
            if state['exp_avg'].shape!=param.shape or state['exp_avg_sq'].shape!=param.shape:raise ValueError('Adam moment tensor shape differs')
        model.eval()
        batch={'aircraft':torch.zeros(1,2,8,device='cuda'),'candidates':torch.zeros(1,2,8,device='cuda'),
            'aircraft_mask':torch.ones(1,2,dtype=torch.bool,device='cuda'),'candidate_mask':torch.ones(1,2,dtype=torch.bool,device='cuda'),
            'visible':torch.full((1,2),-1,dtype=torch.long,device='cuda'),'legal':torch.ones(1,2,2,dtype=torch.bool,device='cuda')}
        with torch.inference_mode():left=model(batch)['logits'].cpu()
        model.load_state_dict(final['model'],strict=True)
        with torch.inference_mode():right=model(batch)['logits'].cpu()
        if not torch.isfinite(right).all() or not torch.equal(left,right):raise ValueError('engineering forward nonfinite/different')
        torch.cuda.synchronize()
        result.update(group='S1-G16',run_id=job['run_id'],scientific_commit=SCIENCE,verified=True,status='passed',
            gpu_available=True,cuda_available=True,strict=True,strict_load=True,strict_model_load=True,
            final_sha256=digest(models/'final.pt'),checkpoint_sha256=digest(models/'checkpoint.pt'),
            model_sha256=digest(models/'final.pt'),teacher_source_sha256=teacher['sha256'],validation_input_sha256=validation['sha256'],
            member_hashes_verified=True,members=members,validation_node_receipts=node_receipts,
            worker_exited=True,torch_version=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
            interpreter=str(Path(sys.executable).resolve()),actual_package=str(package),engineering_forward_only=True,
            parameter_updates_performed=0,new_evaluations_performed=0,endpoint_audit_module_sha256=digest(__file__))
        return result


def execute(plan_path,state_path,base):
    from . import ops
    from .strict_audit import JobZip
    from satellite_rl.workbench.geoload_runner import exclusive_lock
    base=Path(base);plan=read(plan_path);state=read(state_path)
    train=check_readiness(plan,state)
    receipts=base/'preflight';receipts.mkdir(exist_ok=True)
    with exclusive_lock(receipts/'endpoint_audit.lock'):
        for task in state['tasks'].values():
            actual=ops.api(task['port'],'/api/runs/'+task['run_id'])
            if actual['status'].get('stage')!='completed' or actual['status'].get('process_alive') is not False:
                raise ValueError('real API task nonterminal/worker alive')
        existing_audit=receipts/'final_learning_audit.json';existing_export=receipts/'model_export_receipt.json'
        if existing_audit.exists() or existing_export.exists():
            if not existing_audit.exists() or not existing_export.exists():raise ValueError('incomplete prior receipts require explicit recovery')
            saved=read(existing_audit);export=read(existing_export)
            if saved.get('verified') is not True or saved.get('run_id')!=train['run_id'] or saved.get('train_zip_sha256')!=state['archives']['train_mlm']['sha256'] or saved.get('endpoint_audit_module_sha256')!=digest(__file__) or digest(export['model_path'])!=export['model_sha256'] or digest(export['metadata_path'])!=export['metadata_sha256']:
                raise ValueError('prior receipts/source/final differ; never silently overwrite')
            return {'audit':str(existing_audit),'model_export':str(existing_export),'status':'passed','safe_reuse':True}
        archive=JobZip(state['archives']['train_mlm'],SCIENCE)
        try:
            manifest=archive.manifest;job=archive.job
            module=Path(__file__).resolve();target=ops.BASE+'/source_closure/postops/endpoint_audit_'+digest(module)+'.py'
            receipt=ops.upload(module,target,20473)
            if receipt['remote']!=digest(module):raise ValueError('audit module transport SHA differs')
            jobroot=job['archive_root']+'/outputs/workbench/jobs/'+job['run_id']
            # Launch only DQN; immutable science package is first on sys.path.
            code='''import importlib.util,json,sys
sys.path.insert(0,%r)
spec=importlib.util.spec_from_file_location('g16_endpoint_postops',%r);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
print(json.dumps(m.remote_audit(%r,%r,%r)))
'''%(job['source_root']+'/src',target,jobroot,manifest,job['run_id'])
            actual=ops.remote_python(code,20473,timeout=300)
            if actual['exit_code']:raise RuntimeError(actual['stderr'][-2000:])
            audit=json.loads(actual['stdout'].splitlines()[-1]);audit.update(train_zip_sha256=state['archives']['train_mlm']['sha256'],train_zip_path=state['archives']['train_mlm']['local_path'])
            destination=Path('D:/paper/models/runs/first_stage_geometry_load_20261012')/job['run_id']/'final';destination.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(destination).free<8*(1<<30)+archive.manifest['members']['model/final.pt']['bytes']:
                raise RuntimeError('final export lacks8GiB reserve')
            target_final=destination/'final.pt';expected=audit['final_sha256']
            if target_final.exists() and digest(target_final)!=expected:raise ValueError('existing final export differs; never overwrite')
            if not target_final.exists():
                # Read final directly from already verified immutable ZIP. An
                # interrupted partial is rewritten; no incomplete file is reused.
                temp=target_final.with_suffix('.part')
                with archive.zip.open('model/final.pt') as source,temp.open('wb') as out:
                    shutil.copyfileobj(source,out,1048576)
                if digest(temp)!=expected:raise ValueError('exported final SHA differs')
                temp.replace(target_final)
            metadata={'group':'S1-G16','run_id':job['run_id'],'format':'s1_g16_geometry_mlm_v1','file':'final.pt','sha256':expected,
                'scientific_commit':SCIENCE,'seed':51001,'update':10000,'from_scratch':True,'protocol_sha256':audit['protocol_sha256'],
                'teacher_source_sha256':audit['teacher_source_sha256'],'validation_input_sha256':audit['validation_input_sha256'],
                'generated_export_metadata':True,'native_model_json_existed':False,'checkpoint_source':'original_train_task_ZIP',
                'checkpoint_sha256':audit['checkpoint_sha256']}
            metadata_path=destination/'model.json';metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
            export={'group':'S1-G16','run_id':job['run_id'],'verified':True,'status':'passed','scientific_commit':SCIENCE,
                'model_path':str(target_final),'model_sha256':expected,'metadata_path':str(metadata_path),'metadata_sha256':digest(metadata_path),
                'checkpoint_exported':False,'checkpoint_zip_path':state['archives']['train_mlm']['local_path'],'checkpoint_member':'model/checkpoint.pt',
                'checkpoint_sha256':audit['checkpoint_sha256'],'train_zip_sha256':state['archives']['train_mlm']['sha256']}
            (receipts/'final_learning_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
            (receipts/'model_export_receipt.json').write_text(json.dumps(export,indent=2)+'\n',encoding='utf-8')
            return {'audit':str(receipts/'final_learning_audit.json'),'model_export':str(receipts/'model_export_receipt.json'),'status':'passed'}
        finally:archive.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--plan',required=True);parser.add_argument('--state',required=True);parser.add_argument('--base',required=True);args=parser.parse_args()
    print(json.dumps(execute(args.plan,args.state,args.base)))


if __name__=='__main__':main()
