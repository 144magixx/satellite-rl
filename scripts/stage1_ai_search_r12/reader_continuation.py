"""One reader-only attempt; immutable original numerical modules are reused.

No science service, producer, solver, engineering or retry entry exists here.
All actual qualification and seed IO run inside the sole17-second attempt.
"""
import time
ENTRY_MONOTONIC=time.monotonic()
ENTRY_UTC=time.time()
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ACCEPTANCE='4238e241eebe06af8c6518ad2b203a62baf06ae9acac2e9c8306a176f68bb0f7'
OLD='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20473/round12/v2_01/resume_20261015_0858'
NEW=OLD+'/reader_continuation_01'
CAP=134217728
TEMP=67108864
RESERVE=1048576

def pinned(ref):
    p=Path(ref['path']);info=p.lstat()
    if not stat.S_ISREG(info.st_mode) or type(ref.get('bytes')) is not int or not 0<=ref['bytes']<=8388608:raise ValueError('bounded regular exact input')
    with p.open('rb') as f:raw=f.read(ref['bytes']+1)
    if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('exact file changed')
    return raw

def load(ref):return json.loads(pinned(ref))

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def same(ref):return (ref['sha256'],ref['bytes'])

def cached_prefix_step(key,available,missing):
    if key in available and missing:raise ValueError('cached audits must be an original fixed-order prefix')
    return missing or key not in available

def qualify(plan,filemap,stop):
    """Metadata/cache binding gate; never computes a ray, pair, metric or gate."""
    from scripts.stage1_ai_search_r12.worker import load_case
    from scripts.stage1_ai_search_r12.admission import INDICES
    expected={row['remote_path']:row for row in filemap}
    if len(expected)!=len(filemap) or len(expected)!=188:raise ValueError('exact complete188 original filemap')
    raw={}
    for path,row in expected.items():
        if stop():raise TimeoutError('qualification stop')
        raw[path]=pinned(dict(path=path,bytes=row['bytes'],sha256=row['sha256']))
    reader=Path(OLD)/'science/reader'
    actual={str(p) for p in reader.iterdir()}
    wanted={p for p in expected if str(Path(p).parent)==str(reader)}
    if actual!=wanted or any(Path(p).suffix=='.pending' for p in wanted):raise ValueError('unknown reader member or pending')
    docs={Path(p).stem:json.loads(raw[p]) for p in wanted if p.endswith('.json')}
    if any(not (k in ('reader_budget','reader_cursor') or k.startswith(('scene_','geometry_','audit_','mother_'))) for k in docs):raise ValueError('foreign reader checkpoint')
    budget=docs['reader_budget']
    if any(budget.get(k)!=v for k,v in dict(rays=891,raypairs=28170,layouts=5,prior_seconds=10,inflight=None).items()):raise ValueError('original exact charged budget')
    scenes={k:v for k,v in docs.items() if k.startswith('scene_')}
    geometries={k:v for k,v in docs.items() if k.startswith('geometry_')}
    audits={k:v for k,v in docs.items() if k.startswith('audit_')}
    mothers={k for k in docs if k.startswith('mother_')}
    if set(scenes)!={'scene_'+str(i) for i in INDICES[:4]} or len(geometries)!=4 or len(audits)!=8 or mothers!={'mother_'+str(i) for i in INDICES[:3]}:raise ValueError('exact completed checkpoint inventory')
    if docs['reader_cursor'].get('last_completed') not in audits:raise ValueError('cursor must reference durable receipt')
    totals=dict(rays=0,raypairs=0,layouts=0);ordered=[];required_geometry=set();missing=False
    for case in plan['cases']:
        if stop():raise TimeoutError('cache qualification stop')
        record=load_case(case['input_adapter_ref']);index=case['original_index']
        parent=canonical(record['origin']['identity']);gkey='geometry_'+parent
        scene=scenes.get('scene_'+str(index))
        if scene is not None:
            if scene.get('checkpoint_content_SHA')!=canonical({k:v for k,v in scene.items() if k!='checkpoint_content_SHA'}):raise ValueError('scene canonical identity')
            binding=dict(input_content_identity=same(record['source_original_input_ref']),source_content_identity=same(plan['source_manifest_ref']),physical_content_SHA=canonical(record['physical_parameters']),source_scene_content_SHA=canonical(record['scene']))
            if canonical(scene.get('binding'))!=canonical(binding) or any(scene['scene'][k]!=record['scene'][k] for k in ('aircraft_ids','positions','demands','input_sha256')):raise ValueError('scene input/source/physics binding')
            totals['rays']+=len(scene['scene']['aircraft_ids']);required_geometry.add(gkey)
            cache=geometries.get(gkey)
            if cache is None or cache.get('parent_cache_SHA')!=parent or cache.get('cache_content_SHA')!=canonical({k:v for k,v in cache.items() if k!='cache_content_SHA'}):raise ValueError('complete canonical parent geometry')
            if set(cache.get('layouts',{}))!=set(cache.get('exact',{})):raise ValueError('geometry layout/exact cache keys')
        tree=json.loads(raw[str(Path(OLD)/'science'/('mother_'+str(index))/'tree.json')])
        for path in ('0','1','00','01','10','11'):
            semantic=tree['paths'][path]['semantic_SHA']
            node=json.loads(raw[str(Path(OLD)/'science'/('mother_'+str(index))/('node_'+semantic+'.json'))])['payload']['node']
            binding=dict(node_SHA=canonical(node),source=same(plan['source_manifest_ref']),input=same(record['source_original_input_ref']),parent_cache_SHA=parent)
            key='audit_'+canonical(binding)
            if key in ordered:continue
            ordered.append(key)
            missing=cached_prefix_step(key,audits,missing)
            if key not in audits:continue
            receipt=audits[key];cache=geometries.get(gkey,{})
            if receipt.get('schema')!='R12IndependentAuditCheckpointV2' or canonical(receipt.get('binding'))!=canonical(binding):raise ValueError('audit source/input/node/parent binding')
            result=receipt.get('result',{})
            if any(result.get(k) is not True for k in ('passed','geometry_verified','identity_verified')):raise ValueError('cached audit incomplete')
            for field,store in (('validated_layout_SHA_metrics','layouts'),('validated_layout_SHA_exact','exact')):
                refs=result.get(field)
                if not isinstance(refs,dict) or any(cache.get(store,{}).get(k)!=v for k,v in refs.items()):raise ValueError('cached referenced layout missing or changed')
            charge=receipt['reserved_cost']
            if set(charge)!=set(totals) or any(type(v) is not int or v<0 for v in charge.values()) or charge['rays']!=0:raise ValueError('exact audit predebit')
            for k in totals:totals[k]+=charge[k]
    if set(geometries)!=required_geometry or set(audits)-set(ordered) or any(totals[k]!=budget[k] for k in totals):raise ValueError('all completed GEO reservations reconcile')
    return {Path(p).name:raw[p] for p in wanted if Path(p).suffix=='.json'},budget

def seed(destination,files,budget,stop,remaining_bytes):
    copied=dict(files)
    copied['reader_budget.json']=json.dumps(dict(budget,seconds=103),sort_keys=True,separators=(',',':')).encode()
    if sum(len(v) for v in copied.values())>remaining_bytes:raise RuntimeError('seed whole-round space ceiling')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    for name,raw in copied.items():
        if stop():raise TimeoutError('seed copy stop')
        with (destination/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        if (destination/name).read_bytes()!=raw:raise ValueError('seed copy identity')

def attempt_bounds(start,now,grant):
    deadline=min(start+16.75,grant['attempt_deadline_monotonic_seconds'],now+grant['attempt_deadline_unix_seconds']-time.time())
    if deadline<=now:raise TimeoutError('attempt lease already spent')
    return deadline,min(start+14,deadline-2.75)

def bounded_stop(plan,stop_new,utc_stop):
    from scripts.stage1_ai_search_r12 import runtime_epoch
    cached=[float('-inf'),False]
    def stop(*args,force=False,**kwargs):
        tick=time.monotonic()
        if tick>=stop_new or time.time()>=utc_stop:return True
        if not force and tick-cached[0]<.1:return cached[1]
        if Path(NEW+'/stop.request').exists():return True
        try:runtime_epoch.validate_authority_and_markers(plan);result=False
        except Exception:result=True
        cached[:]=[tick,result];return result
    return stop

def reap_owned(proc,registry,deadline):
    import psutil
    owned=[]
    for pid,birth in registry.items():
        try:
            p=psutil.Process(pid)
            if p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE:owned.append(p);p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=max(0.,min(.5,deadline-time.monotonic())))
    for p in alive:
        try:
            if p.create_time()==registry[p.pid]:p.kill()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(alive,timeout=max(0.,deadline-time.monotonic()))
    if proc is not None:proc.join(timeout=max(0.,deadline-time.monotonic()))
    return [dict(pid=p.pid,birth_unix=registry[p.pid]) for p in alive]

def journal_adapter(destination,journal_limit,temporary_remaining):
    from scripts.stage1_ai_search_r12.persistence import Journal,encoded
    class RoutedJournal(Journal):
        def __init__(self,root,*args,**kwargs):
            if Path(root)!=Path(OLD)/'science/reader':raise ValueError('sole original reader Journal route')
            super().__init__(destination,limit=journal_limit)
        def write(self,key,value):
            if len(encoded(value))>temporary_remaining:raise RuntimeError('temporary original64MiB ceiling')
            return super().write(key,value)
    return RoutedJournal

def child(grant,deadline,stop_new,parent):
    # Imports and all IO are charged by the already running parent outer timer.
    from scripts.stage1_ai_search_r12.worker import bound_parent,one_thread
    bound_parent(parent);one_thread()
    from scripts.stage1_ai_search_r12 import runtime_epoch
    from scripts.stage1_ai_search_r12.persistence import Journal,encoded
    plan=load(grant['original_plan_ref'])
    if plan.get('content_SHA')!=canonical({k:v for k,v in plan.items() if k!='content_SHA'}):raise ValueError('original unchanged plan content')
    runtime_epoch.validate_authority_and_markers(plan)
    stop=bounded_stop(plan,stop_new,grant['attempt_stop_new_unix_seconds'])
    if stop(force=True):raise TimeoutError('before cache qualification')
    files,budget=qualify(plan,load(grant['filemap_ref']),stop)
    destination=Path(NEW)/'reader'
    journal_limit=CAP-grant['static_baseline_bytes']-RESERVE
    seed(destination,files,budget,stop,journal_limit)
    RoutedJournal=journal_adapter(destination,journal_limit,TEMP-grant['static_temporary_bytes'])
    from scripts.stage1_ai_search_r12 import collect as collector
    from scripts.stage1_ai_search_r12.worker import verify_imports
    verify_imports(plan)
    collector.Journal=RoutedJournal
    runtime_epoch.stopped=stop  # declared process-local operational binding only
    result=collector.collect(plan,str(Path(OLD)/'science'))
    return result

def main():
    import argparse
    import multiprocessing as mp
    import shutil
    parser=argparse.ArgumentParser()
    for key in ('grant','grant-sha256'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--grant-bytes',required=True,type=int);args=parser.parse_args()
    registry={};proc=None;receipt=dict(schema='R12OneReaderContinuationOuterReceiptV1',complete=False,scientific_calls=0,prior_reader_seconds=103,outer_hard_seconds=17,new_GEO_stop_seconds=14,retry_allowed=False,charged_outer_seconds=17,final_receipt_io_requires_external_exit_observation=True)
    root=Path(NEW);grant=None;intent_created=False;deadline=ENTRY_MONOTONIC+16.75
    try:
        grant=load(dict(path=args.grant,sha256=args.grant_sha256,bytes=args.grant_bytes))
        if grant.get('schema')!='RootR12OneReaderContinuationExecutionGrantV1' or grant.get('execution_authorized') is not True:raise ValueError('separate explicit audit-only root grant')
        acceptance_ref=grant['design_acceptance_ref']
        if acceptance_ref.get('sha256')!=ACCEPTANCE or load(acceptance_ref).get('actual_continuation_execution_authorized') is not False:raise ValueError('exact prospective design binding')
        if grant.get('prior_reader_debit_seconds')!=103 or grant.get('outer_hard_seconds')!=17 or grant.get('parent_reserve_bytes')!=RESERVE:raise ValueError('unaltered one-attempt caps')
        if grant.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip() or grant.get('uid')!=os.getuid():raise ValueError('fresh same owner/boot')
        if not -5<=ENTRY_UTC-grant['observed_unix_seconds']<=120:raise ValueError('fresh stopped writer ownership')
        if grant.get('all_original_writers_exit_verified') is not True:raise ValueError('old writers must be verified gone')
        import psutil
        old=grant['original_process_identities']
        if len(old)<3:raise ValueError('all original service/worker/reader generations')
        for owner in old:
            if owner['boot_id']!=grant['boot_id']:continue
            try:
                if psutil.Process(owner['pid']).create_time()==owner['birth_unix']:raise ValueError('original generation still exists')
            except psutil.NoSuchProcess:pass
        if not 0<=grant['static_temporary_bytes']<TEMP:raise ValueError('old temporary evidence remains charged')
        original_plan=load(grant['original_plan_ref'])
        actual_refs={(r['path'],r['sha256'],r['bytes']) for r in grant['original_source_refs']}
        expected_refs={(r['path'],r['sha256'],r['bytes']) for r in original_plan['source_files'].values()}
        if actual_refs!=expected_refs or len(actual_refs)!=45:raise ValueError('exact original executed45 source path/content set')
        if Path(grant['entry_ref']['path']).resolve()!=Path(__file__).resolve():raise ValueError('exact standalone entry route')
        pinned(grant['entry_ref'])
        for ref in grant['original_source_refs']:pinned(ref)
        if len(grant['original_source_refs'])!=45:raise ValueError('complete unchanged45 executed source')
        if not 0<=grant['static_baseline_bytes']<CAP-RESERVE or shutil.disk_usage(root.parent).free<8589934592:raise ValueError('original whole-round space reserve')
        # Intent is never erased, even if failure precedes first new GEO.
        root.mkdir(parents=True,exist_ok=True)
        with (root/'attempt_intent.json').open('x') as f:json.dump(dict(grant_sha256=args.grant_sha256,started_utc=ENTRY_UTC),f);f.flush();os.fsync(f.fileno())
        intent_created=True
        if grant['attempt_stop_new_unix_seconds']!=grant['attempt_deadline_unix_seconds']-3:raise ValueError('same immutable dual-clock17s/14s envelope')
        deadline,stop_new=attempt_bounds(ENTRY_MONOTONIC,time.monotonic(),grant)
        ctx=mp.get_context('spawn');proc=ctx.Process(target=child,args=(grant,deadline,stop_new,os.getpid()));proc.start()
        import psutil
        birth=psutil.Process(proc.pid).create_time();registry[proc.pid]=birth;receipt.update(native_pid=proc.pid,native_birth_unix=birth,boot_id=grant['boot_id'])
        while proc.is_alive() and time.monotonic()<stop_new:
            owner=psutil.Process(proc.pid)
            if owner.create_time()!=birth:raise RuntimeError('reader child generation changed')
            descendants=owner.children(recursive=True)
            for descendant in descendants:registry[descendant.pid]=descendant.create_time()
            if descendants:raise RuntimeError('reader-only job cannot spawn descendants')
            if owner.memory_info().rss+psutil.Process().memory_info().rss>2147483648:raise RuntimeError('original parent+child2GiB cap')
            if shutil.disk_usage(root).free<8589934592:raise RuntimeError('original8GiB reserve')
            from scripts.stage1_ai_search_r12.runtime_epoch import validate_authority_and_markers
            validate_authority_and_markers(load(grant['original_plan_ref']))
            proc.join(timeout=min(.05,max(0.,stop_new-time.monotonic())))
        if proc.is_alive():proc.join(timeout=max(0.,deadline-time.monotonic()-1.))
        if proc.is_alive():proc.terminate();proc.join(timeout=min(.5,max(0.,deadline-time.monotonic())))
        if proc.is_alive():proc.kill();proc.join(timeout=max(0.,deadline-time.monotonic()))
        receipt.update(exit_code=proc.exitcode,exit_verified=not proc.is_alive())
        receipt['complete']=proc.exitcode==0 and not proc.is_alive() and (root/'reader/renderer_small_table.json').exists() and time.monotonic()<=deadline
    except BaseException as error:receipt.update(error_type=type(error).__name__,error=str(error))
    finally:
        if proc is not None:
            try:
                import psutil
                if proc.is_alive():
                    current=psutil.Process(proc.pid)
                    if proc.pid not in registry and current.ppid()==os.getpid() and current.create_time()>=ENTRY_UTC-1:registry[proc.pid]=current.create_time()
                    if current.create_time()==registry.get(proc.pid):
                        for descendant in current.children(recursive=True):registry[descendant.pid]=descendant.create_time()
                remaining=reap_owned(proc,registry,deadline)
                receipt['registered_descendants_remaining']=remaining
                if remaining:receipt['complete']=False
            except Exception as error:
                receipt.update(cleanup_error=str(error),complete=False)
        receipt.update(actual_outer_seconds=time.monotonic()-ENTRY_MONOTONIC,actual_outer_utc_seconds=time.time()-ENTRY_UTC,exit_verified=proc is None or not proc.is_alive())
        if receipt['actual_outer_seconds']>17:receipt['complete']=False
        receipt['audit_payload_complete']=receipt['complete']
        receipt['complete']=False
        receipt['closure_pending_external_process_exit_and_final_wallclock']=True
        # Bounded evidence reserve; never recursively inventories active Journal.
        if intent_created:
            with (root/'outer_receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        receipt['actual_outer_including_receipt_save_seconds']=time.monotonic()-ENTRY_MONOTONIC
        if receipt['actual_outer_including_receipt_save_seconds']>17:receipt['complete']=False
        print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__':main()
