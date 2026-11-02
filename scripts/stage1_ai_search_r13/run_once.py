"""One local120s R13 scalar IO job. Import never reads a scientific payload."""
import time
ENTRY=time.monotonic()
import csv
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import socket
import stat
import sys

META_CAP=2097152
COMPRESSED_CAP=33554432
DECODED_CAP=134217728
PER_FILE_CAP=16777216
OUTPUT_CAP=16777216
RESERVE=1048576
AUTH='aa63719a267d0442a7f9849635d4f774184d9195020d4fc074215eb9b562427d'
ACCEPT='c971e6f0df9aaa4a74b2642c0368c59de03422443d64f07e739e81ad04320298'

def digest(raw):return hashlib.sha256(raw).hexdigest()

def raw_ref(ref,limit=META_CAP):
    p=Path(ref['path']);info=p.lstat()
    if not stat.S_ISREG(info.st_mode) or type(ref.get('bytes')) is not int or not 0<=ref['bytes']<=limit:raise ValueError('bounded regular source/input ref')
    with p.open('rb') as f:raw=f.read(ref['bytes']+1)
    if len(raw)!=ref['bytes'] or digest(raw)!=ref['sha256']:raise ValueError('content identity mismatch')
    return raw

class Metadata:
    def __init__(self,prior_bytes=0):self.used=prior_bytes;self.cache={}
    def read(self,ref):
        key=str(Path(ref['path']).resolve());signature=(ref['sha256'],ref['bytes'])
        if key in self.cache:
            if self.cache[key][0]!=signature:raise ValueError('metadata path alias changed identity')
            return self.cache[key][1]
        if type(ref.get('bytes')) is not int or self.used+ref['bytes']>META_CAP:raise ValueError('metadata2MiB prospective ceiling')
        self.used+=ref['bytes'];raw=raw_ref(ref);self.cache[key]=(signature,raw);return raw

class Payloads:
    def __init__(self,rows,stop):
        if len(rows)!=24:raise ValueError('exact24 single-pass originals')
        self.rows=rows;self.stop=stop;self.read=set();self.compressed=0;self.decoded=0
        if sum(r['local_ref']['bytes'] for r in rows)>COMPRESSED_CAP:raise ValueError('compressed32MiB precharge')
        if sum(r['logical_output_ref']['uncompressed_bytes'] for r in rows)>DECODED_CAP:raise ValueError('decoded128MiB precharge')
        if any(r['logical_output_ref']['uncompressed_bytes']>PER_FILE_CAP for r in rows):raise ValueError('perfile16MiB precharge')
        identities=[(r['local_ref']['path'],r['local_ref']['sha256']) for r in rows]
        if len(set(identities))!=24:raise ValueError('no duplicated/implicit payload reread')
    def load(self,row):
        import stage_a as a
        ref=row['local_ref'];logical=row['logical_output_ref'];key=(ref['path'],ref['sha256'])
        if key in self.read:raise ValueError('single-pass input already entered; no replay')
        if self.stop():raise TimeoutError('stop before input entry')
        self.read.add(key);self.compressed+=ref['bytes']
        if self.compressed>COMPRESSED_CAP:raise ValueError('compressed group ceiling')
        raw=raw_ref(ref,COMPRESSED_CAP)
        if logical.get('compression')!='gzip' or logical.get('sha256')!=ref['sha256'] or logical.get('compressed_bytes')!=len(raw):raise ValueError('typed original compressed identity')
        output=bytearray()
        with gzip.GzipFile(fileobj=io.BytesIO(raw),mode='rb') as stream:
            while True:
                if self.stop():raise TimeoutError('stop during unique gzip stream')
                block=stream.read(min(262144,PER_FILE_CAP-len(output)+1))
                if not block:break
                self.decoded+=len(block)
                if len(output)+len(block)>PER_FILE_CAP or self.decoded>DECODED_CAP:raise ValueError('decoded hard ceiling; returned overflow bytes remain charged')
                output.extend(block)
        if len(output)!=logical['uncompressed_bytes']:raise ValueError('exact decoded byte count')
        if self.stop():raise TimeoutError('stop before original JSON parse/canonical binding')
        value=a.parse(output)
        if a.sha(value)!=logical['canonical_json_sha256']:raise ValueError('canonical original JSON identity')
        return value

class Writer:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(exist_ok=False);self.sizes={}
    def write(self,name,raw):
        if '/' in name or '\\' in name or not name.endswith(('.json','.csv')):raise ValueError('bounded result filename')
        if len(raw)>OUTPUT_CAP or sum(self.sizes.values())+len(raw)>OUTPUT_CAP-RESERVE:raise ValueError('prospective canonical/replacement output16MiB ceiling')
        p=self.root/name;temp=p.with_suffix('.pending')
        with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temp,p);self.sizes[name]=len(raw)
        return dict(path=str(p),bytes=len(raw),sha256=digest(raw))

def bind_inputs(metadata,manifest):
    import stage_a as a
    need=a.need
    need(manifest.get('schema')=='R13ExactStageAInputsV1','exact input-map schema')
    # FALSE is a historical preparation label; only separate root execution review
    # approves the byte-identical input map. It is not a second execution switch.
    for ref in manifest['source_metadata_refs']:metadata.read(ref)
    old_map=a.parse(metadata.read(manifest['source_map_ref']))
    logical=a.parse(metadata.read(manifest['logical_table_ref']))
    rows=manifest['rows'];need([(r['original_index'],r['stage']) for r in rows]==[(i,s) for i in a.INDICES for s in ('common','SH-C')],'fixed12 common/SH-C slots')
    old={r['remote_path']:r for r in old_map['files']}
    for row in rows:
        slot=next(v for v in logical['rows'] if v['original_index']==row['original_index'] and v['stage']==row['stage'])
        need(slot['selected_epoch']==row['selected_epoch'] and slot['executor_port']==row['executor_port'] and slot['selected_stage_view']['phase']=='complete' and slot['selected_stage_view']['output_ref']==row['logical_output_ref'],'exact previously audited logical source slot')
        saved=old[row['logical_output_ref']['path']];local=row['local_ref']
        need((saved['local_path'],saved['bytes'],saved['sha256'])==(local['path'],local['bytes'],local['sha256']),'accepted canonical source/content locator')
        certificate=a.parse(metadata.read(row['certificate_ref']))
        need(certificate.get('schema')=='RootR10CompleteScientificGateAcceptanceV1' and certificate.get('complete12x5') is True and certificate.get('scientific_branch_closed') is True,'universal old full12x5 independent scientific certificate')
    return rows

def marker_stop(config,counter=None):
    if counter is not None:counter['marker_probe_calls']=counter.get('marker_probe_calls',0)+1
    for path in config['watched_stop_paths']:
        try:Path(path).lstat();return True
        except FileNotFoundError:pass
        except OSError:return True
    for ref in config['historical_marker_refs']:
        try:
            if counter is not None:counter['marker_bytes_precharged']=counter.get('marker_bytes_precharged',0)+ref['bytes']
            raw_ref(ref,16384)
        except Exception:return True
    return False

def _child_inner(config,prior_meta,stop_new,parent_pid,parent_birth,writer,meta,budget_context):
    import psutil
    import stage_a as a
    # No original project numerical module is imported or executed.
    guard_IO={};budget_context['guard_IO']=guard_IO
    cached=[float('-inf'),False]
    def stop():
        now=time.monotonic()
        if now>=stop_new:return True
        if now-cached[0]>=.1:
            try:owner_ok=psutil.Process(parent_pid).create_time()==parent_birth
            except psutil.NoSuchProcess:owner_ok=False
            cached[:]=[now,not owner_ok or marker_stop(config,guard_IO)]
        return cached[1]
    for ref in config['source_files']:meta.read(ref)
    authority=a.parse(meta.read(config['human_resume_ref']))
    a.need(config['human_resume_ref']['sha256']==AUTH and authority.get('group')=='S1-AI17' and authority.get('observed_after_0700_pause') is True and authority.get('current_round')=='R12','exact unchanged same-group human authority; no fabricated R13 human message')
    prospective=a.parse(meta.read(config['design_acceptance_ref']))
    a.need(config['design_acceptance_ref']['sha256']==ACCEPT and prospective.get('design_and_code_preparation_accepted') is True,'root compact certificate/source design choice')
    manifest=a.parse(meta.read(config['input_manifest_ref']));rows=bind_inputs(meta,manifest)
    loader=Payloads(rows,stop);budget_context['loader']=loader;summaries=[];all_rows=[];prefix_csv=[]
    counters=dict(prefix_rows=0,method_records=0,screenings=0,aircraft_records=0)
    try:
        for i,index in enumerate(a.INDICES):
            if stop():raise TimeoutError('110second stop before new mother')
            common_row,method_row=rows[2*i:2*i+2]
            common=loader.load(common_row);method=loader.load(method_row)
            result=a.analyze_case(common,method,common_row['logical_output_ref'],method_row['logical_output_ref'],method_row['certificate_ref'],stop)
            counters['prefix_rows']+=len(result['prefix_rows']);counters['method_records']+=result['full_attempts'];counters['screenings']+=result['screens'];counters['aircraft_records']+=result['aircraft_count']
            if counters['prefix_rows']>396 or counters['method_records']>384 or counters['screenings']>316800:raise ValueError('original scalar/record ceilings')
            case_ref=writer.write('mother_'+str(index)+'.json',a.canonical(result));all_rows.append(result)
            summaries.append(dict(original_index=index,complete=True,full_attempts=result['full_attempts'],first_safe_prefix=result['first_safe_prefix'],strong_fixed_envelope=result['strong_fixed_envelope'],case_ref=case_ref))
            for p in result['prefix_rows']:
                prefix_csv.append(dict(original_index=index,prefix_index=p['prefix_index'],last_accepted_trial=p['last_accepted_trial_zero_based'],layout_SHA=p['layout_SHA'],quality_preserving_own_teacher=p['quality_preserving_own_teacher'],**p['metrics'],screens_already_paid=p['screens_already_paid'],inner_flow_queries_proxy_paid=p['inner_flow_queries_proxy_paid']))
        a.need(counters['aircraft_records']==4715 and len(loader.read)==24,'all12 original4715 aircraft and24 unique original payloads')
        gate=a.opportunity_gate(all_rows,config['instrumentation_boundary_identified'])
        csv_buffer=io.StringIO(newline='');csv_writer=csv.DictWriter(csv_buffer,fieldnames=list(prefix_csv[0]));csv_writer.writeheader();csv_writer.writerows(prefix_csv)
        writer.write('prefixes.csv',csv_buffer.getvalue().encode('utf-8-sig'))
        writer.write('result.json',a.canonical(dict(schema='R13StageAResultV1',complete=True,mothers=summaries,gate=gate,counters=counters,
            IO=dict(unique_payloads=len(loader.read),compressed_bytes=loader.compressed,decoded_bytes=loader.decoded,source_control_metadata_bytes=meta.used,source_control_byte_scope='source/control reads plus parent duplicate reads; excludes repeated guard marker checks',guard_marker_IO=guard_IO),
            development_only=True,actual_acceleration_measured=False,new_solver_GEO_model_training=0,instrumentation_boundary=config['instrumentation_boundary'])))
    except BaseException as error:
        writer.write('failure.json',a.canonical(dict(complete=False,error_type=type(error).__name__,error=str(error),counters=counters,unique_payloads_entered=len(loader.read),compressed_bytes_entered=loader.compressed,decoded_bytes=loader.decoded,metadata_bytes=meta.used,automatic_retry=False)))
        raise

def child(config,prior_meta,stop_new,parent_pid,parent_birth):
    import stage_a as a
    writer=Writer(Path(config['output_root'])/'data');meta=Metadata(prior_meta);context={}
    try:
        _child_inner(config,prior_meta,stop_new,parent_pid,parent_birth,writer,meta,context)
    except BaseException as error:
        if 'failure.json' not in writer.sizes:
            loader=context.get('loader')
            writer.write('failure.json',a.canonical(dict(complete=False,error_type=type(error).__name__,error=str(error),metadata_bytes=meta.used,
                unique_payloads_entered=0 if loader is None else len(loader.read),compressed_bytes_entered=0 if loader is None else loader.compressed,
                decoded_bytes_returned=0 if loader is None else loader.decoded,guard_marker_IO=context.get('guard_IO',{}),source_control_byte_scope='excludes repeated guard marker checks; guard IO charged in outer wall',automatic_retry=False)))
        raise

def cleanup(proc,registry,deadline):
    import psutil
    owned=[]
    for pid,birth in registry.items():
        try:
            p=psutil.Process(pid)
            if p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE:p.terminate();owned.append(p)
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=max(0,min(2,deadline-time.monotonic())))
    for p in alive:
        try:
            if p.create_time()==registry[p.pid]:p.kill()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(alive,timeout=max(0,deadline-time.monotonic()))
    if proc is not None:proc.join(timeout=max(0,deadline-time.monotonic()))
    return [dict(pid=p.pid,birth_unix=registry[p.pid]) for p in alive]

def operational_bounds(entry,now):
    if now>=entry+110:raise TimeoutError('stop-new boundary already consumed')
    return dict(stop_new=entry+110,reap_deadline=entry+119,outer_deadline=entry+120)

def claim_once(root,config_ref,review_ref):
    root=Path(root);root.mkdir(exist_ok=False)
    with (root/'attempt_intent.json').open('x') as f:
        json.dump(dict(config_ref=config_ref,review_ref=review_ref,started_unix=time.time(),one_attempt=True),f);f.flush();os.fsync(f.fileno())

def validate_route(route,now,host,boot,executable):
    from math import isfinite
    delta=boot-route['boot_time']
    if not isfinite(delta) or abs(delta)>1.0 or route['host_name']!=host or route['python']!=executable or not -5<=now-route['observed_unix_seconds']<=120:
        raise ValueError('fresh actual local host/boot/interpreter')
    return dict(route_boot_time_raw=route['boot_time'],actual_boot_time_raw=boot,boot_time_delta_seconds=delta,boot_time_tolerance_seconds=1.0)

def main():
    import argparse
    import psutil
    import stage_a as a
    parser=argparse.ArgumentParser()
    for key in ('config','config-sha256','review','review-sha256'):parser.add_argument('--'+key,required=True)
    for key in ('config-bytes','review-bytes'):parser.add_argument('--'+key,required=True,type=int)
    args=parser.parse_args();proc=None;registry={};root=None;intent=False;deadline=ENTRY+119
    receipt=dict(schema='R13StageAOuterReceiptV1',complete=False,execution_attempts=0,automatic_retry=False,new_GEO_solver_model_training=0,closure_pending_external_process_exit=True)
    try:
        config_ref=dict(path=args.config,sha256=args.config_sha256,bytes=args.config_bytes);review_ref=dict(path=args.review,sha256=args.review_sha256,bytes=args.review_bytes)
        meta=Metadata();config=a.parse(meta.read(config_ref));review=a.parse(meta.read(review_ref))
        a.need(config.get('schema')=='R13StageAExecutionConfigV1' and config.get('scientific_execution_authorized') is True,'FALSE preparation config cannot execute')
        a.need(review.get('schema')=='RootR13StageAExecutionAdmissionV1' and review.get('execution_authorized') is True and review.get('config_ref')==config_ref,'separate byte-bound root once execution review')
        route=review['fresh_local_route'];receipt['fresh_boot_route']=validate_route(route,time.time(),socket.gethostname(),psutil.boot_time(),sys.executable)
        root=Path(config['output_root']);a.need(root.is_absolute() and config['own_stop_marker']==str(root/'stop.request') and config['own_stop_marker'] in config['watched_stop_paths'] and config['global_stop_marker'] in config['watched_stop_paths'] and config['global_stop_marker']!=config['own_stop_marker'],'actual distinct own/global stop markers')
        a.need(len(config['historical_marker_refs'])==3 and {(r['sha256'],r['bytes']) for r in config['historical_marker_refs']}=={('6902ca368bdf459105131625f620b3062aa3abc283e25701d0b7e33f78fc9ddd',517),('c41dade127fed70b8a5b6f7a1054d768789063e8760a1235bf524d513fe0053e',1641),('20f2829f9c2c8efc3bbebcc5c474db850dad5f9a1a00fa19c344eda9eebc8064',1641)} and not marker_stop(config),'historical byte generations and new pause priority')
        a.need(config['outer_hard_seconds']==120 and config['stop_new_seconds']==110 and config['minimum_free_bytes']==8589934592,'frozen finite job caps')
        a.need(shutil.disk_usage(root.parent).free>=8589934592,'actual local space reserve')
        for ref in config['source_files']:meta.read(ref)
        a.need({Path(__file__).resolve(),Path(a.__file__).resolve()}<={Path(ref['path']).resolve() for ref in config['source_files']},'actual entry and imported scalar module in exact source manifest')
        claim_once(root,config_ref,review_ref)
        intent=True;receipt['execution_attempts']=1
        bounds=operational_bounds(ENTRY,time.monotonic());deadline=bounds['reap_deadline']
        proc=mp.get_context('spawn').Process(target=child,args=(config,meta.used,ENTRY+110,os.getpid(),psutil.Process().create_time()));proc.start()
        birth=psutil.Process(proc.pid).create_time();registry[proc.pid]=birth;receipt.update(actual_child_pid=proc.pid,actual_child_birth=birth)
        peak=0;reason=None;parent_guard_IO={}
        while proc.is_alive() and time.monotonic()<ENTRY+110:
            process=psutil.Process(proc.pid);a.need(process.create_time()==birth,'owned child generation')
            children=process.children(recursive=True)
            for p in children:registry[p.pid]=p.create_time()
            if children:reason='unexpected_descendants';break
            peak=max(peak,process.memory_info().rss+psutil.Process().memory_info().rss)
            if peak>2147483648:reason='parent_children_RSS';break
            if shutil.disk_usage(root).free<8589934592:reason='free_space';break
            if marker_stop(config,parent_guard_IO):reason='new_pause_or_historical_change';break
            proc.join(timeout=.05)
        if reason is None and proc.is_alive():proc.join(timeout=max(0,ENTRY+115-time.monotonic()))
        if proc.is_alive():receipt['remaining_registered']=cleanup(proc,registry,deadline)
        receipt.update(parent_guard_marker_IO=parent_guard_IO,reason=reason,exit_code=proc.exitcode,peak_parent_children_RSS_bytes=peak,
            result_candidate=reason is None and proc.exitcode==0 and not proc.is_alive() and not receipt.get('remaining_registered') and (root/'data/result.json').exists() and time.monotonic()-ENTRY<=120)
    except BaseException as error:receipt.update(error_type=type(error).__name__,error=str(error),result_candidate=False)
    finally:
        if proc is not None:
            try:
                if proc.is_alive():
                    current=psutil.Process(proc.pid)
                    if proc.pid not in registry and current.ppid()==os.getpid():registry[proc.pid]=current.create_time()
                    if current.create_time()==registry[proc.pid]:
                        for p in current.children(recursive=True):registry[p.pid]=p.create_time()
                receipt['remaining_registered']=cleanup(proc,registry,deadline)
            except Exception as error:receipt.update(cleanup_error=str(error),result_candidate=False)
        if receipt.get('remaining_registered') or receipt.get('cleanup_error') or receipt.get('reason') or time.monotonic()-ENTRY>120:receipt['result_candidate']=False
        receipt.update(actual_outer_seconds_before_receipt=time.monotonic()-ENTRY,registered_identities=[dict(pid=p,birth_unix=b) for p,b in registry.items()],owned_child_exit_verified=proc is None or not proc.is_alive())
        if intent:
            with (root/'outer_receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        receipt['actual_outer_seconds_including_receipt_save']=time.monotonic()-ENTRY
        print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__':main()
