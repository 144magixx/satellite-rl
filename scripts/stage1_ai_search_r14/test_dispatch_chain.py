"""Actual dispatcher/collector with fake child outputs; no payload/math/fit/process."""
import copy,json,tempfile,threading,time
from pathlib import Path
import unittest
from unittest.mock import patch
from .runtime import ObjectStore,atomic_bytes,json_bytes,exclusive
from .dispatcher import command_for_task,collect_task,aggregate,check_dependencies
from .bindings import bind_task,bind_fit_labels,validate_cold_pairing,read_json
from .lane import run_child,validate_task_order
from .test_lane import FakeChild,FakeGuard
from .pipeline import stage_plan
from .data_manifest import PARTITIONS

class DispatchChainTests(unittest.TestCase):
    def test_positive_fixed_routing_and_actual_derived_evidence_chain(self):
        with tempfile.TemporaryDirectory() as d,patch('subprocess.Popen',side_effect=AssertionError('real process forbidden')):
            root=Path(d);store=ObjectStore(root/'objects');case_ref=store.put_json({'fake':'case-scalar-no-aircraft'})
            cache_ref=store.put_json({'fake':'cache-no-rays'});policy_ref=store.put_json({'fake':'policy-no-model'})
            bundle_ref=store.put_json({'policy_refs':{'AI':policy_ref,'baseline':policy_ref}})
            seen=[];sealed={}
            def one(kind,idx=0,arm=None,phase=None,deps=None,extra=None):
                task_id=f'{kind}_{len(seen)}';out=root/(task_id+'_output');out.mkdir()
                request={'task_id':task_id,'kind':kind,'phase':phase or kind,'recipe_index':idx,'arm':arm,
                    'output_root':str(out),'shared_object_root':str(store.root),'source_refs':[],
                    'case_ref':case_ref,'B_ref':sealed.get('B_body',case_ref),'cache_ref':cache_ref,
                    'dependency_receipts':deps or {},**(extra or {})}
                task={k:request[k] for k in ('task_id','kind','phase','recipe_index','arm')};task['outer_seconds']=120
                task['request_ref']=store.put_json(request);bind_task(task,request);check_dependencies(request)
                command=command_for_task({'interpreter':'FAKE_DQN'},task);seen.append((kind,command[4]))
                ticks=[0.];child=FakeChild(ticks,0)
                def fake_popen(*a,**kw):
                    if kind=='data':atomic_bytes(out/'typed_case_derivation.json',json_bytes({'case_ref':case_ref}))
                    elif kind=='B':
                        body={'case_id':'fake_case','recipe_index':idx,'status':'success','qualified':True,
                            'proposal_success':True,'independent_layouts':['raw_W','Wenv','D5terminal'],
                            'cache_manifest':{'cache_ref':cache_ref,'inner_cache_SHA':'f'*64,'created_before_teacher':True}}
                        ref=store.put_json(body);sealed['B_body']=ref
                        atomic_bytes(out/'objects.json',json_bytes([{'label':'qualified_B_before_parent_exit','ref':ref}]))
                    elif kind in ('teacher','cal','dev'):
                        record={'case_id':'fake_case','task_id':task_id,'arm':arm,'complete':True,'features':[1]*12,
                            'selected_policy':{'action':'cap32','node_id':'fake'},'own_terminal_audit':{'metrics':{'N':1}}}
                        atomic_bytes(out/'result.json',json_bytes(store.put_json(record)))
                        if kind=='teacher':atomic_bytes(out/'training_label_row.json',json_bytes(store.put_json({'case_id':'fake_case','source_task_id':task_id})))
                    else:
                        filename={'fit':'published_bundle_ref.json','veto':'published_bundle_ref.json','offline':'offline_receipt_ref.json','engineering':'engineering_receipt_ref.json','data_registry':'evidence_receipt_ref.json'}[kind]
                        ref=store.put_json({'authority_envelope_ref':case_ref}) if kind=='data_registry' else bundle_ref
                        atomic_bytes(out/filename,json_bytes(ref))
                    return child
                facts=run_child(command,task,root/(task_id+'_control'),FakeGuard(),popen=fake_popen,
                    clock=lambda:ticks[0],sleeper=lambda _:None,identify=lambda pid:{'pid':pid,'process_start_identity':'FAKE_ONLY','boot_id':'FAKE'},
                    terminate=lambda *a,**kw:0,sample=lambda *a:None)
                with self.assertRaises(ValueError):collect_task(task,request,{**facts,'request_ref':{**task['request_ref'],'sha256':'0'*64}})
                receipt,ref=collect_task(task,request,facts);sealed[kind]=ref;return receipt,ref
            def fake_group(role,base,kind,indices):
                refs=[];ids=[]
                for i in indices:
                    item=synthetic_leaf(base,f'FAKE_{role}_{i}',kind=kind,recipe_index=i,fixture_index=i)
                    refs.append(store.put_json(item));ids.append(item['task_id'])
                return aggregate(role,refs,expected_ids=ids,store=store)
            def synthetic_leaf(base,task_id,**fields):
                item={**base,**fields,'task_id':task_id}
                request={**read_json(base['request_ref']),**fields,'task_id':task_id}
                item['request_ref']=store.put_json(request)
                item['actual_exit']={**base['actual_exit'],'task_id':task_id,'kind':item['kind'],'request_ref':item['request_ref']}
                return item
            eng,engref=one('engineering')
            eng_group=fake_group('engineering_all8',eng,'engineering',range(8))
            data,_=one('data',deps={'engineering_all8':eng_group})
            Dall=fake_group('data_all140',data,'data',[i for a in PARTITIONS.values() for i in a])
            registry,registryref=one('data_registry',deps={'data_all140':Dall})
            B,Bref=one('B',deps={'data':sealed['data'],'data_authority':registryref},extra={'authority_envelope_ref':case_ref})
            teacher,_=one('teacher',arm='full_SHC',phase='train_labels',deps={'B':Bref,'engineering_all8':eng_group},extra={'sealed_B_receipt_ref':Bref})
            Ball=fake_group('B_all140',B,'B',[i for a in PARTITIONS.values() for i in a])
            Tall=fake_group('teacher_all96',teacher,'teacher',PARTITIONS['train'])
            fit,fitref=one('fit',deps={'B_all140':Ball,'teacher_all96':Tall})
            cal,calref=one('cal',arm='AI',deps={'initial_fit':fitref},extra={'policy_bundle_ref':bundle_ref,'policy_ref':policy_ref})
            # Veto group fixtures are scalar receipts; no calibration geometry occurs.
            cal_members=[];calids=[]
            for i in PARTITIONS['calibration']:
                for a in ('AI','strong_causal_baseline','full_SHC'):
                    item=synthetic_leaf(cal,f'FAKE_cal_{i}_{a}',recipe_index=i,arm=a);cal_members.append(store.put_json(item));calids.append(item['task_id'])
            calgroup=aggregate('cal_all24x3',cal_members,expected_ids=calids,store=store)
            off,offref=one('offline',deps={'cold_complete':calref})
            offgroup=fake_group('cal_offline24',off,'offline',PARTITIONS['calibration'])
            veto,vetoref=one('veto',deps={'initial_fit':fitref,'cal_all24x3':calgroup,'cal_offline24':offgroup},extra={'initial_policy_bundle_ref':bundle_ref})
            one('dev',arm='AI',deps={'veto':vetoref},extra={'policy_bundle_ref':bundle_ref,'policy_ref':policy_ref})
            expected={'data':'data_worker','B':'B_worker','teacher':'cold_worker','fit':'fit_worker','cal':'cold_worker','veto':'fit_worker','dev':'cold_worker','offline':'offline_worker','engineering':'engineering_worker','data_registry':'evidence_worker'}
            self.assertTrue(all(module.endswith(expected[kind]) for kind,module in seen))
            wrong={'kind':'B','case_ref':{**case_ref,'sha256':'0'*64},'dependency_receipts':{'data':sealed['data']}}
            with self.assertRaises(ValueError):check_dependencies(wrong)

    def test_ref_arm_teacher_and_latin_counterexamples(self):
        task={'task_id':'same','kind':'dev','phase':'development','recipe_index':576,'arm':'AI'}
        with self.assertRaises(ValueError):bind_task(task,{**task,'arm':'full_SHC'})
        ref={'sha256':'a'*64,'bytes':1};exit={'raw_returncode':0,'reaped':True,'descendants_live':0,'process_start_identity':'fake','elapsed_seconds':1}
        rows=[{'case_id':str(i),'source_task_id':str(i)} for i in range(96)]
        receipts=[{'label_ref':ref,'actual_exit':exit,'case_id':str(i),'source_task_id':str(i),'all_positions_audited':True} for i in range(96)]
        self.assertTrue(bind_fit_labels([ref]*96,rows,receipts,[str(i) for i in range(96)]))
        with self.assertRaises(ValueError):bind_fit_labels([{**ref,'sha256':'b'*64}]+[ref]*95,rows,receipts,[str(i) for i in range(96)])
        with self.assertRaises(ValueError):validate_cold_pairing([{'arm':'AI'},{'arm':'strong_causal_baseline'}],{})
        tasks=stage_plan()['development'][:18]
        with self.assertRaises(ValueError):validate_task_order(tasks[9:]+tasks[:9])
        with self.assertRaises(ValueError):validate_task_order(tasks[:1]+tasks[9:]+tasks[1:9])

    def test_shared_lock_wait_and_unknown_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            lock=Path(d)/'lock';entered=threading.Event()
            def holder():
                with exclusive(lock):entered.set();time.sleep(.04)
            thread=threading.Thread(target=holder);thread.start();entered.wait()
            with exclusive(lock,timeout=.3):pass
            thread.join();self.assertFalse(lock.exists())
            lock.write_text('unknown owner')
            with self.assertRaises(TimeoutError):
                with exclusive(lock,timeout=.02):pass
            self.assertEqual(lock.read_text(),'unknown owner')

    def test_chunked_oversize_and_new_pause_tripwire(self):
        with tempfile.TemporaryDirectory() as d:
            store=ObjectStore(d);value={'synthetic_bytes_only':'x'*(33*1024**2)}
            ref=store.put_json(value);self.assertEqual(read_json(ref),value)
            self.assertTrue(all(p.stat().st_size<=32*1024**2 for p in Path(d).iterdir() if len(p.name)==64))
        from .audit import AuditLedger,build_ray_cache,AuditError
        case={'identity':{'case_id':'pure_control'},'aircraft_ids':['fake'],'positions':[[0.,0.]],'demands':[21600000],
            'source_ref':{'sha256':'a'*64,'bytes':1},'physical_parameters':{'version':'FAKE_ONLY','earth_radius_km':1,
            'orbit_radius_km':2,'satellite_longitude_deg':0,'min_elevation_deg':0,'latitude_bounds_deg':[-1,1],'longitude_bounds_deg':[-1,1]}}
        ledger=AuditLedger()
        with patch('scripts.stage1_ai_search_r14.audit._project_rays_visibility',side_effect=AssertionError('GEO forbidden')):
            with self.assertRaises(AuditError):build_ray_cache(case,ledger,stop=lambda:True)
        self.assertEqual(ledger.state['rays'],0)

    def test_popen_identity_failure_still_reaps_direct_handle(self):
        ticks=[0.];child=FakeChild(ticks,999);child.kill=lambda:setattr(child,'killed',True)
        count=[0]
        def identify(pid):
            count[0]+=1
            if count[0]>1:raise RuntimeError('identity unavailable')
            return {'pid':pid,'process_start_identity':'FAKE'}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):run_child(['FAKE'],{'outer_seconds':120},Path(d)/'task',FakeGuard(),
                popen=lambda *a,**k:child,clock=lambda:ticks[0],identify=identify)
            self.assertTrue(child.killed);self.assertTrue(child.waited)

if __name__=='__main__':unittest.main()
