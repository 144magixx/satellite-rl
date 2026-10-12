import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.workbench.coverage_geoload import default_config, validate_config, pin
from satellite_rl.workbench.geoload_runner import load_cases, _run_mother, exclusive_lock, result_key
from satellite_rl.workbench.storage import digest

class GeometryRuntimeTests(unittest.TestCase):
    def test_frozen_budget_rejects_extra_fields(self):
        c=default_config();c['search']['physical_calls']=12
        with self.assertRaises(ValueError):validate_config(c)
    def test_seed_immutable(self):
        c=default_config();c['training']['seed']=51002
        with self.assertRaises(ValueError):validate_config(c)
    def test_unknown_metric_config_rejected(self):
        c=default_config();c['coverage']['capacity']=600
        with self.assertRaises(ValueError):validate_config(c)
    def test_sha_pin(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_text('first');r={'path':str(p),'sha256':digest(p)}
            self.assertEqual(pin(r),p.resolve());p.write_text('second')
            with self.assertRaises(ValueError):pin(r)
    def test_packed_dataset_groups_mothers(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'inputs.npz'
            np.savez(p,positions=np.array([[45.,5.],[45.1,5.],[45.,5.],[45.1,5.]]),
                demands=np.array([1.,2.,2.,1.]),offsets=np.array([0,2,4]),
                case_ids=np.array(['a_base','a_hot']),mother_ids=np.array(['a','a']),demand_profiles=np.array(['base','one_hotspot']))
            groups=load_cases({'path':str(p),'sha256':digest(p),'count':2})
            self.assertEqual(len(groups),1);self.assertEqual(len(groups[0]),2)
            rows=_run_mother(groups[0],['BH'],{'seconds':0.,'max_attempts':0},'evaluate')
            self.assertTrue(all(x['metrics']['evidence']['valid'] for x in rows))
            self.assertEqual(rows[0]['initial_sha256'],rows[1]['initial_sha256'])
    def test_kernel_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with exclusive_lock(Path(d)/'lock'):
                with self.assertRaises(OSError):
                    with exclusive_lock(Path(d)/'lock'):pass
    def test_keys_do_not_alias(self):
        self.assertNotEqual(result_key('a','BC'),result_key('aB','C'))
    def test_dispatch_rejects_cycle_and_other_lane(self):
        from scripts.stage1_geoload.controller import validate_plan
        p={'group':'S1-G16','authorized':True,'tasks':{'a':{'port':20424,'body':{'request_id':'a'},'depends':['a']}}}
        with self.assertRaises(ValueError):validate_plan(p)
        p['tasks']['a']['depends']=[];p['tasks']['a']['port']=20418
        with self.assertRaises(ValueError):validate_plan(p)
    def test_reference_gate_uses30s_and_both_halves(self):
        from scripts.stage1_geoload.pipeline import make_gate
        state={'tasks':{}}
        for port in (20424,20473):
            for mode in ('dev','reference'):state['tasks'][mode+'_'+str(port)]={'port':port,'mode':mode}
        def read(task):
            rows=[];parity=0 if task['port']==20424 else 1
            for m in range(parity,20,2):
                for profile in range(4):
                    key=f'{m:02}_{profile}';mother=f'{m:02}'
                    a={'P':2.,'T':1.8,'evidence':{'valid':True}}
                    b={**a,'P':1.9 if m%10<2 else 2.}
                    row={'case_id':key,'mother_id':mother,'method':'09-LEX','metrics':a}
                    if task['mode']=='reference':
                        row['method']='JOINT';row['reference_nodes']=[{'seconds_limit':10,'metrics':a},{'seconds_limit':30,'metrics':b}]
                    rows.append(row)
            return {'rows':rows,'index_sha256':'0'*64}
        with tempfile.TemporaryDirectory() as d,patch('scripts.stage1_geoload.pipeline.raw_rows',side_effect=read):
            gate,_=make_gate(state,d)
            self.assertTrue(gate['passed']);self.assertEqual(gate['reference_seconds'],30)
            self.assertEqual(gate['half_fractions'],[.2,.2])
    def test_create_job_idempotency_and_source_pin(self):
        from types import SimpleNamespace
        from satellite_rl.workbench.storage import Store
        from satellite_rl.workbench.coverage_geoload import create_job
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store=Store(root/'archive',source_root=root);commit='a'*40
            plan=root/'plan.json';plan.write_text(json.dumps({'group':'S1-G16','authorized':True,'scientific_commit':commit}))
            partition=root/'partition.json';partition.write_text(json.dumps({'group':'S1-G16','roles':{
                r:{'dates':[r],'count':80,'mother_count':20} for r in ('train','development','validation','test')}}))
            dataset=root/'inputs.npz';dataset.write_bytes(b'bounded test source')
            c=default_config();c['datasets']={'development':{'path':str(dataset),'sha256':digest(dataset),'role':'development','count':80}}
            b={'request_id':'unique','kind':'evaluate','mode':'evaluate','dataset_role':'development','config':c,'methods':['BH'],
                'plan':{'path':str(plan),'sha256':digest(plan)},'partition':{'path':str(partition),'sha256':digest(partition)}}
            with patch('satellite_rl.workbench.coverage_geoload.deployment',return_value='20424'),patch.object(store,'freeze_source',return_value=(root,commit)):
                job=create_job(SimpleNamespace(store=store),b)
                self.assertEqual(create_job(SimpleNamespace(store=store),b)['run_id'],job['run_id'])
                b['methods']=['07']
                with self.assertRaises(ValueError):create_job(SimpleNamespace(store=store),b)

if __name__=='__main__':unittest.main()
