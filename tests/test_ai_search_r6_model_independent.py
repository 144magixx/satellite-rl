"""Independent future modelops checks: pure metadata/mocks only.

No forward, backward, optimizer, D5, preparation, T or real input reads.
"""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
import numpy as np

class IndependentModelOpsTests(unittest.TestCase):
    def test_node_gate_binds_all_passed_training_endpoint_refs(self):
        from scripts.stage1_ai_search_r6model.plan_builder import node_plan,DESIGN_SHA,GROUP
        import satellite_rl.workbench.coverage_ai_search_r6_model as subject
        def ref(name):return dict(path=name,sha256='a'*64)
        inputs=dict(design_ref=dict(path='design',sha256=DESIGN_SHA),identity_ref=ref('identity'),qualification_refs={'engineering':ref('qualification')},prerequisite_gate_ref=ref('gate'),model_identity='one',dataset_ref=dict(role='development',count=80,sha256='9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b'))
        for name in ('checkpoint','normalization','initialization_receipt','registry','source_equivalence'):inputs[name+'_ref']=ref(name)
        plan=node_plan('b'*40,inputs);gate=dict(passed=True,phase='train2000',model_identity='one',updates=2000,checkpoint_ref=inputs['checkpoint_ref'],normalization_ref=inputs['normalization_ref'],initialization_receipt_ref=inputs['initialization_receipt_ref'])
        review=dict(passed=True,group=GROUP,scientific_commit='b'*40,plan_sha256='a'*64,design_sha256=DESIGN_SHA)
        qualification=dict(passed=True,scientific_commit='b'*40,design_sha256=DESIGN_SHA,source_file_sha256={'src/code.py':'file'})
        def read(path,*args):return review if str(path)=='review' else gate if str(path)=='gate' else qualification
        with patch.object(subject,'pin',side_effect=lambda value:Path(value['path'])),patch.object(subject,'read_json',side_effect=read),patch.object(subject,'digest',return_value='file'):
            subject.validate_admission(plan,ref('plan'),ref('review'))
            task=next(iter(plan['tasks'].values()))
            for key in ('checkpoint','normalization','initialization_receipt'):
                original=task['model_inputs'][key];task['model_inputs'][key]=ref('other')
                with self.subTest(key=key),self.assertRaises(ValueError):subject.validate_admission(plan,ref('plan'),ref('review'))
                task['model_inputs'][key]=original
    def test_actual_engineering_source_equivalence_all_keys(self):
        from satellite_rl.workbench.r6_model_io import verify_model_source_equivalence
        actual=dict(model_graph_source_SHA='model',feature_graph_source_SHA='feature',learning_source_SHA='learn',adapter_source_SHA='adapter',b7_decoder_source_SHA='b7',runtime_source_SHA='runtime')
        initial=dict(model_graph_source_SHA='model',feature_graph_source_SHA='feature',training_runtime_identity={'runtime_source_sha256':'runtime'})
        engineering={'sha256':'accepted'};record=dict(passed=True,accepted_engineering_admission_sha256='accepted',immutable_model_source_sha256=actual)
        with patch('satellite_rl.workbench.r6_model_io.signed_json',return_value=record),patch('satellite_rl.workbench.r6_model_io.model_source_identity',return_value=actual):self.assertEqual(verify_model_source_equivalence({},engineering,initial),actual)
        for key in actual:
            changed=copy.deepcopy(actual);changed[key]='changed'
            with self.subTest(key=key),patch('satellite_rl.workbench.r6_model_io.signed_json',return_value=record),patch('satellite_rl.workbench.r6_model_io.model_source_identity',return_value=changed):
                with self.assertRaises(ValueError):verify_model_source_equivalence({},engineering,initial)

    def test_controlled_real_program_and_foreign_bootstrap_scope(self):
        from satellite_rl.workbench.r6_resources import controlled_science_command,belongs_to_archive
        own='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round6/learning/archive'
        foreign='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20418/round6/archive'
        bootstrap=f'/envs/DQN/bin/python3.11 -B -u /immutable/r6_runtime_bootstrap.py --job-dir {foreign}/outputs/jobs/id'
        self.assertTrue(controlled_science_command(bootstrap));self.assertFalse(belongs_to_archive(bootstrap,own))
        for module in ('satellite_rl.workbench.runner','satellite_rl.workbench.ai_search_r6_model_runner','satellite_rl.workbench.r6_model_eval_stage','satellite_rl.workbench.r6_model_d5_stage'):
            command=f'/envs/DQN/bin/python -B -m {module} --request {own}/job/request.json'
            self.assertTrue(controlled_science_command(command));self.assertTrue(belongs_to_archive(command,own))
        for command in (f'python -c pass -m satellite_rl.workbench.runner --job-dir {foreign}',f'python other.py --note /immutable/r6_runtime_bootstrap.py --job-dir {foreign}','sshd python -m satellite_rl.workbench.runner --job-dir /foreign','python - --job-dir /foreign'):
            self.assertFalse(controlled_science_command(command),command)
        self.assertFalse(belongs_to_archive(f'python -m satellite_rl.workbench.runner --job-dir {own}2/job',own))

    def test_node_gate_missing_denominator_and_unpublished_never_credit(self):
        from satellite_rl.workbench.r6_node_audit import node_gate
        metrics=dict(maxload=100.,top10_mean=100.,CV=.1)
        rows=[]
        for i in range(24):
            methods={m:dict(valid=True,published=True,nonfallback=True,identity_closed=True,budget_closed=True,metrics=metrics)for m in ('AI','RULE','RANDOM')}
            rows.append(dict(case_index=i,methods=methods,total_request=1.,costs=dict(AI=1.,RULE=10.,RANDOM=10.),preparation_seconds=0.,teacher_metrics=metrics,teacher_identity_and_cost_closed=True))
        self.assertTrue(node_gate(rows)['passed'])
        with self.assertRaises(ValueError):node_gate(rows[:-1])
        for case in rows:case['methods']['AI'].update(published=False,nonfallback=False,metrics=None,budget_closed=False)
        result=node_gate(rows);self.assertFalse(result['passed']);self.assertEqual(result['denominator'],24);self.assertEqual(result['nonfallback'],0)

    def test_independent_aggregation_reaudits_raw_and_rejects_other_checkpoint(self):
        import scripts.stage1_ai_search_r6model.analyze_node as subject
        from scripts.stage1_ai_search_r6model.plan_builder import NODE_INDICES
        checkpoint={'path':'good','sha256':'goodsha'};model='one';science='science';registry={'cases':[]};cases={};documents={};expected_audit=dict(valid=True,published=True,nonfallback=True,identity_closed=True,budget_closed=True,metrics=dict(maxload=100.,top10_mean=100.,CV=.1))
        for i in NODE_INDICES:
            ident=dict(case_id=str(i),role='development',input_sha256='input')
            registry['cases'].append(dict(input_case_index=i,identity=ident,scene_source={'raw_scene':{'path':'scene','sha256':'scene'}},teacher={'publication_cost':{'path':'oldcost','sha256':'oldcost'}}))
            refs={}
            for method in ('AI','RULE','RANDOM'):
                key=f'row{i}_{method}';ref=dict(path=key,sha256=key,cost_endpoint_ref=dict(path='endpoint_'+key,sha256='endpoint_'+key));refs[method]=ref
                request=dict(checkpoint_ref=checkpoint,model_identity=model)
                documents[key]=dict(identity=ident,method=method,source_commit=science,checkpoint_ref=checkpoint if method=='AI' else None,request=request,result_ref={'path':'result','sha256':'result'},audit=expected_audit)
                documents['endpoint_'+key]=dict(raw_ref=ref,identity=ident,cost_known=True,complete_cold_seconds=1. if method=='AI' else 10.)
            cases[str(i)]=dict(identity=ident,raw_refs=refs)
        manifest=dict(phase='node2000',checkpoint_ref=checkpoint,model_identity=model,scientific_commit=science,cases=cases)
        documents.update(registry=registry,manifest=manifest,oldcost={'components':{'prepare':0.},'unknown_cost':False,'unknown_algorithm_spend':False})
        @contextmanager
        def raw_scene(*args,**kwargs):yield {'positions':np.array([[0.,0.]]),'demands':np.array([1.]),'aircraft_ids':np.array(['a']),'profile':np.array('europe_v1')}
        def read(path,*args):return documents[str(path)]
        prepared={'Ncap':1,'Omega_cap':1.};teacher={'layout':{},'cost':{}}
        with tempfile.TemporaryDirectory(dir='E:/paper_archives') as tmp,patch.object(subject,'pin',side_effect=lambda ref:Path(ref['path'])),patch.object(subject,'pin_tree'),patch.object(subject,'read_json',side_effect=read),patch.object(subject,'original_records',return_value=(prepared,teacher)),patch.object(subject,'identity',side_effect=lambda source:source['identity']),patch.object(subject.np,'load',side_effect=raw_scene),patch.object(subject,'scene_sha',return_value='input'),patch.object(subject,'audit_layout',return_value={'valid':True,'metrics':expected_audit['metrics']}),patch.object(subject,'audit_ledger',return_value={'valid':True}),patch.object(subject,'read_asset',return_value={}),patch.object(subject,'audit_row',return_value=expected_audit) as audit:
            result=subject.analyze([{'path':'manifest','sha256':'manifest'}],{'path':'registry','sha256':'registry'},checkpoint,model,Path(tmp)/'gate.json')
            self.assertTrue(result['passed']);self.assertEqual(audit.call_count,72)
            first=documents[f'row{NODE_INDICES[0]}_AI'];first['request']=dict(first['request'],checkpoint_ref={'path':'other2000','sha256':'other'})
            with self.assertRaises(ValueError):subject.analyze([{'path':'manifest','sha256':'manifest'}],{'path':'registry','sha256':'registry'},checkpoint,model,Path(tmp)/'gate2.json')

if __name__=='__main__':unittest.main()
