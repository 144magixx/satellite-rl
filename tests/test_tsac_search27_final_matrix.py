import unittest
from scripts.tsac_search27.final_matrix import build_candidate

class FinalContractTests(unittest.TestCase):
    def test_json_roundtrip_catalog_candidate_lock_and_policy_metadata(self):
        """No model, PHY, training, or remote calls: exercise transport boundaries."""
        import copy,json
        from unittest.mock import patch
        from scripts.tsac_search27.final_matrix import policy_reference_catalog
        from satellite_rl.tsac_search27.known_evaluation import validate_lock,validate_policy_metadata
        from satellite_rl.tsac_search27.agent import schema_identity
        from satellite_rl.t0_width.agent import schema_identity as old_schema
        from satellite_rl.t0_width.logging import content_sha
        wire=lambda x:json.loads(json.dumps(x))
        recipe=self.recipe([]);science=recipe['science'];plans={}
        for port,variant in zip((20418,20424,20627),('B','E','LP')):
            science['jobs'][variant].update(port=port,task_key='train_'+variant)
            config=dict(variant=variant,group='S10',run_id='train_'+variant,model_dir='/models/train_'+variant,source_sha='CORE9f',data_manifest_sha='logical6343')
            plans[str(port)]=dict(source_commit=science['source_commit'],tasks={'train_'+variant:dict(config=config)})
        original=dict(group='S10',source_sha='CORE9f',data_manifest_sha='history5d')
        old=dict(schema='W26AuthorizedPlanV1',authorized=True,source_root='/oldsource',source_manifest={'a':'sha'},tasks={'S10':dict(mode='train',config=original)})
        confirmation=dict(tasks={'evaluate':dict(mode='evaluate',config=dict(terminal_policies={'S10':dict(path='/models/oldrun/policy_150000.pt',sha256='p')},confirmation_manifest_ref=recipe['known_manifest_ref'],confirmation_unlock=recipe['w26_unlock']))})
        catalog=policy_reference_catalog(science,plans,old,confirmation,50000)
        for name,item in catalog['inventory_candidates'].items():
            recipe['inventory'][name].update(schema=wire(item['schema']),config_SHA=item['config_SHA'])
        candidate=build_candidate(**recipe)
        producer=dict(original,run_id='oldrun',model_dir='/models/oldrun')
        oldref=recipe['inventory']['HIST']['original_plan_ref']
        job=dict(config_sha256=content_sha(producer),status=dict(run_id='oldrun',model_dir='/models/oldrun',plan_sha256=oldref['sha256']),source_manifest=old['source_manifest'],source_root=old['source_root'])
        config=candidate['config'];lock=candidate['lock_candidate'];lock['root_approved']=True
        for key,path in [('historical_producer_config_ref','producer'),('historical_producer_job_ref','job')]:config[key]={'path':path};lock[key]=config[key]
        config=wire(config);lock=wire(lock)
        refs={'futurelock':lock,'decision':recipe['decision'],'oldplan':old,'producer':producer,'job':job}
        with patch('satellite_rl.tsac_search27.known_evaluation.read_json_ref',side_effect=lambda ref:refs[ref['path']]):
            validate_lock(config)
        for name,p in config['policies'].items():
            expected=old_schema('S10') if name=='HIST' else schema_identity(name)
            policy=dict(schema=expected,env_step=p['env_step'],config_sha=p['config_SHA'],source_sha=p['source_SHA'],data_manifest_sha=p['data_manifest_SHA'])
            validate_policy_metadata(policy,p,expected)
            for key in ('env_step','config_sha','source_sha','data_manifest_sha'):
                bad=dict(policy);bad[key]='wrong'
                with self.assertRaises(ValueError):validate_policy_metadata(bad,p,expected)
            bad=copy.deepcopy(policy);bad['schema']['dimension']+=1
            with self.assertRaises(ValueError):validate_policy_metadata(bad,p,expected)
            bad=copy.deepcopy(policy);bad['schema']['extra_field']='forbidden'
            with self.assertRaises(ValueError):validate_policy_metadata(bad,p,expected)

    def recipe(self,passed):
        node=150000 if passed else 50000;names=['B']+passed+['HIST'] if passed else ['B','E','LP','HIST']
        science=dict(source_commit='a'*40,jobs={m:{'run_id':'train_'+m} for m in ('B','E','LP')})
        inventory={m:dict(root_approved=True,env_step=node,fixed_node_not_best=True,owner_exit_verified=True,policy_file_SHA_verified=True,
                      executor_port=20418,stage='completed' if passed else 'pilot_complete',origin_run_id='train_'+m,source_commit='a'*40,
                      source_SHA='CORE9f',data_manifest_SHA='logical6343' if m!='HIST' else 'history5d',policy_ref={'path':'policy_'+m,'sha256':'p'},schema=(__import__('satellite_rl.t0_width.agent',fromlist=['schema_identity']).schema_identity('S10') if m=='HIST' else __import__('satellite_rl.tsac_search27.agent',fromlist=['schema_identity']).schema_identity(m)),config_SHA='c',group='S10',historical_W26=m=='HIST',original_plan_ref={'path':'oldplan','sha256':'oldsha'}) for m in names}
        manifest={'path':'old40','sha256':'m'}
        return dict(ready=dict(stage='ready_for_known_matrix',evaluation_started=False,node=node,passed_variants=passed),
             decision=dict(root_approved=True,node=50000,passed_variants=passed,known40_opened=False),science=science,inventory=inventory,
             evaluation_commit='b'*40,executor_port=20418,rootpauseepoch='e',run_id='known',decision_ref={'path':'decision','sha256':'d'},
             terminal_lock_ref={'path':'futurelock','sha256':'future'},known_manifest_ref=manifest,
             w26_unlock={'manifest_ref':manifest,'manifest_SHA':'m','receipt_ref':{'path':'unlock','sha256':'u'}})
    def test_exact_final_matrix_cases_and_historical_provenance(self):
        for passed,budget in [([],680),(['E'],560),(['LP'],560),(['E','LP'],680)]:
            result=build_candidate(**self.recipe(passed));self.assertEqual(result['budget_episodes'],budget)
            self.assertEqual(result['config']['policies']['HIST']['data_manifest_SHA'],'history5d')
            self.assertEqual(result['config']['source_sha'],'CORE9f');self.assertFalse(result['lock_candidate']['root_approved'])
    def test_export_audits_paired_metadata_steps_and_timing_without_environment(self):
        import json,gzip,tempfile,hashlib
        from pathlib import Path
        from unittest.mock import patch
        from scripts.tsac_search27.final_matrix import export_results
        from satellite_rl.tsac_search27.known_evaluation import jobs_for
        from satellite_rl.t0_width.logging import content_sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);logs=root/'logs';logs.mkdir()
            metadata={'rows':[dict(case_id=str(m),layout_content_SHA='layout'+str(m),case_content_SHA='case'+str(m)) for m in range(40)]}
            meta=root/'known.json';meta.write_text(json.dumps(metadata))
            config=dict(node=150000,budget_episodes=560,policies={'B':{},'E':{},'HIST':{}},known_manifest_ref={'path':str(meta),'sha256':hashlib.sha256(meta.read_bytes()).hexdigest()},
                        training_source_commit='a'*40,evaluation_source_commit='b'*40,source_sha='core',data_manifest_sha='logical')
            rows=[];steps=[]
            for m in range(40):
                identity=dict(case_id=str(m),layout_SHA='layout'+str(m),aircraft_SHA='case'+str(m),physics_SHA='physics')
                for method,repeat in jobs_for(['E'],150000):
                    base=dict(mother=m,repeat=repeat,node=150000,phase='known_'+method,identity=identity,action_seed=275000+m+repeat*1000000)
                    rows.append(dict(base,allocation_wire=dict(beam_ids=['beam'],actual_start_slot=[0],actual_slot_count=[1],actual_whole_beam_power_w=[5]),
                                     timing=dict(total_s=1,algorithm_with_terminal_s=1,wall_to_terminal_s=.5,resumed_from_WAL=True,previous_algorithm_s=.7)))
                    steps.append(dict(base,beam_id='beam',requested_action=[0,0,0],actual_action=dict(start=0,count=1,power_w=5)))
            for name,data in [('evaluations',rows),('steps',steps)]:
                with gzip.open(logs/(name+'.jsonl.gz'),'wt',encoding='utf8') as stream:
                    for row in data:stream.write(json.dumps(row)+'\n')
            offsets={'evaluations':(logs/'evaluations.jsonl.gz').stat().st_size,'steps':(logs/'steps.jsonl.gz').stat().st_size,'episodes':0}
            (root/'known_cursor.json').write_text(json.dumps(dict(cursor=560,task_sha=content_sha(config),log_offsets=offsets)))
            metrics=dict(SI_mean=.95,SGM=.6,total_power_w=5,throughput_bps=1,complete=1,constraint_violations=0)
            with patch('scripts.tsac_search27.final_matrix.audit_terminal',return_value=(metrics,None,None)),patch('scripts.tsac_search27.final_matrix.decode_array',side_effect=lambda value:value):
                summary=export_results(config,root,root/'report',meta)
                self.assertEqual(summary['actual_decisions'],560);self.assertEqual(summary['timing']['B']['resumed_episodes'],120)
                self.assertEqual(len(summary['paired_contrasts']),10)
            with patch('scripts.tsac_search27.final_matrix.audit_terminal',side_effect=[(metrics,('demand1',),None),(metrics,('demand2',),None)]),patch('scripts.tsac_search27.final_matrix.decode_array',side_effect=lambda value:value):
                with self.assertRaisesRegex(ValueError,'demand/ID signature'):export_results(config,root,root/'bad-report',meta)

    def test_mixed_nodes_or_wrong_run_rejected(self):
        for field,value in [('env_step',50000),('origin_run_id','other'),('source_commit','c'*40),('owner_exit_verified',False)]:
            recipe=self.recipe(['E']);recipe['inventory']['E'][field]=value
            with self.assertRaises(ValueError):build_candidate(**recipe)

if __name__=='__main__':unittest.main()
