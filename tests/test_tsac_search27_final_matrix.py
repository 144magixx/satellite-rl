import unittest
from scripts.tsac_search27.final_matrix import build_candidate

class FinalContractTests(unittest.TestCase):
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
