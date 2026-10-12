"""Static round expansion and no-duplicate/four-profile shard checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from scripts.stage1_ai_search.plan_builder import build,audit_plan,add_training_contract


class AiSearchPlanTests(unittest.TestCase):
    def fixture(self):
        design=Path(__file__).resolve().parents[1]/'scripts/stage1_ai_search/round1_design.json'
        ref={'path':str(design),'sha256':hashlib.sha256(design.read_bytes()).hexdigest()}
        profiles=['base','one_hotspot','two_hotspots','edge_heavy'];identities={};sources={'sources':{}}
        for role,count in [('train',576),('development',20),('validation',20)]:
            mothers=[f'{role}:{i:04}' for i in range(count)]
            identities[role]={'mother_ids':[m for m in mothers for p in profiles],
                'profiles':profiles*count,'case_ids':[m+':'+p for m in mothers for p in profiles]}
            sources['sources'][role]={'path':'/immutable/'+role+'.npz','sha256':'a'*64,
                'case_count':count*4,'mother_count':count,'format_version':'geoload_aircraft_v1'}
        return ref,sources,identities

    def test_full_initial_matrix_and_teacher_ordinals_are_unique(self):
        ref,sources,identities=self.fixture()
        plan,audit=build('a'*40,ref,sources,identities,('development','reference','bridge','teacher'))
        self.assertEqual(audit['trajectory_counts'],{'development':480,'reference':80,'bridge':320,'teacher':768})
        self.assertEqual(len(plan['teacher_mother_order']),192)
        self.assertEqual(len(set(plan['teacher_mother_order'])),192)
        for task in plan['tasks'].values():
            if task['mode']=='teacher':self.assertEqual(task['config']['search']['max_attempts'],3072)

    def test_duplicate_and_split_mother_are_rejected(self):
        ref,sources,identities=self.fixture();plan,audit=build('a'*40,ref,sources,identities)
        duplicate=deepcopy(plan);key=next(iter(duplicate['tasks']));duplicate['tasks']['duplicate']=deepcopy(duplicate['tasks'][key])
        with self.assertRaisesRegex(ValueError,'duplicate semantic'):audit_plan(duplicate,identities)
        split=deepcopy(plan);keys=list(split['tasks']);first=split['tasks'][keys[0]];other=split['tasks'][keys[1]]
        moved=first['case_indices'].pop(0);other['case_indices'].append(moved)
        with self.assertRaisesRegex(ValueError,'mother profiles split'):audit_plan(split,identities)

    def test_asynchronous_nodes_keep_240_extra_and_one_final560_matrix(self):
        ref,sources,identities=self.fixture();plan,audit=build('a'*40,ref,sources,identities,('teacher',))
        validation={'path':'/new/validation.npz','sha256':'b'*64,'role':'validation','count':80}
        plan=add_training_contract(plan,{'path':'/new/teacher.json','sha256':'c'*64},
            {'path':'/new/gate.json','sha256':'d'*64},validation,identities['validation'])
        self.assertEqual(list(plan['tasks']),['r1_train_ranker'])
        self.assertEqual(plan['node_budget_audit']['counts'],{'2500':80,'5000':80,'7500':80,'10000':560})
        for node in ('2500','5000','7500'):
            self.assertNotIn(20418,{t['executor_port'] for t in plan['node_templates'][node].values()})
            self.assertTrue(all(t['methods']==['AI'] for t in plan['node_templates'][node].values()))
        self.assertTrue(all(len(t['methods'])==7 for t in plan['node_templates']['10000'].values()))


if __name__=='__main__':unittest.main()
