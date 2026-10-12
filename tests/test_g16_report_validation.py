"""Synthetic science495 _train directory/schema checks; no science evaluation."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from scripts.stage1_geoload.report import validation_tables
from scripts.stage1_geoload.strict_audit import audit_layout


class ValidationTablesTests(unittest.TestCase):
    def setUp(self):
        scene=dict(positions=[[45,13]],demands=[4])
        layout=dict(centers=[[45,13]],assignment=[0])
        metric=audit_layout(scene,layout)
        profiles=('base','one_hotspot','two_hotspots','edge_heavy')
        self.scenes={}
        for mother in range(20):
            name='validation:2024-02-13T'+str(mother).zfill(2)+':02:00Z'
            for profile in profiles:
                key=name+':'+profile
                self.scenes[key]=dict(scene,mother_id=name,demand_profile=profile)
        class Data:
            pass
        self.data=Data();self.data.ids=list(self.scenes);self.data.ref={'sha256':'input_sha'}
        self.data.get=lambda key:self.scenes[key]
        self.initials={key:metric for key in self.scenes}
        class Archive:
            def read(self,name):return copy.deepcopy(self.content[name])
        self.archive=Archive();self.archive.path=Path('synthetic.zip');self.archive.content={}
        self.archive.job=dict(scientific_commit='science495',references=dict(teacher=dict(sha256='teacher_sha')))
        refs={}
        for node in (2500,5000,7500,10000):
            prefix='model/validation_'+str(node).zfill(5)+'/'
            index=[]
            for key,case in self.scenes.items():
                member=prefix+hashlib.sha256((key+'\0MLM').encode()).hexdigest()+'.json'
                row=dict(case_id=key,mother_id=case['mother_id'],demand_profile=case['demand_profile'],
                    layout=layout,metrics=metric,journal=[],budgets=dict(seconds=.01,attempts=0,
                        accepted=0,geometry_calls=0,late_candidates=0,seconds_limit=2))
                self.archive.content[member]=row
                sha=hashlib.sha256(json.dumps(row).encode()).hexdigest()
                refs[member]=dict(sha256=sha);index.append(dict(case_id=key,sha256=sha))
            done=prefix+'node_done.json'
            self.archive.content[done]=dict(update=node,count=80,index=index,validation_sha256='input_sha')
            refs[done]=dict(sha256='node_done_'+str(node))
        refs['model/checkpoint.pt']=dict(sha256='cp_sha');refs['model/final.pt']=dict(sha256='final_sha')
        self.archive.manifest=dict(members=refs)
        protocol=dict(scientific_commit='science495',validation_sha256='input_sha',teacher_source_sha256='teacher_sha')
        sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        self.cp=dict(format='s1_g16_geometry_mlm_v1',update=10000,seed=51001,from_scratch=True,protocol_sha256=sha,dataset_sha256='evidence_sha',
                     protocol=protocol,nodes=[2500,5000,7500,10000])
        self.final={k:v for k,v in self.cp.items() if k not in ('protocol','nodes')}

    def tables(self):
        return validation_tables(self.archive,self.data,self.initials,lambda _: (self.cp,self.final))

    def test_actual_directory_schema_and_mother_aggregation(self):
        cases,curves,issues=self.tables()
        self.assertEqual(issues,[])
        self.assertEqual(len(cases),320);self.assertEqual(len(curves),4)
        self.assertEqual([r['node'] for r in curves],[2500,5000,7500,10000])
        self.assertTrue(all(r['mothers']==20 and r['cases']==80 and r['P']==1 for r in curves))
        self.assertAlmostEqual(curves[0]['search_seconds_total'],.8)
        self.assertEqual(cases[0]['checkpoint_sha256'],'cp_sha')
        self.assertEqual(cases[0]['scope'],'training_diagnostic_not_common_timing')

    def test_missing_node_never_fabricated(self):
        del self.archive.content['model/validation_02500/node_done.json']
        cases,curves,issues=self.tables()
        self.assertTrue(issues);self.assertEqual(len(curves),3)
        self.assertFalse(any(r['node']==2500 for r in cases))

    def test_wrong_node_member_sha_rejected(self):
        member=next(n for n in self.archive.manifest['members'] if n.startswith('model/validation_02500/') and not n.endswith('node_done.json'))
        self.archive.manifest['members'][member]['sha256']='wrong'
        cases,curves,issues=self.tables()
        self.assertTrue(issues);self.assertEqual(len(curves),3)

    def test_duplicate_case_node_rejected(self):
        done=self.archive.content['model/validation_02500/node_done.json']
        done['index'][-1]=done['index'][0]
        _,curves,issues=self.tables()
        self.assertTrue(issues);self.assertEqual(len(curves),3)

    def test_endpoint_checkpoint_protocol_source_mismatch_rejected(self):
        self.final['protocol_sha256']='wrong'
        cases,curves,issues=self.tables()
        self.assertTrue(issues);self.assertEqual(cases,[]);self.assertEqual(curves,[])

    def test_checkpoint_missing_completed_node_rejected(self):
        self.cp['nodes']=[2500,5000,7500]
        cases,curves,issues=self.tables()
        self.assertTrue(issues);self.assertEqual(cases,[]);self.assertEqual(curves,[])


if __name__=='__main__':unittest.main()
