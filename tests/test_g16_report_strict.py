import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from scripts.stage1_geoload.strict_audit import (
    AuditError, JobZip, audit_layout, audit_search, load_metrics, paired_bootstrap, digest)


class StrictReportTests(unittest.TestCase):
    def setUp(self):
        self.scene=dict(positions=[[45,13],[45,13]],demands=[1,3])
        self.layout=dict(centers=[[45,13],[45,13]],assignment=[0,1])
        self.metric=audit_layout(self.scene,self.layout)

    def test_independent_load_algebra(self):
        self.assertEqual(self.metric['P'],1.5)
        self.assertEqual(self.metric['Q'],1.25)
        self.assertEqual(self.metric['CV'],.5)
        self.assertEqual(load_metrics([0,0])['P'],None)

    def test_geometry_owner_and_demand_rejected(self):
        for layout in [dict(centers=[[45,13]],assignment=[0,-1]),
                       dict(centers=[[45,13]],assignment=[0,.5]),
                       dict(centers=[[25,-40]],assignment=[0,0]),
                       dict(centers=[[45,13],[45,13]],assignment=[0,0])]:
            with self.assertRaises(AuditError):audit_layout(self.scene,layout)
        bad=copy.deepcopy(self.metric);bad['P']=2
        with self.assertRaises(AuditError):audit_layout(self.scene,self.layout,bad)

    def row(self):
        return dict(method='JOINT',layout=self.layout,metrics=self.metric,initial_metrics=self.metric,
                    journal=[],budgets=dict(attempts=0,accepted=0,seconds_limit=2))

    def test_late_acceptance_forbidden(self):
        row=self.row();row['journal']=[dict(elapsed_seconds=2.01,reason='accepted',
                candidate_metrics=self.metric,coverage_evidence=dict(valid=True,loads=[1,3]))]
        row['budgets']['attempts']=1
        with self.assertRaisesRegex(AuditError,'late'):audit_search(self.scene,row)

    def test_reference_no_future(self):
        row=self.row();row['reference_nodes']=[dict(kind='paired',seconds_limit=2,candidate_limit=4096,
                   state_seconds=2.001,state_attempts=0,layout=self.layout,metrics=self.metric)]
        with self.assertRaisesRegex(AuditError,'future'):audit_search(self.scene,row)

    def test_worse_acceptance_rejected(self):
        row=self.row();row['journal']=[dict(elapsed_seconds=.1,reason='accepted',candidate_metrics=self.metric,
                                          coverage_evidence=dict(valid=True,loads=[1,3]))]
        with self.assertRaisesRegex(AuditError,'acceptance'):audit_search(self.scene,row)

    def test_mother_bootstrap_four_profiles_and_strata(self):
        rows=[]
        for mother,date in [('a','2024-02-14'),('b','2024-02-15')]:
            for profile in ('base','one_hotspot','two_hotspots','edge_heavy'):
                for method,p in [('MLM',1.8),('rule',2.)]:
                    rows.append(dict(mother_id=mother,date=date,demand_profile=profile,method=method,P=p,outcome='completed'))
        value=paired_bootstrap(rows,'MLM','rule')
        self.assertAlmostEqual(value['lower95'],.1)
        self.assertEqual(value['draws'],5000)
        self.assertFalse(value['broad_significance_claim_allowed'])
        with self.assertRaises(AuditError):paired_bootstrap(rows[:-1],'MLM','rule')

    def test_zip_receipt_and_manifest_stream_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'job.zip';source='a'*40
            contents={'job/job.json':json.dumps(dict(run_id='r',scientific_commit=source)).encode(),
                      'job/status.json':b'{"stage":"completed"}'}
            manifest=dict(group='S1-G16',run_id='r',scientific_commit=source,
                          members={n:dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in contents.items()})
            with zipfile.ZipFile(path,'w') as z:
                for n,b in contents.items():z.writestr(n,b)
                z.writestr('archive_manifest.json',json.dumps(manifest))
            receipt=dict(local_path=str(path),sha256=digest(path),bytes=path.stat().st_size,verified=True,run_id='r')
            archive=JobZip(receipt,source);archive.close()
            receipt['sha256']='0'*64
            with self.assertRaises(AuditError):JobZip(receipt,source)

    def test_incomplete_snapshot_never_formal_ready(self):
        from scripts.stage1_geoload.report import build
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json';state=root/'state.json'
            plan.write_text(json.dumps(dict(group='S1-G16',scientific_commit='a'*40,
                preparations={},mother_counts=dict(train=576,development=20,validation=20,test=40))))
            state.write_text(json.dumps(dict(group='S1-G16',plan_sha256=digest(plan),tasks={})))
            result=build(plan,state,root/'report')
            self.assertFalse(result['formal_ready'])
            self.assertEqual(result['status'],'data_unknown')
            self.assertTrue((root/'report'/'candidate.csv').exists())


if __name__=='__main__':unittest.main()
