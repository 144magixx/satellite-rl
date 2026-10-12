"""Bounded synthetic streams; no production archive scans or evaluations."""
import json
import tempfile
import tracemalloc
import unittest
import zipfile
from pathlib import Path
from scripts.stage1_geoload.strict_audit import _read_case,audit_search,audit_layout


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'synthetic.zip'
        self.scene=dict(positions=[[45,13]],demands=[2]);self.layout=dict(centers=[[45,13]],assignment=[0])
        self.metric=audit_layout(self.scene,self.layout)
    def tearDown(self):self.temp.cleanup()
    def row(self,count=5000):
        events=[dict(attempt=i+1,elapsed_seconds=.001*(i+1),started_seconds=.001*i,
            reason='objective_reject',candidate_metrics=self.metric,
            coverage_evidence=dict(valid=True,loads=[2]),details=dict(padding='x'*200)) for i in range(count)]
        return dict(method='JOINT',layout=self.layout,metrics=self.metric,initial_metrics=self.metric,journal=events,
                    budgets=dict(attempts=count,accepted=0,late_candidates=0,seconds_limit=100,candidate_limit=60000),
                    ranking_journal=[dict(unused='x'*1000)]*count)
    def archive(self,value):
        with zipfile.ZipFile(self.path,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('case.json',json.dumps(value))
        class Archive:pass
        a=Archive();a.zip=zipfile.ZipFile(self.path);return a

    def test_all_events_replayed_without_retaining_journal(self):
        archive=self.archive(self.row())
        try:
            tracemalloc.start()
            row=_read_case(archive,'case.json')
            self.assertFalse(isinstance(row['journal'],list))
            result,counts=audit_search(self.scene,row,compact=True)
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            self.assertEqual(counts,dict(known_completed_candidates=5000,accepted=0,completed_late=0))
            self.assertEqual(result,self.metric)
            self.assertLess(peak,8*(1<<20))
        finally:archive.zip.close()

    def test_variable_entries_streams_select_exact_entry(self):
        value=dict(entries=[self.row(2),self.row(3)])
        archive=self.archive(value)
        try:
            row=_read_case(archive,'case.json')
            self.assertEqual(len(list(row['entries'][0]['journal'])),2)
            self.assertEqual(len(list(row['entries'][1]['journal'])),3)
            _,counts=audit_search(self.scene,row['entries'][1],compact=True)
            self.assertEqual(counts['known_completed_candidates'],3)
        finally:archive.zip.close()

    def test_late_tamper_at_final_event_not_omitted(self):
        row=self.row(30);row['journal'][-1]['reason']='accepted'
        row['journal'][-1]['elapsed_seconds']=101
        archive=self.archive(row)
        try:
            with self.assertRaisesRegex(ValueError,'late'):
                audit_search(self.scene,_read_case(archive,'case.json'),compact=True)
        finally:archive.zip.close()

    def test_stream_and_list_produce_same_audit_counts(self):
        original=self.row(25);archive=self.archive(original)
        try:
            a,old=audit_search(self.scene,original)
            b,new=audit_search(self.scene,_read_case(archive,'case.json'),compact=True)
            self.assertEqual(a,b);self.assertEqual(new['known_completed_candidates'],len(old))
            self.assertEqual(new['accepted'],sum(x['accepted'] for x in old))
        finally:archive.zip.close()

    def test_teacher_commit_in_middle_of_pool_uses_pre_pool_delta(self):
        scene=dict(positions=[[45,13]]*4,demands=[1,1,1,1])
        initial=dict(centers=[[45,13],[45,13]],assignment=[0,0,0,1])
        best=dict(centers=[[45,13],[45,13]],assignment=[0,0,1,1])
        values=audit_layout(scene,best);old=audit_layout(scene,initial)
        common=dict(stage=0,started=.01,ended=.02,late=False,geometry=dict(valid=True,loads=[2,2]),metrics=values)
        first=dict(common,pool_committed=True,delta=dict(users=[2],owners=[1],beams=[],centers=[]))
        # Another equally balanced trial is generated against initial, not best.
        second=dict(common,started=.03,ended=.04,pool_committed=False,
                    delta=dict(users=[0],owners=[1],beams=[],centers=[]))
        row=dict(method='JOINT',input=dict(initial=initial),layout=best,metrics=values,
            journal=[first,second],budget=dict(attempts=2,limit_seconds=10,late_completed_candidates=0))
        _,counts=audit_search(scene,row,compact=True)
        self.assertEqual(counts['accepted'],1)

    def test_all_three_reference_nodes_survive_streaming(self):
        row=self.row(25)
        row['reference_nodes']=[dict(kind='paired',seconds_limit=t,candidate_limit=c,
            state_seconds=0,state_attempts=0,layout=self.layout,metrics=self.metric)
            for t,c in [(2,4096),(10,20480),(30,61440)]]
        archive=self.archive(row)
        try:
            streamed=_read_case(archive,'case.json')
            self.assertEqual(len(streamed['reference_nodes']),3)
            audit_search(self.scene,streamed,compact=True)
            streamed['reference_nodes'][-1]['metrics']['P']=2
            with self.assertRaisesRegex(ValueError,'metric differs'):
                audit_search(self.scene,streamed,compact=True)
        finally:archive.zip.close()


if __name__=='__main__':unittest.main()
