"""Partial worker35 / complete worker36 can fold old pinned bytes, never new ones."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.stage1_nondrl12 import execution_sources as sources
from scripts.stage1_nondrl12.strict_audit import Evidence
from test_nondrl_execution_sources import lineage_fixture, raw, sha


def folded_fixture(root):
    job, contract, payloads = lineage_fixture(root)
    provenance = job['source_repair_contract']['origin_provenance']
    parent_before = {'format':'s1_nondrl_teacher_v1','route':'01','provenance':provenance,
                     'rows':492,'problems':35,'trials':123,'elapsed_seconds':43.29680798202753,
                     'shards':[],'problem_receipts':[],
                     'reservations':{str(i):{'max_trials':12,'max_seconds':1.} for i in (35,36)}}
    current = deepcopy(parent_before)
    current.update(rows=512,problems=37,trials=128,elapsed_seconds=45.,reservations={})
    current['execution_segments']=[{'execution_source':{
          'git_commit':job['git_commit'],'source_sha256':job['source_history'][-1]['source_manifest_sha256'],
          'repair_id':sources.REPAIR_ID,'origin_commit':sources.ORIGIN_COMMIT}}]
    artifacts = []
    def old_artifact(name,value,kind='artifact'):
        payloads[name]=value;payloads['source_lineage/origin/artifacts/'+name]=value
        artifacts.append({'path':'/original/'+name,'relative_path':name,'sha256':sha(value),'kind':kind})
    for i,field in ((35,'shards'),(36,'problem_receipts')):
        folder=f'teacher/workers/problem_{i:06d}'
        name='samples_00000.jsonl.gz' if field=='shards' else 'problem_receipts/problem_000000.json.gz'
        value=gzip.compress(raw({'route':'01','mother_id':f'mother{i}','old_complete_bytes':True}),mtime=0)
        record={'path':name,'sha256':sha(value)}
        if field=='shards':record['rows']=4
        else:record.update(mother_id=f'mother{i}',demand_multiplier=1.)
        child={'format':'s1_nondrl_teacher_v1','route':'01','provenance':provenance,
               'status':'generating' if i==35 else 'completed','problems':0 if i==35 else 1,
               'rows':4,'trials':1,'shards':[record] if field=='shards' else [],
               'problem_receipts':[record] if field=='problem_receipts' else []}
        old_artifact(folder+'/'+name,value)
        old_artifact(folder+'/manifest.json',raw(child),'teacher_manifest')
        current[field].append({**record,'path':f'workers/problem_{i:06d}/'+name})
    old_artifact('teacher/manifest.json',raw(parent_before),'teacher_manifest')
    payloads['teacher/manifest.json']=raw(current)
    job['source_repair_contract']['artifacts']=artifacts
    contract['runs'][job['run_id']]['artifacts']=artifacts
    payload=raw(contract);payloads['source_lineage/contract.json']=payload
    pin=sha(payload);job['source_repair_contract']['contract_sha256']=pin
    job['source_history'][-1]['contract_sha256']=pin
    current['execution_segments'][0]['execution_source']['contract_sha256']=pin
    payloads['teacher/manifest.json']=raw(current);payloads['job.json']=raw(job)
    for name,value in payloads.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    return job,parent_before,current,payloads


class DeferredOriginFoldTests(unittest.TestCase):
    def test_actual_shape_partial35_and_completed36_references_are_origin_not_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,before,current,_=folded_fixture(root)
            proof=sources.audit_lineage(job,Evidence(root))
            folds=proof['deferred_original_worker_folds']
            self.assertEqual(len(folds),2)
            self.assertEqual({r['original_worker_index'] for r in folds},{35,36})
            self.assertTrue(all(r['execution_commit']==sources.ORIGIN_COMMIT for r in folds))
            self.assertEqual(json.loads((root/'teacher/manifest.json').read_bytes()),current)
            self.assertEqual(before['rows'],492)

    def test_current_bytes_changed_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,_,_,_=folded_fixture(root)
            (root/'teacher/workers/problem_000035/samples_00000.jsonl.gz').write_bytes(b'new unmarked scientific bytes')
            with self.assertRaisesRegex(ValueError,'immutable origin|artifact bytes changed'):
                sources.audit_lineage(job,Evidence(root))

    def test_wrong_reference_metadata_is_rejected_even_when_raw_bytes_are_pinned(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,_,current,_=folded_fixture(root)
            current['shards'][0]['rows']=5
            (root/'teacher/manifest.json').write_bytes(raw(current))
            with self.assertRaisesRegex(ValueError,'original worker reference'):
                sources.audit_lineage(job,Evidence(root))

    def test_unpinned_new_shard_cannot_pass_as_old(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,_,current,_=folded_fixture(root)
            value=gzip.compress(b'new execution without a marker',mtime=0)
            path=root/'teacher/workers/problem_000035/new.jsonl.gz';path.write_bytes(value)
            current['shards'][0].update(path='workers/problem_000035/new.jsonl.gz',sha256=sha(value))
            (root/'teacher/manifest.json').write_bytes(raw(current))
            with self.assertRaisesRegex(ValueError,'signed original artifact'):
                sources.audit_lineage(job,Evidence(root))

    def test_original_bytes_without_original_manifest_reference_still_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,before,current,_=folded_fixture(root)
            pins={a['relative_path']:a for a in job['source_repair_contract']['artifacts']}
            child='teacher/workers/problem_000035/manifest.json'
            payload=json.loads((root/'source_lineage/origin/artifacts'/child).read_bytes());payload['shards']=[]
            updated=raw(payload);(root/'source_lineage/origin/artifacts'/child).write_bytes(updated)
            pins[child]['sha256']=sha(updated)
            with self.assertRaisesRegex(ValueError,'original worker reference'):
                sources.deferred_origin_fold(job,Evidence(root),'teacher/manifest.json',before,'shards',current['shards'][0],pins)

    def test_no_pending_reservation_or_wrong_field_or_traversal_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,before,current,_=folded_fixture(root)
            pins={a['relative_path']:a for a in job['source_repair_contract']['artifacts']}
            for altered,field,record in (({**before,'reservations':{}},'shards',current['shards'][0]),
                                        (before,'rows',current['shards'][0]),
                                        (before,'shards',{**current['shards'][0],'path':'../workers/problem_000035/x'})):
                with self.subTest(field=field,path=record['path']),self.assertRaises(ValueError):
                    sources.deferred_origin_fold(job,Evidence(root),'teacher/manifest.json',altered,field,record,pins)


if __name__=='__main__':unittest.main()
