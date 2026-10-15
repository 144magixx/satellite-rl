"""Pure fake metadata/seed checks, no real checkpoint arrays or GEO."""
import json
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch
from . import reader_continuation as c

def full_fake():
    """Synthetic checkpoint grammar only; values never enter scientific code."""
    from .admission import INDICES
    source=dict(path='fake_source',sha256='1'*64,bytes=1)
    plan=dict(source_manifest_ref=source,cases=[]);records={};payloads={};audits=[]
    def save(path,value):payloads[str(path)]=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    reader=Path(c.OLD)/'science/reader'
    for position,index in enumerate(INDICES):
        adapter=dict(path='adapter_'+str(index),sha256='2'*64,bytes=1)
        plan['cases'].append(dict(original_index=index,input_adapter_ref=adapter))
        n=(293,216,204,178)[position] if position<4 else 1
        original=dict(path='fake_original_'+str(index),sha256=str(position+3).zfill(64),bytes=1)
        scene=dict(aircraft_ids=['fake_'+str(i) for i in range(n)],positions=[],demands=[],input_sha256='fake')
        origin=dict(identity=dict(fake_index=index))
        record=dict(source_original_input_ref=original,scene=scene,origin=origin,physical_parameters=dict(fake=True))
        records[adapter['path']]=record;parent=c.canonical(origin['identity'])
        nodes={letter:dict(fake_index=index,semantic_SHA=letter*64) for letter in ('a','b')}
        tree=dict(paths={path:dict(semantic_SHA=('a' if path in ('0','00','10') else 'b')*64) for path in ('0','1','00','01','10','11')})
        mother=Path(c.OLD)/'science'/('mother_'+str(index));save(mother/'tree.json',tree)
        for letter,node in nodes.items():save(mother/('node_'+letter*64+'.json'),dict(payload=dict(node=node)))
        if position>=4:continue
        binding=dict(input_content_identity=c.same(original),source_content_identity=c.same(source),physical_content_SHA=c.canonical(record['physical_parameters']),source_scene_content_SHA=c.canonical(scene))
        checkpoint=dict(binding=binding,scene=dict(scene,rays=[]));checkpoint['checkpoint_content_SHA']=c.canonical(checkpoint)
        save(reader/('scene_'+str(index)+'.json'),checkpoint)
        cache=dict(parent_cache_SHA=parent,layouts={},exact={});cache['cache_content_SHA']=c.canonical(cache)
        save(reader/('geometry_'+parent+'.json'),cache)
        for letter,node in nodes.items():
            binding=dict(node_SHA=c.canonical(node),source=c.same(source),input=c.same(original),parent_cache_SHA=parent)
            key='audit_'+c.canonical(binding);audits.append(key)
            charge=dict(rays=0,raypairs=28170 if position==0 and letter=='a' else 0,layouts=5 if position==0 and letter=='a' else 0)
            save(reader/(key+'.json'),dict(schema='R12IndependentAuditCheckpointV2',binding=binding,reserved_cost=charge,result=dict(passed=True,geometry_verified=True,identity_verified=True,validated_layout_SHA_metrics={},validated_layout_SHA_exact={})))
        if position<3:save(reader/('mother_'+str(index)+'.json'),dict(fake=True))
    save(reader/'reader_budget.json',dict(rays=891,raypairs=28170,layouts=5,prior_seconds=10,inflight=None,seconds=11))
    save(reader/'reader_cursor.json',dict(last_completed=audits[-1]));payloads[str(reader/'owner.lock')]=b'fake'
    while len(payloads)<188:save(Path(c.OLD)/'service'/('fake_'+str(len(payloads))+'.json'),{})
    filemap=[dict(remote_path=path,bytes=len(raw),sha256='0'*64) for path,raw in payloads.items()]
    return plan,records,payloads,filemap,reader,audits

class ContinuationTests(unittest.TestCase):
    def qualify_fake(self,change=None):
        plan,records,payloads,filemap,reader,audits=full_fake()
        if change:
            path=str(reader/(audits[0]+'.json'))
            receipt=json.loads(payloads[path])
            if change=='source':receipt['binding']['source']=['wrong',1]
            elif change=='input':receipt['binding']['input']=['wrong',1]
            elif change=='charge':receipt['reserved_cost']['raypairs']+=1
            elif change=='cursor':
                path=str(reader/'reader_cursor.json');receipt=dict(last_completed='unknown')
            payloads[path]=json.dumps(receipt).encode()
        files=[Path(p) for p in payloads if Path(p).parent==reader]
        with patch('scripts.stage1_ai_search_r12.worker.load_case',side_effect=lambda ref:records[ref['path']]),patch.object(c,'pinned',side_effect=lambda ref:payloads[ref['path']]),patch.object(Path,'iterdir',side_effect=lambda:iter(files)):
            return c.qualify(plan,filemap,lambda:False)
    def test_full_synthetic_188_map_four_cache_eight_audit_success(self):
        files,budget=self.qualify_fake()
        self.assertEqual(budget['rays'],891);self.assertEqual(budget['raypairs'],28170);self.assertEqual(budget['layouts'],5)
        self.assertNotIn('owner.lock',files);self.assertEqual(sum(name.startswith('audit_') for name in files),8)
    def test_full_synthetic_source_input_charge_cursor_tamper_rejected(self):
        for change in ('source','input','charge','cursor'):
            with self.subTest(change=change),self.assertRaises(ValueError):self.qualify_fake(change)
    def test_14_17_bounds_and_expired_lease(self):
        grant=dict(attempt_deadline_monotonic_seconds=117,attempt_deadline_unix_seconds=1017)
        with patch.object(c.time,'time',return_value=1000):
            self.assertEqual(c.attempt_bounds(100,100,grant),(116.75,114))
            with self.assertRaises(TimeoutError):c.attempt_bounds(100,118,grant)
    def test_pause_precedence_and_new_geo_stop(self):
        plan={}
        with patch.object(c.time,'monotonic',return_value=100),patch.object(c.time,'time',return_value=1000),patch.object(Path,'exists',return_value=False),patch('scripts.stage1_ai_search_r12.runtime_epoch.validate_authority_and_markers',side_effect=ValueError('new pause')):
            self.assertTrue(c.bounded_stop(plan,114,1014)(force=True))
        with patch.object(c.time,'monotonic',return_value=114),patch.object(c.time,'time',return_value=1000):
            self.assertTrue(c.bounded_stop(plan,114,1014)())
    def test_complete_cache_prefix_and_gap_rejected(self):
        missing=False
        for key in ('a','b','c','d'):missing=c.cached_prefix_step(key,{'a','b'},missing)
        self.assertTrue(missing)
        with self.assertRaises(ValueError):
            missing=False
            for key in ('a','b','c'):missing=c.cached_prefix_step(key,{'a','c'},missing)
    def test_journal_replacement_peak_and_temporary_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter=c.journal_adapter(Path(directory),50,20)
            journal=adapter(Path(c.OLD)/'science/reader')
            try:
                journal.write('cached','a'*10)
                with self.assertRaises(RuntimeError):journal.write('cached','b'*25)
                self.assertFalse((Path(directory)/'cached.pending').exists())
                journal.limit=20
                with self.assertRaises(RuntimeError):journal.write('cached','c'*10)
            finally:journal.close()
    def test_guard_cleanup_only_verified_generations(self):
        first=types.SimpleNamespace(pid=1,create_time=lambda:1,status=lambda:'running',terminate=unittest.mock.Mock(),kill=unittest.mock.Mock())
        reused=types.SimpleNamespace(pid=2,create_time=lambda:9,status=lambda:'running',terminate=unittest.mock.Mock(),kill=unittest.mock.Mock())
        fake=types.SimpleNamespace(Process=lambda pid:{1:first,2:reused}[pid],STATUS_ZOMBIE='zombie',NoSuchProcess=ProcessLookupError,wait_procs=unittest.mock.Mock(return_value=([],[])))
        proc=types.SimpleNamespace(join=unittest.mock.Mock())
        with patch.dict('sys.modules',{'psutil':fake}):self.assertEqual(c.reap_owned(proc,{1:1,2:2},c.time.monotonic()+1),[])
        first.terminate.assert_called_once();reused.terminate.assert_not_called();reused.kill.assert_not_called();proc.join.assert_called_once()
    def test_seed_preserves_paid_counts_and_prior_but_debits103(self):
        budget=dict(rays=891,raypairs=28170,layouts=5,seconds=11,prior_seconds=10,inflight=None)
        files={'reader_budget.json':json.dumps(budget).encode(),'audit_fake.json':b'{"cached":true}'}
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'new';c.seed(dest,files,budget,lambda:False,4096)
            result=json.loads((dest/'reader_budget.json').read_bytes())
            self.assertEqual(result,dict(budget,seconds=103));self.assertFalse((dest/'owner.lock').exists())
            self.assertEqual((dest/'audit_fake.json').read_bytes(),files['audit_fake.json'])
            self.assertEqual(budget['seconds'],11)
    def test_seed_unknown_pause_does_not_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'new'
            with self.assertRaises(TimeoutError):c.seed(dest,{'reader_budget.json':b'{}'},dict(seconds=11),lambda:True,4096)
            self.assertEqual(list(dest.iterdir()),[])
    def test_seed_quota_is_prospective_and_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'new'
            with self.assertRaises(RuntimeError):c.seed(dest,{'fake.json':b'x'*100},dict(seconds=11),lambda:False,50)
            self.assertFalse(dest.exists())
            c.seed(dest,{},dict(seconds=11),lambda:False,4096)
            with self.assertRaises(FileExistsError):c.seed(dest,{},dict(seconds=11),lambda:False,4096)
    def test_incomplete_or_duplicate_filemap_fails_without_GEO(self):
        with self.assertRaisesRegex(ValueError,'188'):c.qualify({},[],lambda:False)
        rows=[dict(remote_path='duplicate',bytes=0,sha256='0'*64)]*188
        with self.assertRaisesRegex(ValueError,'188'):c.qualify({},rows,lambda:False)
    def test_inflight_or_missing_cache_not_qualified_by_null_alone(self):
        reader=Path(c.OLD)/'science/reader'
        budgetpath=str(reader/'reader_budget.json');lockpath=str(reader/'owner.lock')
        paths=[budgetpath,lockpath]+[str(Path(c.OLD)/'science'/('fake_'+str(i)+'.json')) for i in range(186)]
        rows=[dict(remote_path=p,bytes=2,sha256='0'*64) for p in paths]
        for inflight,match in (('unknown','charged budget'),(None,'checkpoint inventory')):
            budget=dict(rays=891,raypairs=28170,layouts=5,prior_seconds=10,inflight=inflight)
            def fake(ref):return json.dumps(budget).encode() if ref['path']==budgetpath else b'0'
            with patch.object(c,'pinned',side_effect=fake),patch.object(Path,'iterdir',return_value=iter([Path(budgetpath),Path(lockpath)])):
                with self.assertRaisesRegex(ValueError,match):c.qualify({},rows,lambda:False)
    def test_qualification_pause_is_before_any_checkpoint_parse(self):
        rows=[dict(remote_path=str(i),bytes=0,sha256='0'*64) for i in range(188)]
        with patch.object(c,'pinned') as pin:
            with self.assertRaises(TimeoutError):c.qualify({},rows,lambda:True)
            pin.assert_not_called()

if __name__=='__main__':unittest.main()
