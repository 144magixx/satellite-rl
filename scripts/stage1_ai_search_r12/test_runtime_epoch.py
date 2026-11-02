"""Pure operational tests: no geometry, solver, model, SSH or child execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from . import runtime_epoch as e

def fixture():
    epoch=dict(schema='R12ExactOperationalEpochV1',epoch='r12_resume_20261015_0858',
        resume_ref=dict(path='authority',sha256=e.AUTH_SHA,bytes=965),
        historical_bindings_ref=dict(path='bindings',sha256=e.BINDINGS_SHA,bytes=3178),boot_id='boot',
        issued_unix_seconds=10000,issued_monotonic_seconds=100,
        deadline_unix_seconds=12550,deadline_monotonic_seconds=2650,
        stop_new_unix_seconds=12430,stop_new_monotonic_seconds=2530)
    plan=dict(runtime_epoch=epoch,watched_pause_paths=list(e.WATCH),own_stop_marker=e.ROOT+'/stop.request',
        executor_port=20473,run_id='r12_v2_scientific_20261015_20473_01',
        archive_root=e.ROOT+'/science',control_root=e.ROOT+'/service',request_id='request_resume_20261015_0858_01')
    markers={p:dict(present=True,sha256=sha,bytes=size) for p,(sha,size) in e.HISTORICAL.items()}
    markers.update({p:None for p in e.WATCH if p not in markers})
    authority=dict(schema='S1AI17ExplicitHumanResumeAuthorityV1',group='S1-AI17',current_round='R12',
        observed_after_0700_pause=True,user_instruction_verbatim='继续',future_pause_always_wins=True)
    binding=dict(resume_ref=dict(sha256=e.AUTH_SHA),historical_pause_refs=[dict(absolute_path=p,sha256=sha,bytes=size) for p,(sha,size) in e.HISTORICAL.items()])
    def docs(ref,sha):
        if ref.get('sha256')!=sha:raise ValueError('binding')
        return authority if sha==e.AUTH_SHA else binding
    return plan,markers,docs

class EpochTests(unittest.TestCase):
    def setUp(self):
        e._STOP_CACHE.clear();e._DOCUMENTS.clear()
        self.plan,self.markers,self.docs=fixture()
        self.patches=[patch.object(e,'pinned_json',side_effect=self.docs),patch.object(e,'marker_values',return_value=self.markers)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
    def valid(self,**kwargs):
        return e.validate(self.plan,now=10000,monotonic=100,markers=self.markers,boot_id='boot',**kwargs)
    def test_exact_history_and_finite_lease(self):
        self.assertEqual(self.valid()['stop_remaining_seconds'],2430)
        self.assertEqual(e.validate_authority_and_markers(self.plan),self.plan['runtime_epoch'])
    def test_missing_or_incorrect_authority(self):
        for bad in ({},dict(self.plan['runtime_epoch'],resume_ref=dict(sha256='0'*64))):
            self.plan['runtime_epoch']=bad
            with self.assertRaises((ValueError,KeyError)):self.valid()
    def test_all_historical_changes_fail_closed(self):
        original=copy.deepcopy(self.markers)
        for p in e.HISTORICAL:
            for value in (None,{},dict(error='PermissionError'),dict(present=True,bytes=0,sha256=hashlib.sha256(b'null').hexdigest())):
                self.markers.clear();self.markers.update(copy.deepcopy(original));self.markers[p]=value
                with self.assertRaises(ValueError):self.valid()
    def test_all_new_and_old_own_markers_any_content_stop(self):
        for p in e.WATCH:
            if p in e.HISTORICAL:continue
            for raw in (b'',b'null',b'{malformed',b'{"paused":false}'):
                self.markers[p]=dict(present=True,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
                with self.assertRaises(ValueError):self.valid()
                self.markers[p]=None
    def test_boot_expiry_clock_rollback_and_no_extension(self):
        for now,mono,boot in ((10000,100,'other'),(12551,100,'boot'),(10000,2651,'boot'),(9000,200,'boot')):
            with self.assertRaises(ValueError):e.validate(self.plan,now=now,monotonic=mono,markers=self.markers,boot_id=boot)
        self.plan['runtime_epoch']['deadline_monotonic_seconds']+=1
        with self.assertRaises(ValueError):self.valid()
    def test_cache_at_most100ms_and_force_boundary(self):
        with patch.object(e.time,'monotonic',side_effect=[10.,10.099,10.100001,10.101]),patch.object(e,'validate',side_effect=[dict(stop_remaining_seconds=1),dict(stop_remaining_seconds=1),ValueError('new pause')]) as validate:
            self.assertFalse(e.stopped(self.plan));self.assertFalse(e.stopped(self.plan))
            self.assertFalse(e.stopped(self.plan));self.assertTrue(e.stopped(self.plan,force=True))
            self.assertEqual(validate.call_count,3)
    def test_post_requires_remaining2310_and_legacy_date_still_blocks(self):
        from datetime import datetime,timezone
        from .admission import check,EXECUTION_DESIGN_VERSION,DESIGN_SHA,DESIGN_ACCEPTANCE_SHA
        self.plan.update(mathematical_design_version=EXECUTION_DESIGN_VERSION,design_ref=dict(sha256=DESIGN_SHA),root_design_acceptance_ref=dict(sha256=DESIGN_ACCEPTANCE_SHA))
        with patch.object(e,'validate',return_value=dict(stop_remaining_seconds=2309)):
            with self.assertRaisesRegex(ValueError,'2310'):check(self.plan,{},now=datetime.now(timezone.utc),marker_values=self.markers)
        self.plan.pop('runtime_epoch')
        with self.assertRaisesRegex(ValueError,'hard pause'):check(self.plan,{},now=datetime(2026,10,15,tzinfo=timezone.utc),marker_values=self.markers)
    def test_prior_intent_blocks_and_duplicate_post_cannot_spawn(self):
        from .server import Endpoint
        with patch.object(e.Path,'exists',return_value=True):
            with self.assertRaisesRegex(ValueError,'prior dispatch'):e.prior_dispatch_absent(self.plan)
        endpoint=Endpoint.__new__(Endpoint);endpoint.plan=dict(self.plan,content_SHA='x');endpoint.path='/task'
        endpoint.journal=types.SimpleNamespace(has=lambda key:True)
        fake=unittest.mock.Mock()
        with self.assertRaisesRegex(ValueError,'never replay'):endpoint.post('/task/start',dict(request_id=self.plan['request_id'],plan_content_SHA='x'),popen=fake)
        fake.assert_not_called()
    def test_idle_service_retires_without_worker(self):
        from .server import Endpoint
        endpoint=Endpoint.__new__(Endpoint);endpoint.plan=self.plan;endpoint.worker=None;endpoint.started=1;endpoint.finished=False
        endpoint.journal=unittest.mock.Mock()
        with patch.dict('sys.modules',{'psutil':types.SimpleNamespace()}),patch('scripts.stage1_ai_search_r12.server.stopped',return_value=False),patch('scripts.stage1_ai_search_r12.server.time.monotonic',return_value=121):endpoint.monitor()
        self.assertTrue(endpoint.finished);endpoint.journal.write.assert_called_once()

class RawMarkerTests(unittest.TestCase):
    def test_raw_null_malformed_empty_are_present(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'marker'
            for raw in (b'',b'null',b'{broken'):
                p.write_bytes(raw);value=e.marker_values(dict(watched_pause_paths=[str(p)]))[str(p)]
                self.assertTrue(value['present']);self.assertEqual(value['bytes'],len(raw))
            p.unlink();self.assertIsNone(e.marker_values(dict(watched_pause_paths=[str(p)]))[str(p)])
    def test_exact_document_content_required(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'authority';p.write_bytes(b'null')
            with self.assertRaises(ValueError):e.pinned_json(dict(path=str(p),sha256=e.AUTH_SHA,bytes=965),e.AUTH_SHA)

if __name__=='__main__':unittest.main()
