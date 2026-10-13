"""Pure operations fixtures; no real task/API/scene/science calls."""
import unittest,tempfile,errno,json,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest,read_json
import satellite_rl.workbench.r7_recovery_ops as ops

class NarrowProcExit(unittest.TestCase):
    def test_confirmed_disappeared_ESRCH_only_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pid=root/'42';pid.mkdir()
            real=Path.read_text
            def read(path,*args,**kwargs):
                if path==pid/'status':raise ProcessLookupError(errno.ESRCH,'gone')
                return real(path,*args,**kwargs)
            with patch.object(Path,'read_text',read):self.assertEqual(ops.rss_tree(42,root,page_size=4096),0)
    def test_ESRCH_on_still_live_pid_remains_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pid=root/'42';pid.mkdir();(pid/'stat').write_text('42 (worker) R 1 2')
            real=Path.read_text
            def read(path,*args,**kwargs):
                if path==pid/'status':raise ProcessLookupError(errno.ESRCH,'ambiguous live error')
                return real(path,*args,**kwargs)
            with patch.object(Path,'read_text',read),self.assertRaises(ProcessLookupError):ops.rss_tree(42,root,page_size=4096)
    def test_permission_and_malformed_RSS_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pid=root/'42';pid.mkdir();(pid/'status').write_text('VmRSS: bogus kB')
            with self.assertRaises(ValueError):ops.rss_tree(42,root,page_size=4096)
            with patch.object(Path,'read_text',side_effect=PermissionError('denied')),self.assertRaises(PermissionError):ops.rss_tree(42,root,page_size=4096)
    def test_same_kernel_RSS_meaning_for_live_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pid=root/'42';(pid/'task'/'42').mkdir(parents=True);(pid/'status').write_text('VmRSS: 123 kB\n');(pid/'task'/'42'/'children').write_text('')
            self.assertEqual(ops.rss_tree(42,root,page_size=4096),123*1024)

class SameIDSidecar(unittest.TestCase):
    def test_original_job_config_owner_bytes_and_idempotent_resume_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/ops.RUN;directory.mkdir();job={'run_id':ops.RUN,'scientific_commit':ops.SCIENCE,'owner_path':str(root/'owner.json')}
            for name,value in (('job.json',job),('config.json',{'pure_mock':True}),('status.json',{'stage':'failed','pid':1}),('launch.json',{}),('case_index.json',{}),('failure.json',{'old':'preserve'})):atomic_json(directory/name,value)
            atomic_json(root/'owner.json',{'run_id':ops.RUN,'job_sha256':digest(directory/'job.json')})
            before={name:(directory/name).read_bytes() for name in ('job.json','config.json')};before_owner=(root/'owner.json').read_bytes();metadata={'contract_ref':{'path':'mock','sha256':'a'*64},'root_review_ref':{'path':'mock2','sha256':'b'*64}}
            contract={'original_job_ref':{'sha256':digest(directory/'job.json')},'original_case_index_ref':{'sha256':digest(directory/'case_index.json')},'original_pid_births':[],'repair_id':'pure_mock_repair'}
            app=SimpleNamespace(store=SimpleNamespace(job_dir=lambda run:directory,archive_root=root),scheduler=SimpleNamespace(lock=threading.RLock()))
            with patch.object(ops,'validate',return_value=contract),patch.object(ops,'exclusive_lock',side_effect=lambda *a,**k:nullcontext()),patch('satellite_rl.workbench.runtime.pid_alive',return_value=False),patch('satellite_rl.workbench.r7_resources.observe',return_value={'foreign_workers':[]}):
                first=ops.resume(app,metadata);second=ops.resume(app,metadata)
            self.assertEqual(first['run_id'],ops.RUN);self.assertFalse(first['idempotent']);self.assertTrue(second['idempotent']);self.assertEqual(read_json(directory/'status.json')['stage'],'queued')
            for name,raw in before.items():self.assertEqual((directory/name).read_bytes(),raw)
            self.assertEqual((root/'owner.json').read_bytes(),before_owner);self.assertEqual(read_json(directory/'operations_recovery.json'),metadata)
    def test_remaining_exact139_and_failed2_never_return_to_queue(self):
        self.assertEqual(len(ops.REMAINING),139);self.assertEqual(ops.REMAINING,list(range(299,576,2)));self.assertFalse(set(ops.FAILED)&set(ops.REMAINING));self.assertFalse(set(range(1,295,2))&set(ops.REMAINING))

if __name__=='__main__':unittest.main()
