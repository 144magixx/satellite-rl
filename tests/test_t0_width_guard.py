import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from satellite_rl.t0_width.guard import proc_stat,run_stage,rss_tree,terminate_owned,birth

class GuardTests(unittest.TestCase):
    def test_proc_exit_races_only(self):
        for code in (errno.ENOENT,errno.ESRCH):
            with patch.object(Path,'read_text',side_effect=OSError(code,'exit')):self.assertIsNone(proc_stat(123))
        with patch.object(Path,'read_text',side_effect=PermissionError(errno.EACCES,'denied')):
            with self.assertRaises(PermissionError):proc_stat(123)
        with patch.object(Path,'read_text',return_value='bad stat'):
            with self.assertRaises((IndexError,ValueError)):proc_stat(123)

    def test_completed_receipt_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'stage_output.json'
            command=[sys.executable,'-c','from pathlib import Path; Path('+repr(str(output))+').write_text("{}")']
            first=run_stage(command,temp,5,4*1024**3)
            self.assertEqual(first['status'],'completed')
            self.assertEqual(run_stage(command,temp,5,4*1024**3),first)
            with self.assertRaises(ValueError):run_stage(command,temp,6,4*1024**3)

    def test_actual_time_and_memory_limits(self):
        for kwargs,reason in [({},'hard_time_guard'),({'memory':lambda pid:4*1024**3+1},'owned_tree_memory_guard')]:
            with tempfile.TemporaryDirectory() as temp:
                result=run_stage([sys.executable,'-c','import time; time.sleep(4)'],temp,.15,4*1024**3,**kwargs)
                self.assertEqual(result['reason'],reason);self.assertIsNotNone(result['exit_code'])
                self.assertTrue((Path(temp)/'guard_receipt.json').exists())

    def test_unknown_wal_blocks_reexecution(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'stage.wal.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unknown'):run_stage([],temp,1,1024)

    def test_monitor_permission_error_preserves_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            def fail(pid):raise PermissionError(errno.EACCES,'denied')
            result=run_stage([sys.executable,'-c','import time; time.sleep(3)'],temp,2,4*1024**3,memory=fail)
            self.assertEqual(result['reason'],'monitor_error');self.assertTrue(result['hard_interface_error'])
            self.assertIsNotNone(result['exit_code'])

    @unittest.skipIf(os.name=='nt','Linux PGID race test')
    def test_kill_esrch_reaps_exit(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.1)'],start_new_session=True)
        identity=birth(child.pid)
        try:
            def race(*args):
                child.wait(timeout=2)
                raise OSError(errno.ESRCH,'exited before signal')
            with patch('os.killpg',side_effect=race):terminate_owned(child,identity)
            self.assertIsNotNone(child.returncode)
        finally:
            if child.poll() is None:child.kill()
            child.wait()

    def test_complete_parent_descendant_rss(self):
        self.assertGreater(rss_tree(os.getpid()),0)

if __name__=='__main__':unittest.main()
