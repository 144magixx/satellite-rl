"""Fake process lifecycle tests. Never starts a real process or geometry."""
import tempfile
from pathlib import Path
import unittest
from .lane import run_child,validate_task_order
from .runtime import BudgetLedger
from .pipeline import stage_plan


class FakeGuard:
    def check(self):pass


class FakeChild:
    pid=987654321
    def __init__(self,ticks,end):self.ticks=ticks;self.end=end;self.killed=False;self.waited=False
    def poll(self):return 0 if self.killed or self.ticks[0]>=self.end else None
    def wait(self,timeout):self.waited=True;return -9 if self.killed else 0


class LaneTests(unittest.TestCase):
    def test_fixed_mother_complete_latin_only(self):
        tasks=stage_plan()['development'][:9]
        self.assertTrue(validate_task_order(tasks))
        with self.assertRaises(ValueError):validate_task_order(tasks[:8])
        with self.assertRaises(ValueError):validate_task_order(tasks[::-1])
    def run_fake(self,end):
        ticks=[0.];child=FakeChild(ticks,end);calls=[]
        def sleep(seconds):ticks[0]+=1
        def kill(process,identity,hard):calls.append(hard);process.killed=process.poll() is None;return 0
        with tempfile.TemporaryDirectory() as d:
            facts=run_child(['never-executed'],{'outer_seconds':120},Path(d)/'task',FakeGuard(),
                popen=lambda *a,**k:child,clock=lambda:ticks[0],sleeper=sleep,
                identify=lambda pid:{'pid':pid,'process_start_identity':'fake','boot_id':'fake'},
                terminate=kill,sample=lambda *a:None)
            stop=(Path(d)/'task/stop.request').exists()
            self.assertTrue((Path(d)/'task/exit.json').exists());self.assertTrue(child.waited)
            return facts,stop,calls

    def test_normal_direct_reap(self):
        facts,stop,_=self.run_fake(2)
        self.assertEqual(facts['raw_returncode'],0);self.assertEqual(facts['elapsed_seconds'],2)
        self.assertFalse(stop);self.assertTrue(facts['reaped'])

    def test_hard_boundary_cooperative_stop_then_reap(self):
        facts,stop,calls=self.run_fake(999)
        self.assertEqual(facts['raw_returncode'],-9);self.assertTrue(stop)
        self.assertEqual(facts['elapsed_seconds'],118);self.assertTrue(all(calls))
        self.assertEqual(facts['stop_reason'],'outer_hard_deadline')

    def test_parallel_outer_reservations_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=BudgetLedger(d),BudgetLedger(d)
            a.reserve('lane73','B',1800);b.reserve('lane74','B',1800)
            with self.assertRaises(ValueError):a.reserve('extra','B',1)

    def test_overlapping_audit_cap_does_not_relax_outer_or_double_group(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=BudgetLedger(d);ledger.reserve('fake','B',120,{'audit':120})
            facts={'pid':1,'process_start_identity':'FAKE','raw_returncode':0,'reaped':True,'descendants_live':0}
            ledger.finish('fake',20.,facts,{'B':20.,'audit':7.})
            self.assertEqual(ledger.available('B'),3580.)
            self.assertEqual(ledger.available('audit'),893.)
            self.assertEqual(sum(t['elapsed_seconds'] for t in ledger._load()['tasks'].values()),20.)


if __name__=='__main__':unittest.main()
