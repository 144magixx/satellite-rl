"""All operations are Python stubs: zero real PHY/kernel calls."""
import tempfile
import json
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from satellite_rl.bottleneck29.ledger import Halt, Ledger


class ControlTests(unittest.TestCase):
    @contextmanager
    def directory(self):
        self.ledgers=[]
        with tempfile.TemporaryDirectory() as root:
            try: yield root
            finally: self.ledgers.clear()

    def make(self,root):
        ledger=Ledger(root,dict(phy=2,phy_prefix=2,kernel=1,seconds=60),dict(source='a'))
        self.ledgers.append(ledger)
        return ledger

    def test_completed_reuse_and_cap(self):
        with self.directory() as root:
            ledger=self.make(root);ledger.begin();called=[]
            self.assertEqual(ledger.meter('phy_prefix','a',lambda: called.append(1) or [42]),[42])
            self.assertEqual(ledger.meter('phy_prefix','a',lambda: self.fail('replayed')),[42])
            self.assertEqual(called,[1]);ledger.finish()
            ledger=self.make(root);ledger.begin()
            ledger.meter('phy_prefix','b',lambda: 2)
            with self.assertRaises(Halt): ledger.meter('phy_prefix','c',lambda: self.fail('over budget'))
            ledger.finish()

    def test_unknown_stops_resume(self):
        with self.directory() as root:
            ledger=self.make(root);ledger.begin()
            def crash(): raise ValueError('stub interrupted')
            with self.assertRaises(ValueError):ledger.meter('phy_prefix','a',crash)
            self.assertEqual(ledger.counts()['phy_prefix'],1);ledger.finish()
            with self.assertRaises(Halt):self.make(root).begin()

    def test_crashed_clock_conservatively_reserves_remainder(self):
        with self.directory() as root:
            self.make(root).begin()
            with self.assertRaises(Halt):self.make(root).begin()

    def test_identity_change(self):
        with self.directory() as root:
            self.make(root)
            with self.assertRaises(Halt):Ledger(root,dict(seconds=90),dict(source='b'))

    def test_result_corruption(self):
        with self.directory() as root:
            ledger=self.make(root);ledger.begin();ledger.meter('kernel','a',lambda: dict(stub=True))
            next((Path(root)/'operations').iterdir()).write_bytes(b'bad')
            with self.assertRaises(Halt):ledger.meter('kernel','a',lambda: self.fail('corrupt replay'))
            ledger.finish()

    def test_torn_journal_blocks(self):
        with self.directory() as root:
            ledger=self.make(root)
            with ledger.path.open('ab') as stream:stream.write(b'{')
            with self.assertRaises(Halt):self.make(root)

    def test_storage_bound_and_actual_account(self):
        with self.directory() as root:
            ledger=self.make(root);ledger.begin()
            ledger.artifact('a',Path(root)/'a.pkl',dict(number=123))
            actual=sum(p.stat().st_size for p in Path(root).iterdir() if p.is_file())
            self.assertEqual(actual,ledger.storage_bytes())
            ledger.storage_limit=ledger.storage_bytes()+10
            with self.assertRaises(Halt):ledger.artifact('b',Path(root)/'b.pkl','x'*100)
            ledger.storage_limit=512*1024**2;ledger.finish()

    def test_missing_ledger_does_not_reset_spend(self):
        with self.directory() as root:
            ledger=self.make(root);ledger.begin();ledger.meter('kernel','a',lambda: 1);ledger.finish()
            ledger.path.unlink()
            with self.assertRaises(Halt):self.make(root)

    def test_root_orphan_and_atomic_temporary_block(self):
        for name in ('cursor.pkl','evidence.pkl.tmp'):
            with self.directory() as root:
                ledger=self.make(root);(Path(root)/name).write_bytes(b'orphan')
                with self.assertRaises(Halt):ledger.begin()

    def test_admission_config_and_throughput(self):
        from satellite_rl.bottleneck29.workbench import admit
        with self.directory() as root:
            receipt_path=Path(root)/'admission.json'
            binding=dict(source_SHA='s',data_SHA='d',contract_SHA='c',authority_SHA='a')
            config=dict(science_admission=str(receipt_path),binding=binding,lane=0,
                receipt_max_age_seconds=300,_config_sha='config',pinned_files={},stop_paths=[])
            receipt=dict(root_approved=True,mode='science',checked_unix=time.time(),binding=binding,
                lane=0,identity={'stub':True},config_SHA='config',engineering_accepted=True,
                full_loop_seconds_upper_bound=899,remaining_science_seconds=900)
            receipt_path.write_text(json.dumps(receipt));admit(config,'science',actual={'stub':True})
            config['_config_sha']='changed'
            with self.assertRaises(Halt):admit(config,'science',actual={'stub':True})
            config['_config_sha']='config';receipt['full_loop_seconds_upper_bound']=901
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(Halt):admit(config,'science',actual={'stub':True})


if __name__=='__main__':unittest.main()
