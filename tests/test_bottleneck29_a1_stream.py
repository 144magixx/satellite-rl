"""Pure append-store transaction tests, no real physics/model/kernel."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from satellite_rl.bottleneck29.a1_stream import StreamStore
from satellite_rl.bottleneck29.ledger import Halt


class StreamTests(unittest.TestCase):
    def make(self,root,read_only=False):
        return StreamStore(root,dict(source_SHA='mock',Q_status='missing'),
            dict(phy=300,forward=2,coefficient=300,seconds=60),minimum_free_bytes=0,read_only=read_only)

    def test_numeric_lossless_reuse_and_alias(self):
        with tempfile.TemporaryDirectory() as root:
            store=self.make(root);store.begin()
            value=dict(capacity_bps=np.asarray([0.,np.nextafter(1.,2.),12345678.901234567]),logpi=float('-inf'),actual=[3,2,5.],reward=np.float64(-1.234))
            store.transact('a',dict(phy=1),lambda:value,state='s')
            result=store.transact('a',dict(phy=1),lambda:self.fail('recomputed'),state='s')
            self.assertEqual(result['capacity_bps'].tobytes(),value['capacity_bps'].tobytes())
            self.assertEqual(result['logpi'],float('-inf'))
            store.alias('A0/a',dict(source_SHA='old',snapshot_SHA='snapshot',row_id='old/a',record_SHA='hash'),state='s')
            self.assertEqual(store.counts,dict(phy=1));store.close_state('s');store.finish()
            store=self.make(root);store.begin()
            self.assertEqual(store.get('a')['capacity_bps'].tobytes(),value['capacity_bps'].tobytes())
            with self.assertRaises(Halt):store.transact('b',dict(phy=1),lambda:1,state='s')
            store.finish()

    def test_unknown_cost_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            store=self.make(root);store.begin()
            def interrupted():raise RuntimeError('stub interruption')
            with self.assertRaises(RuntimeError):store.transact('a',dict(phy=1),interrupted,state='s')
            self.assertEqual(store.counts['phy'],1);store.finish()
            other=self.make(root)
            with self.assertRaises(Halt):other.begin()
            self.assertEqual(other.counts['phy'],1);other.finish()

    def test_chunk_bound_seal_and_readback(self):
        with tempfile.TemporaryDirectory() as root:
            store=self.make(root);store.begin()
            for i in range(257):store.transact(str(i),dict(phy=1),lambda i=i:dict(number=i),state='s')
            self.assertEqual(sorted(c['rows'] for c in store.chunks.values()),[1,256])
            store.close_state('s');store.finish()
            store=self.make(root);store.begin();self.assertEqual(store.get('256')['number'],256)
            actual=sum(p.stat().st_size for p in Path(root).iterdir() if p.is_file())+sum(p.stat().st_size for p in (Path(root)/'chunks').iterdir())
            self.assertEqual(actual,store.bytes);store.finish()

    def test_crashed_clock_reserved(self):
        with tempfile.TemporaryDirectory() as root:
            store=self.make(root);store.begin();store.journal.close()
            other=self.make(root)
            with self.assertRaises(Halt):other.begin()
            other.finish()

    def test_torn_result_and_missing_ledger_reject(self):
        with tempfile.TemporaryDirectory() as root:
            store=self.make(root);store.begin();store.transact('a',dict(phy=1),lambda:42,state='s');store.finish()
            chunk=next((Path(root)/'chunks').iterdir())
            with chunk.open('ab') as stream:stream.write(b'partial')
            store=self.make(root)
            with self.assertRaises(Halt):store.begin()
            store.finish()
            (Path(root)/'a1.journal').unlink()
            with self.assertRaises(Halt):self.make(root)

    def test_read_only_never_creates(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'absent'
            with self.assertRaises(Halt):self.make(path,True)
            self.assertFalse(path.exists())

    def test_budget_and_pause_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            store=self.make(root);store.begin()
            with self.assertRaises(Halt):store.transact('bad',dict(phy=301),lambda:self.fail('over cap'),state='s')
            stop=Path(root)/'STOP';stop.touch();store.stop_paths=(stop,)
            with self.assertRaises(Halt):store.transact('paused',dict(phy=1),lambda:self.fail('after pause'),state='s')
            self.assertEqual(store.counts,{});store.finish()

    def test_audited_time_debit_is_cumulative_and_once(self):
        with tempfile.TemporaryDirectory() as root:
            store=self.make(root);store.debit_time('old_dispatch_sha',10);store.debit_time('old_dispatch_sha',10)
            store.begin();self.assertEqual(store.remaining,50);store.finish()
            other=self.make(root);self.assertEqual(len(other.debits),1)
            with self.assertRaises(Halt):other.debit_time('old_dispatch_sha',9)
            other.finish()


if __name__=='__main__':unittest.main()
