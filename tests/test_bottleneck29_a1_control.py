"""A1 admission and status tests with injected identity, no remote/hardware."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from satellite_rl.bottleneck29.a1_workbench import admit
from satellite_rl.bottleneck29.a1_stream import StreamStore,_frame
from satellite_rl.bottleneck29.ledger import Halt,digest


class A1ControlTests(unittest.TestCase):
    def fixture(self,root,lane=0):
        binding=dict(source_SHA='s',data_SHA='d',contract_SHA='c',authority_SHA='a',A0_source_SHA='old',A0_summary_SHA='summary',actor_SHA='actor',Q_status='unavailable_same_node_150000')
        limits=dict(phy=158925,enumeration=128925,suffix=30000,forward=30015,kernel=0,prefix=0,seconds=3600)
        config=dict(binding=binding,lane=lane,_config_sha='config',pinned_files={},stop_paths=[],receipt_max_age_seconds=300,
            science_limits=limits,science_storage_bytes=400*1024**2,engineering_storage_bytes=8*1024**2,science_admission=str(Path(root)/'science.json'))
        config['storage_allocation']=dict(total_working_cap_bytes=2*1024**3,existing_working_bytes=350*1024**2,
            other_reserved_bytes=16*1024**2,lane_science_working_bytes=[400*1024**2]*4,lane_engineering_working_bytes=[8*1024**2]*4)
        receipt=dict(root_approved=True,phase='A1',mode='science',binding=binding,lane=lane,checked_unix=time.time(),
            config_SHA='config',limits=limits,identity=dict(stub=True),engineering_accepted=True,
            full_loop_seconds_upper_bound=3500,remaining_science_seconds=3600,storage_allocation=config['storage_allocation'])
        Path(config['science_admission']).write_text(json.dumps(receipt))
        return config,receipt

    def test_exact_missing_Q_caps_and_config_SHA(self):
        with tempfile.TemporaryDirectory() as root:
            config,receipt=self.fixture(root);admit(config,'science',actual=dict(stub=True))
            config['_config_sha']='changed'
            with self.assertRaises(Halt):admit(config,'science',actual=dict(stub=True))
            config['_config_sha']='config';config['science_limits']['phy']=164925
            receipt['limits']=config['science_limits'];Path(config['science_admission']).write_text(json.dumps(receipt))
            with self.assertRaises(Halt):admit(config,'science',actual=dict(stub=True))

    def test_overall_storage_and_stale_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            config,receipt=self.fixture(root);config['science_storage_bytes']+=1
            with self.assertRaises(Halt):admit(config,'science',actual=dict(stub=True))
            config['science_storage_bytes']-=1;receipt['checked_unix']-=301
            Path(config['science_admission']).write_text(json.dumps(receipt))
            with self.assertRaises(Halt):admit(config,'science',actual=dict(stub=True))

    def test_read_only_incremental_tail_waits_for_partial_frame(self):
        with tempfile.TemporaryDirectory() as root:
            binding=dict(source='mock');limits=dict(phy=3,seconds=60)
            writer=StreamStore(root,binding,limits,minimum_free_bytes=0);writer.begin()
            reader=StreamStore(root,binding,limits,read_only=True)
            position=reader.journal_offset
            event=dict(type='metadata',key='progress',value=2);frame=_frame(json.dumps(event).encode())
            with writer.journal_path.open('ab',buffering=0) as stream:stream.write(frame[:20])
            reader.refresh();self.assertEqual(reader.journal_offset,position)
            with writer.journal_path.open('ab',buffering=0) as stream:stream.write(frame[20:])
            reader.refresh();self.assertEqual(reader.metadata['progress'],2)
            writer.finish()


if __name__=='__main__':unittest.main()
