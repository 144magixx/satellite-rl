"""Derived cache proofs only; this candidate is not connected to formal jobs."""
import copy
import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.coverage.nondrl.evidence_cache import build_cache, CompactEvidenceDataset, digest
from satellite_rl.coverage.nondrl.learning import EvidenceDataset, collate
from satellite_rl.coverage.nondrl.models import make_model, supervised_loss
from satellite_rl.coverage.nondrl.teachers import EvidenceWriter
from test_nondrl_learning import examples


def evidence(root, route):
    writer = EvidenceWriter(root, route, shard_rows=1)
    sample = examples()[route]
    for mother in ('z', 'a', 'z', 'b'):
        row = copy.deepcopy(sample);row['mother_id'] = mother
        # Large audit-only payload is intentionally not a training input.
        row['teacher_trial'] = {'full_physical_audit': [100.123456789]*100}
        writer.append(row)
    writer.manifest['status'] = 'completed';writer.flush()
    return writer.folder/'manifest.json'


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):return torch.equal(a,b)
    if isinstance(a, dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


class EvidenceCacheTests(unittest.TestCase):
    def test_all_routes_draw_positions_rng_and_tensors_identical(self):
        for route in ('01','02','03','04','05'):
            with self.subTest(route=route),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=evidence(root/'source',route)
                originals={str(p):p.read_bytes() for p in (root/'source').rglob('*') if p.is_file()}
                path=build_cache(source,root/'cache',route)
                old=EvidenceDataset(source,route);new=CompactEvidenceDataset(path,expected_sha256=digest(path),verify_original=True)
                self.assertEqual(old.sha256,new.sha256);self.assertEqual(old.mothers,new.mothers)
                a,b=np.random.default_rng(51001),np.random.default_rng(51001)
                for _ in range(4):
                    x,px=old.draw(a,8);y,py=new.draw(b,8)
                    self.assertEqual(px,py);self.assertEqual(a.bit_generator.state,b.bit_generator.state)
                    self.assertTrue(nested_equal(collate(x,route),collate(y,route)))
                    self.assertNotIn('teacher_trial',y[0]);self.assertFalse(y[0]['aircraft'].flags.writeable)
                    self.assertTrue(np.array_equal(np.asarray(x[0]['aircraft']),y[0]['aircraft']))
                self.assertEqual(originals,{str(p):p.read_bytes() for p in (root/'source').rglob('*') if p.is_file()})
                self.assertEqual(build_cache(source,root/'cache',route),path)
                new.close();del y,new;gc.collect()

    def test_two_real_updates_preserve_parameters_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=evidence(root/'source','01');path=build_cache(source,root/'cache')
            datasets=(EvidenceDataset(source,'01'),CompactEvidenceDataset(path))
            torch.manual_seed(51001);model=make_model('01',{'hidden':16,'layers':1,'heads':4})
            initial=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone();outcomes=[]
            for dataset in datasets:
                torch.set_rng_state(rng);model=make_model('01',{'hidden':16,'layers':1,'heads':4});model.load_state_dict(initial)
                optimizer=torch.optim.Adam(model.parameters(),lr=.0003);sampler=np.random.default_rng(51001)
                for _ in range(2):
                    samples,_=dataset.draw(sampler,4);batch=collate(samples,'01');optimizer.zero_grad()
                    loss=supervised_loss('01',model(batch),batch,samples);loss.backward();optimizer.step()
                outcomes.append((copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict()),sampler.bit_generator.state,torch.get_rng_state()))
            self.assertTrue(nested_equal(outcomes[0],outcomes[1]))
            datasets[1].close();del samples,datasets;gc.collect()

    def test_source_and_derived_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=evidence(root/'source','01');path=build_cache(source,root/'cache');pin=digest(path)
            data=path.parent/'data.bin';raw=data.read_bytes();data.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaisesRegex(ValueError,'derived artifact'):CompactEvidenceDataset(path,expected_sha256=pin)
            data.write_bytes(raw)
            record=json.loads(source.read_text(encoding='utf-8'));shard=source.parent/record['shards'][0]['path'];shard.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'original shard SHA'):CompactEvidenceDataset(path,verify_original=True)
            with self.assertRaisesRegex(ValueError,'shard path or SHA'):build_cache(source,root/'different-cache')

    def test_filtered_rows_keep_original_shard_row_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=evidence(root/'source','04')
            # Retain a real valid row plus a fully known tied row which the
            # original ranking Dataset filters out; cached draw keys must agree.
            writer=EvidenceWriter(root/'source','04',resume=True,shard_rows=1)
            row=copy.deepcopy(examples()['04']);row['utility']=[0.]*len(row['actions']);writer.append(row);writer.flush()
            path=build_cache(source,root/'cache','04');old=EvidenceDataset(source,'04');new=CompactEvidenceDataset(path)
            self.assertEqual(new.filtered,1);self.assertEqual(old.mothers,new.mothers);new.close()

    def test_frozen_manifest_pin_and_readonly_mmap_prevent_silent_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=evidence(root/'source','01');path=build_cache(source,root/'cache');pin=digest(path)
            dataset=CompactEvidenceDataset(path,expected_sha256=pin)
            samples,_=dataset.draw(np.random.default_rng(51001),1)
            with self.assertRaises(ValueError):samples[0]['labels'][0]=100
            original=copy.deepcopy(samples[0]['aircraft']);dataset.close()
            # Close relinquishes our mmap ownership; extant views stay valid.
            self.assertTrue(np.array_equal(samples[0]['aircraft'],original))
            del samples,dataset;gc.collect()
            record=json.loads(path.read_text(encoding='utf-8'));record['filtered']+=1
            path.write_text(json.dumps(record),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'frozen compact manifest SHA'):
                CompactEvidenceDataset(path,expected_sha256=pin)


if __name__=='__main__':unittest.main()
