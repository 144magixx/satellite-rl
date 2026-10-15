"""Adversarial cached-source checks with synthetic bytes only; no physics."""
import hashlib,pickle,unittest
from pathlib import Path
from satellite_rl.bottleneck29 import core
from satellite_rl.bottleneck29.a1_inputs import A0Facts
from test_bottleneck29_a1_core import fixture


class CachedSourceTests(unittest.TestCase):
    def fixture(self):
        ctx,_=fixture(3);snapshot=core.initial_snapshot(ctx);raw=pickle.dumps(snapshot)
        sha=hashlib.sha256(raw).hexdigest();sid='stub/0'
        facts=A0Facts.__new__(A0Facts);facts.root=Path('.');facts.binding=dict(source_SHA='A0')
        facts.bundle=dict(states={sid:{}},A0_archive_SHA='archive')
        package=dict(schema='W29A1A0StateCacheV1',state_id=sid,snapshot_raw=raw,
            snapshot_source_ref=dict(record_SHA=sha,source_SHA='A0',snapshot_SHA=sha,row_id=sid+'/snapshot'),rows={},row_sources={})
        facts._bundle_value=lambda ref:package
        return facts,ctx,dict(case_id='stub'),package,sha

    def test_snapshot_wrong_source_rejected(self):
        facts,ctx,mother,package,_=self.fixture();package['snapshot_source_ref']['source_SHA']='wrong'
        with self.assertRaisesRegex(ValueError,'snapshot source metadata'):facts.state(ctx,mother,0)

    def test_row_wrong_snapshot_rejected(self):
        facts,ctx,mother,package,sha=self.fixture();key='stub/0/minimum5/0'
        row=dict(row_id=key,snapshot_sha='wrong',binding=facts.binding,spec={},result=dict(status='invalid'))
        raw=pickle.dumps(row);package['rows'][key]=raw
        package['row_sources'][key]=dict(record_SHA=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError,'row identity'):facts.state(ctx,mother,0)

    def test_row_wrong_source_rejected(self):
        facts,ctx,mother,package,sha=self.fixture();key='stub/0/minimum5/0'
        row=dict(row_id=key,snapshot_sha=sha,binding=facts.binding,spec={},result=dict(status='invalid'))
        raw=pickle.dumps(row);package['rows'][key]=raw
        package['row_sources'][key]=dict(record_SHA=hashlib.sha256(raw).hexdigest(),source_SHA='wrong')
        with self.assertRaisesRegex(ValueError,'row source metadata'):facts.state(ctx,mother,0)


if __name__=='__main__':unittest.main()
