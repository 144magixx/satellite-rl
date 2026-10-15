"""Synthetic scalar/history and bounded-IO tests; no real historical payloads."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from . import stage_a as a
from . import run_once as io

def normal(index=0):
    identity=dict(case_id='fake_case_'+str(index),role='train',profile='uniform21600000',input_sha256='fake_input_'+str(index))
    static=dict(aircraft_ids=['a','b'],positions=[[0.,0.],[1.,1.]],demands=[21600000]*2,input_sha256=identity['input_sha256'],profile='europe_v1')
    base=dict(beam_ids=['old'],anchor_aircraft_ids=['a'],centers=[[0.,0.]],widths_deg=[.3],assignment=[0,0])
    m=dict(maxload=43200000,top10_mean=43200000.,CV=0.,N=1,actual_Omega=1e-6,total_request=43200000)
    origin=dict(identity=identity,layout=base,layout_SHA=a.sha(base),metrics=m,Ncap=2,Omega_cap=1e-3)
    common=dict(schema='R10CommonV2',original_index=index,design_sha256=a.DESIGN,origin=origin,identity=identity,origin_SHA=origin['layout_SHA'],q=1,common_ledger={})
    common['common_SHA']=a.sha(common)
    common_ref=dict(path='fake_common_'+str(index),sha256='1'*64,compression='gzip',compressed_bytes=1,uncompressed_bytes=1,canonical_json_sha256='2'*64)
    input_ref=dict(path='fake_source',sha256='3'*64)
    wrapper=dict(run_id='fake_run',original_index=index,identity=identity,actual_stage='common',source_input_ref=input_ref,common=common,static=static)
    screens=[];trials=[];current=base
    def descriptor(target,eligible=True):
        state=a.state_sha(identity,static['aircraft_ids'],current);region=sorted(current['beam_ids'])
        value=dict(state_SHA=state,region_beam_ids=region,target_k=target,member_ids=['a','b'],structurally_eligible=eligible)
        value['semantic_SHA']=a.sha([state,region,target,a.RECIPE]);value['descriptor_SHA']=a.sha(value);return value
    skipped=descriptor(3,False);screens.append(dict(descriptor=skipped,screening='known_unavailable'))
    d=descriptor(1);screens.append(dict(descriptor=d,screening='kernel_attempted'))
    trials.append(dict(attempt=0,semantic_SHA=d['semantic_SHA'],kernel_entered=True,status='rejected',ledger=dict(flow_queries=1)))
    screens.append(dict(descriptor=d,screening='duplicate_semantics'))
    d=descriptor(2);screens.append(dict(descriptor=d,screening='kernel_attempted'))
    anchors=['a','b'];newids=[json.dumps([identity['case_id'],'R10_seq',d['state_SHA'],d['semantic_SHA'],anchor],ensure_ascii=False,separators=(',',':')) for anchor in anchors]
    terminal=dict(beam_ids=newids,anchor_aircraft_ids=anchors,centers=static['positions'],widths_deg=[.3,.3],assignment=[0,1])
    next_m=dict(m,maxload=21600000,top10_mean=21600000.,N=2,actual_Omega=2e-6)
    trials.append(dict(attempt=1,semantic_SHA=d['semantic_SHA'],kernel_entered=True,status='admissible',ledger=dict(flow_queries=2),selected_anchor_ids=anchors,
        terminal=terminal,raw_layout=terminal,metrics=next_m,raw_metrics=next_m,new_beam_lineage=dict(removed_region_beam_ids=d['region_beam_ids'],new_ids=newids)))
    current=terminal
    for target,status in ((1,'rejected'),(2,'no_improvement'),(3,'rejected')):
        d=descriptor(target);screens.append(dict(descriptor=d,screening='kernel_attempted'))
        trials.append(dict(attempt=len(trials),semantic_SHA=d['semantic_SHA'],kernel_entered=True,status=status,ledger=dict(flow_queries=1)))
    result=dict(schema='R10MethodV2',method='SH-C',complete=True,partial_status=None,identity=identity,original_index=index,common_SHA=common['common_SHA'],origin_SHA=origin['layout_SHA'],q=1,common_ledger={},new_RULE_D5_T_W_model_calls=0,
        results=trials,screenings=screens,kernel_attempts=5,termination='no_progress_pool_exhausted',selected=dict(terminal=terminal,metrics=next_m))
    method=dict(run_id='fake_run',original_index=index,identity=identity,actual_stage='SH-C',source_input_ref=input_ref,common_ref=common_ref,result=result)
    return wrapper,method,common_ref,dict(path='fake_method',sha256='4'*64),dict(path='fake_old_certificate',sha256='5'*64)

class ScalarTests(unittest.TestCase):
    def test_full_history_zero_skips_acceptance_reset_and_natural_end(self):
        row=a.analyze_case(*normal())
        self.assertEqual(len(row['prefix_rows']),6);self.assertEqual(row['first_safe_prefix'],2)
        self.assertEqual(row['prefix_rows'][0]['last_attempt_status'],'original_B')
        self.assertFalse(row['prefix_rows'][0]['quality_preserving_own_teacher']);self.assertTrue(row['prefix_rows'][2]['quality_preserving_own_teacher'])
        policies={p['policy']:p['stop_prefix'] for p in row['policies']}
        self.assertEqual(policies,{'full':5,'cap_0':0,'cap_4':4,'cap_8':5,'cap_16':5,'cap_32':5,'window_2':4,'window_4':5,'window_8':5})
        self.assertEqual(row['prefix_rows'][1]['screens_already_paid'],2)
        self.assertEqual(row['prefix_rows'][2]['screens_already_paid'],4)
        self.assertEqual(row['prefix_rows'][3]['layout_SHA'],row['prefix_rows'][2]['layout_SHA'])
    def test_own_teacher_and_wrapper_lineage_not_other_method(self):
        for mutate in (lambda x:x[1].update(actual_stage='S2'),lambda x:x[1].update(run_id='other'),lambda x:x[1]['result'].update(method='S2'),lambda x:x[1].update(common_ref={}),lambda x:x[1]['result']['selected'].update(terminal=x[0]['common']['origin']['layout'])):
            fixture=list(normal());mutate(fixture)
            with self.assertRaises(a.Unknown):a.analyze_case(*fixture)
    def test_unknown_or_missing_trial_never_zero_filled(self):
        for status in ('entered_unknown','unknown',None):
            fixture=normal();fixture[1]['result']['results'][0]['status']=status
            with self.assertRaises(a.Unknown):a.analyze_case(*fixture)
        fixture=normal();fixture[1]['result']['results'].pop();fixture[1]['result']['kernel_attempts']=4
        with self.assertRaises(a.Unknown):a.analyze_case(*fixture)
    def test_changed_semantic_and_original_ownership_rejected(self):
        fixture=normal();fixture[1]['result']['screenings'][0]['descriptor']['state_SHA']='bad'
        with self.assertRaises(a.Unknown):a.analyze_case(*fixture)
    def test_quality_resource_tolerance_exact(self):
        b=dict(maxload=100.,top10_mean=80.,CV=.5,N=2,actual_Omega=.01,total_request=100)
        self.assertTrue(a.quality_preserving(dict(b,maxload=100+1e-6+1e-10,CV=.5+1e-9),b))
        for change in (dict(maxload=100.01),dict(top10_mean=80.01),dict(CV=.50001),dict(N=3),dict(actual_Omega=.01000001)):
            self.assertFalse(a.quality_preserving(dict(b,**change),b))
    def test_32_cap_and_windows_reset(self):
        statuses=['rejected']*31+['admissible'];stops=a.policy_stops(statuses)
        self.assertEqual(stops['cap_32'],32);self.assertEqual(stops['window_2'],2)
        self.assertEqual(a.policy_stops(['rejected','admissible','rejected','rejected'])['window_2'],4)
    def test_all12_gate_and_zero_comparator_denominator(self):
        rows=[a.analyze_case(*normal(index)) for index in a.INDICES]
        gate=a.opportunity_gate(rows,True);self.assertFalse(gate['passed']);self.assertEqual(gate['aggregate_oracle_attempts'],24)
        for row in rows:row['strong_fixed_envelope']=row['first_safe_prefix']=0
        gate=a.opportunity_gate(rows,True);self.assertFalse(gate['conditions'][3]);self.assertIsNone(gate['aggregate_proxy_saving_fraction'])
        rows.pop()
        with self.assertRaises(a.Unknown):a.opportunity_gate(rows,True)
    def test_strong_envelope_gate_positive_and_no_wall_claim(self):
        rows=[]
        for j,index in enumerate(a.INDICES):
            rows.append(dict(original_index=index,complete=True,full_attempts=32,first_safe_prefix=2+j%2,strong_fixed_envelope=8,prefix_rows=[dict(quality_preserving_own_teacher=False),dict(quality_preserving_own_teacher=True)]))
        gate=a.opportunity_gate(rows,True);self.assertTrue(gate['passed']);self.assertFalse(gate['wall_acceleration_claim']);self.assertFalse(gate['training_authorized'])

class IOTests(unittest.TestCase):
    def test_120_outer_110_stop_and_once_intent(self):
        self.assertEqual(io.operational_bounds(100,209.9),dict(stop_new=210,reap_deadline=219,outer_deadline=220))
        with self.assertRaises(TimeoutError):io.operational_bounds(100,210)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'one';io.claim_once(root,{},{});original=(root/'attempt_intent.json').read_bytes()
            with self.assertRaises(FileExistsError):io.claim_once(root,{},{});
            self.assertEqual((root/'attempt_intent.json').read_bytes(),original)
    def test_boot_time_windows_one_second_boundary(self):
        route=dict(host_name='fake',boot_time=1000.,python='fakepython',observed_unix_seconds=2000.)
        for delta in (-1.,0.,.423,1.):
            actual=io.validate_route(route,2000.,'fake',1000.+delta,'fakepython')
            self.assertEqual(actual['actual_boot_time_raw'],1000.+delta)
        for delta in (-1.00001,1.00001,1000.):
            with self.assertRaises(ValueError):io.validate_route(route,2000.,'fake',1000.+delta,'fakepython')
    def test_synthetic_gzip_one_pass_and_exact_decoded_identity(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{'stage_a':a}):
            rows=[]
            for i in range(24):
                value=dict(synthetic=i);raw=a.canonical(value);packed=gzip.compress(raw);p=Path(directory)/(str(i)+'.gz');p.write_bytes(packed)
                rows.append(dict(local_ref=dict(path=str(p),bytes=len(packed),sha256=hashlib.sha256(packed).hexdigest()),logical_output_ref=dict(compression='gzip',sha256=hashlib.sha256(packed).hexdigest(),compressed_bytes=len(packed),uncompressed_bytes=len(raw),canonical_json_sha256=a.sha(value))))
            loader=io.Payloads(rows,lambda:False)
            self.assertEqual(loader.load(rows[0]),dict(synthetic=0))
            with self.assertRaises(ValueError):loader.load(rows[0])
            self.assertEqual(len(loader.read),1)
    def test_prospective_metadata_compressed_and_decoded_caps(self):
        with self.assertRaises(ValueError):io.Metadata(io.META_CAP).read(dict(path='unused',bytes=1,sha256='0'*64))
        rows=[dict(local_ref=dict(path=str(i),bytes=io.COMPRESSED_CAP,sha256=str(i)),logical_output_ref=dict(uncompressed_bytes=1)) for i in range(24)]
        with self.assertRaises(ValueError):io.Payloads(rows,lambda:False)
        for row in rows:row['local_ref']['bytes']=1;row['logical_output_ref']['uncompressed_bytes']=io.PER_FILE_CAP+1
        with self.assertRaises(ValueError):io.Payloads(rows,lambda:False)
    def test_any_new_marker_bytes_stop_and_old_change_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'stop';config=dict(watched_stop_paths=[str(p)],historical_marker_refs=[])
            self.assertFalse(io.marker_stop(config))
            for raw in (b'',b'null',b'{malformed'):
                p.write_bytes(raw);self.assertTrue(io.marker_stop(config))
            p.unlink();config['historical_marker_refs']=[dict(path=str(p),bytes=1,sha256='0'*64)]
            self.assertTrue(io.marker_stop(config))
    def test_writer_prospective_quota_and_no_live_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=io.Writer(Path(directory)/'new');writer.write('small.json',b'{}')
            with patch.object(io,'OUTPUT_CAP',io.RESERVE+3):
                with self.assertRaises(ValueError):writer.write('small.json',b'{}')
            self.assertFalse((writer.root/'small.pending').exists())

if __name__=='__main__':unittest.main()
