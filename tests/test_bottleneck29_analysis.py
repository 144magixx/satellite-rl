"""Small synthetic reader fixtures only; no worker/core/model/physical calls."""
import json
from pathlib import Path
import tempfile
import unittest
import pickle
import numpy as np
from satellite_rl.bottleneck29 import analysis as a


def fixture():
    mother=dict(mother=0,case_id='synthetic',N=1,layout_content_SHA='layout',
        processing_index_to_beam_id=['b0'],actions_indices_by_processing_order=[[0,0,0]],
        _layout_beam_ids=['b0'],_demands=[21600000.])
    snapshot=dict(beam_idx=0,identity=dict(layout_SHA='layout',physics_SHA=a.PHYSICS_SHA,order=['b0']),
        occupied=np.zeros((8,100),dtype=bool),powers=np.zeros(1,dtype=np.float64))
    binding=dict(source_SHA='synthetic-source-only');rows={}
    points=[('minimum5',[(f,l,5.) for f in range(100) for l in range(1,11) if f+l<=100]),
            ('fine_power',[(0,1,k/2.) for k in range(1,101)])]
    for table,values in points:
        for sequence,(f,l,p) in enumerate(values):
            key='synthetic/0/'+table+'/'+str(sequence)
            domain='original' if p in range(5,51,5) else ('lower_bound_relaxation' if p<5 else 'granularity_relaxation')
            spec=dict(start=f,count=l,power_w=p,family=table,valid=True,invalid_reasons=[],domain=domain)
            rate=np.asarray([p*1e6],dtype=np.float64);demand=np.asarray([21600000.])
            si=np.minimum(rate/demand,1);sgm=a.sgm_values(rate,demand,float(demand.mean()))
            action=dict(beam_id='b0',start=f,count=l,power_w=p)
            result=dict(status='completed',physical_calls=1,spec=spec,
                base_snapshot_SHA=a.content_sha(snapshot),capacity_bps=rate,demand_bps=demand,SI=si,SGM=sgm,
                beam_id='b0',prefix=dict(indices=np.asarray([0],dtype=np.int64),SI_sum=float(si[0]),SGM_sum=float(sgm[0])),
                current=dict(capacity_bps=float(rate[0]),demand_bps=float(demand[0])),
                requested_action=action,actual_action=action,
                constraints=dict(requested_execution_equal=True,slot_conflicts=0,total_power_w=p,future_reserved_power_w=0),reward=0.)
            rows[key]=dict(row_id=key,snapshot_sha='snapshot',binding=binding,spec=spec,result=result)
    # Same physical point, different table: cache must resolve original numeric source.
    rows['synthetic/0/fine_power/9']['result']=dict(status='cache',reference='synthetic/0/minimum5/0',physical_calls=0)
    return mother,snapshot,binding,rows


class ReaderTests(unittest.TestCase):
    def test_complete_state_and_baseline_cache(self):
        mother,snapshot,binding,rows=fixture()
        summary,table=a.summarize_state(rows,snapshot,mother,'snapshot',binding)
        self.assertEqual(summary['rows'],1055)
        self.assertEqual(summary['baseline']['current_capacity_bps'],5e6)
        self.assertEqual(summary['minimum5_capacity_bps'],5e6)
        self.assertTrue(summary['gate_booleans']['early_under_supply_residual'])
        self.assertEqual(summary['power_groups']['original_discrete']['valid_rows'],10)
        self.assertEqual(summary['power_groups']['fine_same_floor']['valid_rows'],81)
        self.assertEqual(summary['power_groups']['relaxed_lower_floor']['valid_rows'],9)
        self.assertEqual(len(table),1055)

    def test_missing_row_cannot_pass(self):
        mother,snapshot,binding,rows=fixture();rows.pop(next(iter(rows)))
        with self.assertRaisesRegex(a.EvidenceError,'exactly'):
            a.summarize_state(rows,snapshot,mother,'snapshot',binding)

    def test_cross_state_cache_and_unknown_rejected(self):
        mother,snapshot,binding,rows=fixture()
        rows['synthetic/0/fine_power/9']['result']['reference']='other/0/minimum5/0'
        with self.assertRaises(a.EvidenceError):a.resolve_rows(rows,'snapshot',binding)
        rows['synthetic/0/fine_power/9']['result']=dict(status='unknown',physical_calls=1)
        with self.assertRaises(a.EvidenceError):a.resolve_rows(rows,'snapshot',binding)

    def test_cache_wrong_power_and_binding_rejected(self):
        _,_,binding,rows=fixture();rows['synthetic/0/fine_power/9']['spec']['power_w']=5.5
        with self.assertRaisesRegex(a.EvidenceError,'point mismatch'):a.resolve_rows(rows,'snapshot',binding)
        _,_,binding,rows=fixture();rows['synthetic/0/minimum5/0']['binding']={}
        with self.assertRaises(a.EvidenceError):a.resolve_rows(rows,'snapshot',binding)

    def test_invalid_is_missing_not_zero(self):
        rows={'x':dict(row_id='x',snapshot_sha='s',binding={},spec=dict(valid=False,invalid_reasons=['occupied']),
                      result=dict(status='invalid',physical_calls=0))}
        self.assertIsNone(a.resolve_rows(rows,'s',{})['x'])

    def test_saved_metric_tamper_rejected(self):
        mother,snapshot,binding,rows=fixture()
        rows['synthetic/0/minimum5/0']['result']['prefix']['SI_sum']=1.
        with self.assertRaisesRegex(a.EvidenceError,'aggregate'):
            a.summarize_state(rows,snapshot,mother,'snapshot',binding)

    def test_incomplete_four_lanes_never_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=root/'manifest.json';contract=root/'contract.json'
            manifest.write_text(json.dumps(dict(rows=[dict(mother=i,case_id=str(i),N=80,snapshot_positions=[10,40,70]) for i in range(20)])))
            contract.write_text(json.dumps(dict(schema='S2W29A0DesignContractV2',inputs=dict(expected_states=60))))
            result=a.analyze_lanes([root/str(i) for i in range(4)],manifest,contract)
            self.assertFalse(result['complete']);self.assertFalse(result['eligible_for_A1_admission']);self.assertFalse(result['training_admitted'])
            self.assertEqual(result['calls']['PHY'],0)

    def test_torn_and_unfinished_journal(self):
        with self.assertRaisesRegex(a.EvidenceError,'torn'):
            a.read_ledger('ignored',lambda path:b'{}')
        def event(payload):
            encoded=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
            return json.dumps(dict(payload=payload,sha256=a._digest(encoded))).encode()+b'\n'
        raw=event(dict(type='binding',value=dict(binding={},limits={})))+event(dict(type='operation',id='q',kind='phy',status='intent'))
        self.assertEqual(a.read_ledger('ignored',lambda path:raw)['unresolved'],['q'])

    def test_complete_small_journal_cursor_completion_and_tamper(self):
        root=Path('synthetic-lane');files={};events=[]
        def append(payload):
            encoded=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
            events.append(json.dumps(dict(payload=payload,sha256=a._digest(encoded))).encode()+b'\n')
        append(dict(type='binding',value=dict(binding={'source_SHA':'synthetic'},limits={})))
        append(dict(type='session',id=0,reserved=1.,elapsed=.1))
        for key,value in [('cursor',dict(status='state_complete')),('completed',dict(status='completed')),
                          ('stopped',dict(status='stopped')),('case/0/snapshot',{})]:
            name=key+'.pkl' if '/' not in key else 'snapshot.pkl'
            raw=pickle.dumps(value);files[str(root/name)]=raw
            append(dict(type='artifact',id=key,path=name,sha=a._digest(raw)))
        journal=a.read_ledger('ledger',lambda path:b''.join(events))
        self.assertFalse(journal['unresolved'])
        a.check_lane_artifacts(journal,root,{'case/0/snapshot'},lambda path:files[str(path)])
        files[str(root/'completed.pkl')]=pickle.dumps(dict(status='uncompleted'))
        with self.assertRaisesRegex(a.EvidenceError,'checksum'):
            a.check_lane_artifacts(journal,root,{'case/0/snapshot'},lambda path:files[str(path)])


if __name__=='__main__':unittest.main()
