"""Actual publisher/reader wiring with synthetic structural-only mocks; no geometry/PHY."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest,read_json
from satellite_rl.workbench.r7_publish import publish_terminal,read_for_stage2
from satellite_rl.shared_interface.content import seal,content_sha,wire_bytes,read_wire
from satellite_rl.shared_interface.geometry import GEOMETRY_SHA
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
from satellite_rl.shared_interface.contracts import validate_layout
from scripts.stage1_ai_search_r4.audit import canonical_sha
from satellite_rl.workbench.r7_postprocess_source import new_output

POSTSOURCE=dict(postprocess_source_epoch='a'*40,actual_postprocess_source_file_sha256={'pure_mock.py':'b'*64},original_algorithm_source_is_separately_retained=True)
ALG_SOURCE='13702ff5d4c2aac8c559ddf79bdd2d8202dec45f'

def fixture(root):
    case=seal(dict(schema='UniformAircraftCaseV1',case_id='synthetic_mock_pub',actual_role='synthetic_engineering',source_role='synthetic_engineering',
                   original_mother_id='synthetic_mock',UTC='2024-01-08T00:02:00Z',source_asset_SHA='1'*64,aircraft_ids=['A','B'],
                   lat_lon_deg=np.array([[50.,8.],[51.,9.]],dtype='<f8'),fixed_request_bps=21600000,request_bps_int64=np.array([21600000,21600000],dtype='<i8'),
                   total_request_bps_int64=43200000,geometry_config_SHA=GEOMETRY_SHA,units=dict(position='lat_lon_degrees',request='bit/s'),source={'pure_mock':True}))
    layout=dict(centers=[[50.,8.],[51.,9.]],widths_deg=[.3,.7],assignment=[0,1],beam_ids=['real_bA','real_bB'],anchor_aircraft_ids=['A','B'])
    prepared=dict(Ncap=2,Omega_cap=.01);audit=dict(valid=True,parent_case_SHA=case['content_sha256'],layout_sha256=canonical_sha(layout),metrics={'actual_Omega':.001})
    case_path=root/'case.json';case_path.write_bytes(wire_bytes(case));case_ref=dict(path=str(case_path),sha256=digest(case_path))
    registry=root/'authority.json';atomic_json(registry,dict(schema='R7PublishedParentRegistryV1',case_sha_by_id={case['case_id']:case['content_sha256']},layout_shas=[],allocation_shas=[]))
    source=root/'original_cost.json';atomic_json(source,{'pure_mock':True});refs=dict(original_terminal_asset_SHA=dict(path=str(source),sha256=digest(source)),original_cost_endpoint_SHA=dict(path=str(source),sha256=digest(source)))
    return case,case_ref,dict(path=str(registry),sha256=digest(registry)),layout,prepared,audit,refs

class ActualTypedPublication(unittest.TestCase):
    def test_real_publisher_shared_layout_wire_registry_and_new_S2_reader(self):
        import satellite_rl.workbench.r7_publish as subject
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root);published={}
            with patch.object(subject,'shared_layout',wraps=subject.shared_layout) as actual_helper:
                for method in ('AI','RULE','RANDOM','T'):published[method]=publish_terminal(*args,root/method,method,postprocess_source=POSTSOURCE,original_algorithm_source_epoch=ALG_SOURCE)
                self.assertEqual(actual_helper.call_count,4)
            for method,result in published.items():
                record=read_wire(Path(result['layout_ref']['path']).read_bytes());self.assertEqual(record['beam_ids'],['real_bA','real_bB']);self.assertEqual(record['N'],2)
                self.assertEqual(record['full_width_deg'].tolist(),[.3,.7]);self.assertEqual(record['request_bps_by_beam_int64'].tolist(),[21600000,21600000])
                self.assertEqual(record['owner_aircraft_to_beam_id'],[['A','real_bA'],['B','real_bB']]);self.assertGreaterEqual(result['inner_descriptor_through_seconds'],0.)
                cost=read_json(result['cost_ref']['path'],{});self.assertTrue(cost['requires_caller_whole_function_elapsed_for_complete_cost']);self.assertEqual(cost['method'],method)
                descriptor=read_json(result['publication_ref']['path'],{});audit=read_json(descriptor['audit_ref']['path'],{})
                for output in (descriptor,audit,cost):self.assertEqual(output['postprocess_source_epoch'],'a'*40);self.assertEqual(output['original_algorithm_source_epoch'],ALG_SOURCE);self.assertEqual(output['actual_postprocess_source_file_sha256'],POSTSOURCE['actual_postprocess_source_file_sha256'])
                # Real new S2 constructor consumes actual published wire and
                # authority; only geometry re-evaluation is explicitly mocked.
                with patch('satellite_rl.shared_interface.stages.validate_layout',side_effect=lambda c,l,a:validate_layout(c,l,a,geometry_check=False)):
                    stage2=read_for_stage2(result['publication_ref'],PHYSICS)
                self.assertEqual(stage2.layout['content_sha256'],result['layout_content_SHA']);self.assertEqual(stage2.processing_index_to_beam_id,['real_bA','real_bB'])
    def test_wrong_layout_rejected_by_existing_audit_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,cref,authority,layout,prepared,audit,refs=fixture(root);layout['widths_deg'][0]=.6
            with self.assertRaisesRegex(ValueError,'already executed geometry audit'):publish_terminal(case,cref,authority,layout,prepared,audit,refs,root/'wrong','AI',postprocess_source=POSTSOURCE,original_algorithm_source_epoch=ALG_SOURCE)
    def test_rehashed_wrong_case_is_not_parent_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,cref,authority,layout,prepared,audit,refs=fixture(root);case=dict(case,case_id='different');case=seal(case)
            with self.assertRaises(ValueError):publish_terminal(case,cref,authority,layout,prepared,audit,refs,root/'wrong','T',postprocess_source=POSTSOURCE,original_algorithm_source_epoch=ALG_SOURCE)
    def test_old_gate_or_qualified_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prior=root/'prior';prior.mkdir();(prior/'qualified_train.json').write_text('preserve')
            with self.assertRaises(ValueError):new_output(prior/'gate.json')
            self.assertEqual((prior/'qualified_train.json').read_text(),'preserve')
            new=root/'new';self.assertEqual(new_output(new/'gate.json'),(new/'gate.json').resolve())

if __name__=='__main__':unittest.main()
