"""Pure serialization/rejection mocks; no actual geometry, RF or fixed12 run."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.shared_interface.content import seal,content_sha,wire_bytes,read_wire,declared_array,typed_tree
from satellite_rl.shared_interface.contracts import Authority,validate_case,validate_layout,validate_allocation,CONSTRAINT_VERSION
from satellite_rl.shared_interface.geometry import GEOMETRY_SHA
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS,PHYSICS_SHA
from satellite_rl.shared_interface.stages import make_case,make_layout,UniformStage2Input,UniformStage3FixedInput
from satellite_rl.shared_interface import geometry
from satellite_rl.shared_interface.spherical_gaussian import validate_physics


class PureUniformInterfaceChecks(unittest.TestCase):
    def records(self):
        case=make_case(dict(case_id='pure',actual_role='synthetic_engineering',source_role='synthetic_engineering',
            original_mother_id='pure',UTC='2026-10-13T00:00:00Z',source_asset_SHA='a'*64,source=dict(synthetic=True)),['a'],[[52.,8.]])
        layout=make_layout(case,beam_ids=['b'],source_slot_ids=['original_slot'],anchor_aircraft_ids=['a'],centres_lat_lon_deg=[[52.,8.]],
            full_width_deg=[.5],owner_aircraft_to_beam_id=[['a','b']],Ncap=1,geometry_Omega_cap_sr=1.,geometry_Omega_sr=.1,
            geometry_audit=dict(status='mock_no_geometry'),source_and_cost=dict(synthetic=True,geometry_calls=0))
        allocation=seal(dict(schema='DiscreteAllocationV1',parent_case_SHA=case['content_sha256'],parent_layout_SHA=layout['content_sha256'],
            physics_config_SHA=PHYSICS_SHA,beam_ids=['b'],processing_index_to_beam_id=['b'],group_by_beam_id={'b':0},
            actual_start_slot=np.array([0],dtype='<i8'),actual_slot_count=np.array([1],dtype='<i8'),actual_whole_beam_power_w=np.array([10.],dtype='<f8'),
            requested_actions_separate=dict(status='pure_mock'),constraint_version=CONSTRAINT_VERSION,units=dict(slot='integer25MHz_slot',power='whole_beam_watt')))
        registry=dict(schema='R7PublishedParentRegistryV1',case_sha_by_id={'pure':case['content_sha256']},layout_shas=[layout['content_sha256']],allocation_shas=[allocation['content_sha256']])
        return case,layout,allocation,Authority(registry,content_sha(registry))

    def test_typed_hash_units_bits_and_strict_integer(self):
        case,layout,allocation,authority=self.records()
        restored=read_wire(wire_bytes(case));self.assertEqual(restored['content_sha256'],case['content_sha256'])
        np.testing.assert_array_equal(restored['request_bps_int64'],case['request_bps_int64'])
        self.assertNotEqual(content_sha(dict(x=0.)),content_sha(dict(x=-0.)))
        self.assertNotEqual(content_sha(dict(x=np.array([1.],dtype='<f8'))),content_sha(dict(x=np.array([[1.]],dtype='<f8'))))
        with self.assertRaises(ValueError):declared_array([1.5],'request_bps_int64')
        with self.assertRaises(ValueError):typed_tree(dict(x=np.nan))
        with self.assertRaises(UnicodeEncodeError):typed_tree('\ud800')
        self.assertEqual(typed_tree(True),True)

    def test_rehashed_case_or_parent_changes_are_not_authority(self):
        case,layout,allocation,authority=self.records();validate_case(case,authority,geometry_check=False)
        altered=copy.deepcopy(case);altered['aircraft_ids']=['renamed'];altered=seal(altered)
        with self.assertRaisesRegex(ValueError,'not root-authorized'):validate_case(altered,authority,geometry_check=False)
        altered=copy.deepcopy(case);altered['request_bps_int64'][0]=21599999;altered['total_request_bps_int64']=21599999;altered=seal(altered)
        with self.assertRaisesRegex(ValueError,'reject_uniform_request'):validate_case(altered,authority,geometry_check=False)
        changed=copy.deepcopy(allocation);changed['parent_layout_SHA']='f'*64;changed=seal(changed)
        with self.assertRaisesRegex(ValueError,'parent/physics'):validate_allocation(case,layout,changed,authority,PHYSICS_SHA,geometry_check=False)

    def test_owner_width_actual_fields_fail_closed_before_geometry(self):
        case,layout,allocation,authority=self.records()
        changed=copy.deepcopy(layout);changed['owner_aircraft_to_beam_id']=[];changed=seal(changed)
        with self.assertRaisesRegex(ValueError,'reject_missing_aircraft'):validate_layout(case,changed,authority,geometry_check=False)
        changed=copy.deepcopy(layout);changed.pop('full_width_deg');changed=seal(changed)
        with self.assertRaisesRegex(ValueError,'reject_missing_width'):validate_layout(case,changed,authority,geometry_check=False)
        changed=copy.deepcopy(allocation);changed['actual_slot_count'][0]=0;changed=seal(changed)
        with self.assertRaisesRegex(ValueError,'slots/power'):validate_allocation(case,layout,changed,authority,PHYSICS_SHA,geometry_check=False)
        changed=copy.deepcopy(allocation);changed['group_by_beam_id']['b']=True;changed=seal(changed)
        with self.assertRaisesRegex(ValueError,'integer groups'):validate_allocation(case,layout,changed,authority,PHYSICS_SHA,geometry_check=False)

    def test_real_stage_caller_routes_only_shared_core_pure_mocks(self):
        case,layout,allocation,authority=self.records();fake=SimpleNamespace(beam_ids=('b',),powers_initial=np.array([10.]))
        with patch('satellite_rl.shared_interface.geometry.validate_visible') as visibility,patch('satellite_rl.shared_interface.geometry.audit_coverage') as coverage,patch('satellite_rl.shared_interface.stages.build_link_coefficients',return_value=fake) as builder,patch('satellite_rl.shared_interface.stages.capacity_bps',return_value=np.array([0.])) as capacity:
            second=UniformStage2Input.from_wire(wire_bytes(case),wire_bytes(layout),authority,PHYSICS).terminal_fixed_evaluation(allocation)
            third=UniformStage3FixedInput.from_wire(wire_bytes(case),wire_bytes(layout),wire_bytes(allocation),authority,PHYSICS).fixed_initial_evaluation()
            self.assertEqual(builder.call_count,2);self.assertEqual(capacity.call_count,2)
            self.assertEqual(second['layout_SHA'],third['layout_SHA']);self.assertEqual(second['allocation_SHA'],third['allocation_SHA'])
            self.assertTrue(visibility.called and coverage.called)  # mocked, no physical work

    def test_actual_input_bool_and_global_config_mutations_rejected_pure(self):
        case,layout,allocation,authority=self.records()
        with patch('satellite_rl.shared_interface.geometry.validate_visible'),patch('satellite_rl.shared_interface.geometry.audit_coverage'):
            stage=UniformStage2Input(case,layout,authority,PHYSICS)
            for field in ('group','start','count','power_w'):
                row=dict(group=0,start=0,count=1,power_w=10.);row[field]=False if field!='count' else True
                with self.assertRaises(ValueError):stage.export_actual_allocation({'b':row},requested_actions_separate={})
        previous=geometry.GEOMETRY['earth_radius_m']
        try:
            geometry.GEOMETRY['earth_radius_m']=previous+1
            with self.assertRaisesRegex(ValueError,'mutated'):geometry.assert_geometry_config()
        finally:geometry.GEOMETRY['earth_radius_m']=previous
        old=PHYSICS['slot_Hz']
        try:
            PHYSICS['slot_Hz']=old+1
            with self.assertRaises(ValueError):validate_physics(copy.deepcopy(PHYSICS))
        finally:PHYSICS['slot_Hz']=old


if __name__=='__main__':unittest.main()
