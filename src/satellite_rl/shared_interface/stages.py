"""Real new-version S1 publish, S2 import/actual export and S3 fixed-slot paths."""
import copy
import numpy as np
from .content import seal,read_wire,wire_bytes
from .contracts import prepare_record,validate_case,validate_layout,validate_allocation,processing_order,require_int,REQUEST_BPS,CONSTRAINT_VERSION
from .geometry import GEOMETRY_SHA,cone_omega
from .spherical_gaussian import PHYSICS,PHYSICS_SHA,validate_physics
from .link import build_link_coefficients,capacity_bps


def make_case(metadata,aircraft_ids,lat_lon_deg,*,request_bps=REQUEST_BPS,plain=False):
    """Pure record construction; caller pins its SHA before algorithm access."""
    required={'case_id','actual_role','source_role','original_mother_id','UTC','source_asset_SHA','source'}
    if set(metadata)!=required:raise ValueError('explicit complete case metadata required')
    ids=list(aircraft_ids);n=len(ids)
    record=dict(schema='UniformAircraftCaseV1',**copy.deepcopy(metadata),aircraft_ids=ids,
        lat_lon_deg=np.asarray(lat_lon_deg,dtype='<f8'),fixed_request_bps=request_bps,
        request_bps_int64=np.full(n,request_bps,dtype='<i8'),total_request_bps_int64=n*request_bps,
        geometry_config_SHA=GEOMETRY_SHA,units=dict(position='lat_lon_degrees',request='bit/s'))
    return dict(record, plain=True) if plain else seal(record)


def make_layout(case,*,beam_ids,source_slot_ids,anchor_aircraft_ids,centres_lat_lon_deg,full_width_deg,
                owner_aircraft_to_beam_id,Ncap,geometry_Omega_cap_sr,geometry_Omega_sr,geometry_audit,source_and_cost,
                stack_id=None,plain=False):
    """Pure S1 output record; no planning/reassignment or width fallback."""
    ids=list(beam_ids);N=len(ids)
    pairs=copy.deepcopy(owner_aircraft_to_beam_id)
    counts=np.zeros(N,dtype='<i8')
    for _,bid in pairs:
        if bid not in ids:raise ValueError('owner unknown beam')
        counts[ids.index(bid)]+=1
    record=prepare_record(dict(schema='BeamLayoutV1',parent_case_SHA=case.get('content_sha256',case['case_id']),
        aircraft_asset_SHA=case['source_asset_SHA'],owner_aircraft_to_beam_id=pairs,beam_ids=ids,
        source_slot_ids=list(source_slot_ids),anchor_aircraft_ids=list(anchor_aircraft_ids),centres_lat_lon_deg=centres_lat_lon_deg,
        full_width_deg=full_width_deg,N=N,aircraft_count_by_beam=counts,request_bps_by_beam_int64=counts*case['fixed_request_bps'],
        Ncap=Ncap,geometry_Omega_cap_sr=float(geometry_Omega_cap_sr),geometry_Omega_sr=float(geometry_Omega_sr),
        geometry_audit=copy.deepcopy(geometry_audit),source_and_cost=copy.deepcopy(source_and_cost),
        units=dict(position='lat_lon_degrees',full_width='degrees',request='bit/s',geometry_Omega='sr')))
    if stack_id is not None: record['stack_id']=np.asarray(stack_id,dtype=np.int64)
    return dict(record,plain=True) if plain else seal(record)


def algorithm_scene(case,authority):
    """Exact integer bps represented as float64 for unchanged S1 algorithms."""
    validate_case(case,authority)
    demands=case['request_bps_int64'].astype(np.float64)
    if not np.array_equal(demands.astype(np.int64),case['request_bps_int64']):raise ValueError('float64 request representation loses integer precision')
    from satellite_rl.coverage.ai_search.core import Scene
    return Scene(case['lat_lon_deg'],demands,tuple(case['aircraft_ids']),'europe_v1')


def s1_publish_import(case,layout,authority):
    """Actual S1 published terminal input audit; immutable parent publication."""
    validate_layout(case,layout,authority)
    return dict(case_SHA=case['content_sha256'],layout_SHA=layout['content_sha256'],N=layout['N'],
        total_request_bps_int64=int(layout['request_bps_by_beam_int64'].sum()))


class UniformStage2Input:
    def __init__(self,case,layout,authority,physics_config,*,plain=False):
        self.plain=plain
        if not plain: validate_layout(case,layout,authority)
        validate_physics(physics_config)
        self.case=copy.deepcopy(case);self.layout=copy.deepcopy(layout);self.authority=authority;self.physics=copy.deepcopy(physics_config)
        self.processing_index_to_beam_id=processing_order(layout)
        self.group_by_beam_id={bid:i%8 for i,bid in enumerate(self.processing_index_to_beam_id)}

    @classmethod
    def from_wire(cls,case_bytes,layout_bytes,authority,physics_config):
        return cls(read_wire(case_bytes),read_wire(layout_bytes),authority,physics_config)

    def export_actual_allocation(self,actual_by_beam_id,*,requested_actions_separate):
        if set(actual_by_beam_id)!=set(self.layout['beam_ids']):raise ValueError('one actual action per stable beam required')
        ids=list(self.processing_index_to_beam_id);rows=[actual_by_beam_id[bid] for bid in ids]
        if any(set(row)!=set(('group','start','count','power_w')) or row['group']!=self.group_by_beam_id[bid] for bid,row in zip(ids,rows)):raise ValueError('actual action fields/group')
        for row in rows:
            for name in ('group','start','count'):require_int(row[name])
            power=row['power_w']
            if isinstance(power,(bool,np.bool_)) or not isinstance(power,(int,float,np.integer,np.floating)) or not np.isfinite(power):raise ValueError('actual power finite numeric nonbool required')
        record=prepare_record(dict(schema='DiscreteAllocationV1',parent_case_SHA=self.case.get('content_sha256',self.case['case_id']),
            parent_layout_SHA=self.layout.get('content_sha256',self.case['case_id']),physics_config_SHA=PHYSICS_SHA,beam_ids=ids,
            processing_index_to_beam_id=ids.copy(),group_by_beam_id=self.group_by_beam_id.copy(),
            actual_start_slot=np.asarray([r['start'] for r in rows]),actual_slot_count=np.asarray([r['count'] for r in rows]),
            actual_whole_beam_power_w=np.asarray([r['power_w'] for r in rows],np.float64),
            requested_actions_separate=copy.deepcopy(requested_actions_separate),constraint_version=CONSTRAINT_VERSION,
            units=dict(slot='integer25MHz_slot',power='whole_beam_watt')))
        if self.plain:
            record.pop('physics_config_SHA')
            return dict(record,plain=True,physics_version=self.physics['version'])
        record=seal(record)
        validate_allocation(self.case,self.layout,record,self.authority,PHYSICS_SHA,published=False)
        # Caller publishes this independently validated actual record in its
        # audited parent registry before S3 consumption; no auto trust by rehash.
        return record

    def terminal_fixed_evaluation(self,allocation):
        validate_allocation(self.case,self.layout,allocation,self.authority,PHYSICS_SHA)
        coefficients=build_link_coefficients(self.layout,allocation,self.physics)
        return dict(actual_method='S2_uniform_terminal',case_SHA=self.case['content_sha256'],layout_SHA=self.layout['content_sha256'],
            allocation_SHA=allocation['content_sha256'],beam_ids=list(coefficients.beam_ids),coefficients=coefficients,
            capacity_bps=capacity_bps(coefficients),initial_power_w=coefficients.powers_initial.copy())


class UniformStage3FixedInput:
    @classmethod
    def from_layout_and_allocation(cls,case,layout,allocation,authority,physics_config):
        validate_physics(physics_config);validate_allocation(case,layout,allocation,authority,PHYSICS_SHA)
        result=cls();result.case=copy.deepcopy(case);result.layout=copy.deepcopy(layout);result.allocation=copy.deepcopy(allocation)
        result.authority=authority;result.physics=copy.deepcopy(physics_config);return result

    @classmethod
    def from_wire(cls,case_bytes,layout_bytes,allocation_bytes,authority,physics_config):
        return cls.from_layout_and_allocation(read_wire(case_bytes),read_wire(layout_bytes),read_wire(allocation_bytes),authority,physics_config)

    def fixed_initial_evaluation(self):
        validate_allocation(self.case,self.layout,self.allocation,self.authority,PHYSICS_SHA)
        coefficients=build_link_coefficients(self.layout,self.allocation,self.physics)
        return dict(actual_method='S3_uniform_fixed_initial',case_SHA=self.case['content_sha256'],layout_SHA=self.layout['content_sha256'],
            allocation_SHA=self.allocation['content_sha256'],beam_ids=list(coefficients.beam_ids),coefficients=coefficients,
            capacity_bps=capacity_bps(coefficients),initial_power_w=coefficients.powers_initial.copy())
