"""Strict versioned records and parent authority; no legacy coercion/fallback."""
from datetime import datetime
from types import MappingProxyType
import numpy as np
from .content import ARRAY_DTYPES,declared_array,seal,verify_self,content_sha
from . import geometry

REQUEST_BPS=21600000
CONSTRAINT_VERSION='R7DiscreteSlotsPowerV1'
CASE_FIELDS={'schema','case_id','actual_role','source_role','original_mother_id','UTC','source_asset_SHA',
    'aircraft_ids','lat_lon_deg','fixed_request_bps','request_bps_int64','total_request_bps_int64','geometry_config_SHA',
    'units','source','content_sha256'}
LAYOUT_FIELDS={'schema','parent_case_SHA','aircraft_asset_SHA','owner_aircraft_to_beam_id','beam_ids','source_slot_ids',
    'anchor_aircraft_ids','centres_lat_lon_deg','full_width_deg','N','aircraft_count_by_beam','request_bps_by_beam_int64',
    'Ncap','geometry_Omega_cap_sr','geometry_Omega_sr','geometry_audit','source_and_cost','units','content_sha256'}
ALLOCATION_FIELDS={'schema','parent_case_SHA','parent_layout_SHA','physics_config_SHA','beam_ids',
    'processing_index_to_beam_id','group_by_beam_id','actual_start_slot','actual_slot_count',
    'actual_whole_beam_power_w','requested_actions_separate','constraint_version','units','content_sha256'}


def require_sha(value):
    if not isinstance(value,str) or len(value)!=64 or set(value)-set('0123456789abcdef'):raise ValueError('SHA256 required')


def require_fields(record,expected,schema):
    if not isinstance(record,dict):raise ValueError('versioned object dictionary required')
    if schema=='BeamLayoutV1' and 'full_width_deg' not in record:raise ValueError('reject_missing_width')
    if set(record)!=expected or record.get('schema')!=schema:raise ValueError('new schema fields required; legacy forbidden')


def require_ids(value):
    if not isinstance(value,(list,tuple)) or any(not isinstance(x,str) or not x for x in value):raise ValueError('ordered stable string IDs required')
    for item in value:item.encode('utf8',errors='strict')
    if len(set(value))!=len(value):raise ValueError('duplicate stable ID')


def require_int(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):raise ValueError('integer scalar required')
    if not np.iinfo(np.int64).min<=int(value)<=np.iinfo(np.int64).max:raise ValueError('int64 overflow')
    return int(value)


def prepare_record(record):
    """Normalize declared numerical arrays without lossy integer conversion."""
    value=dict(record)
    for name in ARRAY_DTYPES:
        if name in value:value[name]=declared_array(value[name],name)
    return value


class Authority:
    """An external frozen manifest or independently audited publish registry.

    Constructor requires its caller's root-pinned manifest SHA. Self-rehashing
    a child cannot change this registry; transport paths live outside it.
    """
    def __init__(self,manifest,expected_manifest_sha):
        require_sha(expected_manifest_sha)
        if content_sha(manifest)!=expected_manifest_sha:raise ValueError('authority manifest SHA')
        if manifest.get('schema')!='R7PublishedParentRegistryV1':raise ValueError('authority schema')
        self.case_by_id=MappingProxyType(dict(manifest['case_sha_by_id']))
        self.layout_shas=frozenset(manifest['layout_shas'])
        self.allocation_shas=frozenset(manifest['allocation_shas'])
        self.manifest_sha=expected_manifest_sha

    def case(self,record):
        if self.case_by_id.get(record['case_id'])!=record['content_sha256']:raise ValueError('case not root-authorized; rehash not authority')

    def layout(self,record):
        if record['content_sha256'] not in self.layout_shas:raise ValueError('layout not audited published parent')

    def allocation(self,record):
        if record['content_sha256'] not in self.allocation_shas:raise ValueError('actual allocation not published authority')


def validate_case(record,authority,*,geometry_check=True):
    require_fields(record,CASE_FIELDS,'UniformAircraftCaseV1')
    if require_int(record['fixed_request_bps'])!=REQUEST_BPS:raise ValueError('reject_uniform_request')
    ids=record['aircraft_ids'];require_ids(ids);n=len(ids)
    if n<1 or n>np.iinfo(np.int64).max//REQUEST_BPS:raise ValueError('positive nonempty safe aircraft count')
    if not isinstance(record['request_bps_int64'],np.ndarray) or record['request_bps_int64'].dtype!=np.dtype('<i8') or record['request_bps_int64'].shape!=(n,) or not (record['request_bps_int64']==REQUEST_BPS).all():raise ValueError('reject_uniform_request')
    if require_int(record['total_request_bps_int64'])!=n*REQUEST_BPS:raise ValueError('integer total request conservation')
    if not isinstance(record['lat_lon_deg'],np.ndarray) or record['lat_lon_deg'].dtype!=np.dtype('<f8') or record['lat_lon_deg'].shape!=(n,2) or not np.isfinite(record['lat_lon_deg']).all():raise ValueError('original aircraft position array')
    if record['actual_role'] not in ('train','development','validation','test','synthetic_engineering') or record['source_role'] not in ('train','development','validation','test','synthetic_engineering'):raise ValueError('real role required')
    for key in ('case_id','original_mother_id','UTC'):
        if not isinstance(record[key],str) or not record[key]:raise ValueError('case identity fields')
    stamp=datetime.fromisoformat(record['UTC'].replace('Z','+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds()!=0:raise ValueError('explicit UTC required')
    require_sha(record['source_asset_SHA'])
    if record['geometry_config_SHA']!=geometry.GEOMETRY_SHA or record['units']!=dict(position='lat_lon_degrees',request='bit/s'):raise ValueError('geometry/units identity')
    verify_self(record);authority.case(record)
    if geometry_check:geometry.validate_visible(record['lat_lon_deg'])
    return record


def validate_layout(case,record,authority,*,geometry_check=True,published=True):
    validate_case(case,authority,geometry_check=geometry_check)
    require_fields(record,LAYOUT_FIELDS,'BeamLayoutV1')
    N=require_int(record['N']);n=len(case['aircraft_ids']);cap=require_int(record['Ncap'])
    if not 1<=N<=min(200,n,cap) or not 1<=cap<=min(200,n):raise ValueError('reject_N')
    ids=record['beam_ids'];require_ids(ids)
    if len(ids)!=N or len(record['source_slot_ids'])!=N or len(record['anchor_aircraft_ids'])!=N:raise ValueError('beam identity lengths')
    if any(not isinstance(v,str) or not v for v in record['source_slot_ids']+record['anchor_aircraft_ids']):raise ValueError('explicit slot/anchor strings')
    if record['parent_case_SHA']!=case['content_sha256'] or record['aircraft_asset_SHA']!=case['source_asset_SHA']:raise ValueError('layout parent/source mismatch')
    pairs=record['owner_aircraft_to_beam_id']
    if not isinstance(pairs,list) or any(not isinstance(pair,(list,tuple)) or len(pair)!=2 for pair in pairs):raise ValueError('owner pairs required')
    owner_ids=[p[0] for p in pairs]
    if len(set(owner_ids))!=len(owner_ids):raise ValueError('reject_duplicate_missing_aircraft')
    if owner_ids!=list(case['aircraft_ids']):raise ValueError('reject_missing_aircraft')
    if any(p[1] not in ids for p in pairs):raise ValueError('owner references unknown beam')
    owner=np.asarray([ids.index(p[1]) for p in pairs],np.int64);counts=np.bincount(owner,minlength=N)
    if not counts.all():raise ValueError('empty beam forbidden')
    for name,dtype,shape in [('centres_lat_lon_deg','<f8',(N,2)),('full_width_deg','<f8',(N,)),('aircraft_count_by_beam','<i8',(N,)),('request_bps_by_beam_int64','<i8',(N,))]:
        array=record[name]
        if not isinstance(array,np.ndarray) or array.dtype!=np.dtype(dtype) or array.shape!=shape or not np.isfinite(array).all():raise ValueError('layout declared array '+name)
    if not ((record['full_width_deg']>=.3)&(record['full_width_deg']<=1.)).all():raise ValueError('invalid full width')
    if not np.array_equal(record['aircraft_count_by_beam'],counts) or not np.array_equal(record['request_bps_by_beam_int64'],counts*REQUEST_BPS) or int(record['request_bps_by_beam_int64'].sum())!=case['total_request_bps_int64']:raise ValueError('exact integer per-beam request conservation')
    for index,anchor in enumerate(record['anchor_aircraft_ids']):
        if anchor not in case['aircraft_ids']:raise ValueError('anchor not original aircraft')
        if owner[case['aircraft_ids'].index(anchor)]!=index:raise ValueError('anchor is not a member of its beam')
    for name in ('geometry_Omega_cap_sr','geometry_Omega_sr'):
        if not isinstance(record[name],(float,np.floating)) or not np.isfinite(record[name]) or record[name]<=0:raise ValueError('explicit finite positive geometric Omega')
    if record['units']!=dict(position='lat_lon_degrees',full_width='degrees',request='bit/s',geometry_Omega='sr'):raise ValueError('layout units')
    verify_self(record)
    if geometry_check:geometry.audit_coverage(case,record,owner)
    if published:authority.layout(record)
    return record


def processing_order(layout):
    return sorted(layout['beam_ids'],key=lambda bid:(-int(layout['request_bps_by_beam_int64'][layout['beam_ids'].index(bid)]),bid))


def validate_allocation(case,layout,record,authority,physics_sha,*,geometry_check=True,published=True):
    validate_layout(case,layout,authority,geometry_check=geometry_check)
    require_fields(record,ALLOCATION_FIELDS,'DiscreteAllocationV1')
    if record['parent_case_SHA']!=case['content_sha256'] or record['parent_layout_SHA']!=layout['content_sha256'] or record['physics_config_SHA']!=physics_sha:raise ValueError('actual allocation parent/physics mismatch')
    ids=record['beam_ids'];require_ids(ids);N=layout['N']
    if len(ids)!=N or set(ids)!=set(layout['beam_ids']):raise ValueError('actual allocation beam set')
    order=processing_order(layout)
    if record['processing_index_to_beam_id']!=order:raise ValueError('initial processing order is immutable')
    groups={bid:i%8 for i,bid in enumerate(order)}
    if not isinstance(record['group_by_beam_id'],dict) or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in record['group_by_beam_id'].values()):raise ValueError('integer groups required')
    if record['group_by_beam_id']!=groups:raise ValueError('group changed during permutation')
    for name,dtype in [('actual_start_slot','<i8'),('actual_slot_count','<i8'),('actual_whole_beam_power_w','<f8')]:
        array=record[name]
        if not isinstance(array,np.ndarray) or array.dtype!=np.dtype(dtype) or array.shape!=(N,) or not np.isfinite(array).all():raise ValueError('actual allocation declared arrays')
    starts=record['actual_start_slot'];counts=record['actual_slot_count'];power=record['actual_whole_beam_power_w']
    if (counts<0).any() or (counts>10).any() or (starts<0).any() or (starts+counts>100).any() or (power<0).any() or ((counts==0)&((starts!=0)|(power!=0))).any() or float(power.sum())>3000:raise ValueError('actual slots/power constraints')
    for i in range(N):
        for k in range(i):
            if groups[ids[i]]==groups[ids[k]] and counts[i] and counts[k] and max(starts[i],starts[k])<min(starts[i]+counts[i],starts[k]+counts[k]):raise ValueError('reject_samegroup_overlap')
    if record['constraint_version']!=CONSTRAINT_VERSION or record['units']!=dict(slot='integer25MHz_slot',power='whole_beam_watt'):raise ValueError('constraint/units version')
    verify_self(record)
    if published:authority.allocation(record)
    return record
