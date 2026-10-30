"""The only R7 S2/S3 coefficient and slot-capacity implementation."""
from dataclasses import dataclass
import numpy as np
from .geometry import directions_and_distance
from .spherical_gaussian import PHYSICS,PHYSICS_V2,PHYSICS_SHA,validate_physics,linear_gain


@dataclass(frozen=True)
class LinkCoefficients:
    beam_ids: tuple
    slot_owner: np.ndarray
    occupied_slots: np.ndarray
    signal: np.ndarray
    interference: np.ndarray
    noise: np.ndarray
    powers_initial: np.ndarray
    active: np.ndarray
    demand_bps: np.ndarray
    metadata: dict


def prepare_slot_kernel(layout,physics_config,case=None):
    """Allocation-independent physical cache, not an observation channel."""
    validate_physics(physics_config)
    rays,distance=directions_and_distance(layout['centres_lat_lon_deg'],validate=physics_config['version']!=PHYSICS_V2['version'])
    if physics_config['version']==PHYSICS_V2['version']:
        aircraft_rays,_=directions_and_distance(case['lat_lon_deg'],validate=False)
        squared=np.sum((aircraft_rays[:,None,:]-rays[None,:,:])**2,axis=-1)
        angles=2*np.arcsin(np.minimum(np.sqrt(squared)/2,1))
        aircraft_gain,_=linear_gain(angles,np.deg2rad(layout['full_width_deg'])[None,:],validate=False)
        ids=list(layout['beam_ids'])
        by_id={aid:i for i,aid in enumerate(case['aircraft_ids'])}
        members=[[] for _ in ids]
        for aid,bid in layout['owner_aircraft_to_beam_id']:members[ids.index(bid)].append(by_id[aid])
        gain=np.asarray([aircraft_gain[m].mean(axis=0) for m in members])
        frequency=physics_config['f0_Hz']+(np.arange(100)+.5)*physics_config['slot_Hz']
        path=(physics_config['c_m_per_s']/(4*np.pi*distance[:,None]*frequency[None,:]))**2
        return dict(gain=gain,path=path,physics_version=physics_config['version'])
    squared=np.sum((rays[:,None,:]-rays[None,:,:])**2,axis=-1)
    angles=2*np.arcsin(np.minimum(np.sqrt(squared)/2,1))
    gain,_=linear_gain(angles,np.deg2rad(layout['full_width_deg'])[None,:])
    frequency=physics_config['f0_Hz']+(np.arange(100)+.5)*physics_config['slot_Hz']
    path=(physics_config['c_m_per_s']/(4*np.pi*distance[:,None]*frequency[None,:]))**2
    gain.setflags(write=False);path.setflags(write=False)
    return dict(layout_SHA=layout['content_sha256'],physics_SHA=PHYSICS_SHA,gain=gain,path=path)


def raw_slot_coupling(layout, groups, physics_config, receiver_indices=None, transmitter_indices=None, slot_indices=None, kernel=None):
    """K[receiver, transmitter, slot], per 1 W/slot, without allocations.

    Optional index selections avoid materializing the full N*N*100 tensor.
    """
    validate_physics(physics_config)
    N=len(layout['beam_ids']);groups=np.asarray(groups,np.int64)
    if groups.shape!=(N,) or ((groups<0)|(groups>7)).any():raise ValueError('one actual group per beam')
    receivers=np.arange(N) if receiver_indices is None else np.asarray(receiver_indices,np.int64)
    transmitters=np.arange(N) if transmitter_indices is None else np.asarray(transmitter_indices,np.int64)
    kernel=prepare_slot_kernel(layout,physics_config) if kernel is None else kernel
    if physics_config['version']!=PHYSICS_V2['version']:
        if kernel['layout_SHA']!=layout['content_sha256'] or kernel['physics_SHA']!=PHYSICS_SHA:raise ValueError('physical cache identity')
    gain=kernel['gain'][np.ix_(receivers,transmitters)]
    slots=np.arange(100) if slot_indices is None else np.asarray(slot_indices,np.int64)
    path=kernel['path'][np.ix_(receivers,slots)]
    eligible=(groups[receivers,None]%2==groups[None,transmitters]%2)&(groups[receivers,None]!=groups[None,transmitters])
    return gain[:,:,None]*10**(physics_config['rx_gain_dBi']/10)*path[:,None,:]*eligible[:,:,None]


def build_link_coefficients(layout,allocation,physics_config,kernel=None):
    validate_physics(physics_config)
    ids=tuple(layout['beam_ids']);N=len(ids);mapping=[allocation['beam_ids'].index(bid) for bid in ids]
    starts=allocation['actual_start_slot'][mapping];counts=allocation['actual_slot_count'][mapping]
    powers=allocation['actual_whole_beam_power_w'][mapping]
    groups=np.asarray([allocation['group_by_beam_id'][bid] for bid in ids],np.int64)
    rays,distance=directions_and_distance(layout['centres_lat_lon_deg'],validate=physics_config['version']!=PHYSICS_V2['version'])
    squared=np.sum((rays[:,None,:]-rays[None,:,:])**2,axis=-1)
    angles=2*np.arcsin(np.minimum(np.sqrt(squared)/2,1))
    legacy=physics_config['version']!=PHYSICS_V2['version']
    tx_gain,underflows=linear_gain(angles,np.deg2rad(layout['full_width_deg'])[None,:],validate=legacy)
    tx_peak,_=linear_gain(np.zeros(N),np.deg2rad(layout['full_width_deg']),validate=legacy)
    rx_gain=10**(physics_config['rx_gain_dBi']/10)
    owner=np.repeat(np.arange(N,dtype=np.int64),counts)
    occupied=np.concatenate([np.arange(starts[i],starts[i]+counts[i],dtype=np.int64) for i in range(N)])
    active=counts>0;center=physics_config['f0_Hz']+(starts+counts/2)*physics_config['slot_Hz']
    signal_beam=np.zeros(N,np.float64)
    signal_beam[active]=(10**(-physics_config['signal_loss_dB']/10)*tx_peak[active]*rx_gain*
        (physics_config['c_m_per_s']/(4*np.pi*distance[active]*center[active]))**2/counts[active])
    if physics_config['version']==PHYSICS_V2['version']:
        signal_beam[active]*=np.diag(kernel['gain'])[active]/tx_peak[active]
    signal=signal_beam[owner]
    frequency=physics_config['f0_Hz']+(occupied+.5)*physics_config['slot_Hz']
    path=(physics_config['c_m_per_s']/(4*np.pi*distance[owner]*frequency))**2
    overlap=((groups[owner,None]%2==groups[None,:]%2)&(groups[owner,None]!=groups[None,:])&
             (occupied[:,None]>=starts[None,:])&(occupied[:,None]<starts[None,:]+counts[None,:]))
    interference=np.zeros((len(owner),N),np.float64)
    kernel=prepare_slot_kernel(layout,physics_config) if kernel is None else kernel
    # Same allocation-independent primitive as potential relation observations.
    # Gather occupied rows without retaining a full scene tensor.
    for receiver in np.unique(owner):
        rows=np.flatnonzero(owner==receiver)
        raw=raw_slot_coupling(layout,groups,physics_config,[receiver],np.flatnonzero(active),occupied[rows],kernel)[0]
        interference[np.ix_(rows,np.flatnonzero(active))]=raw.T/counts[None,active]
    interference*=overlap
    noise=np.full(len(owner),physics_config['k_J_per_K']*physics_config['noise_temperature_K']*physics_config['slot_Hz'])
    arrays=[owner,occupied,signal,interference,noise,powers.copy(),active.copy(),layout['request_bps_by_beam_int64'].copy()]
    for array in arrays:array.setflags(write=False)
    return LinkCoefficients(ids,*arrays,dict(physics_config_SHA=PHYSICS_SHA if legacy else None,parent_layout_SHA=layout.get('content_sha256'),
        allocation_SHA=allocation.get('content_sha256'),physics_config=physics_config,linear_gain_tail_underflows=underflows,
        matrix='occupied receiver-slot row, transmitter beam column',full_width='transmitter column width'))


def capacity_bps(coefficients,power_w=None):
    config=coefficients.metadata.get('physics_config',PHYSICS)
    validate_physics(config)
    power=coefficients.powers_initial if power_w is None else np.asarray(power_w,np.float64)
    if power.shape!=(len(coefficients.beam_ids),) or not np.isfinite(power).all() or (power<0).any() or (power[~coefficients.active]!=0).any() or float(power.sum())>3000:raise ValueError('fixed power actual constraints')
    denominator=coefficients.noise+coefficients.interference@power
    desired=coefficients.signal*power[coefficients.slot_owner]
    config=coefficients.metadata.get('physics_config',PHYSICS)
    if config['version']==PHYSICS_V2['version']:
        efficiency=np.minimum(np.log2(1+desired/denominator*10**(-config['implementation_loss_dB']/10)),config['spectral_efficiency_cap'])
        slot_capacity=config['slot_Hz']*efficiency
    else:
        slot_capacity=PHYSICS['slot_Hz']/np.log(2)*np.log1p(desired/denominator)
    total=np.bincount(coefficients.slot_owner,weights=slot_capacity,minlength=len(coefficients.beam_ids))
    if not np.isfinite(total).all() or (total<0).any():raise ValueError('finite capacity required')
    return total
