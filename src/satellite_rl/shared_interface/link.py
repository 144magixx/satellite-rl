"""The only R7 S2/S3 coefficient and slot-capacity implementation."""
from dataclasses import dataclass
import numpy as np
from .geometry import directions_and_distance
from .spherical_gaussian import PHYSICS,PHYSICS_SHA,validate_physics,linear_gain


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


def build_link_coefficients(layout,allocation,physics_config):
    validate_physics(physics_config)
    ids=tuple(layout['beam_ids']);N=len(ids);mapping=[allocation['beam_ids'].index(bid) for bid in ids]
    starts=allocation['actual_start_slot'][mapping];counts=allocation['actual_slot_count'][mapping]
    powers=allocation['actual_whole_beam_power_w'][mapping]
    groups=np.asarray([allocation['group_by_beam_id'][bid] for bid in ids],np.int64)
    rays,distance=directions_and_distance(layout['centres_lat_lon_deg'])
    squared=np.sum((rays[:,None,:]-rays[None,:,:])**2,axis=-1)
    angles=2*np.arcsin(np.minimum(np.sqrt(squared)/2,1))
    tx_gain,underflows=linear_gain(angles,np.deg2rad(layout['full_width_deg'])[None,:])
    tx_peak,_=linear_gain(np.zeros(N),np.deg2rad(layout['full_width_deg']))
    rx_gain=10**(physics_config['rx_gain_dBi']/10)
    owner=np.repeat(np.arange(N,dtype=np.int64),counts)
    occupied=np.concatenate([np.arange(starts[i],starts[i]+counts[i],dtype=np.int64) for i in range(N)])
    active=counts>0;center=physics_config['f0_Hz']+(starts+counts/2)*physics_config['slot_Hz']
    signal_beam=np.zeros(N,np.float64)
    signal_beam[active]=(10**(-physics_config['signal_loss_dB']/10)*tx_peak[active]*rx_gain*
        (physics_config['c_m_per_s']/(4*np.pi*distance[active]*center[active]))**2/counts[active])
    signal=signal_beam[owner]
    frequency=physics_config['f0_Hz']+(occupied+.5)*physics_config['slot_Hz']
    path=(physics_config['c_m_per_s']/(4*np.pi*distance[owner]*frequency))**2
    overlap=((groups[owner,None]%2==groups[None,:]%2)&(groups[owner,None]!=groups[None,:])&
             (occupied[:,None]>=starts[None,:])&(occupied[:,None]<starts[None,:]+counts[None,:]))
    interference=np.zeros((len(owner),N),np.float64)
    interference[:,active]=tx_gain[owner][:,active]*rx_gain*path[:,None]/counts[None,active]
    interference*=overlap
    noise=np.full(len(owner),physics_config['k_J_per_K']*physics_config['noise_temperature_K']*physics_config['slot_Hz'])
    arrays=[owner,occupied,signal,interference,noise,powers.copy(),active.copy(),layout['request_bps_by_beam_int64'].copy()]
    for array in arrays:array.setflags(write=False)
    return LinkCoefficients(ids,*arrays,dict(physics_config_SHA=PHYSICS_SHA,parent_layout_SHA=layout['content_sha256'],
        allocation_SHA=allocation['content_sha256'],linear_gain_tail_underflows=underflows,
        matrix='occupied receiver-slot row, transmitter beam column',full_width='transmitter column width'))


def capacity_bps(coefficients,power_w=None):
    validate_physics(PHYSICS)
    power=coefficients.powers_initial if power_w is None else np.asarray(power_w,np.float64)
    if power.shape!=(len(coefficients.beam_ids),) or not np.isfinite(power).all() or (power<0).any() or (power[~coefficients.active]!=0).any() or float(power.sum())>3000:raise ValueError('fixed power actual constraints')
    denominator=coefficients.noise+coefficients.interference@power
    desired=coefficients.signal*power[coefficients.slot_owner]
    slot_capacity=PHYSICS['slot_Hz']/np.log(2)*np.log1p(desired/denominator)
    total=np.bincount(coefficients.slot_owner,weights=slot_capacity,minlength=len(coefficients.beam_ids))
    if not np.isfinite(total).all() or (total<0).any():raise ValueError('finite capacity required')
    return total
