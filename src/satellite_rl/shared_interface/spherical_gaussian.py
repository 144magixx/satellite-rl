"""New idealized spherical transmit pattern; never a legacy/ITU equivalence."""
import numpy as np
from .content import content_sha
from .geometry import GEOMETRY,GEOMETRY_SHA

PHYSICS=dict(schema='R7SharedPhysicsV1',version='spherical_gaussian_tx_v1',geometry=GEOMETRY,
    geometry_config_SHA=GEOMETRY_SHA,reference_full_width_deg=.5,reference_peak_gain_dBi=50.,
    rx_gain_dBi=40.,noise_temperature_K=290.,slot_Hz=25000000.,f0_Hz=17700000000.,slots=100,
    signal_loss_dB=5.,c_m_per_s=299792458.,k_J_per_K=1.380649e-23,power_budget_w=3000.,
    receiver='beam-centre representative, aligned independent terminal',width='full_half_power_degrees',
    classification='idealized engineering pattern; no sidelobes scan loss or hardware feasibility claim')
PHYSICS_SHA=content_sha(PHYSICS)
PHYSICS_V2=dict(PHYSICS, schema='R7SharedPhysicsV2',version='spherical_gaussian_tx_v2',
    noise_temperature_K=631.,spectral_efficiency_cap=4.5,implementation_loss_dB=1.,
    receiver='member-average off-axis gain')


def validate_physics(config):
    if config.get("version")==PHYSICS_V2["version"]:
        if config!=PHYSICS_V2:raise ValueError("exact v2 physics config required")
        return
    from .geometry import assert_geometry_config
    assert_geometry_config()
    if content_sha(PHYSICS)!=PHYSICS_SHA or content_sha(config)!=PHYSICS_SHA:raise ValueError('exact new shared physics config required; mutated global forbidden')


def kappa(width_rad,validate=True):
    if validate:validate_physics(PHYSICS)
    width=np.asarray(width_rad,np.float64)
    if not np.isfinite(width).all() or not ((width>=np.deg2rad(.3))&(width<=np.deg2rad(1.))).all():raise ValueError('full width outside .3..1 degrees')
    return np.log(2)/(2*np.sin(width/4)**2)


def pattern_solid_angle(width_rad,validate=True):
    concentration=kappa(width_rad,validate)
    return 2*np.pi*(-np.expm1(-2*concentration))/concentration


def gain_dbi(theta_rad,width_rad,validate=True):
    theta=np.asarray(theta_rad,np.float64)
    if not np.isfinite(theta).all() or ((theta<0)|(theta>np.pi)).any():raise ValueError('off-axis radians0..pi required')
    concentration=kappa(width_rad,validate)
    reference=pattern_solid_angle(np.deg2rad(.5),validate);angle=pattern_solid_angle(width_rad,validate)
    result=50.+10*np.log10(reference/angle)-(10/np.log(10))*2*concentration*np.sin(theta/2)**2
    if not np.isfinite(result).all():raise ValueError('finite log-domain gain required')
    return result


def linear_gain(theta_rad,width_rad,validate=True):
    db=gain_dbi(theta_rad,width_rad,validate)
    with np.errstate(under='ignore'):linear=np.exp(db*np.log(10)/10)
    return linear,int((linear==0).sum())
