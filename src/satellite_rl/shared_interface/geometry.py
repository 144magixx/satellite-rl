"""One R7 metre/radian ground-projection geometry, without RF objectives."""
import numpy as np
from .content import content_sha

GEOMETRY=dict(schema='R7GroundGeometryV1',profile='europe_v1',earth_radius_m=6371000,
    orbit_radius_m=42164000,satellite_lon_deg=13,satellite_lat_deg=0,min_elevation_deg=10,
    latitude_range_deg=[25,80],longitude_range_deg=[-40,50],altitude='source metadata retained; ground projection in every stage')
GEOMETRY_SHA=content_sha(GEOMETRY)


def assert_geometry_config():
    if content_sha(GEOMETRY)!=GEOMETRY_SHA:raise ValueError('shared geometry global config mutated')


def locations(value):
    assert_geometry_config()
    array=np.asarray(value,dtype=np.float64)
    if array.ndim!=2 or array.shape[1]!=2 or not np.isfinite(array).all():raise ValueError('finite lat_lon_deg[n,2] required')
    return array


def directions_and_distance(value):
    array=locations(value);lat,lon=np.deg2rad(array).T
    ground=GEOMETRY['earth_radius_m']*np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    longitude=np.deg2rad(GEOMETRY['satellite_lon_deg'])
    satellite=GEOMETRY['orbit_radius_m']*np.array([np.cos(longitude),np.sin(longitude),0.])
    vectors=ground-satellite;distance=np.linalg.norm(vectors,axis=1)
    return vectors/distance[:,None],distance


def validate_visible(value):
    array=locations(value)
    lat=np.deg2rad(array[:,0]);delta=np.deg2rad(array[:,1]-GEOMETRY['satellite_lon_deg'])
    cosine=np.cos(lat)*np.cos(delta)
    elevation=np.rad2deg(np.arctan2(cosine-GEOMETRY['earth_radius_m']/GEOMETRY['orbit_radius_m'],np.sqrt(np.maximum(0.,1-cosine*cosine))))
    if not ((array[:,0]>=25)&(array[:,0]<=80)&(array[:,1]>=-40)&(array[:,1]<=50)&(elevation>=10)).all():raise ValueError('outside shared GEO visibility')


def cone_omega(width_deg):
    assert_geometry_config()
    return float(np.sum(2*np.pi*(1-np.cos(np.deg2rad(np.asarray(width_deg,np.float64))/2))))


def audit_coverage(case,layout,owners):
    validate_visible(case['lat_lon_deg']);validate_visible(layout['centres_lat_lon_deg'])
    aircraft,_=directions_and_distance(case['lat_lon_deg']);beams,_=directions_and_distance(layout['centres_lat_lon_deg'])
    squared=np.sum((aircraft-beams[owners])**2,axis=1)
    limit=4*np.sin(np.deg2rad(layout['full_width_deg'][owners])/4)**2
    if not (squared<=limit+1e-14).all():raise ValueError('aircraft geometric coverage failed')
    actual=cone_omega(layout['full_width_deg'])
    if abs(actual-layout['geometry_Omega_sr'])>1e-12 or actual>layout['geometry_Omega_cap_sr']+1e-12:raise ValueError('geometric Omega mismatch/cap')
    return dict(valid=True,aircraft_count=len(owners),actual_Omega_sr=actual,coverage_tolerance=1e-14,Omega_tolerance_sr=1e-12)
