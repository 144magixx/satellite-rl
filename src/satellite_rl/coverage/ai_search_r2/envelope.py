"""R1-equivalent envelope, with separate precharged visibility and angle work."""
import numpy as np
from ..ai_search.config import GeometryConfig
from ..geometry import visible_numpy,satellite_rays_numpy,ray_angle_numpy

def minimum_enclosing_width(scene,members,center,geometry=GeometryConfig(),budget=None):
    members=np.asarray(members,dtype=int)
    if not len(members):return None
    center=np.asarray(center).reshape(1,2)
    if budget:budget.geometry(0,'envelope_center_visibility')
    if not visible_numpy(center,scene.profile).all():return None
    ray=budget.center_rays(center,scene.profile)[0] if budget else satellite_rays_numpy(center,scene.profile)[0]
    if budget:budget.geometry(int(len(members)),'envelope_member_angles')
    required=2*float(ray_angle_numpy(scene.rays[members],ray).max())
    if required>geometry.width_max_deg+1e-12:return None
    return float(min(geometry.width_max_deg,max(geometry.width_min_deg,np.nextafter(required,np.inf))))
