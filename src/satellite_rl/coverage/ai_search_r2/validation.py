import numpy as np
from ..ai_search.core import metrics
from ..ai_search.config import GeometryConfig
from ..geometry import satellite_rays_numpy,visible_numpy

def validate_owned(scene,layout,geometry=GeometryConfig(),budget=None):
    """Same R1 predicate/slack, computing only the unique owner ray pair."""
    n=len(layout.centers);a=layout.assignment;reasons=[]
    if scene.profile!=geometry.profile:reasons.append('profile_mismatch')
    if n>geometry.max_beams or (len(scene.demands) and n<1):reasons.append('beam_limit')
    if layout.widths.shape!=(n,):reasons.append('width_length')
    if not np.isfinite(layout.centers).all():reasons.append('nonfinite_center')
    if not np.isfinite(layout.widths).all() or (layout.widths<geometry.width_min_deg).any() or (layout.widths>geometry.width_max_deg).any():reasons.append('width_bounds')
    valid_ids=len(a)==len(scene.demands) and bool(((a>=0)&(a<n)).all())
    if not valid_ids:reasons.append('unassigned_or_invalid_owner')
    if n:
        if budget:budget.geometry(0,'final_center_visibility')
        if not visible_numpy(layout.centers,scene.profile).all():reasons.append('center_visibility')
    loads=np.zeros(n);slack=None
    if valid_ids and not reasons:
        if len(a):
            rays=budget.center_rays(layout.centers,scene.profile) if budget else satellite_rays_numpy(layout.centers,scene.profile)
            if budget:budget.geometry(int(len(a)),'final_owner_validation')
            chord=np.sum((scene.rays-rays[a])**2,axis=1)
            limits=4*np.sin(np.deg2rad(layout.widths[a]/2)/2)**2
            if not (chord<=limits+geometry.tolerance).all():reasons.append('geometric_coverage')
            loads=np.bincount(a,weights=scene.demands,minlength=n)
            if (np.bincount(a,minlength=n)==0).any():reasons.append('empty_beam')
            slack=float((limits-chord).min())
        elif n:reasons.append('empty_beam')
    conserved=valid_ids and bool(np.isclose(loads.sum(),scene.demands.sum(),rtol=1e-12,atol=1e-6))
    if not conserved:reasons.append('demand_conservation')
    return dict(valid=not reasons,reasons=reasons,beam_count=n,unique_assignment=valid_ids,demand_conserved=conserved,coverage_100=valid_ids and not reasons,loads=loads.tolist(),minimum_squared_ray_slack=slack,geometry_version='ai17_per_beam_ray_v1')
