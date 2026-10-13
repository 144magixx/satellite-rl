"""External layout group projection onto fixed R3 slots, never teacher centers."""
import time
import numpy as np
from .pool import WIDTHS,omega
from ..ai_search_r2.validation import validate_owned
from ..geometry import satellite_rays_numpy

def audit_projection(p,layout):
    started=time.monotonic();audit=validate_owned(p.scene,layout)
    eligible=audit['valid'] and len(layout.centers)<=p.ncap and float(omega(layout.widths).sum())<=p.omega_cap+1e-12
    if not eligible:return dict(status='resource_ineligible',source_audit=audit,seconds=time.monotonic()-started)
    # Finite bipartite assignment minimizes menu solid angle, not layout quality.
    from scipy.optimize import linear_sum_assignment
    slots=p.available;cost=np.full((len(layout.centers),len(slots)),np.inf)
    for b in range(len(layout.centers)):
        members=np.flatnonzero(layout.assignment==b)
        covers=p.coverage[slots//6,(slots//2)%3][:,members].all(1)
        cost[b,covers]=omega(WIDTHS[(slots[covers]//2)%3])
    unmatched=np.flatnonzero(~np.isfinite(cost).any(1)).tolist()
    matched=None;reason=None
    if not unmatched:
        try:
            r,c=linear_sum_assignment(cost)
            if len(r)==len(layout.centers) and np.isfinite(cost[r,c]).all():matched=slots[c].tolist()
            else:reason='finite_copy_matching_incomplete'
        except ValueError:reason='finite_copy_matching_infeasible'
    minimum=float(sum(omega(WIDTHS[(np.asarray(matched)//2)%3]))) if matched is not None else None
    return dict(status='projectable' if matched is not None and minimum<=p.omega_cap+1e-12 else 'projection_failed',
        resource_eligible=True,unmatched_groups=unmatched,reason=reason,matching_slot_ids=matched,
        minimum_projected_menu_Omega=minimum,menu_resource_gap=None if minimum is None else minimum-p.omega_cap,
        copy_limit=2,source_audit=audit,geometry_groups_checked=len(layout.centers),seconds=time.monotonic()-started)
