"""Enlarged spherical-grid relaxation; only proved solver bounds raise LB.

Latitude/longitude here are coordinates of unit satellite rays, not ground
locations. Their spherical metric is bounded by the Euclidean coordinate
metric, so rounding moves a centre by at most g/sqrt(2) degrees. Both models
allow multiple centres per candidate (x<=N), independently of spacing.
"""
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeWarning, linprog, milp
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import cKDTree


def grid_edges(directions, g=.2):
    """Return enlarged candidate centres and aircraft/candidate edge indices."""
    directions = np.asarray(directions)
    latitude = np.rad2deg(np.arcsin(directions[:, 2]))
    longitude = np.rad2deg(np.unwrap(np.arctan2(directions[:, 1], directions[:, 0])))
    radius = 1.01 * (.5 + g / np.sqrt(2))
    lat = np.arange(np.floor((latitude.min() - radius) / g),
                    np.ceil((latitude.max() + radius) / g) + 1) * g
    # Satellite rays for Europe lie close to the equator in these coordinates.
    # The longitude margin includes the spherical latitude scale factor.
    lon_margin = radius / np.cos(np.deg2rad(np.max(np.abs(latitude)) + radius))
    lon = np.arange(np.floor((longitude.min() - lon_margin) / g),
                    np.ceil((longitude.max() + lon_margin) / g) + 1) * g
    la, lo = np.deg2rad(np.meshgrid(lat, lon, indexing="ij"))
    candidates = np.column_stack((np.cos(la.ravel()) * np.cos(lo.ravel()),
                                  np.cos(la.ravel()) * np.sin(lo.ravel()),
                                  np.sin(la.ravel())))
    neighbours = cKDTree(candidates).query_ball_point(
        directions, 2 * np.sin(np.deg2rad(radius) / 2))
    aircraft = np.repeat(np.arange(len(directions)), [len(row) for row in neighbours])
    candidate = np.concatenate(neighbours).astype(int)
    used, candidate = np.unique(candidate, return_inverse=True)
    return candidates[used], aircraft, candidate


def _model(n, candidates, aircraft, candidate, peak):
    c, e = len(candidates), len(aircraft)
    edge = np.arange(e)
    assignment = coo_matrix((np.ones(e), (aircraft, c + edge)), shape=(n, c + e)).tocsr()
    capacity = coo_matrix((np.r_[-np.full(c, peak), np.ones(e)],
                          (np.r_[np.arange(c), candidate],
                           np.r_[np.arange(c), c + edge])), shape=(c, c + e)).tocsr()
    opened = coo_matrix((np.r_[-np.ones(e), np.ones(e)],
                        (np.r_[edge, edge], np.r_[candidate, c + edge])),
                       shape=(e, c + e)).tocsr()
    return np.r_[np.ones(c), np.zeros(e)], assignment, vstack((capacity, opened), format="csr")


def compute_bound(frame_or_rays, N_budget, upper_peak, g=.2, milp_seconds=120):
    """Lower bound ignoring spacing in both LP and integer strengthening.

    ``upper_peak`` must come from a feasible layout. Unknown/time-limited
    solves leave the last proved lower bound unchanged. MILP time is shared
    across its tested peaks, rather than renewed after every binary step.
    """
    from .geom import rays
    directions = (rays(frame_or_rays.positions) if hasattr(frame_or_rays, "positions")
                  else np.asarray(frame_or_rays))
    n = len(directions)
    average = int(np.ceil(n / N_budget))
    if not n:
        return dict(LB_avg=0, LB_lp=0, LB_milp=0, g=g, lp_seconds=0., status="empty")
    started = perf_counter()
    candidates, aircraft, candidate = grid_edges(directions, g)
    c = len(candidates)
    statuses = []

    def relaxation(peak, integer=False, seconds=None):
        objective, equality, inequality = _model(n, candidates, aircraft, candidate, peak)
        upper = np.r_[np.full(c, N_budget), np.ones(len(aircraft))]
        if integer:
            constraints = [LinearConstraint(equality, np.ones(n), np.ones(n)),
                           LinearConstraint(inequality, -np.inf, 0)]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=RuntimeWarning)
                result = milp(objective, integrality=np.r_[np.ones(c), np.zeros(len(aircraft))],
                              bounds=Bounds(np.zeros(len(objective)), upper),
                              constraints=constraints, options={"time_limit": seconds, "threads": 1})
            proof = getattr(result, "mip_dual_bound", None)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=OptimizeWarning)
                result = linprog(objective, A_ub=inequality, b_ub=np.zeros(inequality.shape[0]),
                                 A_eq=equality, b_eq=np.ones(n),
                                 bounds=np.column_stack((np.zeros(len(objective)), upper)),
                                 method="highs", options={"threads": 1, "time_limit": 300.})
            proof = result.fun if result.success else None
        # A finite incumbent above N never proves impossibility at this T.
        ruled_out = result.status == 2 or (proof is not None and proof > N_budget + 1e-6)
        feasible = result.fun is not None and result.fun <= N_budget + 1e-7
        statuses.append(f"{'milp' if integer else 'lp'}:{peak}:{result.status}")
        return ruled_out, feasible

    low, high = average, int(upper_peak)
    while low < high:
        middle = (low + high) // 2
        excluded, feasible = relaxation(middle)
        if excluded:
            low = middle + 1
        elif feasible:
            high = middle
        else:
            break
    lp = low
    lp_seconds = perf_counter() - started
    if n < 600 and milp_seconds > 0:
        integer_start = perf_counter()
        low, high = lp, int(upper_peak)
        while low < high and perf_counter() - integer_start < milp_seconds:
            middle = (low + high) // 2
            excluded, feasible = relaxation(middle, True,
                                           milp_seconds - (perf_counter() - integer_start))
            if excluded:
                low = middle + 1
            elif feasible:
                high = middle
            else:
                break
    return dict(LB_avg=average, LB_lp=lp, LB_milp=low, g=g,
                lp_seconds=lp_seconds, status=";".join(statuses) or "average_equals_upper")
