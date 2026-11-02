"""Convert literature geometry to ordinary downstream JSON records."""
from datetime import datetime
import json
import numpy as np
from satellite_rl.coverage.s1fix.data import Frame
from satellite_rl.coverage.s1fix import geom
from satellite_rl.s2fix.data import cone_omega, _json_value
from satellite_rl.shared_interface.stages import make_case, make_layout, UniformStage2Input
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_V2
from .hexgrid import hex_layout
from .kmeans_mec import kmeans_layout
from .voronoi_lb import voronoi_layout

METHODS = {"A1": hex_layout, "A2": kmeans_layout, "A3": voronoi_layout}


def solve_layout(frame, N_budget, method, seed=51001):
    return METHODS[method](frame, N_budget, seed) if method == "A2" else METHODS[method](frame, N_budget)


def frame_from_case(case):
    n = len(case["aircraft_ids"])
    utc = datetime.fromisoformat(case["UTC"].replace("Z", "+00:00"))
    timestamp = utc.timestamp()
    index = int(case["source"]["frame_index"] if "frame_index" in case["source"]
                else (timestamp - utc.replace(hour=0, minute=0, second=0).timestamp() - 120) / 300)
    return Frame(case["UTC"][:10], timestamp, index, np.asarray(case["aircraft_ids"]),
                 np.asarray(case["lat_lon_deg"]), np.zeros(n), np.zeros(n))


def to_stage2(result, case, request_bps):
    metadata = {key: case[key] for key in ("case_id", "actual_role", "source_role", "original_mother_id",
                                         "UTC", "source_asset_SHA", "source")}
    case = make_case(metadata, case["aircraft_ids"], case["lat_lon_deg"], request_bps=request_bps, plain=True)
    output = result.layout
    ids = [f"lit:{int(i):04d}" for i in output.beam_ids]
    distances = geom.angles(geom.rays(output.centers_latlon), geom.rays(case["lat_lon_deg"]))
    anchors = []
    for j in range(len(ids)):
        members = np.flatnonzero(output.owner == j)
        anchors.append(case["aircraft_ids"][members[distances[j, members].argmin()]])
    budget = result.N_budget
    layout = make_layout(case, beam_ids=ids, source_slot_ids=ids, anchor_aircraft_ids=anchors,
        centres_lat_lon_deg=output.centers_latlon, full_width_deg=output.widths,
        owner_aircraft_to_beam_id=[[aid, ids[j]] for aid, j in zip(case["aircraft_ids"], output.owner)],
        Ncap=budget, geometry_Omega_cap_sr=cone_omega(np.ones(budget)),
        geometry_Omega_sr=cone_omega(output.widths), geometry_audit=dict(method="literature", spacing=False),
        source_and_cost=dict(seconds=result.t_total), stack_id=np.full(len(ids), -1), plain=True)
    UniformStage2Input(case, layout, None, PHYSICS_V2, plain=True)
    return tuple(json.loads(json.dumps(record, default=_json_value)) for record in (case, layout))
