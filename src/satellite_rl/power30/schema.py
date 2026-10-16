"""Selected W30 fields have actual projection columns, never placeholders."""
import hashlib
import json
from .menus import action_schema
GROUPS = {'M0': False, 'M1': True}

def state_width(group):
    return 208 + 100 * int(GROUPS[group])

def observation_schema(group):
    fields = dict(geo=[0,2], rate=[2,3], power=[3,4], interference=[4,104],
                  occupancy=[104,204], global_context=[204,207], width=[207,208])
    if GROUPS[group]:
        fields['margin_summary']=[208,308]
    return dict(version='S2W30ObservationV1', group=group, state_width=state_width(group), fields=fields)

def schema_identity(menu, group):
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
    value = dict(version='S2W30CheckpointV1', observation=observation_schema(group),
                 action=action_schema(menu), dimension=state_width(group), tokens=14,
                 readout='first_geo', physics_SHA=PHYSICS_SHA,
                 reward='legacy_repaired_T0_threshold_50Mbps_v1',
                 summary='log1p_sum_prefix_exp_neg_abs_capacity_demand_minus1_kappa_over_noise_plus_I')
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value
