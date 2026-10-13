"""Single semantic contract for the environment, model and checkpoints."""
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA

TOKEN_DIM, RELATION_DIM, GLOBAL_DIM, ACTION_DIM = 16, 4, 12, 1001
STATE_SCHEMA = 'LayoutResourceTokensV1'
ACTION_SCHEMA = 'JointStartCountNoOpV1'
PHYSICS_SCHEMA = 'spherical_gaussian_tx_v1'
POWER_RULE_W_PER_SLOT = 3.75
REWARD_SCHEMA = 'all_N_si_delta'
TOKEN_FEATURES = ('latitude_over_90', 'longitude_over_180', 'full_width_deg',
    'peak_gain_over_1e5', 'distance_over_4e7', 'demand_Gbps', 'demand_share',
    'group_over_7', 'processed', 'current', 'start_over_100', 'count_over_10',
    'power_over_3000', 'capacity_Gbps', 'si', 'sgm')
RELATION_FEATURES = ('angle_over_pi', 'transmitter_gain_over_1e5',
    'cross_group_same_polarization', 'actual_slot_overlap_over_10')
GLOBAL_FEATURES = ('N_over_200', 'current_index_over_N', 'power_over_3000',
    'count_over_800', 'SI', 'SGM', 'total_demand_Gbps', 'power_rule_over_3000',
    'occupancy_block0', 'occupancy_block1', 'occupancy_block2', 'occupancy_block3')


def observation_semantics():
    return dict(state_schema=STATE_SCHEMA, action_schema=ACTION_SCHEMA,
        physics_schema=PHYSICS_SCHEMA, physics_SHA=PHYSICS_SHA,
        power_w_per_slot=POWER_RULE_W_PER_SLOT, reward_schema=REWARD_SCHEMA,
        token_features=TOKEN_FEATURES, relation_features=RELATION_FEATURES,
        global_features=GLOBAL_FEATURES)
