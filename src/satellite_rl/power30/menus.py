"""Frozen physical watt menus; indices never imply a watt conversion."""
import hashlib
import json
import numpy as np

MENUS = {
    'P00': tuple(float(x) for x in range(5, 51, 5)),
    'P10': tuple(x / 2 for x in range(10, 101)),
    'P01': (0.5,) + tuple(float(x) for x in range(5, 51, 5)),
    'P11': tuple(x / 2 for x in range(1, 101)),
}

def power_table(menu):
    if menu not in MENUS:
        raise ValueError('Unknown W30 power menu')
    return MENUS[menu]

def action_schema(menu, plain=False):
    table = power_table(menu)
    value = dict(version='S2W30ActionV1', menu=menu, power_table_w=list(table),
                 action_dims=[100, 10, len(table)], unit='whole_beam_watt',
                 minimum_power_w=min(table), maximum_power_w=50.0, power_budget_w=3000.0,
                 future_reserve_power_w=5.0, future_reserve_slots=1,
                 zero_slot_allowed=False, clipping_allowed=False)
    if not plain:
        value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value

def legal_power_mask(menu, used_power_w, future_count):
    if not np.isfinite(used_power_w) or used_power_w < 0 or not isinstance(future_count, (int, np.integer)) or future_count < 0:
        raise ValueError('invalid power resource ledger')
    return np.asarray(power_table(menu)) <= 3000.0 - used_power_w - 5.0 * future_count

def validate_actual_power(menu, values, *, active=None):
    values = np.asarray(values, dtype=np.float64)
    selected=np.ones(values.shape,dtype=bool) if active is None else np.asarray(active)
    if selected.dtype!=np.bool_ or selected.shape!=values.shape:
        raise ValueError('active power selection must be aligned boolean mask')
    if (not np.isfinite(values).all() or not np.isin(values[selected],power_table(menu)).all()
            or (values[~selected]!=0).any() or float(values.sum())>3000):
        raise ValueError('actual power is outside W30 physical menu/budget')
