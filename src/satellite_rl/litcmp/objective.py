"""True service evaluations and objective-specific incumbent tracking."""
import numpy as np
from satellite_rl.s2fix.evaluate import metrics
from .moves import Alloc

POINTS = (32, 64, 128, 320, 640, 1280, 3200)


def j_gap(result):
    demand = result['demand_bps']
    return float(np.abs(result['capacity_bps'] - demand).sum() / demand.sum())


def j_w(result):
    return 1 - result['W']


class Probe:
    def __init__(self, stage_input, objective=j_gap):
        self.stage_input = stage_input
        self.objective = objective
        self.calls = 0
        self.best = None
        self.best_at = 0
        self.best_value = np.inf
        self.best_result = None
        self.history = []
        self.curve = {}

    def __call__(self, alloc, count=True):
        result = metrics(self.stage_input, *alloc)
        result['demand_bps'] = np.asarray(self.stage_input.layout['request_bps_by_beam_int64'])
        self.calls += int(count)
        value = self.objective(result)
        if value < self.best_value:
            self.best = Alloc(*(x.copy() for x in alloc))
            self.best_at, self.best_value, self.best_result = self.calls, value, result
        if count:
            self.history.append(dict(calls=self.calls, W=result['W'], gap=j_gap(result), objective=value))
            if self.calls in POINTS:
                self.curve[self.calls] = self.best_result['W']
        return result
