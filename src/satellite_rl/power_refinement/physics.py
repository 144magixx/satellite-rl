"""Frozen-slot physical capacity and terminal service metrics.

Power is whole-beam watts. The formulas mirror ``envs.satellite``: desired
signal uses the allocated interval's centre, whereas cross-beam interference
uses each overlapping slot centre. The five dB SINR loss applies to the signal.
"""

from dataclasses import dataclass

import numpy as np


_C = 2.99792458e8
_K = 1.380649e-23
_BW = 2.5e7
_F0 = 1.77e10
_LOSS = 10 ** (-5 / 10)


def _pattern(gain_db, angle_deg):
    """The legacy antenna main lobe and Gaussian continuation in dBi."""
    width = 0.5
    sigma = width / (2 * np.sqrt(2 * np.log10(2)))
    angle = np.abs(angle_deg)
    return np.where(
        angle <= width / 2,
        gain_db - 3 * (angle / (width / 2)) ** 2,
        gain_db - 3 - 10 / np.log(10) * (angle ** 2 - (width / 2) ** 2)
        / (2 * sigma ** 2),
    )


def _directions(locations, profile):
    if profile == "europe_v1":
        longitude, earth, orbit = 13.0, 6371.0, 42164.0
    elif profile == "china_legacy":
        longitude, earth, orbit = 122.2, 6371.0, 42164.0
    else:
        raise ValueError(f"Unsupported physical profile: {profile!r}")
    lat, lon = np.deg2rad(locations).T
    ground = earth * np.column_stack((np.cos(lat) * np.cos(lon),
                                      np.cos(lat) * np.sin(lon), np.sin(lat)))
    satellite = np.array([orbit * np.cos(np.deg2rad(longitude)),
                          orbit * np.sin(np.deg2rad(longitude)), 0.0])
    vector = ground - satellite
    return vector / np.linalg.norm(vector, axis=1)[:, None]


def _sgm(capacity, demand):
    """Original SGM and its derivative with respect to capacity (bps)."""
    result = np.zeros_like(capacity)
    derivative = np.zeros_like(capacity)
    beta = float(demand.mean())
    positive = capacity > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        c, d = capacity[positive], demand[positive]
        ratio = c / d
        real = np.where(ratio >= 1, ratio - 1, 1 - 1 / ratio)
        real_prime = np.where(ratio >= 1, 1 / d, d / (c * c))
        imag = (c - d) / beta
        magnitude = np.hypot(real, imag)
        decay = np.exp(-magnitude)
        result[positive] = 1 - (1 - decay) ** 3
        slope_m = -3 * (1 - decay) ** 2 * decay
        slope_c = np.divide(real * real_prime + imag / beta, magnitude,
                            out=np.zeros_like(c), where=magnitude > 0)
        derivative[positive] = slope_m * slope_c
    # The right derivative at zero and derivative at exact demand are zero.
    return np.nan_to_num(result, nan=0.0), np.nan_to_num(derivative, nan=0.0)


@dataclass(frozen=True)
class PhysicalEvaluation:
    capacity_bps: np.ndarray
    si: float
    sgm: float
    beam_si: np.ndarray
    beam_sgm: np.ndarray
    capacity_jacobian: np.ndarray | None = None
    si_jacobian: np.ndarray | None = None
    sgm_jacobian: np.ndarray | None = None


class FixedAllocationPhysics:
    """Vectorized fixed-slot link model for a complete frozen scene."""

    def __init__(self, demand_bps, groups, starts, slots, signal, interference,
                 noise, slot_owner, power_budget_w, spectral_efficiency_cap=None,
                 implementation_loss_dB=0.):
        self.spectral_efficiency_cap = spectral_efficiency_cap
        self.implementation_loss_dB = implementation_loss_dB
        self.demand_bps = demand_bps
        self.groups = groups
        self.starts = starts
        self.slots = slots
        self.active = slots > 0
        self.signal = signal
        self.interference = interference
        self.noise = noise
        self.slot_owner = slot_owner
        self.beam_count = len(demand_bps)
        self.power_budget_w = power_budget_w

    @classmethod
    def from_scene(cls, scene, config, beams):
        """Freeze actual actions, rejecting requests or incomplete action logs."""
        n = len(beams)
        if n == 0 or set(row.get("beam_id") for row in beams) != set(range(n)):
            raise ValueError("beams must contain one actual allocation per beam")
        rows = sorted(beams, key=lambda row: row["beam_id"])
        def field(name, shape):
            value = np.asarray(scene[name], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid scene field: {name}")
            return value
        locations = field("beam_location", (n, 2))
        tx = field("beam_G_Tx_max", (n, 1)).reshape(-1)
        rx = field("UE_G_Rx_max", (n, 1)).reshape(-1)
        distance = field("distances", (n,)).reshape(-1) * 1000
        demand = field("rate", (n, 1)).reshape(-1)
        temperature = field("noise_temperature", (n, 1)).reshape(-1)
        if np.any(distance <= 0) or np.any(demand <= 0) or np.any(temperature <= 0):
            raise ValueError("Distance, demand, and noise temperature must be positive")
        groups, starts, slots = [], [], []
        for i, row in enumerate(rows):
            group, start, count = row.get("group"), row.get("freq"), row.get("slots")
            actual_power = row.get("power_w")
            if (not isinstance(group, (int, np.integer)) or group != i % 8
                    or not isinstance(count, (int, np.integer)) or not 0 <= count <= 10
                    or (count > 0 and (not isinstance(start, (int, np.integer))
                                       or not 0 <= start <= 100 - count))
                    or not isinstance(actual_power, (int, float, np.number))
                    or not np.isfinite(actual_power) or actual_power < 0
                    or (count == 0 and actual_power != 0)):
                raise ValueError(f"Invalid actual allocation for beam {i}")
            groups.append(int(group))
            starts.append(int(start) if count > 0 else 0)
            slots.append(int(count))
        groups = np.asarray(groups)
        starts = np.asarray(starts)
        slots = np.asarray(slots)
        for i in range(n):
            if not slots[i]:
                continue
            for j in range(i):
                if groups[i] == groups[j] and slots[j] and (
                    max(starts[i], starts[j]) < min(starts[i] + slots[i], starts[j] + slots[j])
                ):
                    raise ValueError("Actual allocations overlap within one group")
        profile = config.get("environment", {}).get("profile", "china_legacy")
        direction = _directions(locations, profile)
        cosines = np.clip(direction @ direction.T, -1, 1)
        angles = np.rad2deg(np.arccos(cosines))
        # Every row denotes one occupied receiver slot; signal has a single
        # interval-centre coefficient, interference has slot-centre coefficients.
        owner = np.repeat(np.arange(n), slots)
        occupied = np.concatenate([np.arange(starts[i], starts[i] + slots[i])
                                   for i in range(n)])
        centre = _F0 + (starts + slots / 2) * _BW
        signal_beam = np.zeros(n)
        active = slots > 0
        signal_beam[active] = (_LOSS * 10 ** ((tx[active] + rx[active]) / 10)
                               * (_C / (4 * np.pi * distance[active] * centre[active])) ** 2
                               / slots[active])
        signal = signal_beam[owner]
        slot_frequency = _F0 + (occupied + 0.5) * _BW
        path = (_C / (4 * np.pi * distance[owner] * slot_frequency)) ** 2
        gain = 10 ** ((_pattern(tx[None, :], angles[owner, :])
                       + rx[owner, None]) / 10)
        overlap = ((groups[owner, None] % 2 == groups[None, :] % 2)
                   & (groups[owner, None] != groups[None, :])
                   & (occupied[:, None] >= starts[None, :])
                   & (occupied[:, None] < starts[None, :] + slots[None, :]))
        interference = np.zeros_like(gain)
        interference[:, active] = gain[:, active] * path[:, None] / slots[None, active]
        interference *= overlap
        noise = _K * temperature[owner] * _BW
        budget = float(config.get("environment", {}).get("power_budget_w", 3000.0))
        if not np.isfinite(budget) or not 0 < budget <= 3000:
            raise ValueError("Invalid power budget")
        return cls(demand, groups, starts, slots, signal, interference, noise,
                   owner, budget)

    def evaluate(self, power_w, with_jacobian=False):
        power = np.asarray(power_w, dtype=np.float64)
        if (power.shape != (self.beam_count,) or not np.isfinite(power).all()
                or (power < 0).any() or (power[~self.active] != 0).any()):
            raise ValueError("Power must be finite, nonnegative, and zero for empty slots")
        denominator = self.noise + self.interference @ power
        signal = self.signal * 10 ** (-self.implementation_loss_dB / 10)
        desired = signal * power[self.slot_owner]
        capacity_slot = _BW / np.log(2) * np.log1p(desired / denominator)
        if self.spectral_efficiency_cap is not None:
            capped = capacity_slot >= _BW * self.spectral_efficiency_cap
            capacity_slot = np.minimum(capacity_slot, _BW * self.spectral_efficiency_cap)
        capacity = np.bincount(self.slot_owner, weights=capacity_slot,
                               minlength=self.beam_count)
        beam_si = np.minimum(capacity / self.demand_bps, 1.0)
        beam_sgm, sgm_prime = _sgm(capacity, self.demand_bps)
        if not with_jacobian:
            return PhysicalEvaluation(capacity, float(beam_si.mean()),
                                      float(beam_sgm.mean()), beam_si, beam_sgm)
        row_jac = (-_BW / np.log(2) * desired[:, None] * self.interference
                   / (denominator[:, None] * (denominator + desired)[:, None]))
        row_jac[np.arange(len(self.slot_owner)), self.slot_owner] += (
            _BW / np.log(2) * signal / (denominator + desired))
        if self.spectral_efficiency_cap is not None:
            row_jac[capped] = 0
        jac = np.zeros((self.beam_count, self.beam_count))
        np.add.at(jac, self.slot_owner, row_jac)
        si_prime = np.where(capacity < self.demand_bps, 1 / self.demand_bps, 0)
        return PhysicalEvaluation(capacity, float(beam_si.mean()),
                                  float(beam_sgm.mean()), beam_si, beam_sgm,
                                  jac, si_prime @ jac / self.beam_count,
                                  sgm_prime @ jac / self.beam_count)
