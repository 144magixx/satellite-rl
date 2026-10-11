"""Pure frequency/length construction with exact all-beam physical rewards."""
from copy import deepcopy
import time
import numpy as np
from satellite_rl.power_refinement.physics import FixedAllocationPhysics

SCENE_FIELDS = ("beam_location", "beam_G_Tx_max", "UE_G_Rx_max", "distances", "rate", "noise_temperature")
OBSERVATION_CONTRACT = {"version": "f20_full_markov_v1", "beam_width": 24, "resource_width": 803,
    "normalization": {"latitude_deg": 90., "longitude_deg": 180., "beamwidth_deg": 1.,
        "demand_bps": 1e9, "capacity_bps": 1e9, "processing_index": 200., "frequency_start": 100.,
        "slot_count": 10., "power_w": 3000., "gain_db": 100., "distance_km": 50000., "temperature_k": 1000.},
    "normalization_source": "fixed physical units; no validation/test fitted statistics"}


def baseline_action(env, kind="greedy", rng=None):
    legal = np.flatnonzero(env.action_mask())
    if kind == "random":
        if rng is None:
            raise ValueError("Random baseline requires independent explicit RNG")
        return int(rng.choice(legal))
    if kind != "greedy":
        raise ValueError("Unknown baseline")
    requested = int(np.clip(np.ceil(env.scene["rate"][env.index, 0]/1e8), 1, 5))
    current_group = env.index % 8
    cross = env.occupancy[np.arange(8) != current_group].sum(axis=0)
    prefix = np.r_[0, np.cumsum(cross)]
    starts, lengths = legal//10, legal % 10+1
    overlap = prefix[starts+lengths]-prefix[starts]
    return int(legal[np.lexsort((legal, overlap, np.abs(lengths-requested)))[0]])


class Fair20Env:
    def __init__(self, scene, config=None):
        self.config = deepcopy(config or {})
        self.profile = self.config.get("profile", self.config.get("environment", {}).get("profile", "europe_v1"))
        if self.profile != "europe_v1":
            raise ValueError("F20 requires europe_v1")
        self.n = len(scene["rate"])
        if not 50 <= self.n <= 200:
            raise ValueError("F20 requires 50..200 positive-demand beams")
        ids = np.asarray(scene.get("original_indices", np.arange(self.n)), dtype=np.int64)
        if len(set(ids.tolist())) != self.n:
            raise ValueError("Beam identities must be unique")
        order = np.lexsort((ids, -np.asarray(scene["rate"]).reshape(-1)))
        self.scene = {k: np.asarray(scene[k], dtype=np.float64)[order].copy() for k in SCENE_FIELDS}
        shapes = {"beam_location": (self.n, 2), "distances": (self.n,)}
        for key, arr in self.scene.items():
            if arr.shape != shapes.get(key, (self.n, 1)) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid scene field {key}")
        for key in ("rate", "noise_temperature", "distances"):
            if np.any(self.scene[key] <= 0):
                raise ValueError(f"Nonpositive {key}")
        self.scene["original_indices"] = ids[order]
        self.groups = np.arange(self.n) % 8
        self.reset()

    def reset(self):
        self.index = 0
        self.starts = np.zeros(self.n, dtype=np.int64)
        self.slots = np.zeros(self.n, dtype=np.int64)
        self.power = np.zeros(self.n)
        self.occupancy = np.zeros((8, 100), dtype=bool)
        self.capacity = np.zeros(self.n)
        self.beam_si = np.zeros(self.n)
        self.beam_sgm = np.zeros(self.n)
        self.si = self.sgm = self.reward_sum = self.physics_seconds = 0.0
        self.trace = []
        return self.observation()

    def mask_diagnostics(self):
        if self.index == self.n:
            return {"mask": np.zeros(1000, bool), "legal_count": 0}
        group = self.index % 8
        f = np.repeat(np.arange(100), 10)
        lengths = np.tile(np.arange(1, 11), 100)
        bounds = f + lengths <= 100
        prefix = np.r_[0, np.cumsum(self.occupancy[group])]
        overlap = prefix[np.minimum(f + lengths, 100)] - prefix[f] > 0
        future = int(np.sum(self.groups[self.index + 1:] == group))
        reserve = 100 - int(self.occupancy[group].sum()) - lengths >= future
        mask = bounds & ~overlap & reserve
        if not mask.any():
            raise RuntimeError("F20 empty legal set: contract failure")
        return {"mask": mask, "legal_count": int(mask.sum()), "bounds_blocked": int((~bounds).sum()),
                "overlap_blocked": int((bounds & overlap).sum()),
                "reserve_blocked": int((bounds & ~overlap & ~reserve).sum()), "future_group_beams": future}

    def action_mask(self):
        return self.mask_diagnostics()["mask"]

    def beam_results(self):
        return [{"beam_id": i, "original_beam_id": int(self.scene["original_indices"][i]),
                 "group": int(self.groups[i]), "freq": int(self.starts[i]), "slots": int(self.slots[i]),
                 "power_w": float(self.power[i]), "rate_bps": float(self.capacity[i]),
                 "required_bps": float(self.scene["rate"][i, 0]), "si": float(self.beam_si[i]),
                 "sgm": float(self.beam_sgm[i])} for i in range(self.n)]

    def observation(self):
        b = np.zeros((200, 20), dtype=np.float32)
        i = np.arange(self.n)
        b[:self.n, :2] = self.scene["beam_location"] / np.array([90., 180.])
        b[:self.n, 2] = .5
        b[i, 3 + self.groups] = 1
        b[:self.n, 11] = self.scene["rate"].reshape(-1) / 1e9
        b[:self.n, 12] = i / 200
        b[:self.n, 13] = self.starts / 100
        b[:self.n, 14] = self.slots / 10
        b[:self.n, 15] = self.power / 3000
        b[:self.n, 16] = self.capacity / 1e9
        b[:self.n, 17] = self.beam_si
        b[:self.n, 18] = i < self.index
        b[:self.n, 19] = i == self.index
        # Full static physical parameters are supplied as additional beam fields.
        physical = np.zeros((200, 4), dtype=np.float32)
        for col, (key, scale) in enumerate((("beam_G_Tx_max", 100.), ("UE_G_Rx_max", 100.),
                                            ("distances", 50000.), ("noise_temperature", 1000.))):
            physical[:self.n, col] = self.scene[key].reshape(-1) / scale
        return {"beams": np.concatenate((b, physical), axis=1),
                "resources": np.r_[self.occupancy.reshape(-1), (3000-self.power.sum())/3000,
                                    self.n/200, (self.n-self.index)/200].astype(np.float32),
                "padding": np.arange(200) >= self.n, "current": min(self.index, self.n-1)}

    def step(self, action):
        diagnostic = self.mask_diagnostics()
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 1000 or not diagnostic["mask"][action]:
            raise ValueError("Illegal F20 action; no hidden repair")
        f, length = divmod(int(action), 10)
        length += 1
        idx = self.index
        old_si = self.si
        self.starts[idx], self.slots[idx], self.power[idx] = f, length, 3000/self.n
        self.occupancy[idx % 8, f:f+length] = True
        self.index += 1
        start = time.perf_counter()
        evaluation = FixedAllocationPhysics.from_scene(self.scene, {"environment": {
            "profile": self.profile, "power_budget_w": 3000}}, self.beam_results()).evaluate(self.power)
        seconds = time.perf_counter()-start
        self.physics_seconds += seconds
        self.capacity, self.beam_si, self.beam_sgm = evaluation.capacity_bps, evaluation.beam_si, evaluation.beam_sgm
        self.si, self.sgm = evaluation.si, evaluation.sgm
        reward = 100*(self.si-old_si)
        self.reward_sum += reward
        info = {k: v for k, v in diagnostic.items() if k != "mask"}
        info.update(requested_action=int(action), actual_action=int(action), beam_id=idx,
                    freq=f, slots=length, power_w=float(self.power[idx]), si=self.si, sgm=self.sgm,
                    physics_seconds=seconds, terminated=self.index == self.n, truncated=False)
        self.trace.append(info.copy())
        return self.observation(), reward, self.index == self.n, info

    def metrics(self):
        return {"si": self.si, "sgm": self.sgm, "si_mean": self.si, "sgm_mean": self.sgm,
                "complete": self.index == self.n, "beam_count": self.n, "total_power_w": float(self.power.sum()),
                "total_slots": int(self.slots.sum()), "reward_sum": self.reward_sum,
                "physics_seconds": self.physics_seconds, "overload": bool(self.power.sum() > 3000+1e-8),
                "clipped_count": 0, "zero_service_fraction": float(np.mean(self.capacity == 0)),
                "basic_fraction": float(np.mean(self.beam_si >= .1)), "basic_threshold_fraction": .1,
                "total_capacity_bps": float(self.capacity.sum())}

    def snapshot(self):
        return {"scene": deepcopy(self.scene), "beams": self.beam_results(), "metrics": self.metrics(),
                "trace": deepcopy(self.trace)}

    def state_dict(self):
        return deepcopy(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(deepcopy(state))
