"""相邻帧重配置；缓存只保存与前帧无关的完整求解。"""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geom import Layout, angles, cone, forecast, metrics, rays, ray_to_latlon, validate_layout
from .io import load_layout, save_layout
from .skeleton import assign, repair_spacing, solve


def copy_layout(layout):
    return Layout(*(np.array(value, copy=True) for value in
                    (layout.centers_latlon, layout.widths, layout.owner, layout.beam_ids)))


def with_members(centers, owner, beam_ids, aircraft_rays):
    """删除空束并按真实成员更新包络，不改变中心。"""
    used = np.unique(owner)
    remap = np.full(len(centers), -1, dtype=int)
    remap[used] = np.arange(len(used))
    owner = remap[owner]
    centers = np.asarray(centers)[used]
    separation = angles(rays(centers), aircraft_rays)
    widths = np.array([max(.3, 2 * separation[j, owner == j].max()) + 1e-9
                       for j in range(len(used))])
    return Layout(centers, widths, owner, np.asarray(beam_ids)[used])


def inherit_ids(previous, current, next_id):
    """先最大化1度内的匹配数，再最小化匹配夹角。"""
    current = copy_layout(current)
    ids = np.full(len(current.widths), -1, dtype=np.int64)
    if previous is not None and len(previous.widths) and len(current.widths):
        distance = angles(rays(previous.centers_latlon), rays(current.centers_latlon))
        old, new = linear_sum_assignment(np.where(distance <= 1., distance, 1e6))
        matched = distance[old, new] <= 1.
        ids[new[matched]] = previous.beam_ids[old[matched]]
    for j in np.flatnonzero(ids < 0):
        ids[j] = next_id
        next_id += 1
    current.beam_ids = ids
    return current, next_id


def costs(old_layout, old_frame, layout, frame):
    _, old, new = np.intersect1d(old_frame.icao, frame.icao, return_indices=True)
    churn = float(np.mean(old_layout.beam_ids[old_layout.owner[old]] !=
                          layout.beam_ids[layout.owner[new]])) if len(old) else 0.
    _, before, after = np.intersect1d(old_layout.beam_ids, layout.beam_ids, return_indices=True)
    movement = np.diag(angles(rays(old_layout.centers_latlon[before]),
                              rays(layout.centers_latlon[after]))) if len(before) else np.array([])
    events = (len(old_layout.widths) + len(layout.widths) - 2 * len(before)
              + np.count_nonzero(movement > .1)
              + np.count_nonzero(np.abs(old_layout.widths[before] - layout.widths[after]) > .05))
    return churn, int(events)


def previous_owner(layout, old_frame, frame):
    """当前飞机的原束列号；新飞机为-1。"""
    result = np.full(len(frame.icao), -1, dtype=int)
    _, old, new = np.intersect1d(old_frame.icao, frame.icao, return_indices=True)
    result[new] = layout.owner[old]
    return result


def patch(layout, old_frame, frame, N_budget=200, next_id=None):
    """中心固定的增量覆盖；仅在预算用尽且确有必要时移动一束。"""
    aircraft = rays(frame.positions)
    centers = np.array(layout.centers_latlon, copy=True)
    ids = list(layout.beam_ids)
    next_id = int(max(ids, default=-1) + 1 if next_id is None else next_id)
    owner = previous_owner(layout, old_frame, frame)
    distance = angles(rays(centers), aircraft)
    valid = owner >= 0
    valid[valid] &= distance[owner[valid], np.flatnonzero(valid)] <= .5
    owner[~valid] = -1
    loads = np.bincount(owner[owner >= 0], minlength=len(centers))
    keep = np.flatnonzero(loads)
    remap = np.full(len(centers), -1, dtype=int)
    remap[keep] = np.arange(len(keep))
    owner[owner >= 0] = remap[owner[owner >= 0]]
    centers, distance, loads = centers[keep], distance[keep], loads[keep]
    ids = list(np.asarray(ids)[keep])
    for a in np.flatnonzero(owner < 0):
        possible = np.flatnonzero(distance[:, a] <= .5)
        if len(possible):
            j = possible[np.argmin(loads[possible])]
        elif len(centers) < N_budget:
            j = len(centers)
            centers = np.vstack((centers, frame.positions[a]))
            ids.append(next_id)
            next_id += 1
            distance = np.vstack((distance, angles(aircraft[a:a + 1], aircraft)))
            loads = np.r_[loads, 0]
        else:
            j = int(np.argmin(distance[:, a]))
            center, radius = cone(aircraft[np.r_[np.flatnonzero(owner == j), a]])
            if radius > .5:
                return None, next_id
            centers[j] = ray_to_latlon(center)
            distance[j] = angles(center[None], aircraft)[0]
        owner[a] = j
        loads[j] += 1
    candidate = with_members(centers, owner, ids, aircraft)
    candidate = repair_spacing(candidate, aircraft, N_budget)
    if candidate is None or not validate_layout(candidate, aircraft, N_budget):
        return None, next_id
    return candidate, max(next_id, int(candidate.beam_ids.max(initial=-1)) + 1)


def predicted_frame(frame):
    positions = forecast(frame.positions[:, 0], frame.positions[:, 1],
                         frame.gs_kt, frame.track_deg, 60.)
    return replace(frame, positions=positions, time=frame.time + 60)


def forecast_peak(layout, frame):
    """预测patch的负载；不改中心，不作间隔修复或未来完整求解。"""
    future = rays(predicted_frame(frame).positions)
    distance = angles(rays(layout.centers_latlon), future)
    centers = rays(layout.centers_latlon)
    owner = np.array(layout.owner, copy=True)
    outside = distance[owner, np.arange(len(owner))] > .5
    loads = np.bincount(owner[~outside], minlength=len(layout.widths))
    keep = np.flatnonzero(loads)
    centers, distance, loads = centers[keep], distance[keep], loads[keep]
    for a in np.flatnonzero(outside):
        possible = np.flatnonzero(distance[:, a] <= .5)
        if len(possible):
            j = possible[np.argmin(loads[possible])]
            loads[j] += 1
        elif len(loads) < 200:
            centers = np.vstack((centers, future[a]))
            distance = np.vstack((distance, angles(future[a:a + 1], future)))
            loads = np.r_[loads, 1]
        else:
            loads[np.argmin(distance[:, a])] += 1
    return int(loads.max(initial=0))


def heavy_region(layout):
    loads = np.bincount(layout.owner, minlength=len(layout.widths))
    heavy = np.argsort(-loads, kind="stable")[:6]
    distance = angles(rays(layout.centers_latlon), rays(layout.centers_latlon))
    return np.any(distance[:, heavy] <= 1., axis=1)


def split_members(members):
    """最远点初始化的两均值；每组中心取成员包络锥中心。"""
    first = np.linalg.norm(members - members[0], axis=1).argmax()
    second = np.linalg.norm(members - members[first], axis=1).argmax()
    centers = members[[first, second]].copy()
    for _ in range(10):
        owner = np.linalg.norm(members[:, None] - centers[None], axis=2).argmin(axis=1)
        if not np.any(owner == 0) or not np.any(owner == 1):
            return None
        updated = np.array([cone(members[owner == j])[0] for j in (0, 1)])
        if np.max(np.linalg.norm(updated - centers, axis=1)) < 1e-10:
            return updated
        centers = updated
    return centers


def repair(layout, frame, region="heavy", iters=10, kappa=1., forecast=True,
           N_budget=200, prev_ids=None, next_id=None):
    """局部交替优化；heavy外的中心、宽度、成员与ID均冻结。"""
    aircraft = rays(frame.positions)
    future = rays(predicted_frame(frame).positions) if forecast else aircraft
    midpoint = aircraft + future
    midpoint /= np.linalg.norm(midpoint, axis=1, keepdims=True)
    active = heavy_region(layout) if region == "heavy" else np.ones(len(layout.widths), bool)
    frozen_ids = layout.beam_ids[~active]
    mobile = active[layout.owner]
    reference = layout.beam_ids[layout.owner] if prev_ids is None else prev_ids
    next_id = int(layout.beam_ids.max(initial=-1) + 1 if next_id is None else next_id)
    best = copy_layout(layout)
    best_score = tuple(metrics(best)[key] for key in ("peak", "top10", "CV"))

    def optimize(centers, ids):
        center_rays = rays(centers)
        distance = angles(center_rays, aircraft).T
        available = ~np.isin(ids, frozen_ids)
        eligible = (distance <= .5) & mobile[:, None] & available[None, :]
        for j, beam_id in enumerate(ids):
            if beam_id in frozen_ids:
                eligible[:, j] = layout.beam_ids[layout.owner] == beam_id
        if forecast:
            predicted = angles(center_rays, future).T <= .5
            both = eligible & predicted
            eligible[mobile & both.any(axis=1)] = both[mobile & both.any(axis=1)]
        old = np.full(len(aircraft), -1, int)
        for j, beam_id in enumerate(ids):
            old[reference == beam_id] = j
        owner = assign(center_rays, aircraft, mode="peak", eligible_mask=eligible)
        if owner is None:
            return None
        owner = assign(center_rays, aircraft, mode="balance", prev_owner=old,
                       kappa=kappa, eligible_mask=eligible)
        if owner is None:
            return None
        centers = np.array(centers, copy=True)
        for j in np.flatnonzero(available):
            if np.any(owner == j):
                center, _ = cone(midpoint[owner == j])
                centers[j] = ray_to_latlon(center)
        candidate = with_members(centers, owner, ids, aircraft)
        candidate = repair_spacing(candidate, aircraft, N_budget)
        if candidate is None or not validate_layout(candidate, aircraft, N_budget):
            return None
        if forecast:
            centers_now = rays(candidate.centers_latlon)
            both = (angles(centers_now, aircraft).T <= .5) & (angles(centers_now, future).T <= .5)
            both[:, np.isin(candidate.beam_ids, frozen_ids)] = False
            selected = both[np.arange(len(aircraft)), candidate.owner]
            if np.any(mobile & both.any(axis=1) & ~selected):
                return None
        for beam_id in frozen_ids:
            old_j = np.flatnonzero(layout.beam_ids == beam_id)[0]
            new_j = np.flatnonzero(candidate.beam_ids == beam_id)
            if not len(new_j):
                return None
            j = new_j[0]
            if (not np.array_equal(candidate.centers_latlon[j], layout.centers_latlon[old_j])
                    or candidate.widths[j] != layout.widths[old_j]
                    or not np.array_equal(candidate.owner == j, layout.owner == old_j)):
                return None
        return candidate

    stale = 0
    for _ in range(iters):
        candidate = optimize(best.centers_latlon, best.beam_ids)
        if candidate is not None:
            score = tuple(metrics(candidate)[key] for key in ("peak", "top10", "CV"))
            if score < best_score:
                best, best_score, stale = candidate, score, 0
            else:
                stale += 1
        else:
            stale += 1
        # 将局部最闲束的名额用于拆最忙束；预算有空位时直接拆。
        loads = np.bincount(best.owner, minlength=len(best.widths))
        moving = np.flatnonzero(~np.isin(best.beam_ids, frozen_ids))
        if len(moving) and loads[moving].max() > 1:
            base = best
            if len(loads) >= N_budget:
                light = moving[np.argmin(loads[moving])]
                base = optimize(np.delete(best.centers_latlon, light, axis=0),
                                np.delete(best.beam_ids, light))
                if base is not None and metrics(base)["peak"] > best_score[0]:
                    base = None
            if base is not None and len(base.widths) < N_budget:
                base_loads = np.bincount(base.owner, minlength=len(base.widths))
                moving = np.flatnonzero(~np.isin(base.beam_ids, frozen_ids))
                j = moving[np.argmax(base_loads[moving])]
                split = split_members(aircraft[base.owner == j])
                if split is not None:
                    centers = np.array(base.centers_latlon, copy=True)
                    centers[j] = ray_to_latlon(split[0])
                    centers = np.vstack((centers, ray_to_latlon(split[1])))
                    ids = np.r_[base.beam_ids, next_id]
                    trial = optimize(centers, ids)
                    if trial is not None:
                        score = tuple(metrics(trial)[key] for key in ("peak", "top10", "CV"))
                        if score < best_score:
                            best, best_score, stale = trial, score, 0
                            next_id += 1
        if stale >= 3:
            break
    return best, next_id


class FullCache:
    """每个日期/帧只解一次；读缓存不计I/O，用原始求解时间。"""
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, frame):
        return self.directory / f"{frame.date}_{frame.index:04d}.npz"

    def build(self, frame):
        path = self.path(frame)
        if not path.exists():
            result = solve(frame, 200, spacing=True, effort=1, time_limit=30.)
            if not result.feasible:
                raise RuntimeError(f"full布局不可行: {frame.date}/{frame.index}")
            save_layout(path, result.layout, frame, t_init=result.t_init, t_opt=result.t_opt,
                        t_check=result.t_check, t_total=result.t_total)
        return path

    def get(self, frame):
        path = self.path(frame)
        layout = load_layout(path)
        with np.load(path) as z:
            elapsed = float(z["t_total"])
        return layout, elapsed


@dataclass
class DynamicEpisode:
    frames: list
    cache: FullCache
    lam: float = .2
    use_forecast: bool = True

    def __post_init__(self):
        initial, _ = self.cache.get(self.frames[0])
        self.layout, self.next_id = inherit_ids(None, initial, 0)
        self.index = 1
        self.last_repair = self.last_full = 0
        self.reference_peak = metrics(self.layout)["peak"]
        self.churn_history = []
        self.prepared = False
        self.decision_seconds = 0.

    def prepare(self):
        if self.prepared:
            return
        start = perf_counter()
        frame = self.frames[self.index]
        old_owner = previous_owner(self.layout, self.frames[self.index - 1], frame)
        distance = angles(rays(self.layout.centers_latlon), rays(frame.positions))
        known = old_owner >= 0
        self.new_ratio = float(np.mean(~known))
        self.outside_ratio = float(np.count_nonzero(distance[old_owner[known], np.flatnonzero(known)] > .5)
                                   / max(len(frame.icao), 1))
        self.patched, self.patch_next_id = patch(self.layout, self.frames[self.index - 1], frame,
                                                 next_id=self.next_id)
        self.prepare_seconds = perf_counter() - start
        self.prepared = True

    def observation(self):
        self.prepare()
        start = perf_counter()
        frame = self.frames[self.index]
        lb = max(1, int(np.ceil(len(frame.icao) / 200)))
        current = self.patched
        # 失败patch的观测仍可计算，step会按同一规则强制full。
        m = metrics(current) if current is not None else metrics(self.layout)
        peak = forecast_peak(current, frame) if current is not None and self.use_forecast else m["peak"]
        first = self.frames[max(0, self.index - 5)]
        delta = (len(frame.icao) - len(first.icao)) / max(len(first.icao), 1)
        hour = datetime.fromtimestamp(frame.time, timezone.utc).hour + (frame.time % 3600) / 3600
        values = [m["peak"] / lb, (m["peak"] - lb) / 10., m["top10"] / lb, m["CV"],
                  len(current.widths if current is not None else self.layout.widths) / 200.]
        if self.use_forecast:
            values.append(peak / lb)
        values += [self.reference_peak / lb, (self.index - self.last_repair) / 30.,
                   (self.index - self.last_full) / 30., self.new_ratio, self.outside_ratio, delta,
                   np.sin(hour * 2 * np.pi / 24), np.cos(hour * 2 * np.pi / 24),
                   np.mean(self.churn_history[-5:]) if self.churn_history else 0.,
                   (len(self.frames) - 1 - self.index) / 30.]
        result = np.asarray(values, dtype=np.float32)
        self.decision_seconds += perf_counter() - start
        return result

    def step(self, action, repair_parameters=None):
        if action != 4:
            self.prepare()
        frame = self.frames[self.index]
        start = perf_counter()
        forced = action != 4 and self.patched is None
        elapsed = (self.prepare_seconds if self.prepared else 0.) + self.decision_seconds
        cache_seconds = solver_seconds = 0.
        if action == 4 or forced:
            reading = perf_counter()
            candidate, solver_seconds = self.cache.get(frame)
            cache_seconds = perf_counter() - reading
            candidate, next_id = inherit_ids(self.layout, candidate, self.next_id)
            self.last_full = self.last_repair = self.index
        else:
            candidate, next_id = self.patched, self.patch_next_id
            if action:
                parameters = repair_parameters or {
                    1: ("heavy", 5, 3.), 2: ("heavy", 15, .5), 3: ("all", 10, .5)}[action]
                old_owner = previous_owner(self.layout, self.frames[self.index - 1], frame)
                prev_ids = np.full(len(frame.icao), -1, dtype=np.int64)
                known = old_owner >= 0
                prev_ids[known] = self.layout.beam_ids[old_owner[known]]
                candidate, next_id = repair(candidate, frame, *parameters, forecast=self.use_forecast,
                                             prev_ids=prev_ids, next_id=next_id)
                self.last_repair = self.index
        churn, events = costs(self.layout, self.frames[self.index - 1], candidate, frame)
        m = metrics(candidate)
        if action or forced:
            self.reference_peak = m["peak"]
        lb = int(np.ceil(len(frame.icao) / 200))
        reward = -(m["peak"] - lb) - self.lam * 100 * churn
        elapsed += perf_counter() - start - cache_seconds + solver_seconds
        row = dict(day=frame.date, seq=self.frames[0].index // 30, frame=self.index,
                   frame_time=frame.time, n=len(frame.icao), LB=lb, peak=m["peak"],
                   top10=m["top10"], CV=m["CV"], N=len(candidate.widths), churn=churn,
                   beam_events=events, action=int(action), forced_full=int(forced), t_frame=elapsed)
        self.layout, self.next_id = candidate, next_id
        self.churn_history.append(churn)
        self.index += 1
        self.prepared = False
        self.decision_seconds = 0.
        return float(reward), self.index == len(self.frames), row


BASELINES = [("B0", 0), ("B1", 0), ("B3", 5), ("B3", 10)] + [
    (arm, theta) for arm in ("B2", "B2+") for theta in (1, 2, 4)]


def evaluate(sequences, cache, arm, param=0, seed=0, policy=None, forecast=True, lam=.2):
    rows = []
    for sequence in sequences:
        episode = DynamicEpisode(sequence, cache, lam, forecast if policy is not None else arm == "B2+")
        while episode.index < len(sequence):
            if policy is not None:
                observation = episode.observation()
                decision = perf_counter()
                action = int(policy(observation))
                episode.decision_seconds += perf_counter() - decision
                options = None
            else:
                action, options = 0, None
                if arm == "B0" or (arm == "B3" and episode.index % int(param) == 0):
                    action = 4
                else:
                    episode.prepare()
                if arm in ("B2", "B2+") and episode.patched is not None:
                    decision = perf_counter()
                    peak = metrics(episode.patched)["peak"]
                    if arm == "B2+":
                        peak = max(peak, forecast_peak(episode.patched, sequence[episode.index]))
                    if peak - episode.reference_peak > param:
                        action, options = 1, ("heavy", 10, 1.)
                    episode.decision_seconds += perf_counter() - decision
            _, _, row = episode.step(action, options)
            row.update(arm=arm, param=param, seed=seed)
            rows.append(row)
    return rows


def summarize(rows, role):
    return dict(role=role, arm=rows[0]["arm"], param=rows[0]["param"], seed=rows[0]["seed"],
                mean_gap=float(np.mean([r["peak"] - r["LB"] for r in rows])),
                mean_peak=float(np.mean([r["peak"] for r in rows])),
                mean_churn=float(np.mean([r["churn"] for r in rows])),
                mean_beam_events=float(np.mean([r["beam_events"] for r in rows])),
                mean_t_frame=float(np.mean([r["t_frame"] for r in rows])))
