"""数据集：布局复用/生成、物理v3张量化、需求抽样、负载分层与集合划分。

目录结构 data/<dataset>/
  config.json            物理、功率预算、需求区间、分层阈值、集合定义
  scenes/<key>.npz       每景张量（与分配无关）
  scenes.csv             场景索引（key、日期、UTC、飞机数、束数、来源）
  splits.json            train/dev/confirm 各自的 (key, per_aircraft_bps, load_ratio, stratum) 列表
场景 key = 日期+UTC，如 20240112T1122。
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import numpy as np

from .physics import PhysicsV3, SceneTensors, build_scene_tensors

BASE_REQUEST_BPS = 43_200_000     # 布局生成时的单机请求（决定每束上限K=26），冻结布局只用它
N_BUDGET = 150
STRATA = ('ample', 'marginal', 'short')   # 宽裕 / 勉强 / 不够


@dataclass
class DatasetConfig:
    name: str
    physics: dict = field(default_factory=lambda: PhysicsV3().to_dict())
    power_budget_w: float = 2000.
    beam_power_cap_w: float = 50.
    reserve_w_per_beam: float = .5
    per_aircraft_bps_range: tuple = (30e6, 80e6)    # 训练时每回合抽样；评价集固定
    load_thresholds: tuple = (.35, .6)              # ρ < t0 宽裕，t0—t1 勉强，≥t1 不够
    split_dates: dict = field(default_factory=lambda: dict(
        train=['2024-01-08', '2024-01-09', '2024-01-12'],
        dev=['2024-01-15'],
        confirm=['2024-01-18', '2024-02-23', '2024-02-24']))
    dev_per_stratum: int = 20
    confirm_per_stratum: int = 100
    train_strata_share: tuple = (.15, .4, .45)      # 训练抽样时三档比例（按 ρ 分层重采样）
    seed: int = 20261024

    def physics_obj(self):
        return PhysicsV3(**self.physics)

    @property
    def capacity_reference_bps(self):
        p = self.physics_obj()
        return p.groups * p.slots * p.slot_Hz * p.spectral_efficiency_cap


def load_ratio(members_total, per_aircraft_bps, config):
    return members_total * per_aircraft_bps / config.capacity_reference_bps


def stratum_of(ratio, config):
    t0, t1 = config.load_thresholds
    return 'ample' if ratio < t0 else ('marginal' if ratio < t1 else 'short')


def scene_key(utc_iso):
    stamp = datetime.fromisoformat(utc_iso.replace('Z', '+00:00')).astimezone(timezone.utc)
    return stamp.strftime('%Y%m%dT%H%M')


# ---------------------------------------------------------------- 布局来源

def existing_s2fix_records(s2fix_data_dir):
    """读取 S2-FIX 已生成的三个集合（都是 43.2 Mbps、150 束、K=26 的 FLOW-AO v2 布局）。"""
    from satellite_rl.s2fix.data import read_records
    rows = []
    for set_name in ('train_d432_n150', 'dev_d432_n150', 'conf_d432_n150'):
        path = Path(s2fix_data_dir) / set_name / 'manifest.json'
        if path.exists():
            rows.extend(read_records(s2fix_data_dir, set_name))
    return rows


def frames_for_dates(frames_dir, dates):
    from satellite_rl.coverage.s1fix.data import load_day
    out = []
    for date in dates:
        if not (Path(frames_dir) / f'frames_{date}.npz').exists():
            continue
        for frame in load_day(frames_dir, date):
            utc = datetime.fromtimestamp(frame.time, timezone.utc).isoformat().replace('+00:00', 'Z')
            out.append((date, utc, frame))
    return out


def _tensorize_job(job):
    """子进程：读 case/layout（或现场生成布局）→ 张量 → npz。返回索引行。"""
    kind, key, payload, physics_dict, out_dir = job
    physics = PhysicsV3(**physics_dict)
    if kind == 'existing':
        from satellite_rl.s2fix.data import read_record
        case = read_record(payload['case_path'])
        layout = read_record(payload['layout_path'])
        source = 's2fix:' + payload['set_name']
    else:
        from satellite_rl.s2fix.data import build_scene
        from satellite_rl.shared_interface.stages import make_case
        date, utc, icao, positions = payload
        metadata = dict(case_id=f's2lab:{utc}', actual_role='s2lab', source_role='frames',
                        original_mother_id=f'{date}:{utc}', UTC=utc, source_asset_SHA=f'frames_{date}.npz',
                        source=dict(frames=date))
        case = make_case(metadata, [str(int(i)) for i in icao], positions, plain=True)
        case, layout, _ = build_scene(case, BASE_REQUEST_BPS, N_BUDGET)
        source = 'flow-ao-v2'
    scene = build_scene_tensors(case, layout, physics, scene_id=key)
    path = Path(out_dir) / f'{key}.npz'
    np.savez_compressed(path, members=scene.members, width_deg=scene.width_deg, lat_lon=scene.lat_lon,
                        distance_m=scene.distance_m, sig=scene.sig, coupling=scene.coupling,
                        adjacent=scene.adjacent, aircraft_count=scene.aircraft_count,
                        date=scene.date, scene_id=key, source=source)
    return dict(key=key, date=scene.date, utc=case['UTC'], aircraft=scene.aircraft_count, beams=scene.n,
                members_total=int(scene.members.sum()), max_members=int(scene.members.max()), source=source)


def _iso_from_case_id(case_id):
    parts = case_id.split(':')
    for i, part in enumerate(parts):
        if len(part) == 13 and part[10] == 'T' and part[4] == '-':
            return ':'.join(parts[i:i + 3])
    raise ValueError(case_id)


def build_scenes(dataset_dir, config, s2fix_data_dir=None, frames_dir=None, workers=1, limit=None, per_date_limit=None, gen_dates=None):
    """生成/复用布局并张量化，可重复运行（已存在的 npz 跳过）。per_date_limit：每个日期只取均匀分布的 k 个时刻（冒烟用）。
    gen_dates：本次只处理这些日期（多容器分日期并行），口径里的日期集合不变。"""
    dataset_dir = Path(dataset_dir)
    scenes_dir = dataset_dir / 'scenes'
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'config.json').write_text(json.dumps(asdict(config), ensure_ascii=False, indent=1), encoding='utf-8')
    wanted_dates = sorted({d for ds in config.split_dates.values() for d in ds})
    if gen_dates:
        wanted_dates = [d for d in wanted_dates if d in gen_dates]
    jobs, seen = [], set()
    if s2fix_data_dir:
        for row in existing_s2fix_records(s2fix_data_dir):
            # case_id 形如 r7:...:2024-01-12T00:37:00Z:uniform...:s2fix:43200000:n150
            key = scene_key(_iso_from_case_id(row['case_id']))
            if row['date'] not in wanted_dates or key in seen:
                continue
            seen.add(key)
            if not (scenes_dir / f'{key}.npz').exists():
                jobs.append(('existing', key, row, config.physics, str(scenes_dir)))
    if frames_dir:
        frames = frames_for_dates(frames_dir, wanted_dates)
        if per_date_limit:
            picked = []
            for date in wanted_dates:
                day = [f for f in frames if f[0] == date]
                picked += [day[int(i)] for i in np.linspace(0, len(day) - 1, min(per_date_limit, len(day)))] if day else []
            frames = picked
        for date, utc, frame in frames:
            key = scene_key(utc)
            if key in seen:
                continue
            seen.add(key)
            if not (scenes_dir / f'{key}.npz').exists():
                jobs.append(('frames', key, (date, utc, frame.icao, frame.positions), config.physics, str(scenes_dir)))
    if limit is not None:
        jobs = jobs[:limit]
    if workers == 1:
        results = list(map(_tensorize_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_tensorize_job, jobs, chunksize=1))
    write_index(dataset_dir)
    return results


def write_index(dataset_dir):
    dataset_dir = Path(dataset_dir)
    rows = []
    for path in sorted((dataset_dir / 'scenes').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            members = z['members']
            rows.append(dict(key=path.stem, date=str(z['date']), aircraft=int(z['aircraft_count']), beams=len(members),
                             members_total=int(members.sum()), max_members=int(members.max()), source=str(z['source'])))
    with (dataset_dir / 'scenes.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['key'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_index(dataset_dir):
    with (Path(dataset_dir) / 'scenes.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in ('aircraft', 'beams', 'members_total', 'max_members'):
            row[name] = int(row[name])
    return rows


def read_config(dataset_dir):
    raw = json.loads((Path(dataset_dir) / 'config.json').read_text(encoding='utf-8'))
    raw['per_aircraft_bps_range'] = tuple(raw['per_aircraft_bps_range'])
    raw['load_thresholds'] = tuple(raw['load_thresholds'])
    raw['train_strata_share'] = tuple(raw['train_strata_share'])
    return DatasetConfig(**raw)


# ---------------------------------------------------------------- 集合划分

def make_splits(dataset_dir, config=None):
    """按日期分 train/dev/confirm；dev/confirm 固定每景的单机需求并按负载分层均衡抽取。"""
    dataset_dir = Path(dataset_dir)
    config = config or read_config(dataset_dir)
    rows = read_index(dataset_dir)
    rng = np.random.default_rng(config.seed)
    lo, hi = config.per_aircraft_bps_range
    splits = {}
    for split, dates in config.split_dates.items():
        pool = [r for r in rows if r['date'] in dates]
        if split == 'train':
            items = []
            for r in pool:
                ratio_lo, ratio_hi = (load_ratio(r['members_total'], v, config) for v in (lo, hi))
                items.append(dict(key=r['key'], load_ratio_range=[ratio_lo, ratio_hi]))
            splits[split] = items
            continue
        per = config.dev_per_stratum if split == 'dev' else config.confirm_per_stratum
        candidates = {s: [] for s in STRATA}
        for r in pool:
            # 每景抽 8 个候选需求值，按其分层归类，后面每档均衡抽样、同一景最多出现一次
            for v in np.sort(rng.uniform(lo, hi, 8)):
                ratio = load_ratio(r['members_total'], float(v), config)
                candidates[stratum_of(ratio, config)].append(dict(key=r['key'], per_aircraft_bps=float(v), load_ratio=float(ratio)))
        chosen, used = [], set()
        for s in reversed(STRATA):        # 先填最稀缺的“不够”档，夜间场景只能进“宽裕”
            options = candidates[s]
            rng.shuffle(options)
            taken = 0
            for item in options:
                if item['key'] in used or taken >= per:
                    continue
                used.add(item['key'])
                chosen.append(dict(item, stratum=s))
                taken += 1
            if taken < per:
                print(f'[splits] {split}/{s}: only {taken}/{per} scenes available', flush=True)
        splits[split] = chosen
    (dataset_dir / 'splits.json').write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding='utf-8')
    return splits


def read_splits(dataset_dir):
    return json.loads((Path(dataset_dir) / 'splits.json').read_text(encoding='utf-8'))


def load_scene(dataset_dir, key):
    with np.load(Path(dataset_dir) / 'scenes' / f'{key}.npz', allow_pickle=False) as z:
        return SceneTensors(key, str(z['date']), z['members'], z['width_deg'], z['lat_lon'], z['distance_m'],
                            z['sig'], z['coupling'], z['adjacent'], int(z['aircraft_count']))
