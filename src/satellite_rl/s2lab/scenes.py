"""鏁版嵁闆嗭細甯冨眬澶嶇敤/鐢熸垚銆佺墿鐞唙3寮犻噺鍖栥€侀渶姹傛娊鏍枫€佽礋杞藉垎灞備笌闆嗗悎鍒掑垎銆?
鐩綍缁撴瀯 data/<dataset>/
  config.json            鐗╃悊銆佸姛鐜囬绠椼€侀渶姹傚尯闂淬€佸垎灞傞槇鍊笺€侀泦鍚堝畾涔?  scenes/<key>.npz       姣忔櫙寮犻噺锛堜笌鍒嗛厤鏃犲叧锛?  scenes.csv             鍦烘櫙绱㈠紩锛坘ey銆佹棩鏈熴€乁TC銆侀鏈烘暟銆佹潫鏁般€佹潵婧愶級
  splits.json            train/dev/confirm 鍚勮嚜鐨?(key, per_aircraft_bps, load_ratio, stratum) 鍒楄〃
鍦烘櫙 key = 鏃ユ湡+UTC锛屽 20240112T1122銆?"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import numpy as np

from .physics import PhysicsV3, SceneTensors, build_scene_tensors

BASE_REQUEST_BPS = 43_200_000     # 甯冨眬鐢熸垚鏃剁殑鍗曟満璇锋眰锛堝喅瀹氭瘡鏉熶笂闄怟=26锛夛紝鍐荤粨甯冨眬鍙敤瀹?N_BUDGET = 150
STRATA = ('ample', 'marginal', 'short')   # 瀹借 / 鍕夊己 / 涓嶅


@dataclass
class DatasetConfig:
    name: str
    physics: dict = field(default_factory=lambda: PhysicsV3().to_dict())
    power_budget_w: float = 2000.
    beam_power_cap_w: float = 50.
    reserve_w_per_beam: float = .5
    per_aircraft_bps_range: tuple = (30e6, 80e6)   # 璁粌鏃舵瘡鍥炲悎鎶芥牱锛涜瘎浠烽泦鍥哄畾
    load_thresholds: tuple = (.35, .6)              # 蟻 < t0 瀹借锛宼0鈥攖1 鍕夊己锛屸墺t1 涓嶅
    split_dates: dict = field(default_factory=lambda: dict(
        train=['2024-01-08', '2024-01-09', '2024-01-12'],
        dev=['2024-01-15'],
        confirm=['2024-01-18', '2024-02-23', '2024-02-24']))
    dev_per_stratum: int = 20
    confirm_per_stratum: int = 100
    train_strata_share: tuple = (.15, .4, .45)      # 璁粌鎶芥牱鏃朵笁妗ｆ瘮渚嬶紙鎸?蟻 鍒嗗眰閲嶉噰鏍凤級
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


# ---------------------------------------------------------------- 甯冨眬鏉ユ簮

def existing_s2fix_records(s2fix_data_dir):
    """璇诲彇 S2-FIX 宸茬敓鎴愮殑涓変釜闆嗗悎锛堥兘鏄?43.2 Mbps銆?50 鏉熴€並=26 鐨?FLOW-AO v2 甯冨眬锛夈€?""
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
    """瀛愯繘绋嬶細璇?case/layout锛堟垨鐜板満鐢熸垚甯冨眬锛夆啋 寮犻噺 鈫?npz銆傝繑鍥炵储寮曡銆?""
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


def build_scenes(dataset_dir, config, s2fix_data_dir=None, frames_dir=None, workers=1, limit=None, per_date_limit=None, gen_dates=None):
    """鐢熸垚/澶嶇敤甯冨眬骞跺紶閲忓寲锛屽彲閲嶅杩愯锛堝凡瀛樺湪鐨?npz 璺宠繃锛夈€俻er_date_limit锛氭瘡涓棩鏈熷彧鍙栧潎鍖€鍒嗗竷鐨?k 涓椂鍒伙紙鍐掔儫鐢級銆?    gen_dates锛氭湰娆″彧澶勭悊杩欎簺鏃ユ湡锛堝瀹瑰櫒鍒嗘棩鏈熷苟琛岋級锛屽彛寰勯噷鐨勬棩鏈熼泦鍚堜笉鍙樸€?""
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
            # case_id 褰㈠ r7:...:2024-01-12T00:37:00Z:uniform...:s2fix:43200000:n150
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


def _iso_from_case_id(case_id):
    parts = case_id.split(':')
    for i, part in enumerate(parts):
        if len(part) == 13 and part[10] == 'T' and part[4] == '-':
            return ':'.join(parts[i:i + 3])
    raise ValueError(case_id)


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


# ---------------------------------------------------------------- 闆嗗悎鍒掑垎

def make_splits(dataset_dir, config=None):
    """鎸夋棩鏈熷垎 train/dev/confirm锛沝ev/confirm 鍥哄畾姣忔櫙鐨勫崟鏈洪渶姹傚苟鎸夎礋杞藉垎灞傚潎琛℃娊鍙栥€?""
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
            # 姣忔櫙鎶?8 涓€欓€夐渶姹傚€硷紝鎸夊叾鍒嗗眰褰掔被锛屽悗闈㈡瘡妗ｅ潎琛℃娊鏍枫€佸悓涓€鏅渶澶氬嚭鐜颁竴娆?            for v in np.sort(rng.uniform(lo, hi, 8)):
                ratio = load_ratio(r['members_total'], float(v), config)
                candidates[stratum_of(ratio, config)].append(dict(key=r['key'], per_aircraft_bps=float(v), load_ratio=float(ratio)))
        chosen, used = [], set()
        for s in reversed(STRATA):        # 鍏堝～鏈€绋€缂虹殑鈥滀笉澶熲€濇。锛屽闂村満鏅彧鑳借繘鈥滃瑁曗€?            options = candidates[s]
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
