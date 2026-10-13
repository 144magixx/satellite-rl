"""T24 historical data preparation; call only inside an admitted frozen worker.

No model is accepted by this interface. Metadata qualification is offline; all
downloads and original aviation calls are deferred until prepare_test_data.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import ast
import json
import math
import shutil
import tarfile
import urllib.request
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

BASELINE = "85f667cd5731e284c19b2590b5dcf5308a58f234"
DATES = ("2024-02-21", "2024-02-22")
RESERVE = 8 * 1024 ** 3
MODEL = dict(persons_per_aircraft=180, satellite_access_fraction=.2,
             active_fraction=.3, rate_mbps_per_active_person=2.,
             aircraft_base_mbps=0, sampling="binomial", lognormal_sigma=.35,
             window_seconds=60)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _pin(ref):
    path = Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("pinned source changed: " + str(path))
    return path


def qualify_metadata(config):
    """Pure local qualification; returns pins, never makes a request."""
    if config.get("experiment_id") != "S2-T24" or config.get("baseline_commit") != BASELINE:
        raise ValueError("historical T0 source identity required")
    if not config.get("owner_run_id") or not config.get("frozen_source_sha"):
        raise ValueError("frozen workbench job identity required")
    authority = json.loads(_pin(config["historical_authority"]).read_text(encoding="utf-8"))
    if config.get("purpose") != "explicit_historical_reproduction" or not authority:
        raise ValueError("explicit historical authority required")
    registry = json.loads(_pin(config["freshness_registry"]).read_text(encoding="utf-8"))
    if registry.get("qualified") is not True:
        raise ValueError("full freshness registry acceptance required")
    model = config["demand_model"]
    if any(model.get(k) != v for k, v in MODEL.items()) or not model.get("parameter_source"):
        raise ValueError("original T0 demand model changed")
    frozen_root = Path(config["frozen_source_root"]).resolve()
    for ref in (config["converter_reference"], config["scanner_reference"]):
        pinned = _pin(ref).resolve()
        if not pinned.is_relative_to(frozen_root):
            raise ValueError("scientific dependencies must reside in frozen source bundle")
    assets = config["source_assets"]
    if tuple(a["date"] for a in assets) != DATES:
        raise ValueError("freeze Feb21/22; replacement dates require a new immutable contract")
    pins = []
    for a in assets:
        day = a["date"]; tag = "v" + day.replace("-", ".") + "-planes-readsb-prod-0"
        raw = _pin(a["metadata_reference"]).read_bytes(); meta = json.loads(raw)
        found = [x for x in meta["assets"] if x["name"] == tag + ".tar"]
        if meta["tag_name"] != tag or len(found) != 1:
            raise ValueError("official prod0 release pin mismatch")
        m = found[0]
        url = "https://github.com/adsblol/globe_history_2024/releases/download/" + tag + "/" + tag + ".tar"
        if (m["id"] != a["asset_id"] or m["size"] != a["expected_bytes"]
                or m["browser_download_url"] != url or a["url"] != url):
            raise ValueError("official asset identity mismatch")
        if day in registry.get("used_dates", []):
            raise ValueError("test date already used")
        pins.append({**a, "release_tag": tag, "metadata_sha256": hashlib.sha256(raw).hexdigest()})
    return pins, registry


def _download(asset, out, progress, stop):
    """Resume exact ranged bytes; SHA remains unknown until all bytes exist."""
    target = out / (asset["release_tag"] + ".tar")
    partial = target.with_suffix(".tar.partial")
    checkpoint = target.with_suffix(".tar.ranges.json")
    total = asset["expected_bytes"]
    if target.exists():
        if target.stat().st_size != total: raise ValueError("archive size changed")
        digest = sha(target)
        if asset.get("sha256") and digest != asset["sha256"]: raise ValueError("archive SHA changed")
        return dict(asset, path=str(target), sha256=digest)
    bound = {k: asset[k] for k in ("date", "asset_id", "expected_bytes", "url", "metadata_sha256")}
    ranges = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"asset": bound, "chunks": []}
    if ranges["asset"] != bound: raise ValueError("partial asset identity changed")
    start = 0
    if partial.exists():
        with partial.open("r+b") as stream:
            for chunk in ranges["chunks"]:
                if chunk["start"] != start: raise ValueError("partial range continuity changed")
                block = stream.read(chunk["bytes"])
                if hashlib.sha256(block).hexdigest() != chunk["sha256"]:
                    raise ValueError("partial downloaded chunk SHA changed")
                start += chunk["bytes"]
            # An interrupted write without an atomic receipt is not accepted.
            stream.truncate(start)
    elif ranges["chunks"]:
        raise ValueError("recorded partial source missing")
    if start > total: raise ValueError("partial exceeds pinned asset size")
    if shutil.disk_usage(out).free < total - start + RESERVE:
        raise OSError("download must retain 8GiB free reserve")
    while start < total:
        if stop(): raise InterruptedError("preparation stopped")
        end = min(start + 8 * 1024 ** 2, total) - 1
        if shutil.disk_usage(out).free < end-start+1 + RESERVE:
            raise OSError("range download must retain8GiB reserve")
        req = urllib.request.Request(asset["url"], headers={"User-Agent": "satellite-rl-S2-T24", "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                raise ValueError("exact pinned HTTP range required")
            block = response.read(end - start + 2)
        if len(block) != end - start + 1: raise ValueError("truncated source range")
        with partial.open("ab") as stream: stream.write(block); stream.flush()
        ranges["chunks"].append({"start": start, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        write(checkpoint, ranges)
        start = end + 1
        progress("download", start, total, {"date": asset["date"], "partial_path": str(partial)})
    digest = sha(partial)
    if asset.get("sha256") and digest != asset["sha256"]: raise ValueError("archive SHA mismatch")
    partial.replace(target)
    return dict(asset, path=str(target), sha256=digest)


def _scan(asset, folder, reference, progress, stop):
    """Use original cleaner functions, preserving member bytes and SHA ledger."""
    # Execute only unchanged original cleaner functions, excluding the old
    # script's mutable machine-specific sys.path and output-directory globals.
    source = _pin(reference).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"inspect_record", "candidate_times", "utc_iso"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names: raise ValueError("original cleaner functions missing")
    from ..envs.regions import get_profile, elevation_degrees
    namespace = dict(math=math, datetime=datetime, timedelta=timedelta,
                     timezone=timezone, PROFILE=get_profile("europe_v1"),
                     elevation_degrees=elevation_degrees)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_pin(reference)), "exec"), namespace)
    cleaner = SimpleNamespace(**namespace)
    day = asset["date"]; times = cleaner.candidate_times(day)
    stamps = [t.timestamp() for t in times]; rows = [[] for _ in times]
    seen = set(); members = []; malformed = []
    raw_members = folder / "members"; raw_members.mkdir(exist_ok=True)
    with tarfile.open(asset["path"], "r|") as archive:
        for member in archive:
            if stop(): raise InterruptedError("scan stopped")
            if not member.isfile(): continue
            if shutil.disk_usage(folder).free < member.size + RESERVE:
                raise OSError("member retention must retain8GiB reserve")
            content = archive.extractfile(member).read()
            digest = hashlib.sha256(content).hexdigest()
            members.append(dict(name=member.name, bytes=len(content), sha256=digest))
            if not member.name.startswith("./traces/") or not member.name.endswith(".json"): continue
            # Content-addressed originals avoid tar path traversal and retain bytes.
            dst = raw_members / digest
            if not dst.exists(): dst.write_bytes(content)
            try:
                data = json.loads(gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content)
                icao = str(data["icao"]).strip().lower(); base = float(data["timestamp"]); trace = data["trace"]
                if not icao or not math.isfinite(base) or not isinstance(trace, list): raise ValueError("bad header")
                if icao in seen: raise RuntimeError("duplicate ICAO " + icao)
                seen.add(icao)
                offsets = [float(r[0]) for r in trace]
                if not all(math.isfinite(o) for o in offsets): raise ValueError("nonfinite offsets")
                if offsets != sorted(offsets):
                    trace = [r for _, r in sorted(zip(offsets, trace), key=lambda pair: pair[0])]; offsets.sort()
                for i, stamp in enumerate(stamps):
                    pos = bisect_right(offsets, stamp - base) - 1
                    if pos < 0 or stamp - base - offsets[pos] > 60: continue
                    reason, geometry = cleaner.inspect_record(trace[pos])
                    if reason == "selected":
                        lat, lon, altitude = geometry
                        rows[i].append([icao, cleaner.utc_iso(base + offsets[pos]), lat, lon, altitude])
            except RuntimeError: raise
            except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as exc:
                malformed.append({"member": member.name, "error": repr(exc)})
            if len(members) % 5000 == 0: progress("scan", len(members), None, {"date": day})
    write(folder / "member_ledger.json", {"archive_sha256": asset["sha256"], "members": members, "malformed": malformed})
    write(folder / "candidate_rows.json", rows)
    return rows, times, {"path": str(folder / "member_ledger.json"), "sha256": sha(folder / "member_ledger.json")}


def prepare_test_data(config, output_dir, *, progress=None, stop_requested=None, adapters=None):
    """Return scene_sets[test] with 40 frozen scenes; resume immutable identity.

    adapters optionally supplies download(asset,out,progress,stop),
    scan(asset,folder,reference,progress,stop), convert(paths,times,**kwargs).
    They exist for pure mock qualification, never constitute science evidence.
    """
    progress = progress or (lambda stage, done, total, detail: None)
    stop = stop_requested or (lambda: False); adapters = adapters or {}
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    assets, registry = qualify_metadata(config)
    # Worker admission is a SHA-pinned receipt bound to this owner/source identity.
    admission = json.loads(_pin(config["admission_reference"]).read_text(encoding="utf-8"))
    if (admission.get("approved") is not True or admission.get("owner_run_id") != config["owner_run_id"]
            or admission.get("frozen_source_sha") != config["frozen_source_sha"]):
        raise ValueError("root worker source and owner admission mismatch")
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sp = out / "preparation_state.json"
    state = json.loads(sp.read_text()) if sp.exists() else dict(identity=identity, status="preparing", files=[], assets=[], rejections=[])
    if state["identity"] != identity: raise ValueError("resume data identity changed")
    for f in state["files"]:
        for ref in (f, f["observations"], f["mother_csv"], f["provenance"], f["member_ledger"]): _pin(ref)
    if state.get("status") == "completed":
        _pin(state["record_reference"])
        record = json.loads(Path(state["record_reference"]["path"]).read_text(encoding="utf-8"))
        return {"status": "completed", "scene_sets": {"test": record}, "state_path": str(sp)}
    write(sp, state)
    if "convert" not in adapters:
        from .aviation import build_aviation_scenes
        converter = Path(__file__).with_name("aviation.py")
        if sha(converter) != config["converter_reference"]["sha256"]: raise ValueError("actual aviation import differs from85f pin")
        adapters["convert"] = build_aviation_scenes
    try:
        for asset in assets:
            day = asset["date"]; folder = out / day; folder.mkdir(exist_ok=True)
            if stop(): raise InterruptedError("preparation stopped")
            old_asset = next((x for x in state["assets"] if x["date"] == day), None)
            receipt = old_asset or adapters.get("download", _download)(asset, folder, progress, stop)
            _pin(receipt)
            if old_asset is None: state["assets"].append(receipt); write(sp, state)
            scanfile = folder / "scan_receipt.json"
            if scanfile.exists():
                scanned = json.loads(scanfile.read_text()); _pin(scanned["rows"]); _pin(scanned["ledger"])
                rows = json.loads(Path(scanned["rows"]["path"]).read_text()); times = [datetime.fromisoformat(x.replace("Z", "+00:00")) for x in scanned["times"]]; ledger = scanned["ledger"]
            else:
                rows, times, ledger = adapters.get("scan", _scan)(receipt, folder, config["scanner_reference"], progress, stop)
                rp = folder / "candidate_rows.json"; write(rp, rows)
                write(scanfile, {"rows": {"path": str(rp), "sha256": sha(rp)}, "times": [x.isoformat() for x in times], "ledger": ledger})
            selected = [x for x in state["files"] if x["source_date"] == day]
            done_indices = {x["grid_index"] for x in selected}
            rejected = {x["grid_index"] for x in state["rejections"] if x["source_date"] == day}
            preferred = [int((j + .5) * 288 / 20) for j in range(20)]
            order = preferred + [i for i in range(288) if i not in preferred]
            for i in order:
                if len(selected) == 20: break
                if i in done_indices or i in rejected: continue
                if stop(): raise InterruptedError("conversion stopped")
                stamp = times[i].isoformat().replace("+00:00", "Z")
                if stamp in registry.get("used_mother_utc", []): reason = "mother_UTC_already_used"
                elif len({(x[2], x[3]) for x in rows[i]}) < 100: reason = "fewer_than100_distinct_positions"
                else: reason = None
                csvpath = folder / f"mother_{i:03d}.csv"
                if reason is None:
                    with csvpath.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream); writer.writerow(("aircraft_id", "timestamp", "lat", "lon", "altitude_m")); writer.writerows(sorted(rows[i]))
                    state["converter_attempts"] = state.get("converter_attempts", 0) + 1
                    state["last_converter_attempt"] = {"date": day, "grid_index": i, "UTC": stamp}
                    write(sp, state)
                    try:
                        arrays, provenance = adapters["convert"]([csvpath], [stamp], source_format="csv_snapshot", units={"timestamp": "utc_iso8601", "lat": "degrees", "lon": "degrees", "altitude_m": "m", "persons": "count"}, demand_model=config["demand_model"], seed=51001, max_record_age_seconds=60, environment_profile="europe_v1", partition_role="test")
                    except Exception as exc:
                        from .aviation import AviationSceneError
                        if not isinstance(exc, AviationSceneError): raise
                        reason = "original_converter_rejected: " + str(exc)
                if reason is not None:
                    state["rejections"].append(dict(source_date=day, grid_index=i, selected_time_utc=stamp, reason=reason)); write(sp, state); continue
                import numpy as np
                data = arrays[0]
                if len(data["rate"]) != 100 or not np.isfinite(data["rate"]).all() or not (data["rate"] > 0).all(): raise ValueError("original converter returned invalid100 demands")
                detail = folder / f"observations_{i:03d}.jsonl.gz"
                observations = provenance["snapshots"][0].pop("source_observations")
                with detail.open("wb") as stream:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
                        for obs in observations: zipped.write((json.dumps(obs, sort_keys=True) + "\n").encode())
                npz = folder / f"scene_{i:03d}.npz"; np.savez_compressed(npz, **data)
                pp = folder / f"provenance_{i:03d}.json"; write(pp, provenance)
                f = dict(path=str(npz), sha256=sha(npz), scene_id=f"t24_{day}_{i:03d}", source_path=str(csvpath), source_date=day, grid_index=i, selected_time_utc=stamp, N=100, mother_csv={"path": str(csvpath), "sha256": sha(csvpath)}, observations={"path": str(detail), "sha256": sha(detail)}, provenance={"path": str(pp), "sha256": sha(pp)}, member_ledger=ledger, sorted_aircraft_ids_sha256=hashlib.sha256(json.dumps(sorted(o["aircraft_id"] for o in observations)).encode()).hexdigest(), archive_sha256=receipt["sha256"])
                state["files"].append(f); selected.append(f); write(sp, state)
                progress("scenes", len(state["files"]), 40, {"date": day, "grid_index": i})
            if len(selected) != 20: raise ValueError("insufficient20 eligible mothers: freeze replacement date before model access")
        record = dict(id="s2_t24_test", name="T0 historical reproduction new40", path=str(out), count=40, files=sorted(state["files"], key=lambda x: x["selected_time_utc"]), profile="europe_v1", source={"experiment_id": "S2-T24", "purpose": "explicit_historical_reproduction", "final_three_stage_interface_claim": False, "uniform_request_qualified": False, "seed": 51001, "demand_model": config["demand_model"], "upstream_assets": state["assets"], "converter_reference": config["converter_reference"], "scanner_reference": config["scanner_reference"], "freshness_registry": config["freshness_registry"], "rejections": state["rejections"], "converter_attempts": state.get("converter_attempts",0), "learning_steps":0, "B_calls":0, "selection": "fixed20 grid then remainingUTC ascending; qualification only; no model/performance selection", "raw_member_bytes_retained": True})
        rp = out / "record.json"; write(rp, record)
        state.update(status="completed", record_reference={"path": str(rp), "sha256": sha(rp)}); write(sp, state)
        return {"status": "completed", "scene_sets": {"test": record}, "state_path": str(sp)}
    except Exception as exc:
        state.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=repr(exc)); write(sp, state); raise
