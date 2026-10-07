"""S2-P19 Feb02 same-asset recovery contracts; no network or large downloads."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from satellite_rl.workbench import s2p19_data as data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def assets():
    rows = []
    for date, asset_id, size in data.OFFICIAL_ASSETS:
        tag = "v" + date.replace("-", ".") + "-planes-readsb-staging-0"
        rows.append({"date_utc": date, "release_tag": tag, "asset_id": asset_id,
                     "asset_name": tag + ".tar",
                     "url": f"https://github.com/adsblol/globe_history_2024/releases/download/{tag}/{tag}.tar",
                     "expected_bytes": size})
    return rows


def fixture(root, wrong_date=False):
    archive = root / "archive"
    old = archive / "outputs" / "workbench" / "jobs" / data.RECOVERY_PARENT_RUN_ID
    raw = old / "mother_sources" / "raw"
    day = old / "mother_sources" / "snapshots" / "final" / "2024-02-01"
    day.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    source_record = write(root / "source_record.json", {"profile": "europe_v1",
        "source": {"seed": data.SEED, "demand_model": {"window_seconds": 60}}})
    official = assets()
    tar = raw / official[0]["asset_name"]
    tar.write_bytes(b"Feb01tar")
    write(raw / "2024-02-01_download.json", {"status": "completed", "asset": official[0],
          "path": str(tar), "bytes": tar.stat().st_size, "sha256": sha(tar), "attempts": []})
    write(raw / "2024-02-02_download.json", {"status": "downloading", "asset": official[1],
          "partial_bytes": 0, "attempts": [{"number": i, "partial_bytes": 0} for i in range(1, 9)]})
    (raw / "2024-02-02_curl.stderr").write_text("curl: (28) timed out\n" * 8, encoding="utf-8")
    start = datetime(2024, 2, 1, 0, 2, tzinfo=timezone.utc)
    candidates = []
    for i in range(288):
        stamp = (start + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z")
        candidates.append({"index": i, "selected_time_utc": stamp, "eligible_200": True,
                           "rejection_reason": None, "selected_aircraft": 205,
                           "positive_demand_aircraft": 205, "unique_positions": 205})
    eligibility = write(day / "eligibility.json", {"date_utc": "2024-02-01",
                        "archive_sha256": sha(tar), "candidates": candidates})
    indices, substitutions = data._selected_indices(candidates)
    times = [candidates[i]["selected_time_utc"] for i in indices]
    if wrong_date:
        times[0] = times[0].replace("2024-02-01", "2024-02-02")
    write(day / "selection_contract.json", {"date_utc": "2024-02-01",
          "archive_sha256": sha(tar), "eligibility_sha256": sha(eligibility),
          "selected_indices": indices, "selected_times_utc": times,
          "substitutions": substitutions})
    csvs = []
    for index in indices:
        csvs.append(day / f"t{index:03d}.csv")
        csvs[-1].write_text("aircraft_id,timestamp,lat,lon,altitude_m\na,2024-02-01T00:02:00Z,45,10,10000\n",
                            encoding="utf-8")
    write(old / "job.json", {"archive_root": str(archive), "run_id": data.RECOVERY_PARENT_RUN_ID,
          "kind": "prepare", "experiment_id": "S2-P19", "research_stage": "s2p19_preparation",
          "request_id": "s2p19-feb01-feb02-mothers-20424"})
    write(old / "config.json", {"research_stage": "s2p19_preparation",
          "preparation": {"preparation_type": "mother_sources",
                          "source_record": {"path": str(source_record), "sha256": sha(source_record)},
                          "official_assets": official}})
    write(old / "status.json", {"stage": "failed", "run_id": data.RECOVERY_PARENT_RUN_ID})
    write(old / "failure.json", {"run_id": data.RECOVERY_PARENT_RUN_ID,
          "error": "official archive did not download after eight resumable attempts"})
    config = {"preparation_type": "mother_sources",
              "source_record": {"path": str(source_record), "sha256": sha(source_record)},
              "seed": data.SEED, "environment_profile": "europe_v1",
              "official_assets": official,
              "recovery_source": data.build_recovery_source(old)}
    return old, config, tar, csvs


class RecoveryTests(unittest.TestCase):
    def test_worker_reuses_feb01_and_only_requests_feb02_api_transport(self):
        with TemporaryDirectory() as temporary, patch.object(data, "OFFICIAL_ASSETS", (
                ("2024-02-01", 149304672, 8), ("2024-02-02", 149522532, 9))):
            root = Path(temporary)
            old, config, tar, csvs = fixture(root)
            original = {str(path): sha(path) for path in (tar, *csvs)}
            new_job = old.parent / "prepare_prepare_20261007_R1"
            frozen = data.freeze_preparation_config(config, new_job / "input_sources")
            feb02_tar = new_job / "mother_sources" / "raw" / assets()[1]["asset_name"]
            feb02_tar.parent.mkdir(parents=True)
            feb02_tar.write_bytes(b"Feb02tar!")
            start = datetime(2024, 2, 2, 0, 2, tzinfo=timezone.utc)
            candidates = []
            for index in range(288):
                stamp = (start + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
                candidates.append({"index": index, "selected_time_utc": stamp,
                                   "eligible_200": True, "rejection_reason": None,
                                   "selected_aircraft": 205, "positive_demand_aircraft": 205,
                                   "unique_positions": 205})
            rows = [[("a", item["selected_time_utc"], 45, 10, 10000)] for item in candidates]

            def fake_download(_raw, asset, _stop, _progress, *, api_asset, max_attempts):
                self.assertEqual(asset["date_utc"], "2024-02-02")
                self.assertTrue(api_asset)
                self.assertEqual(max_attempts, 4)
                return {"path": str(feb02_tar), "sha256": sha(feb02_tar),
                        "bytes": feb02_tar.stat().st_size}

            with (patch.object(data, "_official_metadata", return_value={"asset_id": 149522532}),
                  patch.object(data, "_download_archive", side_effect=fake_download) as download,
                  patch.object(data, "_scan_day", return_value=(rows, candidates, {"source_files": 1}))):
                result = data.run_preparation(new_job, frozen)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(result["count"], 40)
            record = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
            self.assertEqual(record["source"]["source_dates"], ["2024-02-01", "2024-02-02"])
            self.assertEqual(len(record["source"]["files"]), 40)
            self.assertEqual(record["source"]["upstream_archives"][0]["path"], str(tar))
            self.assertTrue(all(sha(path) == value for path, value in original.items()))

    def test_freeze_twenty_mothers_without_writing_old_assets(self):
        with TemporaryDirectory() as temporary, patch.object(data, "OFFICIAL_ASSETS", (
                ("2024-02-01", 149304672, 8), ("2024-02-02", 149522532, 9))):
            root = Path(temporary)
            old, config, tar, csvs = fixture(root)
            old_hashes = {str(path): sha(path) for path in (tar, *csvs)}
            frozen = data.freeze_preparation_config(
                config, old.parent / "prepare_prepare_20261007_R1" / "input_sources")
            self.assertEqual(frozen["recovery_source"]["parent_run_id"], data.RECOVERY_PARENT_RUN_ID)
            self.assertEqual(len(frozen["recovery_source"]["frozen_evidence"]["feb01_csv_files"]), 20)
            self.assertFalse((old.parent / "prepare_prepare_20261007_R1" / "input_sources" /
                              tar.name).exists())
            copied, snapshots, receipt, selection = data._reuse_feb01_mothers(
                old.parent / "prepare_prepare_20261007_R1", frozen["recovery_source"], None)
            self.assertEqual(len(copied), 20)
            self.assertEqual(len(snapshots), 20)
            self.assertEqual(receipt["reused_from_parent_run_id"], data.RECOVERY_PARENT_RUN_ID)
            self.assertEqual(selection["sha256"], config["recovery_source"]["feb01_selection"]["sha256"])
            self.assertTrue(all(sha(path) == value for path, value in old_hashes.items()))
            self.assertEqual(data.validate_preparation_config(frozen)["recovery_source"]["feb01_archive"],
                             config["recovery_source"]["feb01_archive"])

    def test_wrong_date_sha_or_archive_is_rejected_before_freeze(self):
        with TemporaryDirectory() as temporary, patch.object(data, "OFFICIAL_ASSETS", (
                ("2024-02-01", 149304672, 8), ("2024-02-02", 149522532, 9))):
            root = Path(temporary)
            _, config, _, _ = fixture(root, wrong_date=True)
            with self.assertRaises(ValueError):
                data.validate_preparation_config(config)
            _, good, _, _ = fixture(root)
            bad = deepcopy(good)
            bad["recovery_source"]["feb01_archive"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                data.validate_preparation_config(bad)
            with self.assertRaises(ValueError):
                data.freeze_preparation_config(good, root / "other_archive/jobs/new/input_sources")

    def test_api_transport_stops_after_four_new_attempts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = assets()[1]
            class FailProcess:
                returncode = 28

                def poll(self):
                    return self.returncode
            calls = []
            def popen(command, **_kwargs):
                calls.append(command)
                return FailProcess()
            with patch.object(data.subprocess, "Popen", side_effect=popen), patch.object(data.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "four API resumable attempts"):
                    data._download_archive(root, asset, None, None, api_asset=True, max_attempts=4)
            self.assertEqual(len(calls), 4)
            self.assertTrue(all("https://api.github.com/repos/adsblol/globe_history_2024/releases/assets/149522532" in call
                                and "Accept: application/octet-stream" in call for call in calls))
            receipt = json.loads((root / "2024-02-02_download.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["attempt_limit"], 4)
            self.assertEqual(len(receipt["attempts"]), 4)
            self.assertTrue(all(row["partial_bytes"] == 0 for row in receipt["attempts"]))


if __name__ == "__main__":
    unittest.main()
