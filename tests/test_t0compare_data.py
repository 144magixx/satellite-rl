"""Pure mock preparation qualification: zero network/physics/converter calls."""
import json
import io
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from satellite_rl.workbench.t0compare_data import BASELINE, DATES, MODEL, prepare_test_data, qualify_metadata, sha, write, _scan, _download
from unittest.mock import patch
from types import SimpleNamespace
from satellite_rl.workbench.aviation import AviationSceneError


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.out = self.root / "prepared"
        def pin(name, value):
            p = self.root / name; write(p, value); return {"path": str(p), "sha256": sha(p)}
        converter = pin("converter.txt", {"mock": True}); scanner = pin("scanner.txt", {"mock": True})
        self.config = dict(experiment_id="S2-T24", baseline_commit=BASELINE,
            owner_run_id="prepare_owner", frozen_source_sha="frozen_sha",
            frozen_source_root=str(self.root), purpose="explicit_historical_reproduction",
            historical_authority=pin("authority.json", {"human_choice": "option_one"}),
            freshness_registry=pin("freshness.json", {"qualified": True, "used_dates": [], "used_mother_utc": []}),
            converter_reference=converter, scanner_reference=scanner,
            admission_reference=pin("admission.json", {"approved": True, "owner_run_id": "prepare_owner", "frozen_source_sha": "frozen_sha"}),
            demand_model={**MODEL, "parameter_source": "original historical T0"}, source_assets=[])
        for i, day in enumerate(DATES):
            tag = "v" + day.replace("-", ".") + "-planes-readsb-prod-0"
            url = f"https://github.com/adsblol/globe_history_2024/releases/download/{tag}/{tag}.tar"
            meta = {"tag_name": tag, "assets": [{"name": tag + ".tar", "id": i + 1, "size": 4, "browser_download_url": url}]}
            self.config["source_assets"].append(dict(date=day, asset_id=i+1, expected_bytes=4, url=url, metadata_reference=pin(day + ".json", meta)))
        self.convert_calls = 0
        def download(asset, out, progress, stop):
            p = out / "source.tar"; p.write_bytes(b"mock"); return {**asset, "path": str(p), "sha256": sha(p)}
        def scan(asset, folder, reference, progress, stop):
            day = datetime.fromisoformat(asset["date"]).replace(tzinfo=timezone.utc)
            times = [day + timedelta(minutes=2+5*i) for i in range(288)]
            rows = [[['plane%03d'%j, t.isoformat(), 30+j*.001, 10, 1000] for j in range(100)] for t in times]
            p = folder / "ledger.json"; write(p, {"members": [{"sha256": sha(Path(asset['path']))}]})
            return rows, times, {"path": str(p), "sha256": sha(p)}
        def convert(paths, times, **kwargs):
            self.convert_calls += 1
            if self.convert_calls == 1:
                raise AviationSceneError("mock positive100 qualification reject")
            obs = [{"aircraft_id": "plane%03d"%j, "demand_bps": 1, "processing_index": j} for j in range(100)]
            return [{"rate": np.ones((100,1)), "original_indices": np.arange(100)}], {"snapshots": [{"source_observations": obs}], "kwargs": kwargs}
        self.adapters = dict(download=download, scan=scan, convert=convert)

    def test_metadata_is_offline_and_unknown_body_sha(self):
        pins, _ = qualify_metadata(self.config)
        self.assertEqual(len(pins), 2); self.assertNotIn("sha256", pins[0])
        self.assertEqual(self.convert_calls, 0)

    def test_40_scenes_rejection_replacement_idempotence_and_artifact_tamper(self):
        result = prepare_test_data(self.config, self.out, adapters=self.adapters)
        rec = result["scene_sets"]["test"]
        self.assertEqual(rec["count"], 40); self.assertEqual(len(rec["files"]), 40)
        self.assertEqual(len(rec["source"]["rejections"]), 1)
        self.assertFalse(rec["source"]["uniform_request_qualified"])
        calls = self.convert_calls
        again = prepare_test_data(self.config, self.out, adapters=self.adapters)
        self.assertEqual(again, result); self.assertEqual(calls, self.convert_calls)
        Path(rec["files"][0]["observations"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "pinned source changed"):
            prepare_test_data(self.config, self.out, adapters=self.adapters)

    def test_stop_preserves_partial_and_resume_exact_scene_prefix(self):
        stopped = lambda: self.convert_calls >= 4
        with self.assertRaises(InterruptedError):
            prepare_test_data(self.config, self.out, adapters=self.adapters, stop_requested=stopped)
        state = json.loads((self.out/"preparation_state.json").read_text())
        prefix = [(f["selected_time_utc"], f["sha256"]) for f in state["files"]]
        self.assertEqual(state["status"], "stopped"); self.assertEqual(len(prefix), 3)
        result = prepare_test_data(self.config, self.out, adapters=self.adapters)
        indexed = {f["selected_time_utc"]: f["sha256"] for f in result["scene_sets"]["test"]["files"]}
        self.assertTrue(all(indexed[k] == v for k,v in prefix))

    def test_changed_identity_and_owner_block_before_download(self):
        with self.assertRaises(InterruptedError):
            prepare_test_data(self.config, self.out, adapters=self.adapters, stop_requested=lambda: True)
        self.config["owner_run_id"] = "another_owner"
        with self.assertRaisesRegex(ValueError, "owner admission mismatch"):
            prepare_test_data(self.config, self.out, adapters=self.adapters)

    def test_used_date_blocks(self):
        ref = self.config["freshness_registry"]; write(ref["path"], {"qualified": True, "used_dates": [DATES[0]]}); ref["sha256"] = sha(ref["path"])
        with self.assertRaisesRegex(ValueError, "test date already used"): qualify_metadata(self.config)

    def test_request_model_change_blocks(self):
        self.config["demand_model"]["sampling"] = "expected"
        with self.assertRaisesRegex(ValueError, "demand model changed"): qualify_metadata(self.config)

    def test_scanner_retains_original_member_bytes_and_latest_record(self):
        cleaner = self.root / "cleaner.py"
        cleaner.write_text('def candidate_times(day):\n return [datetime.fromisoformat(day).replace(tzinfo=timezone.utc)+timedelta(minutes=2)]\ndef utc_iso(stamp):\n return datetime.fromtimestamp(stamp, timezone.utc).isoformat()\ndef inspect_record(row):\n return ("selected",tuple(row[1:4]))\n')
        archive = self.root / "tiny.tar"; folder = self.root / "scan"; folder.mkdir()
        content = json.dumps({"icao":"id1", "timestamp":datetime(2024,2,21,tzinfo=timezone.utc).timestamp(), "trace":[[0,30,10,1000],[110,31,11,1100]]}).encode()
        with tarfile.open(archive,'w') as tar:
            info=tarfile.TarInfo('./traces/aa/id1.json');info.size=len(content);tar.addfile(info,io.BytesIO(content))
        with patch('satellite_rl.workbench.t0compare_data.shutil.disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            rows,times,ledger=_scan(dict(date=DATES[0],path=str(archive),sha256=sha(archive)),folder,{"path":str(cleaner),"sha256":sha(cleaner)},lambda *args:None,lambda:False)
        self.assertEqual(rows[0][0][2:],[31,11,1100])
        records=json.loads(Path(ledger['path']).read_text())['members']
        self.assertEqual((folder/'members'/records[0]['sha256']).read_bytes(),content)

    def test_existing_partial_without_receipt_is_not_trusted(self):
        folder=self.root/'download';folder.mkdir()
        a={**self.config['source_assets'][0], 'metadata_sha256':'meta','release_tag':'prod0'}
        (folder/'prod0.tar.partial').write_bytes(b'bad')
        # No completed range receipt: discard uncommitted bytes and never infer SHA.
        class Response:
            status=206;headers={'Content-Range':'bytes 0-3/4'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):return b'good'
        with patch('satellite_rl.workbench.t0compare_data.urllib.request.urlopen',return_value=Response()), patch('satellite_rl.workbench.t0compare_data.shutil.disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            result=_download(a,folder,lambda *args:None,lambda:False)
        self.assertEqual(Path(result['path']).read_bytes(),b'good')


if __name__ == "__main__": unittest.main()
