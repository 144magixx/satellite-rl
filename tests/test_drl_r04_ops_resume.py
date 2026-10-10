import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_drl_r04 import data_controller as dc


class PreparationResumeTests(unittest.TestCase):
    def test_existing_prepares_are_verified_without_resubmission(self):
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "preflight").mkdir()
            days = {day for _, day in dc.ASSIGNMENT.values()}
            (root / "preflight/source_metadata_upload.json").write_text(
                json.dumps({"sources": dict.fromkeys(days, {})}), encoding="utf8")
            tasks = {day: {"run_id": f"prepare_{port}", "port": port, "role": role,
                          "request_id": f"s1d12a_prepare_{day.replace('-', '')}_{port}",
                          "git_commit": commit, "kind": "prepare"}
                     for port, (role, day) in dc.ASSIGNMENT.items()}
            receipt = root / "prepares.json"
            receipt.write_text(json.dumps({"scientific_commit": commit, "tasks": tasks}))
            by_port = {v["port"]: (day, v) for day, v in tasks.items()}
            def api(port, path, body=None):
                self.assertIsNone(body)
                day, item = by_port[port]
                self.assertEqual(path, "/api/runs/" + item["run_id"])
                return {"job": {**item, "experiment_id": "S1-D12A",
                                "dataset_preparation": {"date_utc": day,
                                                        "role": item["role"]}}}
            catalog = {"drl_r04_defaults": {"coverage": {"method": "none"}}}
            with (patch.object(dc.ops, "LOCAL", root),
                  patch.object(dc, "RECEIPT", receipt),
                  patch.object(dc, "assert_service", return_value=catalog),
                  patch.object(dc.ops, "api", side_effect=api) as calls):
                result = dc.submit_preparations(commit)
                self.assertEqual(result["tasks"], tasks)
                self.assertEqual(calls.call_count, 3)

if __name__ == "__main__":
    unittest.main()
