"""Prevent rerun transport and archival from falling back to the failed group."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_reconstruction import remote
from scripts.stage1_reconstruction import archive_completed as archive


class GroupIsolationTests(unittest.TestCase):
    def test_api_routes_rerun_and_legacy_to_distinct_ports(self):
        for group, port in (("s1_03", 8894), ("s1_03_r1", 8896)):
            with patch.object(remote, "run_code", return_value={}) as call:
                remote.api(object(), 20418, "coverage/runs", group=group)
                self.assertIn(f"127.0.0.1:{port}/api/", call.call_args.args[1])
        with patch.object(remote, "run_code") as call:
            with self.assertRaises(ValueError):
                remote.api(object(), 20418, "coverage/runs", group="../s1_02")
            call.assert_not_called()

    def test_rerun_archive_rejects_old_plan_and_old_destination(self):
        with tempfile.TemporaryDirectory() as temporary, patch.multiple(
                archive, GROUP="s1_03_r1", LOCAL=Path(temporary)):
            commit = "a" * 40
            old = remote.group_profile("s1_03")
            new = remote.group_profile("s1_03_r1")
            plan = {"experiment_id":old["experiment_id"], "hostroot":old["remote_base"],
                    "ports":[20418,20424], "science_commit":commit}
            def write():
                raw=json.dumps(plan).encode()
                (Path(temporary)/"plan.json").write_bytes(raw)
                (Path(temporary)/"controller_state.json").write_text(json.dumps({
                    "plan_sha256":hashlib.sha256(raw).hexdigest(),
                    "science_commit":commit,"jobs":{}}),encoding="utf-8")
            write()
            with self.assertRaises(ValueError):
                archive.verified_inputs()
            plan.update(experiment_id=new["experiment_id"], hostroot=new["remote_base"])
            write()
            archive.verified_inputs()
            job, model, source = archive.job_paths(20418, "train_ppo_rerun")
            for path in (job,model,source):
                self.assertTrue(path.startswith(new["remote_base"]+"/20418/"))
                self.assertFalse(path.startswith(old["remote_base"]+"/20418/"))

    def test_profiles_do_not_share_local_evidence_or_api_maps(self):
        old,new=remote.group_profile(),remote.group_profile("s1_03_r1")
        for key in ("local","models","remote_base","experiment_id"):
            self.assertNotEqual(old[key],new[key])
        self.assertFalse(set(old["api_ports"].values()) & set(new["api_ports"].values()))
        new["api_ports"][20418]=8894
        self.assertEqual(remote.group_profile("s1_03_r1")["api_ports"][20418],8896)


if __name__ == "__main__":
    unittest.main()
