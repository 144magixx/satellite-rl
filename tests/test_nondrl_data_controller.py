"""Bounded deployment/client checks; no SSH, GPU work or official tasks."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.stage1_nondrl12 import data_controller as data, deploy, launcher, ops
from satellite_rl.workbench import coverage_nondrl as api
from satellite_rl.workbench import nondrl_preparation as preparation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ClientTests(unittest.TestCase):
    def test_prepare_payloads_match_actual_server_validation_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root / "remote_group"
            partition = {"group": "S1-ND12", "roles": {
                "train": {"dates": ["2024-01-08"], "count": 1},
                "development": {"dates": ["2024-01-12"], "count": 1},
                "validation": {"dates": ["2024-02-10"], "count": 1},
                "test": {"dates": ["2024-02-11", "2024-02-12"], "count": 2}}}
            partition_path = parent / "source/partition_manifest.json"
            write_json(partition_path, partition)
            parser = parent / "source/parser.py"; parser.write_text("# frozen fixture")
            refs = {"partition": {"path": str(partition_path), "sha256": data.sha(partition_path)},
                    "parser": {"path": str(parser), "sha256": data.sha(parser)}}
            for day, (role, _) in data.ASSIGNMENT.items():
                base = parent / "source" / role / day
                archive_name = f"v{day.replace('-', '.')}-planes-readsb-prod-0.tar"
                url = "https://example.invalid/" + archive_name
                contract = {"group": "S1-ND12", "role": role, "date_utc": day, "count": 1,
                            "selected_times_utc": [day+"T12:00:00Z"], "source_instance": "prod-0",
                            "no_outcome_based_replacement": True}
                release = {"assets": [{"name": archive_name, "id": 1000, "size": 1, "browser_download_url": url}]}
                write_json(base / "selection_contract.json", contract)
                write_json(base / "release_api.json", release)
                (base / "PREFERRED_RELEASES.txt").write_text(url)
                (base / "LICENSE-ODbL-official.txt").write_text("test metadata")
                pin = {"group": "S1-ND12", "role": role, "date_utc": day, "archive_name": archive_name,
                       "asset_id": 1000, "expected_bytes": 1, "archive_url": url,
                       "selection_contract_sha256": data.sha(base/"selection_contract.json"),
                       "release_api_sha256": data.sha(base/"release_api.json"),
                       "preferred_sha256": data.sha(base/"PREFERRED_RELEASES.txt"),
                       "license_sha256": data.sha(base/"LICENSE-ODbL-official.txt")}
                write_json(base/"source_preflight.json", pin)
            local = root / "local"
            write_json(local/"preflight/source_metadata_upload.json", refs)
            submitted, jobs = [], {}
            class FakeStore:
                def create_job(self, config, *, kind, request_id, extra):
                    submitted.append((config,kind,request_id,extra))
                    run_id = "prepare_" + extra["source_date"].replace("-", "")
                    value = {"run_id": run_id, "git_commit": "a"*40, **extra}
                    jobs[run_id] = value
                    return value
            application = SimpleNamespace(store=FakeStore())
            def call(port, path, body=None, **kwargs):
                if path.endswith("/catalog"):
                    return api.api(application, "GET", ["catalog"], {})
                if path.endswith("/nondrl/runs"):
                    return api.submit(application, body)
                run_id = path.split("/")[-1]
                return {"job": jobs[run_id], "status": {"stage": "completed"},
                        "artifacts": {"raw_archive_sha256": "b"*64}}
            with (patch.object(ops,"LOCAL",local), patch.object(data,"STATE",local/"state.json"),
                 patch.object(ops,"api",side_effect=call), patch.object(api,"deployment",return_value="20418"),
                 patch.object(preparation,"PARENT",parent)):
                first = data.prepare_once("a"*40)
                second = data.prepare_once("a"*40)
            self.assertTrue(first["all_completed"])
            self.assertEqual(first["tasks"],second["tasks"])
            self.assertEqual(len(submitted),3)
            self.assertEqual({x[3]["source_date"] for x in submitted},set(data.ASSIGNMENT))

    def test_existing_registration_and_partition_mismatch(self):
        record = {"id":"dataset_x","partition_role":"test","sha256":"a"*64,"count":100,"path":"/input.npz","partition_manifest_sha256":"b"*64}
        with patch.object(data,"records",return_value=[record]),patch.object(ops,"api") as network:
            result=data.ensure_import(20418,"test",record,{"path":"/partition.json","sha256":"b"*64})
            self.assertEqual(result["id"],"dataset_x");network.assert_not_called()
            with self.assertRaisesRegex(ValueError,"partition"):
                data.ensure_import(20418,"test",record,{"path":"/partition.json","sha256":"c"*64})

    def test_recovery_cap_and_attempt_saved_before_network(self):
        row={"port":20418,"run_id":"prepare_x","stage":"failed","resume_attempts":0,
             "last_status":{"stage":"failed","error":"temporary network"}}
        state={"tasks":{"day":row}};saved=[]
        def save(path,value):saved.append(copy.deepcopy(value))
        def call(port,path,body):
            self.assertEqual(saved[-1]["tasks"]["day"]["resume_attempts"],1)
            self.assertEqual(path,"/api/runs/prepare_x/resume")
            return {"stage":"queued"}
        with patch.object(data,"save",side_effect=save),patch.object(ops,"api",side_effect=call):
            launcher.preparation_recovery(state)
        self.assertEqual(row["stage"],"queued")
        row.update(stage="failed",resume_attempts=2)
        with patch.object(ops,"api") as network:
            with self.assertRaisesRegex(RuntimeError,"cap"):
                launcher.preparation_recovery(state)
            network.assert_not_called()

    def test_launcher_plan_guard_and_no_science_when_upload_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"run_plan.json"
            plan={"experiment_id":"S1-ND12","scientific_commit":"a"*40,"freeze_status":"frozen_for_execution"}
            write_json(path,plan)
            self.assertEqual(launcher.validate_inputs(path,"a"*40)[0],plan)
            with self.assertRaises(ValueError):launcher.validate_inputs(path,"b"*40)
            with (patch.object(ops,"LOCAL",root),patch.object(data,"upload_sources",side_effect=RuntimeError("network")),
                 patch.object(data,"prepare_once") as prepare):
                self.assertEqual(launcher.run(path,"a"*40),"attention_required")
                prepare.assert_not_called()
            self.assertEqual(json.loads((root/"controller/launcher.json").read_text())["phase"],"attention_required")

    def test_upload_rejects_traversal_and_shell_characters(self):
        with patch.object(ops.transport,"upload",return_value={"ok":True}) as send, \
                patch.object(ops,"remote_python",return_value={"exit_code":0,"stdout":"","stderr":""}) as prepare:
            ops.upload(Path("a"),ops.BASE+"/20418/ops/a.bundle",20418)
            self.assertEqual(send.call_count,1)
            self.assertEqual(prepare.call_count,1)
            self.assertIn(".parent.mkdir(parents=True,exist_ok=True)",prepare.call_args.args[0])
            for path in (ops.BASE+"/../old/x",ops.BASE+"_other/x",ops.BASE+"/a b",ops.BASE+"/a;echo",ops.BASE+"/a$(secret)"):
                with self.assertRaises(ValueError):ops.upload(Path("a"),path,20418)
            self.assertEqual(send.call_count,1)
            self.assertEqual(prepare.call_count,1)

    def test_serve_command_matches_current_server_flags_and_pinned_source(self):
        calls=[]
        def remote(port,code,timeout=120):
            compile(code,"serve_lane_remote","exec")
            calls.append(code)
            return {"source_commit":"a"*40}
        with patch.object(deploy,"remote",side_effect=remote):
            deploy.serve_lane(20418,"a"*40,2)
        self.assertIn("--archive-root",calls[0]);self.assertIn("--max-workers",calls[0])
        self.assertIn("SATELLITE_RL_SSH_PORT",calls[0]);self.assertIn("PYTHONDONTWRITEBYTECODE",calls[0])
        self.assertIn(str(ops.API_PORTS[20418]),calls[0]);self.assertIn(ops.DQN,calls[0])


if __name__=="__main__":unittest.main()
