"""Finalizer ordering/resume/attention tests; never starts the real finalizer."""
from contextlib import redirect_stdout
from io import StringIO
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_nondrl12 import finalizer,archive,report,strict_audit
from test_nondrl_postprocessing import make_volume


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def fixture(root,phase="report_pending"):
    plan={"experiment_id":"S1-ND12","scientific_commit":"a"*40,"freeze_status":"frozen_for_execution","name":"真实中文冻结方案"}
    registry={"experiment_id":"S1-ND12","scientific_commit":"a"*40}
    state={"experiment_id":"S1-ND12","phase":phase,"plan_sha256":report.stable_sha(plan),
           "registry_sha256":report.stable_sha(registry),"jobs":{}}
    preparations={"experiment_id":"S1-ND12","scientific_commit":"a"*40,"all_completed":True,
        "tasks":{day:{"stage":"completed","port":port,"run_id":"prepare_"+day.replace("-",""),"role":role}
                 for day,role,port in (("2024-02-10","validation",20473),("2024-02-11","test",20474),("2024-02-12","test",20418))}}
    paths=finalizer.input_paths(root)
    for key,value in (("plan",plan),("state",state),("preparations",preparations),("manifest",{"experiment_id":"S1-ND12"}),("registry",registry)):
        write(paths[key],value)
    return paths


class FinalizerTests(unittest.TestCase):
    def test_launcher_absence_is_not_failure_and_attention_exits(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=fixture(root)
            self.assertTrue(finalizer.readiness(root,paths))
            write(root/"controller/launcher.json",{"phase":"attention_required","error":"明确故障"})
            with self.assertRaisesRegex(finalizer.Attention,"attention"):
                finalizer.readiness(root,paths)

    def test_raw_utf8_snapshot_is_immutable_and_registry_pinned(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=fixture(root)
            raw=paths["plan"].read_bytes()
            with patch.object(finalizer,"reserve"):
                saved=finalizer.freeze_inputs(root,paths)
                self.assertEqual(Path(saved["files"]["plan"]["path"]).read_bytes(),raw)
                finalizer.freeze_inputs(root,paths)
                Path(saved["files"]["plan"]["path"]).write_bytes(b"altered")
                with self.assertRaisesRegex(finalizer.Attention,"changed"):
                    finalizer.freeze_inputs(root,paths)

    def test_wait_uses60s_and_no_launcher_pid_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root,phase="initial")
            with (patch.object(finalizer,"reserve"),patch.object(finalizer,"readiness",side_effect=[False,True]),
                  patch.object(finalizer.time,"sleep") as sleep,patch.object(finalizer,"finalize",return_value={"ok":True}) as complete,
                  redirect_stdout(StringIO())):
                result=finalizer.run(root,wait_hours=1)
            sleep.assert_called_once_with(60);complete.assert_called_once()
            self.assertEqual(result["completion"],{"ok":True})

    def test_attention_no_retry_and_restart_does_not_extend_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root)
            with (patch.object(finalizer,"reserve"),patch.object(finalizer,"readiness",side_effect=finalizer.Attention("engineering")),
                  patch.object(finalizer.time,"sleep") as sleep,redirect_stdout(StringIO())):
                first=finalizer.run(root,wait_hours=1)
                again=finalizer.run(root,wait_hours=48)
            self.assertEqual(first["phase"],"attention_required")
            self.assertEqual(first["deadline_unix"],again["deadline_unix"]);sleep.assert_not_called()

    def test_module_change_stops_without_archive_or_science(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root)
            counter={"calls":0}
            def changed(path):
                counter["calls"]+=1
                return "a"*64 if counter["calls"]<=5 else "b"*64
            with (patch.object(finalizer,"reserve"),patch.object(archive,"digest",side_effect=changed),
                  patch.object(finalizer,"finalize") as complete,redirect_stdout(StringIO())):
                result=finalizer.run(root)
            self.assertEqual(result["phase"],"attention_required");complete.assert_not_called()

    def test_archive_all_before_report_and_not_ready_preserves_issues(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=fixture(root);events=[]
            planned={"tasks":{"one":{"port":20418,"run_id":"jobx"}},"scientific_commit":"a"*40}
            source={"port":20418,"run_id":"shared_sources"};job={"port":20418,"run_id":"jobx"}
            def mark(label,value):
                def call(*args,**kwargs):events.append(label);return value
                return call
            with (patch.object(finalizer,"reserve"),patch.object(archive,"plan",return_value=planned),
                  patch.object(archive,"one_source",side_effect=mark("source",source)),
                  patch.object(archive,"one_task",side_effect=mark("task",job)),
                  patch.object(archive,"verify_source_receipt",side_effect=mark("verify_source",{})),
                  patch.object(archive,"verify_receipt",side_effect=mark("verify_task",{"closed":True})),
                  patch.object(finalizer,"dependency_map",return_value={}),
                  patch.object(report,"build",side_effect=mark("report",{"formal_ready":False,"issues":["真实证据缺口"]})),
                  patch.object(report,"figures") as figures):
                with self.assertRaisesRegex(finalizer.Attention,"not ready"):
                    finalizer.finalize(root,paths,lambda:None,lambda *a,**k:None)
            self.assertEqual(events,["source","task","verify_source","verify_task","report"])
            self.assertEqual(finalizer.read(root/"analysis/final_report/report.json")["issues"],["真实证据缺口"])
            figures.assert_not_called();self.assertFalse((root/"docs").exists())

    def test_volume_batch_cache_streams_once_and_invalidates_after_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=make_volume(root,"volume.zip",{"x":b"real bytes"});cache=set()
            real=archive.verify_volume
            with patch.object(archive,"verify_volume",wraps=real) as verify:
                archive.verify_cached_volume(root/"volume.zip",volume,cache)
                archive.verify_cached_volume(root/"volume.zip",volume,cache)
                self.assertEqual(verify.call_count,1)
                with (root/"volume.zip").open("ab") as stream:stream.write(b"corruption")
                with self.assertRaises(ValueError):archive.verify_cached_volume(root/"volume.zip",volume,cache)
                self.assertEqual(verify.call_count,2)

    def test_canonical_dependency_path_map_reads_zip_without_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job={"kind":"train","model_dir":"/remote/archive/models/runs/jobx"}
            volume=make_volume(root,"job.zip",{"20418/jobx/job.json":json.dumps(job).encode()})
            receipt={"experiment_id":"S1-ND12","port":20418,"run_id":"jobx","volumes":[volume]}
            path=finalizer.receipt_path(root,receipt);write(path,receipt)
            mapping=finalizer.dependency_map(root,[receipt],{},set())
            self.assertEqual(mapping["/remote/archive/models/runs/jobx/"]["member_prefix"],"models/")
            self.assertEqual(strict_audit.Evidence(path).json("job.json"),job)

    def test_fixed_endpoint_export_hash_and_checkpoint_stays_in_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);policy=b"small canonical policy";policy_sha=hashlib.sha256(policy).hexdigest()
            model={"policy_sha256":policy_sha,"selection":"fixed_budget_endpoint"}
            volume=make_volume(root,"model.zip",{"20418/trainx/job.json":b'{"kind":"train"}',
                "20418/trainx/status.json":b'{"stage":"completed"}',"20418/trainx/models/final/policy.pt":policy,
                "20418/trainx/models/final/model.json":json.dumps(model).encode(),
                "20418/trainx/models/checkpoint.pt":b"large recovery state remains canonical"})
            receipt={"experiment_id":"S1-ND12","port":20418,"run_id":"trainx","volumes":[volume]}
            write(finalizer.receipt_path(root,receipt),receipt)
            with patch.object(finalizer,"reserve"):
                result=finalizer.export_endpoints(root,[receipt],set())
                self.assertEqual(result[0]["sha256"],policy_sha)
                self.assertEqual(Path(result[0]["path"]).read_bytes(),policy)
                self.assertFalse((Path(result[0]["path"]).parent/"checkpoint.pt").exists())
                Path(result[0]["path"]).write_bytes(b"corrupt")
                with self.assertRaisesRegex(finalizer.Attention,"differs"):
                    finalizer.export_endpoints(root,[receipt],set())

    def test_single_os_lock_rejects_competing_finalizer(self):
        from satellite_rl.coverage.nondrl.runner import worker_lock
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root);directory=root/"controller/finalizer";directory.mkdir(parents=True)
            with worker_lock(directory),patch.object(finalizer,"reserve"):
                with self.assertRaises(OSError):finalizer.run(root)


if __name__=="__main__":unittest.main()
