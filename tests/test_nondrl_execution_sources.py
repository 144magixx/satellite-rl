"""Offline source-repair corruption and controller boundary checks; no jobs run."""
from copy import deepcopy
import hashlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_nondrl12 import archive, execution_sources as sources, finalizer, report
from scripts.stage1_nondrl12.strict_audit import Evidence
from scripts.stage1_nondrl12.controller import Controller, stable_sha
from scripts.stage1_nondrl12.controller_pipeline import PipelineController
from test_nondrl_controller import inputs
from test_nondrl_postprocessing import make_volume
from test_nondrl_finalizer import fixture as final_fixture


def raw(value):
    return json.dumps(value,sort_keys=True).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def lineage_fixture(root, future=False):
    run = "future_train" if future else "prepare_nondrl_20261012_045319_9ffedc65"
    commit="e"*40
    config=raw({"frozen_budget":10000})
    receipt=raw({"passed":True,"checks":["native deadline","no search-budget extension"]})
    file="src/satellite_rl/coverage/nondrl/core.py"
    origin=raw({"git_commit":sources.ORIGIN_COMMIT,"files":{file:sha(b"old core")}})
    snapshot=raw({"git_commit":commit,"files":{file:sha(b"bounded new core")}})
    provenance={"run_id":run,"git_commit":sources.ORIGIN_COMMIT,"source_sha256":sha(origin),"config_sha256":sha(config)}
    records={"job":raw({"run_id":run,"git_commit":sources.ORIGIN_COMMIT}),
             "status":raw({"stage":"failed"}),"config":config,"launch":raw({"pid":123,"process_start_time":12.})}
    baseline=raw({"provenance":provenance,"trials":14,"elapsed_seconds":10.,"rows":2})
    pin={"config_sha256":sha(config),"origin_provenance":provenance,
         "origin_source_manifest_sha256":sha(origin),
         **{k+"_sha256":sha(v) for k,v in records.items()},
         "artifacts":[{"path":"/old/teacher/manifest.json","relative_path":"teacher/manifest.json","sha256":sha(baseline),"kind":"teacher_manifest"}],
         "reason":"engineering deadline repair","comparison_scope":"preserve_committed_evidence_continue"}
    previous_run="prepare_nondrl_20261012_045319_9ffedc65" if future else run
    contract={"format":sources.FORMAT,"repair_id":sources.REPAIR_ID,"origin_commit":sources.ORIGIN_COMMIT,
        "execution_commit":commit,"allowed_run_ids":[previous_run],"runs":{previous_run:pin},
        "allowed_future_roles":[{"route":"01","kind":"train"}],
        "diff_files":[{"path":file,"origin_sha256":sha(b"old core"),"execution_sha256":sha(b"bounded new core")}],
        "validation_receipts":[{"path":"/remote/check.json","sha256":sha(receipt)}]}
    contract_payload=raw(contract)
    item={**pin,"format":sources.FORMAT,"repair_id":sources.REPAIR_ID,"run_id":run,
          "origin_commit":sources.ORIGIN_COMMIT,"execution_commit":commit,"contract_sha256":sha(contract_payload)}
    if future:item.update(comparison_scope="new_execution_task",origin_provenance=None)
    job={"experiment_id":"S1-ND12","run_id":run,"git_commit":commit,"execution_commit":commit,
         "origin_commit":sources.ORIGIN_COMMIT,"kind":"train" if future else "prepare","variant":"01",
         "preparation_mode":None if future else "teacher","source_repair_contract":item,
         "source_history":[{k:item[k] for k in ("origin_commit","execution_commit","repair_id","contract_sha256")} | {"source_manifest_sha256":sha(snapshot)}]}
    payloads={"job.json":raw(job),"config.json":config,"source_lineage/contract.json":contract_payload,
        "source/source_manifest.json":snapshot,"source/"+file:b"bounded new core",
        sources.receipt_member(contract["validation_receipts"][0]):receipt}
    if not future:payloads["teacher/manifest.json"]=baseline
    if not future:
        payloads.update({"source_lineage/origin/"+k+".json":v for k,v in records.items()})
        payloads["source_lineage/origin/source_manifest.json"]=origin
        payloads["source_lineage/origin/artifacts/teacher/manifest.json"]=baseline
    else:
        payloads["source_lineage/origin/source_manifest.json"]=origin
        payloads["source_lineage/origin_snapshot_pin.json"]=raw({"run_id":previous_run,"sha256":sha(origin)})
    for name,value in payloads.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    return job,contract,payloads


class PolicyTransport:
    def __init__(self,job,contract_sha):
        self.job=job;self.contract_sha=contract_sha;self.requests=[];self.stage="queued";self.scientific_stop=False;self.verifications=0

    def api(self,port,path,body=None):
        self.requests.append((port,path,deepcopy(body)))
        if path.startswith("/api/runs/"):
            return {"job":deepcopy(self.job),"status":{"stage":self.stage,"scientific_stop":self.scientific_stop}}
        if body is None:
            return {"runs":[{"run_id":self.job["run_id"],"stage":self.stage,"scientific_stop":self.scientific_stop}] if port==20418 else []}
        return deepcopy(self.job)

    def remote_python(self,code,port,timeout):
        self.verifications+=1
        self.last_code=code
        return {"exit_code":0,"stderr":"","stdout":json.dumps({"run_id":self.job["run_id"],"passed":True,
            "execution_commit":self.job["git_commit"],"contract_sha256":self.contract_sha})}

    def upload(self,*args):pass


class ExecutionSourcesTests(unittest.TestCase):
    def test_archive_directory_and_zip_preserve_both_sources_and_old_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,contract,payloads=lineage_fixture(root)
            proof=sources.audit_lineage(job,Evidence(root))
            self.assertEqual(proof["origin_commit"],sources.ORIGIN_COMMIT)
            self.assertNotEqual(proof["execution_commit"],sources.ORIGIN_COMMIT)
            prefix="20418/"+job["run_id"]+"/"
            volume=make_volume(root,"proof.zip",{prefix+k:v for k,v in payloads.items()})
            receipt={"experiment_id":"S1-ND12","run_id":job["run_id"],"port":20418,"volumes":[volume]}
            self.assertEqual(sources.audit_lineage(job,Evidence(root/"receipt.json",receipt=receipt)),proof)
            self.assertEqual(json.loads(Evidence(root).read("source_lineage/origin/artifacts/teacher/manifest.json"))["trials"],14)

    def test_corrupt_source_history_original_config_validation_and_current_source_fail_closed(self):
        mutations=("config.json","source_lineage/origin/status.json","source_lineage/origin/artifacts/teacher/manifest.json",
                   "source/src/satellite_rl/coverage/nondrl/core.py","source_lineage/contract.json")
        for name in mutations:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);job,_,_=lineage_fixture(root);(root/name).write_bytes(b"tampered")
                with self.assertRaises((ValueError,json.JSONDecodeError)):
                    sources.audit_lineage(job,Evidence(root))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,_,_=lineage_fixture(root);job["source_history"][-1]["contract_sha256"]="b"*64
            with self.assertRaisesRegex(ValueError,"history"):
                sources.audit_lineage(job,Evidence(root))

    def test_future_roles_and_origin_row_markers_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,contract,_=lineage_fixture(root,future=True)
            self.assertTrue(sources.audit_lineage(job,Evidence(root))["repaired"])
            job["variant"]="05"
            with self.assertRaisesRegex(ValueError,"role"):
                sources.audit_lineage(job,Evidence(root))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,_,_=lineage_fixture(root)
            marker={k:job["source_repair_contract"]["origin_provenance"][k] for k in ("git_commit","source_sha256")}
            self.assertEqual(sources.verify_marker(job,marker),sources.ORIGIN_COMMIT)
            marker["source_sha256"]="0"*64
            with self.assertRaises(ValueError):sources.verify_marker(job,marker)

    def test_future_body_hash_and_request_id_stay_frozen_while_source_is_honest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,contract,_=lineage_fixture(root,future=True)
            path=root/"source_lineage/contract.json";pin=sha(path.read_bytes())
            (root/"source_policy.json").write_bytes(raw({"contract_path":"/remote/contract.json","contract_sha256":pin,"local_contract_path":str(path)}))
            plan,registry=inputs();plan["scientific_commit"]=registry["scientific_commit"]=sources.ORIGIN_COMMIT
            transport=PolicyTransport(job,pin);controller=Controller(root,plan,registry,transport)
            body=controller.body("01","main","train",None,20418,teacher_source={"port":"20418","run_id":"teacher"})
            record=controller.ensure("train/01",20418,body)
            self.assertEqual(record["body_sha256"],stable_sha(body))
            published=transport.requests[0][2]
            self.assertEqual(published["request_id"],"nd12_"+stable_sha({"plan":controller.plan_sha,"key":"train/01"})[:40])
            self.assertEqual(published["config"],body["config"])
            self.assertIn("execution_source",published)
            self.assertEqual(record["source_commit"],job["git_commit"])
            self.assertIn("validate_job_lineage",transport.last_code)

    def test_attention_requires_true_healthy_api_and_preserves_old_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,contract,_=lineage_fixture(root)
            path=root/"source_lineage/contract.json";pin=sha(path.read_bytes())
            (root/"source_policy.json").write_bytes(raw({"contract_path":"/remote/contract.json","contract_sha256":pin,"local_contract_path":str(path)}))
            plan,registry=inputs();plan["scientific_commit"]=registry["scientific_commit"]=sources.ORIGIN_COMMIT
            transport=PolicyTransport(job,pin);controller=PipelineController(root,plan,registry,transport)
            record={"run_id":job["run_id"],"port":20418,"stage":"stopped"}
            controller.state["jobs"]["teacher/01"]=record
            controller.route_attention("01","workbench task explicitly stopped; continuation needs diagnosis",key="teacher/01")
            controller.route_attention("05","unrelated evidence gap",key="teacher/05")
            snapshot={p:{} for p in (20418,20424,20473,20474)}
            snapshot[20418][job["run_id"]]={"stage":"queued"}
            transport.stage="stopped"
            controller.reconcile_source_attention(snapshot)
            self.assertIn("01",controller.state["route_attention"])
            transport.stage="queued";controller.reconcile_source_attention(snapshot)
            self.assertNotIn("01",controller.state["route_attention"])
            self.assertIn("05",controller.state["route_attention"])
            self.assertEqual(controller.state["source_repair_attention_history"][0]["previous_attention"]["key"],"teacher/01")
            controller.route_attention("01","workbench task explicitly stopped; continuation needs diagnosis",key="teacher/01")
            controller.reconcile_source_attention(snapshot)
            self.assertIn("01",controller.state["route_attention"])
            self.assertEqual(len(controller.state["source_repair_attention_history"]),1)
            controller.route_attention("01","measured failure",key="teacher/01")
            controller.state["routes"]["01"]={"status":"science_stopped"}
            controller.reconcile_source_attention(snapshot)
            self.assertIn("01",controller.state["route_attention"])

    def test_final_input_freezes_small_receipts_and_does_not_follow_mutable_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=final_fixture(root)
            job,contract,_=lineage_fixture(root/"proof")
            cp=root/"proof/source_lineage/contract.json";pin=sha(cp.read_bytes())
            check=root/sources.receipt_member(contract["validation_receipts"][0])
            check.parent.mkdir(parents=True,exist_ok=True);check.write_bytes(raw({"passed":True,"checks":["native deadline","no search-budget extension"]}))
            policy={"contract_path":"/remote/contract.json","contract_sha256":pin,"local_contract_path":str(cp),"validation_path_map":{"/remote/check.json":str(check)}}
            (root/"controller/source_policy.json").write_bytes(raw(policy))
            state=json.loads(paths["state"].read_bytes());state["source_policy_sha256"]=pin;paths["state"].write_bytes(raw(state))
            paths=finalizer.input_paths(root)
            with patch.object(finalizer,"reserve"):
                snapshot=finalizer.freeze_inputs(root,paths)
            cp.write_bytes(b"mutated live file")
            refs=report.frozen_source_refs(root,state)
            self.assertEqual(refs["source_contract"]["sha256"],pin)
            Path(refs["source_contract"]["path"]).write_bytes(b"mutated frozen file")
            with self.assertRaisesRegex(ValueError,"SHA"):
                report.frozen_source_refs(root,state)

    def test_remote_archive_builder_compiles_with_new_lineage_branch(self):
        compile("REQUEST_JSON='{}'\n"+archive.REMOTE_BUILD,"remote archive","exec")

    def test_mixed_or_overrun_timing_keeps_quality_and_blocks_efficiency_claim(self):
        mother={"mother_scene_id":"m","load_cv_all":.2,"aircraft_carry":1.,"hard_feasible":True,"total_seconds":1.}
        baseline={"mothers":[{**mother,"load_cv_all":.4,"total_seconds":10.}],"summary":{"source_versions":[sources.ORIGIN_COMMIT]}}
        method={"mothers":[{**mother,"online_budget_overrun":True}],"summary":{"source_versions":[sources.ORIGIN_COMMIT,"e"*40]}}
        result=report.paired_comparison(method,baseline)
        self.assertTrue(result["quality_branch_met"])
        self.assertFalse(result["timing_budget_comparable"])
        self.assertFalse(result["efficiency_branch_met"])

    def test_history_copy_does_not_allow_relabelling_or_budget_rollback(self):
        for key,value in (("trials",0),("elapsed_seconds",0.),("provenance",{"git_commit":"e"*40})):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);job,_,_=lineage_fixture(root)
                current=json.loads((root/"teacher/manifest.json").read_bytes())
                current["format"]="s1_nondrl_teacher_v1"
                baseline=deepcopy(current)
                # Bind a real original format before hashing the contract.
                original=root/"source_lineage/origin/artifacts/teacher/manifest.json"
                original.write_bytes(raw(baseline))
                pin=job["source_repair_contract"]
                pin["artifacts"][0]["sha256"]=sha(raw(baseline))
                contract=json.loads((root/"source_lineage/contract.json").read_bytes())
                contract["runs"][job["run_id"]]["artifacts"]=pin["artifacts"]
                payload=raw(contract);(root/"source_lineage/contract.json").write_bytes(payload)
                pin["contract_sha256"]=sha(payload);job["source_history"][-1]["contract_sha256"]=sha(payload)
                current[key]=value;(root/"teacher/manifest.json").write_bytes(raw(current))
                with self.assertRaisesRegex(ValueError,"budget|provenance"):
                    sources.audit_lineage(job,Evidence(root))

    def test_compact_receipt_bounds_payload_and_is_idempotent(self):
        result={"status":"attention_required","statistics":{"actual_complete_problems":60,"mother_metrics":[{"dataset_index":i} for i in range(20)],
            "problem_metrics":[{"dataset_index":i//3,"result":{"metrics_all":{"load_cv_all":.2,"geometry_feasible":True,"unique_assignment":True,"demand_conserved":True,"beam_demand_bps":[2]*10000},
            "physical":{"aircraft_service":{"demand_weighted_carried_fraction":.99,"per_aircraft":[1]*10000}}}} for i in range(60)]},"sources":[{"sha256":"a"*64,"path":"canonical raw"}]}
        compact=sources.compact_receipt(result)
        self.assertLess(len(raw(compact)),30000)
        self.assertEqual(len(compact["statistics"]["problem_metrics"]),60)
        self.assertEqual(compact,sources.compact_receipt(compact))
        self.assertNotIn("per_aircraft",json.dumps(compact))
        self.assertEqual(compact["statistics"]["problem_metrics"][0]["final_carry"],.99)

    def test_verified_existing_gzip_reused_once_without_raw_recopy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"controller/state.json";path.parent.mkdir()
            state={"route_attention":{"02":{"reason":"gap","receipt":{"statistics":{"problem_metrics":[{"result":{"physical":{"huge":[1]*1000}}}]}}}}}
            payload=raw(state);path.write_bytes(payload)
            saved=root/"preflight/repair_epoch/postops_censored";saved.mkdir(parents=True)
            packed=saved/"original.gz";packed.write_bytes(gzip.compress(payload))
            (saved/"preservation.json").write_bytes(raw([{"source":str(path),"source_sha256":sha(payload),"gzip":str(packed),"gzip_sha256":sha(packed.read_bytes())}]))
            (saved/"history_roundtrip.json").write_bytes(raw({"passed":True,"checks":[{"gzip":str(packed),"uncompressed_sha256":sha(payload),"bytes":len(payload),"passed":True}]}))
            sources.compact_state_history(path,state)
            self.assertEqual(state["payload_compaction_history"][0]["path"],str(packed))
            self.assertFalse((path.parent/"history").exists())
            path.write_bytes(raw(state));sources.compact_state_history(path,state)
            self.assertEqual(len(state["payload_compaction_history"]),1)
            self.assertEqual(gzip.decompress(packed.read_bytes()),payload)

    def test_bounded_disposition_closes_only_exact_exhausted_unknown_probe_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,contract,_=lineage_fixture(root)
            run="prepare_nondrl_20261012_045228_1d87b8e0";job["run_id"]=run;job["variant"]="02";job["preparation_mode"]="teacher_probe"
            path=root/"source_lineage/contract.json";pin=sha(path.read_bytes())
            (root/"source_policy.json").write_bytes(raw({"contract_path":"/remote/contract.json","contract_sha256":pin,"local_contract_path":str(path)}))
            plan,registry=inputs();plan["scientific_commit"]=registry["scientific_commit"]=sources.ORIGIN_COMMIT
            transport=PolicyTransport(job,pin);transport.stage="completed"
            controller=PipelineController(root,plan,registry,transport)
            record={"run_id":run,"port":20418,"stage":"completed"};controller.state["jobs"]["teacher_probe/02"]=record
            receipt={"status":"attention_required","passed":False,"scientific_stop":False,
                "reason":"offline_teacher_search_evidence_gap: bounded unknown/reference-only problems remain",
                "statistics":{"actual_complete_problems":60,"legal_reference_mothers":list(range(20)),"improved_mothers":[0,1],
                    "unknown_search_problems":7,"recorded_teacher_seconds":601.,"true_trials":123},
                "task_source":{"run_id":run,"contract_sha256":pin,"wrapper_sha256":"4"*64,"evidence_manifest_sha256":"5"*64}}
            helper=root/"teacher_reachability/route_02.json";helper.parent.mkdir();helper.write_bytes(raw(receipt))
            entry={"run_id":run,"status":"budget_limited_inconclusive","reason":"authorized evidence insufficiency",
                "teacher_wrapper_sha256":"4"*64,"teacher_evidence_manifest_sha256":"5"*64,"helper_receipt_path":str(helper),"helper_receipt_sha256":sha(helper.read_bytes())}
            scope={"format":"s1_nd12_bounded_dispositions_v1","origin_commit":sources.ORIGIN_COMMIT,"contract_sha256":pin,"scientific_stop":False,"entries":{"02":entry}}
            (root/"bounded_dispositions.json").write_bytes(raw(scope))
            snapshots={20418:{run:{"stage":"completed"}}}
            controller.apply_bounded_dispositions(snapshots)
            gate=controller.state["routes"]["02"]
            self.assertEqual(gate["status"],"budget_limited_inconclusive")
            self.assertFalse(gate["passed"]);self.assertFalse(gate["scientific_stop"])
            calls=transport.verifications;controller.apply_bounded_dispositions(snapshots)
            self.assertEqual(transport.verifications,calls)
            self.assertNotIn("train/02",controller.state["jobs"])
            bad=deepcopy(receipt);bad["statistics"]["recorded_teacher_seconds"]=599.
            with self.assertRaises(ValueError):sources.validate_disposition(scope,"02",bad)
            bad=deepcopy(receipt);bad["statistics"]["improved_mothers"]=list(range(6))
            with self.assertRaises(ValueError):sources.validate_disposition(scope,"02",bad)


if __name__=="__main__":unittest.main()
