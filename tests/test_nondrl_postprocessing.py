import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from io import StringIO
from contextlib import redirect_stdout
import unittest
import zipfile
from unittest.mock import patch
import numpy as np

from satellite_rl.coverage.nondrl import solve
from satellite_rl.coverage.nondrl import metrics_all
from satellite_rl.coverage.nondrl.runner import jsonable, RowStore
from satellite_rl.coverage.scenes import CoverageDataset
from satellite_rl.coverage.variable_dataset import save_variable_dataset
from satellite_rl.coverage.variable_design import Layout
from satellite_rl.coverage.variable_downstream import evaluate_layout
from scripts.stage1_nondrl12 import archive, report, strict_audit
from test_nondrl_core import small_scene


def raw_row(scene, initial, multiplier=1.):
    modified = copy.deepcopy(scene)
    modified.demands = scene.demands * multiplier
    result = jsonable(solve(modified, "I0", {"online_seconds": 2.}, initial))
    return {"key": f"0:{multiplier:g}", "run_id": "micro_run", "dataset_index": 0,
            "mother_scene_id": scene.scene_id, "demand_multiplier": multiplier,
            "result": result, "measured_task_seconds": result["timing"]["total_seconds"],
            "route": "06", "method": "I0"}


def make_volume(root, filename, files):
    path = root / filename
    with zipfile.ZipFile(path, "w") as out:
        for name, value in files.items():
            out.writestr(name, value)
    return {"file": filename, "canonical_path": str(path), "bytes": path.stat().st_size,
            "sha256": archive.digest(path), "members": {name: {"bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest()} for name, value in files.items()}}


class PostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.initial = small_scene((9e6, 1e6, 1e6, 1e6))
        cls.row = raw_row(cls.scene, cls.initial)

    def test_real_TDMA_algebra_and_all_demand_denominator(self):
        result = strict_audit.audit_row(self.row, self.scene)
        self.assertEqual(result["N"], 2)
        self.assertEqual(result["aircraft_carry"], 1.)
        tampered = copy.deepcopy(self.row)
        tampered["result"]["physical"]["aircraft_service"]["demand_weighted_carried_fraction"] = .9
        with self.assertRaisesRegex(ValueError, "denominator"):
            strict_audit.audit_row(tampered, self.scene)

    def test_tampered_assignment_load_or_TDMA_cannot_pass(self):
        for mutation in ("load", "airtime", "center"):
            row = copy.deepcopy(self.row)
            if mutation == "load":
                row["result"]["metrics_all"]["beam_demand_bps"][0] += 1e6
            elif mutation == "airtime":
                row["result"]["physical"]["aircraft_service"]["airtime_fraction"][0] /= 2
            else:
                row["result"]["physical"]["beams"][0]["lat"] += .01
            with self.assertRaises(ValueError):
                strict_audit.audit_row(row, self.scene)

    def test_zero_demand_remains_undefined(self):
        scene = copy.deepcopy(self.scene); scene.demands[:] = 0
        row = raw_row(scene, self.initial)
        result = strict_audit.audit_row(row, scene)
        self.assertIsNone(result["load_cv_all"])
        self.assertIsNone(result["aircraft_carry"])

    def test_complete_RowStore_job_and_shard_forgery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "inputs.npz"
            save_variable_dataset(CoverageDataset([self.scene], {}), data)
            config = {"coverage": {"nondrl": {"route": "06", "method": "I0"}},
                      "datasets": {"test": {"path": str(data), "sha256": archive.digest(data)}},
                      "evaluation": {"demand_multipliers": [.75, 1., 1.25]}}
            job = {"experiment_id": "S1-ND12", "run_id": "micro_run", "kind": "evaluate",
                   "dataset_role": "test", "git_commit": "a"*40, "scene_indices": [0]}
            for name, value in (("job.json", job), ("config.json", config), ("status.json", {"stage": "completed", "run_id": "micro_run"})):
                (root/name).write_text(json.dumps(value))
            store = RowStore(root/"raw_scenes", {"run_id": "micro_run", "git_commit": "a"*40, "config_sha256": "b"*64})
            for multiplier in (.75, 1., 1.25):
                row = raw_row(self.scene, self.initial, multiplier)
                store.append(row["key"], row)
            audit = strict_audit.audit_job(root)
            self.assertEqual(len(audit["rows"]), 3)
            shard = root/"raw_scenes"/store.manifest["rows"][0]["file"]
            with shard.open("ab") as stream:stream.write(b"forgery")
            with self.assertRaisesRegex(ValueError, "SHA"):
                strict_audit.audit_job(root)

    def test_mother_aggregation_is_not_300_row_or_percentage_mean(self):
        base = strict_audit.audit_row(self.row, self.scene)
        rows = []
        for mother, cvs in (("A", [1.,2.,3.]), ("B", [2.,2.,2.])):
            for level, cv in zip((.75,1.,1.25), cvs):
                item = copy.deepcopy(base)
                item.update(mother_scene_id=mother, multiplier=level, load_cv_all=cv,
                            task_seconds=level, total_seconds=level, published_I0=True)
                rows.append(item)
        summary, mothers = report.aggregate(rows)
        self.assertEqual(summary["load_cv_all"]["mean"], 2.)
        self.assertEqual(summary["load_cv_all"]["count"], 2)
        self.assertEqual(summary["total_seconds"]["p95"], 1.)
        with self.assertRaisesRegex(ValueError, "three"):
            report.aggregate(rows[:-1])

    def test_zero_baseline_percentage_not_fabricated(self):
        mother = {"mother_scene_id": "x", "load_cv_all": 0., "aircraft_carry": 1.,
                  "total_seconds": 1., "hard_feasible": True}
        result = report.paired_comparison({"mothers": [mother]}, {"mothers": [mother]})
        self.assertIsNone(result["relative_CV_change_ratio_of_means"])
        self.assertFalse(result["quality_branch_met"])

    def test_gate_cannot_change_original_I0_or_hide_mechanism_gap(self):
        measured = []
        source_rows = []
        for multiplier in (.75,1.,1.25):
            row = raw_row(self.scene,self.initial,multiplier)
            audited = strict_audit.audit_row(row,self.scene)
            measured.append(audited)
            source_rows.append({"index":0,"multiplier":multiplier,"mother_id":self.scene.scene_id,
                "cv":audited["load_cv_all"],"initial_cv":audited["initial_cv"],
                "carry":audited["aircraft_carry"],"initial_carry":audited["initial_carry"],
                "feasible":True,"initial_sha256":audited["initial_sha256"]})
        threshold={"relative_cv_improvement":.01,"minimum_improved_mothers":0}
        audits={"micro_run":{"state":"completed","rows":measured,"raw_manifest_sha256":"b"*64}}
        gate={"experiment_id":"S1-ND12","route":"03","threshold":threshold,"passed":True,
              "sources":[{"run_id":"micro_run","manifest_sha256":"b"*64,"rows":source_rows}],
              "actual_row_count":3,"expected_row_count":3,"mother_count":1,
              "improved_mothers":[],"geometry_pass":True}
        self.assertTrue(report.verify_gate(gate,threshold,audits)["actual_passed"])
        forged=copy.deepcopy(gate);forged["sources"][0]["rows"][0]["initial_cv"]*=2
        with self.assertRaisesRegex(ValueError,"initial_cv"):
            report.verify_gate(forged,threshold,audits)
        required={**threshold,"require_mechanism_audit":True};gate["threshold"]=required
        with self.assertRaisesRegex(ValueError,"engineering gap"):
            report.verify_gate(gate,required,audits)

    def test_shared_BH_does_not_require_route06_scientific_pass(self):
        base=strict_audit.audit_row(self.row,self.scene)
        audits={}
        for port in (20418,20424,20473,20474):
            rows=[]
            for multiplier in (.75,1.,1.25):
                row=copy.deepcopy(base);row.update(mother_scene_id=f"mother{port}",multiplier=multiplier)
                rows.append(row)
            run_id=f"baseline_{port}"
            audits[run_id]={"run_id":run_id,"key":f"test/06/BH/{port}","port":port,"method":"BH",
                            "dataset_role":"test","state":"completed","rows":rows}
        result=report.shared_baseline(audits)
        self.assertEqual(result["summary"]["row_count"],12)
        self.assertEqual(result["summary"]["mother_count"],4)
        self.assertIn("does not count as route06",result["scope"])
        incomplete=copy.deepcopy(audits);incomplete.pop("baseline_20474")
        self.assertIsNone(report.shared_baseline(incomplete))

    def _teacher_probe_fixture(self, root, route="02"):
        root=Path(root)
        mothers=[]
        for index in range(20):
            mother=copy.deepcopy(self.scene)
            mother.scene_id=f"teacher_mother_{index}";mother.source_hash=f"fixture_source_{index}"
            mothers.append(mother)
        dataset_path=root/"scene_inputs/development/inputs.npz"
        save_variable_dataset(CoverageDataset(mothers,{"partition_role":"development"}),dataset_path)
        job={"experiment_id":"S1-ND12","kind":"prepare","preparation_mode":"teacher_probe",
             "run_id":"teacher_probe_fixture","git_commit":"a"*40}
        config={"coverage":{"nondrl":{"route":route}},"datasets":{"development":{"path":str(dataset_path),"sha256":archive.digest(dataset_path)}},
                "evaluation":{"demand_multipliers":[.75,1.,1.25]}}
        provenance={"run_id":job["run_id"],"git_commit":job["git_commit"],"config_sha256":"b"*64}
        templates={}
        for multiplier in (.75,1.,1.25):
            before=raw_row(self.scene,self.initial,multiplier)["result"]
            scaled=copy.deepcopy(self.scene);scaled.demands*=multiplier
            final=Layout.from_record(self.initial.to_record());final.assignment[1]=1
            templates[multiplier]=(before,final.to_record(),metrics_all(scaled,final),evaluate_layout(scaled,final,{}))
        receipts=[]
        for index,mother in enumerate(mothers):
            for multiplier in (.75,1.,1.25):
                before,final,final_metrics,final_physical=templates[multiplier]
                improved=index<6
                problem={"format":"s1_nondrl_teacher_problem_v1","route":route,"provenance":provenance,"target_n":2,
                    "mother_id":mother.scene_id,"source_hash":mother.source_hash,"demand_multiplier":multiplier,
                    "scene_positions":mother.positions.tolist(),"scene_demands_bps":(mother.demands*multiplier).tolist(),
                    "aircraft_ids":mother.aircraft_ids,"initial_layout":before["initial_layout"],
                    "final_layout":final if improved else before["final_layout"],
                    "initial_metrics_all":before["initial_metrics_all"],"final_metrics_all":final_metrics if improved else before["metrics_all"],
                    "initial_physical":before["initial_physical"],"final_physical":final_physical if improved else before["physical"],
                    "initial_sha256":before["initial_sha256"],"config":before["config"],"trace":[],
                    "timing":before["timing"],"calls":before["calls"],"teacher_elapsed_seconds":.01}
                relative=f"problem_receipts/problem_{len(receipts):06d}.json.gz"
                path=root/"teacher"/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(gzip.compress(json.dumps(problem).encode(),mtime=0))
                receipts.append({"path":relative,"sha256":archive.digest(path),"mother_id":mother.scene_id,"demand_multiplier":multiplier})
        manifest={"format":"s1_nondrl_teacher_v1","route":route,"status":"completed","provenance":provenance,
                  "rows":0,"shards":[],"problems":60,"trials":60,"elapsed_seconds":.6,"problem_receipts":receipts}
        wrapper={"dataset_sha256":archive.digest(dataset_path),"sha256":"","completed":True,
                 "evidence_path":str(root/"teacher/manifest.json")}
        for name,value in (("job.json",job),("config.json",config),("status.json",{"run_id":job["run_id"],"stage":"completed"}),
                           ("teacher/manifest.json",manifest)):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        wrapper["sha256"]=archive.digest(root/"teacher/manifest.json")
        (root/"teacher_manifest.json").write_text(json.dumps(wrapper))
        return manifest,wrapper

    def _rewrite_teacher_manifest(self,root,manifest,wrapper):
        (root/"teacher/manifest.json").write_text(json.dumps(manifest))
        wrapper["sha256"]=archive.digest(root/"teacher/manifest.json")
        (root/"teacher_manifest.json").write_text(json.dumps(wrapper))

    def test_full_teacher60_problem_proof_and_true_authority_over_online_proxy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self._teacher_probe_fixture(root)
            audit=strict_audit.audit_job(root);proof=audit["teacher_probe"]
            self.assertEqual(proof["complete_problems"],60)
            self.assertEqual(len(proof["rows"]),60)
            measured={"status":"passed","passed":True,"task_source":{"run_id":audit["run_id"],"wrapper_sha256":proof["wrapper_sha256"],
                      "dataset_sha256":proof["dataset_sha256"],"scientific_commit":audit["git_commit"]},
                      "statistics":{"actual_complete_problems":60,"legal_reference_mothers":list(range(20)),"improved_mothers":list(range(6))}}
            gate={"route":"02","source_commit":"a"*40,"passed":True,"improved_mothers":list(range(6)),
                  "teacher_reachability":measured,"proxy_diagnostics":{"method":"T","improved_mothers":[]}}
            verdict=report.verify_teacher_gate(gate,{audit["run_id"]:audit})
            self.assertTrue(verdict["actual_passed"]);self.assertTrue(verdict["proxy_diagnostics_retained"])
            forged=copy.deepcopy(gate);forged["teacher_reachability"]["statistics"]["improved_mothers"].append(6)
            with self.assertRaisesRegex(ValueError,"statistics"):
                report.verify_teacher_gate(forged,{audit["run_id"]:audit})

    def test_teacher_fullproof_tampering_even_with_recomputed_container_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest,wrapper=self._teacher_probe_fixture(root,"03")
            original_manifest=copy.deepcopy(manifest);raw_path=root/"teacher"/manifest["problem_receipts"][0]["path"]
            original_raw=raw_path.read_bytes();original_problem=json.loads(gzip.decompress(original_raw))
            for mutation in ("demand","airtime","initial_sha","provenance","target_n"):
                problem=copy.deepcopy(original_problem);manifest=copy.deepcopy(original_manifest)
                if mutation=="demand":problem["scene_demands_bps"][0]+=1e6
                elif mutation=="airtime":problem["final_physical"]["aircraft_service"]["airtime_fraction"][0]/=2
                elif mutation=="initial_sha":problem["initial_sha256"]="0"*64
                elif mutation=="provenance":problem["provenance"]["git_commit"]="0"*40
                else:problem["target_n"]=3
                raw_path.write_bytes(gzip.compress(json.dumps(problem).encode(),mtime=0))
                manifest["problem_receipts"][0]["sha256"]=archive.digest(raw_path)
                self._rewrite_teacher_manifest(root,manifest,wrapper)
                with self.assertRaises(ValueError,msg=mutation):strict_audit.audit_job(root)
            raw_path.write_bytes(original_raw)
            missing=copy.deepcopy(original_manifest);missing["problem_receipts"].pop()
            self._rewrite_teacher_manifest(root,missing,wrapper)
            with self.assertRaisesRegex(ValueError,"denominator"):
                strict_audit.audit_job(root)
            over=copy.deepcopy(original_manifest);over["trials"]=7681
            self._rewrite_teacher_manifest(root,over,wrapper)
            with self.assertRaisesRegex(ValueError,"ledger"):
                strict_audit.audit_job(root)

    def test_dynamic_mechanism_pins_utf8_content_and_immutable_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"中文机制核验.json"
            saved={"experiment_id":"S1-ND12","route":"06","status":"science_stopped","passed":False,
                   "reason":"真实机会不足，保留失败", "sources":[{"run_id":"dev06","path":"/jobs/dev06/raw_scenes/manifest.json","sha256":"b"*64}]}
            path.write_text(json.dumps(saved,ensure_ascii=False),encoding="utf-8")
            gate={"route":"06","mechanism":{**saved,"receipt_source":{"path":str(path),"sha256":archive.digest(path)}}}
            audits={"dev06":{"state":"completed","raw_manifest_sha256":"b"*64}}
            self.assertFalse(report.verify_dynamic_mechanism(gate,audits)["passed"])
            bad=copy.deepcopy(gate);bad["mechanism"]["reason"]="伪造已通过"
            with self.assertRaisesRegex(ValueError,"fields"):
                report.verify_dynamic_mechanism(bad,audits)
            with self.assertRaisesRegex(ValueError,"SHA"):
                report.verify_dynamic_mechanism(gate,{"dev06":{"state":"completed","raw_manifest_sha256":"c"*64}})

    def test_all_postops_read_text_uses_explicit_utf8(self):
        import ast
        for module in (report,strict_audit,archive):
            tree=ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="read_text":
                    self.assertTrue(any(k.arg=="encoding" and isinstance(k.value,ast.Constant) and k.value.value=="utf-8" for k in node.keywords),
                                    f"{module.__name__}:{node.lineno}")

    def test_incomplete_or_missing_mechanism_is_not_scientific_failure(self):
        plan = {"experiment_id": "S1-ND12", "scientific_commit": "a"*40,
                "methods": {f"{i:02d}": ["main"] for i in range(1,13)},
                "gate_thresholds": {f"{i:02d}": {} for i in range(1,13)}}
        state = {"experiment_id": "S1-ND12", "plan_sha256": report.stable_sha(plan),
                 "phase": "attention_required", "jobs": {}, "routes": {}}
        with tempfile.TemporaryDirectory() as folder:
            value = report.build(plan,state,folder)
        self.assertFalse(value["formal_ready"])
        self.assertEqual(value["routes"]["03"]["status"], "pending")

    def test_archive_task_and_source_member_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {"20418/micro_run/"+name: b"{}" for name in ("job.json","config.json","status.json")}
            volume = make_volume(root,"job.zip",files)
            receipt = {"experiment_id": "S1-ND12", "run_id": "micro_run", "port": 20418,
                       "source_commit": "a"*40, "volumes": [volume]}
            self.assertTrue(archive.verify_receipt(receipt,root)["closed"])
            bad = copy.deepcopy(receipt);bad["volumes"][0]["members"][next(iter(files))]["sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError,"member SHA"):
                archive.verify_receipt(bad,root)

    def test_source_bundle_exactly_three_days_and_reference_closure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);files={};sources=[]
            for role,day in (("validation","2024-02-10"),("test","2024-02-11"),("test","2024-02-12")):
                prefix=f"shared_sources/{role}/{day}"
                files[prefix+"/raw.tar"] = day.encode()
                files[prefix+"/source_preflight.json"]=b"pin"
                files[prefix+"/selection_contract.json"]=b"selection"
                run_id="prepare_"+day.replace("-","")
                files[prefix+"/source_job_identity.json"]=json.dumps({"experiment_id":"S1-ND12","git_commit":"a"*40,"run_id":run_id}).encode()
                files[prefix+"/source_job_status.json"]=b'{"stage":"completed"}'
                files[prefix+"/source_job_artifacts.json"]=json.dumps({"raw_archive_sha256":hashlib.sha256(day.encode()).hexdigest()}).encode()
                sources.append({"role":role,"date_utc":day,"source_bundle_member":prefix+"/raw.tar",
                                "source_run_id":run_id,"source_port":20418,
                                "bytes":len(day),"sha256":hashlib.sha256(day.encode()).hexdigest(),
                                "source_preflight_sha256":hashlib.sha256(b"pin").hexdigest(),
                                "selection_contract_sha256":hashlib.sha256(b"selection").hexdigest()})
            files["shared_sources/partition_manifest.json"] = b"partition"
            volume=make_volume(root,"source.zip",files)
            receipt={"experiment_id":"S1-ND12","run_id":"shared_sources","source_commit":"a"*40,
                     "volumes":[volume],"source_refs":sources,
                     "partition_manifest_sha256":hashlib.sha256(b"partition").hexdigest()}
            self.assertEqual(len(archive.verify_source_receipt(receipt,root)),19)
            receipt["source_refs"][0]["sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"identity"):
                archive.verify_source_receipt(receipt,root)

    def test_canonical_drive_reserve_and_bounded_readv_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.multiple(archive,LOCAL=root/"D",BACKUP=root/"C",EXTRA=root/"E"):
                result=archive.plan_canonical(20418,{"run_id":"x","volumes":[{"file":"x.zip","bytes":30}]},
                        {"local":archive.RESERVE+10,"backup":archive.RESERVE+10,"extra":archive.RESERVE+40})
                self.assertIn("E",result["x.zip"])
                with self.assertRaises(RuntimeError):
                    archive.plan_canonical(20418,{"run_id":"y","volumes":[{"file":"x.zip","bytes":100}]},
                        {"local":archive.RESERVE+10,"backup":archive.RESERVE+10,"extra":archive.RESERVE+40})
            class Remote:
                def readv(self, chunks):return [b"abcdefghij"[offset:offset+size] for offset,size in chunks]
            self.assertEqual(b"".join(archive.remote_blocks(Remote(),10,3,start_offset=4)),b"efghij")

    def test_remote_builder_offline_same_job_retry_and_no_duplicate_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/"s1_nd12";lane=base/"20418";store=lane/"archive"
            jobdir=store/"outputs/workbench/jobs/micro_run";jobdir.mkdir(parents=True)
            source=store/"source"/('a'*40);source.mkdir(parents=True)
            (source/"source_manifest.json").write_text(json.dumps({"git_commit":"a"*40,"files":{}}))
            data=jobdir/"scene_inputs/test/inputs.npz";data.parent.mkdir(parents=True);data.write_bytes(b"NPZ")
            job={"experiment_id":"S1-ND12","run_id":"micro_run","kind":"evaluate",
                 "git_commit":"a"*40,"archive_root":str(store),"source_root":str(source)}
            config={"datasets":{"test":{"path":str(data),"sha256":archive.digest(data)}}}
            for name,value in (("job.json",job),("config.json",config),("status.json",{"stage":"completed"})):
                (jobdir/name).write_text(json.dumps(value))
            request={"base":str(base),"port":20418,"run_id":"micro_run","archive":str(store),
                     "source_commit":"a"*40,"mode":"task","destination":str(lane/"ops/export"),
                     "reserve":0,"part_bytes":256,"max_files":100,"max_bytes":10000}
            results=[]
            for _ in range(2):
                out=StringIO()
                try:
                    with redirect_stdout(out):exec(compile(archive.REMOTE_BUILD,"remote_builder","exec"),{"REQUEST_JSON":json.dumps(request)})
                except SystemExit as error:self.assertEqual(error.code,0)
                results.append(json.loads(out.getvalue()))
            self.assertEqual(results[0],results[1])
            names=[name for volume in results[0]["volumes"] for name in volume["members"]]
            self.assertEqual(sum(name.endswith("inputs.npz") for name in names),1)
            for volume in results[0]["volumes"]:
                volume["canonical_path"]=volume["remote_path"]
            self.assertTrue(archive.verify_receipt(results[0],root)["closed"])

    def test_forged_dependency_receipt_is_not_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            volume=make_volume(root,"job.zip",{"20418/micro_run/"+name:b"{}" for name in ("job.json","config.json","status.json")})
            receipt={"experiment_id":"S1-ND12","run_id":"micro_run","port":20418,"source_commit":"a"*40,
                     "volumes":[volume],"dependency_refs":[{"sha256":"b"*64,"run_id":"teacher_run"}]}
            self.assertFalse(archive.verify_receipt(receipt,root)["closed"])
            forged={"experiment_id":"S1-ND12","source_commit":"a"*40,"port":20418,"run_id":"teacher_run",
                    "volumes":[{"file":"fake.zip","canonical_path":str(root/"fake.zip"),"members":{"teacher":{"sha256":"b"*64}},
                                "sha256":"b"*64,"bytes":1}]}
            with self.assertRaises(FileNotFoundError):archive.verify_receipt(receipt,root,dependency_receipts=[forged])


if __name__=="__main__":unittest.main()
