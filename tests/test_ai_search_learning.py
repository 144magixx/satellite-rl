"""AI17 engineering fixtures: no scientific training or learned-effect claim."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from satellite_rl.coverage.ai_search.config import GeometryConfig, SearchConfig
from satellite_rl.coverage.ai_search.core import Scene, Layout, layout_sha256, metrics, validate_layout
from satellite_rl.coverage.ai_search.search import Task, Budget, BudgetStop, decode_task, run_search, compare_metrics
from satellite_rl.coverage.ai_search.features import FEATURE_NAMES, FEATURE_SCHEMA, task_features, pool_features, Normalizer
from satellite_rl.coverage.ai_search.teacher import (teacher_case, layout_delta,replay_delta,digest,TEACHER_FORMAT)
from satellite_rl.coverage.ai_search.learning import EvidenceDataset,make_model,loss,train,orderer,load_orderer,isolated_evaluation_rng


def scene_layout():
    scene=Scene([[50.,0.],[50.,.01],[50.,5.],[50.,5.01]],[1e6,3e6,2e6,4e6],("a","b","c","d"))
    current=Layout([[50.,0.],[50.,5.]],[.5,.5],[0,0,1,1])
    return scene,current


def dataset_fixture(root):
    """Synthetic completed/censored rows for loss/resume correctness only."""
    scene,current=scene_layout()
    (root/"scene.npz").write_bytes(b"synthetic scene input fixture")
    (root/"parent.json").write_text(json.dumps(current.to_record()))
    (root/"candidate_deltas.jsonl").write_text("")
    tasks=[Task("merge",(0,1),1,(1.,)),Task("joint_reconstruct",(0,1),2,(.5,.5)),Task("split",(0,1),3,(.3,.3,.3))]
    xs=pool_features(scene,current,tasks,0)
    xs=np.vstack((xs,xs[0]+3))
    rows=[]
    for i in range(4):
        rows.append(dict(mother_id="m1",profile="base",scene_sha256="synthetic",parent_layout_sha256=layout_sha256(current),
            preference_index=0,task_id=str(i),task_predecode_features=xs[i].tolist(),
            completion_status="censored" if i==3 else "completed",zero_total_request=False,
            completed_best_metrics={"N":i+1,"solid_angle_sum_sr":.001,"maxload":1e6,"top10_mean":1e6,"CV":.1},
            new_nondominated=None if i==3 else i==0))
    value=dict(format=TEACHER_FORMAT,status="completed",role="training",feature_schema=FEATURE_SCHEMA,rows=rows,
        scene=dict(path="scene.npz",sha256=digest(root/"scene.npz")),parents=[dict(path="parent.json",sha256=digest(root/"parent.json"))],
        candidate_evidence=dict(path="candidate_deltas.jsonl",sha256=digest(root/"candidate_deltas.jsonl")))
    path=root/"dataset.json";path.write_text(json.dumps(value));return path


class FeaturesTests(unittest.TestCase):
    def test_menu_and_shrink_inputs_distinguish_real_decoder_outcomes(self):
        scene,current=scene_layout()
        menu=Task("joint_reconstruct",(0,1),2,(.5,.5),width_policy="menu")
        shrink=Task("joint_reconstruct",(0,1),2,(.5,.5),width_policy="shrink")
        a=task_features(scene,current,menu);b=task_features(scene,current,shrink)
        self.assertEqual(FEATURE_SCHEMA["version"],"scalar48_preference5_v2")
        self.assertEqual(FEATURE_NAMES[47],"width_policy_shrink")
        self.assertEqual(np.flatnonzero(a!=b).tolist(),[47]);self.assertEqual(a[47],0);self.assertEqual(b[47],1)
        cfg=SearchConfig(seconds=2,max_attempts=64,variants=2,center_iterations=0)
        decoded_menu=decode_task(scene,current,menu,Budget(cfg),config=cfg)
        decoded_shrink=decode_task(scene,current,shrink,Budget(cfg),config=cfg)
        self.assertEqual(decoded_menu.status,"completed");self.assertEqual(decoded_shrink.status,"completed")
        self.assertGreater(decoded_menu.metrics["solid_angle_sum_sr"],decoded_shrink.metrics["solid_angle_sum_sr"])

    def test_input_units_and_no_outcome_leakage(self):
        scene,current=scene_layout(); task=Task("merge",(0,1),1,(1.,),cheap_features={"gain":1e300,"decoded_layout":"forbidden"})
        x=task_features(scene,current,task,4)
        self.assertEqual(len(FEATURE_NAMES),48);self.assertEqual(x.shape,(53,))
        self.assertAlmostEqual(float(x[2]),np.log1p(10),places=6)
        self.assertTrue(np.array_equal(x[-5:],[0,0,0,0,1]))
        same=Task("merge",(0,1),1,(1.,),cheap_features={"gain":-99})
        np.testing.assert_array_equal(x,task_features(scene,current,same,4))
        budget=Budget(SearchConfig(seconds=10,max_geometry_calls=5))
        task_features(scene,current,task,4,budget=budget)
        self.assertEqual(budget.geometry_calls,1); self.assertEqual(budget.ray_pair_checks,6)

    def test_zero_request_and_constant_columns(self):
        scene,current=scene_layout();scene=Scene(scene.positions,np.zeros(4),scene.aircraft_ids)
        x=task_features(scene,current,Task("merge",(0,1),1,(1.,)))
        self.assertTrue(np.isfinite(x).all()); self.assertEqual(x[5],0)
        normalizer=Normalizer.fit(np.stack((x,x)))
        np.testing.assert_array_equal(normalizer.transform(x),np.zeros(53))
        with self.assertRaises(ValueError):Normalizer.fit([x],role="validation")


class TeacherTests(unittest.TestCase):
    def test_variable_N_delta_replay(self):
        scene,current=scene_layout()
        candidate=Layout(scene.positions,np.full(4,.3),np.arange(4),("x","y","z","w"))
        replay=replay_delta(current,layout_delta(current,candidate))
        self.assertEqual(layout_sha256(candidate),layout_sha256(replay));self.assertTrue(validate_layout(scene,replay)["valid"])

    def test_actual_teacher_case_and_stop(self):
        scene,current=scene_layout()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            case=dict(positions=scene.positions,demands=scene.demands,aircraft_ids=scene.aircraft_ids,
                mother_id="fixture",mother_index=0,demand_profile="base")
            result=teacher_case(case,root/"run",SearchConfig(seconds=2,max_attempts=256,pool_size=6,variants=2,center_iterations=1))
            self.assertEqual(result["status"],"completed");self.assertTrue(result["rows"])
            self.assertLessEqual(result["states"],2);self.assertLessEqual(result["candidates"],256)
            self.assertEqual(digest(result["manifest"]["path"]),result["manifest"]["sha256"])
            init=json.loads((root/"run"/result["initialization"]["path"]).read_text())
            expected=run_search(scene,config=SearchConfig(seconds=2,max_attempts=256,pool_size=6,variants=2,center_iterations=1),method="INIT")
            self.assertEqual({r["sha256"] for r in init["result"]["pareto"]},{r["sha256"] for r in expected.pareto})
            archive=init["result"]["pareto"]
            first=result["parents"][0]
            self.assertEqual(min(archive,key=lambda r: r["metrics"]["N"])["metrics"]["N"],first["metrics"]["N"])
            for r in archive:self.assertLessEqual(compare_metrics(first["metrics"],r["metrics"],first["preference_index"]),0)
            self.assertEqual(result["budget"]["attempts"],init["budget_delta"]["attempts"]+sum(r["budget_delta"]["attempts"] for r in result["rows"]))
            self.assertEqual(init["initialization_candidate_limit"],256)
            self.assertEqual(result["budget"]["candidate_limit"],256)
            parents={p["sha256"]:Layout.from_record(json.loads((root/"run"/p["path"]).read_text())) for p in result["parents"]}
            for line in (root/"run"/"candidate_deltas.jsonl").read_text().splitlines():
                row=json.loads(line); replay=replay_delta(parents[row["parent_sha256"]],row["delta"])
                self.assertEqual(layout_sha256(replay),row["result_sha256"])
            stopped=teacher_case(case,root/"stopped",stop=lambda:True)
            self.assertEqual(stopped["status"],"stopped");self.assertEqual(stopped["budget"]["attempts"],0)
            self.assertEqual(stopped["rows"],[]);self.assertEqual(stopped["case_completion_status"],"censored_no_feasible_initialization")


class LearningTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_censor_pair_exclusion_and_train_only_normalizer(self):
        with tempfile.TemporaryDirectory() as name:
            dataset=EvidenceDataset(dataset_fixture(Path(name)))
            self.assertEqual(dataset.counts["censored"],1)
            for i in range(10):
                raw,ids=dataset.draw(np.random.default_rng(i),64)
                self.assertNotIn(3,ids["auxiliary"])
                self.assertTrue(all(3 not in x for x in ids["pairs"]))
            expected=np.stack([r["features"] for r in dataset.rows]).mean(axis=0)
            np.testing.assert_allclose(dataset.normalizer.mean,expected,rtol=1e-6)

    def test_actual_both_head_updates_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); path=dataset_fixture(root)
            protocol=dict(engineering=True,updates=4,batch=64,seed=51001,engineering_validation_nodes=[2,4])
            def request(model,normalizer,update,path):return dict(status="requested",node=update)
            whole=train(path,root/"whole",protocol,device="cpu",validation=request)
            self.assertEqual(whole["nodes"],[]);self.assertEqual(whole["nodes_requested"],[2,4])
            seen=[]
            requested=[]
            def request_once(model,normalizer,update,path):
                requested.append(update);return dict(status="requested",node=update)
            first=train(path,root/"resume",protocol,device="cpu",validation=request_once,
                progress=lambda u,l:seen.append(u),stop=lambda:len(seen)>=2)
            self.assertEqual(first["status"],"stopped");self.assertEqual(first["update"],2)
            # Interrupted WAL beyond the durable model cursor must be counted.
            (root/"resume"/"update_wal.json").write_text(json.dumps(dict(update=3,status="pending",attempt_id=1)))
            recovered=train(path,root/"resume",protocol,device="cpu",resume=True,validation=request_once)
            self.assertEqual(requested,[2,4]);self.assertEqual(recovered["nodes"],[])
            self.assertTrue(all(v["status"]=="requested" for v in recovered["node_status"].values()))
            self.assertTrue((root/"resume"/"checkpoint_00002.pt").exists())
            self.assertEqual(recovered["ledger"]["unknown_attempts"],1)
            a=torch.load(whole["final"],weights_only=False);b=torch.load(recovered["final"],weights_only=False)
            for key in a["model"]:torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
            self.assertEqual(a["optimizer"]["param_groups"],b["optimizer"]["param_groups"])
            for key in a["optimizer"]["state"]:
                for field in a["optimizer"]["state"][key]:
                    torch.testing.assert_close(a["optimizer"]["state"][key][field],b["optimizer"]["state"][key][field],rtol=0,atol=0)
            self.assertEqual(a["sampler_rng"],b["sampler_rng"])
            torch.manual_seed(51001); initial=make_model().state_dict()
            self.assertFalse(torch.equal(initial["score_head.weight"],a["model"]["score_head.weight"]))
            self.assertFalse(torch.equal(initial["improvement_head.weight"],a["model"]["improvement_head.weight"]))
            self.assertEqual(a["update"],4);self.assertEqual(len((root/"whole"/"loss.jsonl").read_text().splitlines()),4)
            with self.assertRaises(ValueError):load_orderer(whole["final"],device="cpu")
            loaded,seconds=load_orderer(whole["final"],device="cpu",allow_engineering=True)
            scene,current=scene_layout();pool=[Task("merge",(0,1),1,(1.,)),Task("split",(0,1),3,(.3,.3,.3))]
            budget=Budget(SearchConfig(seconds=10));loaded.set_budget(budget)
            ordered=loaded(scene,current,pool)
            self.assertEqual(sorted(map(id,pool)),sorted(map(id,ordered)));self.assertGreater(budget.geometry_calls,0)
            # Artificial metadata only exercises schema/node-access checks; it
            # is never a claim that this fixture trained 2500/10000 updates.
            artificial=dict(a,engineering=False,update=2500)
            node=root/"artificial_node_metadata.pt";torch.save(artificial,node)
            with self.assertRaises(ValueError):load_orderer(node,device="cpu")
            intermediate,_=load_orderer(node,device="cpu",allow_intermediate=True,
                geometry=GeometryConfig(),variant_count=8)
            self.assertEqual(sorted(map(id,pool)),sorted(map(id,intermediate(scene,current,pool))))
            old_schema=dict(a["feature_schema"],version="scalar48_preference5_v1")
            old=dict(a,feature_schema=old_schema,engineering=False,update=10000)
            torch.save(old,root/"artificial_v1_metadata.pt")
            with self.assertRaises(ValueError):load_orderer(root/"artificial_v1_metadata.pt",device="cpu")
            with self.assertRaises(ValueError):train(path,root/"whole",protocol,device="cpu")

    def test_validation_restores_randomness_even_on_error(self):
        import random
        rng=np.random.default_rng(51001);baseline=np.random.default_rng(51001)
        random.seed(1);np.random.seed(2);torch.manual_seed(3)
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        with self.assertRaises(RuntimeError):
            with isolated_evaluation_rng(rng,"cpu"):
                rng.random(100);random.random();np.random.random(100);torch.rand(100)
                raise RuntimeError("synthetic validation error")
        self.assertEqual(rng.random(),baseline.random());self.assertEqual(random.getstate(),before[0])
        np.testing.assert_array_equal(np.random.get_state()[1],before[1][1])
        torch.testing.assert_close(torch.get_rng_state(),before[2],rtol=0,atol=0)

    def test_expired_ordering_does_not_start_decode(self):
        scene,current=scene_layout();pool=[Task("merge",(0,1),1,(1.,))]
        inputs=pool_features(scene,current,pool)
        ordered=orderer("AI",make_model(),Normalizer.fit(inputs),device="cpu")
        budget=Budget(SearchConfig(seconds=0));ordered.set_budget(budget)
        with self.assertRaises(BudgetStop):ordered(scene,current,pool)
        self.assertEqual(budget.attempts,0)


if __name__=="__main__":unittest.main()
