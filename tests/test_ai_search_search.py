import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.coverage.ai_search.config import GeometryConfig, SearchConfig
from satellite_rl.coverage.ai_search.core import Scene, Layout, validate_layout, metrics, layout_sha256
from satellite_rl.coverage.ai_search.search import (Task, Budget, BudgetStop, decode_task, generate_tasks,
    run_search, apply_delta, layout_delta, add_archive, dominates)


class AI17SearchTest(unittest.TestCase):
    def fixture(self):
        scene = Scene([[50, 10], [50.1, 10], [50.2, 10], [50.3, 10]], [10, 5, 0, 1], ("a", "b", "c", "d"))
        layout = Layout([[50, 10], [50.3, 10]], [.5, .5], [0, 0, 1, 1])
        return scene, layout

    def test_tasks_are_predecode_and_actual_joint_N_width_change(self):
        scene, parent = self.fixture()
        cfg = SearchConfig(seconds=3, max_attempts=128, variants=4, center_iterations=1)
        tasks = list(generate_tasks(scene, parent, cfg))
        self.assertTrue(any(t.target_k == 1 and t.widths == (1.,) for t in tasks))
        self.assertTrue(any(len(set(t.widths)) > 1 for t in tasks))
        self.assertTrue(all("metrics" not in t.to_record() and "decoded_layout" not in t.cheap_features for t in tasks))
        merged = decode_task(scene, parent, Task("merge", (0, 1), 1, (1.,)), config=cfg)
        self.assertIsNotNone(merged.layout); self.assertEqual(len(merged.layout.centers), 1)
        self.assertEqual(merged.layout.widths[0], 1.)
        self.assertEqual(len(merged.layout.assignment), 4)
        self.assertTrue(validate_layout(scene, merged.layout)["valid"])
        split = decode_task(scene, parent, Task("split", (0, 1), 3, (.3, .5, 1.)), config=cfg)
        self.assertIsNotNone(split.layout); self.assertEqual(len(split.layout.centers), 3)
        self.assertEqual(len(set(split.layout.widths)), 3)
        self.assertTrue(validate_layout(scene, split.layout)["valid"])

    def test_delta_replay_variable_and_sameN(self):
        scene, parent = self.fixture()
        cfg = SearchConfig(seconds=3, variants=2, center_iterations=0)
        for target, widths in ((1, (1.,)), (2, (.3, 1.))):
            result = decode_task(scene, parent, Task("joint_reconstruct", (0, 1), target, widths), config=cfg)
            self.assertIsNotNone(result.layout)
            delta = layout_delta(parent, result.layout)
            self.assertEqual(layout_sha256(apply_delta(parent, delta)), layout_sha256(result.layout))
            with self.assertRaises(ValueError): apply_delta(Layout(parent.centers, [.3, .5], parent.assignment), delta)

    def test_budget_geometry_calls_and_pairs_are_separate(self):
        cfg = SearchConfig(seconds=3, max_geometry_calls=2)
        budget = Budget(cfg)
        budget.geometry(60000); budget.geometry(60000)
        self.assertEqual(budget.geometry_calls, 2); self.assertEqual(budget.ray_pair_checks, 120000)
        with self.assertRaises(BudgetStop): budget.geometry(1)

    def test_final_candidate_budget_and_task_budget(self):
        scene, parent = self.fixture()
        cfg = SearchConfig(seconds=3, max_attempts=1, task_candidate_limit=64)
        budget = Budget(cfg)
        result = decode_task(scene, parent, Task("merge", (0, 1), 1, (1.,)), budget, config=cfg)
        self.assertEqual(budget.attempts, 1); self.assertEqual(len(result.candidates), 1)
        self.assertIsNotNone(result.layout)

    def test_late_candidate_not_published(self):
        scene, parent = self.fixture()
        cfg = SearchConfig(seconds=1, variants=1, center_iterations=0)
        clock = [0.]
        budget = Budget(cfg, clock=lambda: clock[0])
        original = validate_layout
        def slow(*args, **kwargs):
            value = original(*args, **kwargs)
            clock[0] = 2.
            return value
        with patch("satellite_rl.coverage.ai_search.search.validate_layout", side_effect=slow):
            result = decode_task(scene, parent, Task("merge", (0, 1), 1, (1.,)), budget, config=cfg)
        self.assertIsNone(result.layout); self.assertTrue(result.candidates[0]["late"])
        self.assertEqual(budget.late_candidates, 1)

    def test_orderer_strict_permutation_and_budget_hook(self):
        scene, parent = self.fixture()
        class Order:
            def set_budget(self, value): self.budget=value
            def set_deadline(self, value): self.deadline=value
            def set_preference(self, value): self.preference=value
            def __call__(self, scene, current, pool):
                self.budget.geometry(5)
                return list(reversed(pool))
        order = Order()
        result = run_search(scene, parent, SearchConfig(seconds=3, max_attempts=12), orderer=order)
        self.assertGreater(result.budgets["geometry_calls"], 0)
        self.assertTrue(hasattr(order, "deadline")); self.assertTrue(hasattr(order, "preference"))
        with self.assertRaises(ValueError):
            run_search(scene, parent, SearchConfig(seconds=3), orderer=lambda s,c,p: p[:-1])

    def test_stop_before_init_and_empty_scene(self):
        result = run_search(self.fixture()[0], stop=lambda: True)
        self.assertIsNone(result.layout); self.assertEqual(result.budgets["stop_reason"], "stop_requested")
        result = run_search(Scene([], []))
        self.assertEqual(result.metrics["N"], 0); self.assertEqual(result.budgets["quality_scope"], "geometry_only")

    def test_init_common_shrink_and_raw_slice(self):
        scene, _ = self.fixture()
        result = run_search(scene, config=SearchConfig(seconds=3), method="INIT")
        self.assertEqual(result.budgets["tasks_evaluated"], 0)
        self.assertIsNotNone(result.slice_reference)
        self.assertTrue(all(w == .5 for w in result.slice_reference["layout"].widths))
        self.assertTrue(all(validate_layout(scene, r["layout"])["valid"] for r in result.initial_archive))
        self.assertTrue(all("unshrunk_layout" in r for r in result.journal if r.get("phase") == "initial" and "layout" in r))

    def test_pareto_preserves_five_extremes(self):
        scene, layout = self.fixture()
        cfg = SearchConfig(archive_limit=5)
        rows=[]
        for i in range(15):
            value=dict(N=i+1, solid_angle_sum_sr=15-i, maxload=abs(i-4)+1, top10_mean=abs(i-8)+1, CV=abs(i-12)+1)
            row=dict(layout=layout, metrics=value, sha256=f"{i:064x}")
            rows,_=add_archive(rows,row,cfg)
        self.assertLessEqual(len(rows),5)
        self.assertIn("0"*64, [r["sha256"] for r in rows])
        self.assertFalse(dominates(dict(N=2,solid_angle_sum_sr=2,maxload=0,top10_mean=0,CV=None),
                                   dict(N=1,solid_angle_sum_sr=1,maxload=0,top10_mean=0,CV=None),cfg.pareto_fields))

    def test_nested_archive_keeps_early_point_dominated_after_checkpoint(self):
        scene, parent = self.fixture()
        cfg = SearchConfig(seconds=3, nested_seconds=.8, variants=1, center_iterations=0, pool_size=2, max_restarts=1)
        clock=[0.]; calls=[0]; original=validate_layout
        def validate(*args,**kwargs):
            result=original(*args,**kwargs); calls[0]+=1
            if calls[0] == 2: clock[0]=.7
            if calls[0] == 3: clock[0]=1.2
            return result
        def tasks(*args,**kwargs):
            yield Task("merge",(0,1),1,(1.,))
            yield Task("merge",(0,1),1,(.3,))
        with patch("satellite_rl.coverage.ai_search.search.perf_counter",side_effect=lambda:clock[0]), \
             patch("satellite_rl.coverage.ai_search.search.validate_layout",side_effect=validate), \
             patch("satellite_rl.coverage.ai_search.search.generate_tasks",side_effect=tasks):
            result=run_search(scene,parent,cfg)
        self.assertTrue(any(len(r["layout"].centers)==1 and r["layout"].widths[0]==1. for r in result.nested_reference))
        self.assertFalse(any(len(r["layout"].centers)==1 and r["layout"].widths[0]==1. for r in result.pareto))
        self.assertTrue(all(r["completed_seconds"]<=.8 for r in result.nested_reference))
        node=next(r for r in result.reference_nodes if r["seconds_limit"]==.8)
        self.assertTrue(any(len(r["layout"].centers)==1 and r["layout"].widths[0]==1. for r in node["archive"]))

    def test_ordering_timeout_never_starts_decoder(self):
        scene,parent=self.fixture(); clock=[0.]
        def expired(s,c,p): clock[0]=2.; return p
        with patch("satellite_rl.coverage.ai_search.search.perf_counter",side_effect=lambda:clock[0]), \
             patch("satellite_rl.coverage.ai_search.search.decode_task") as decode:
            result=run_search(scene,parent,SearchConfig(seconds=1),orderer=expired)
        decode.assert_not_called()
        self.assertEqual(result.budgets["stop_reason"],"time_budget")

    def test_region_overflow_is_censored_not_plane_subset(self):
        scene,parent=self.fixture(); cfg=SearchConfig(max_region_users=2)
        task=next(generate_tasks(scene,parent,cfg))
        self.assertTrue(task.cheap_features["region_overflow"])
        result=decode_task(scene,parent,task,config=cfg)
        self.assertEqual(result.status,"censored")
        self.assertEqual(result.details["fixed_excess_users"],0)

    def test_scale_aware_compare_and_serialize_restart_evidence(self):
        from satellite_rl.coverage.ai_search.search import compare_metrics,metric_tolerance
        scene,parent=self.fixture(); value=metrics(scene,parent)
        noise=dict(value,maxload=value["maxload"]+1e-7)
        self.assertEqual(compare_metrics(noise,value,"maxload"),0)
        change=dict(value,maxload=value["maxload"]+1.)
        self.assertEqual(compare_metrics(change,value,"maxload"),1)
        self.assertEqual(metric_tolerance("N",10.,10.),0.)
        cfg=SearchConfig(seconds=3,max_attempts=8)
        a=run_search(scene,parent,cfg).to_record(); b=run_search(scene,Layout.from_record(parent.to_record()),cfg).to_record()
        # Complete deterministic bounded reruns for engineering only. Formal
        # partial worker recovery is fail-closed when spent budget is unknown.
        self.assertEqual(a["layout"],b["layout"])
        parents={layout_sha256(parent):parent}
        for row in a["journal"]:
            if row.get("phase") != "task":continue
            origin=parents[row["current_sha256"]]
            for trial in row["candidates"]:
                self.assertNotIn("layout",trial)
                rebuilt=apply_delta(origin,trial["delta"])
                self.assertTrue(validate_layout(scene,rebuilt)["valid"])
                parents[layout_sha256(rebuilt)]=rebuilt

    def test_post_shrink_has_reserved_real_window_and_no_N_change(self):
        from satellite_rl.coverage.geoload.classical import SearchResult as OldResult
        from satellite_rl.coverage.geoload.core import Layout as OldLayout
        scene,parent=self.fixture(); passed=[]
        def fixed_search(oldscene,initial,method,config):
            passed.append(config)
            return OldResult(OldLayout(initial.centers,initial.assignment),{},[],
                dict(attempts=0,geometry_calls=0))
        cfg=SearchConfig(seconds=3,post_shrink_reserve_seconds=1.)
        with patch("satellite_rl.coverage.geoload.classical.run_classical",side_effect=fixed_search):
            result=run_search(scene,config=cfg,method="POST_SHRINK")
        self.assertLessEqual(passed[0].seconds,2.)
        self.assertEqual(result.metrics["N"],result.slice_reference["metrics"]["N"])
        self.assertTrue(all(w<=.5 for w in result.layout.widths))
        self.assertLess(result.metrics["solid_angle_sum_sr"],result.slice_reference["metrics"]["solid_angle_sum_sr"])
        self.assertFalse(result.budgets["nested_trace_complete"])
        self.assertFalse(result.reference_nodes[0]["trace_complete"])
        self.assertTrue(all(r["sha256"] in {x["sha256"] for x in result.initial_archive} for r in result.nested_reference))

    def test_teacher_INIT_uses_shared_master_ledger(self):
        scene,_=self.fixture(); cfg=SearchConfig(seconds=6,max_attempts=3072)
        clock=[0.]; ledger=Budget(cfg,clock=lambda:clock[0]); clock[0]=.1
        result=run_search(scene,config=cfg,method="INIT",budget=ledger)
        self.assertGreater(ledger.geometry_calls,0); self.assertEqual(ledger.attempts,len(result.initial_archive))
        self.assertEqual(ledger.deadline,6.); self.assertEqual(ledger.attempt_limit,3072)
        self.assertEqual(result.budgets["seconds"],.1)


if __name__ == "__main__": unittest.main()
