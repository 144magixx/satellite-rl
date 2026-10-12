import unittest,tempfile,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from satellite_rl.coverage.ai_search.core import Scene,Layout,validate_layout,layout_sha256,metrics
from satellite_rl.coverage.ai_search.config import GeometryConfig
from satellite_rl.coverage.geometry import ray_angle_numpy
from satellite_rl.coverage.ai_search_r2.config import Ledger,TaskConfig,BudgetStop
from satellite_rl.coverage.ai_search_r2.validation import validate_owned
from satellite_rl.coverage.ai_search_r2.initials import freeze_common,CommonStart,digest,file_sha
from satellite_rl.coverage.ai_search_r2.tasks import make_pool,Task
from satellite_rl.coverage.ai_search_r2.decoder import solve_task
from satellite_rl.coverage.ai_search_r2.probe import evaluate_closed_pool,target_hit,prepare_pool,mechanism

def common_fixture(scene,parent):
    ev=validate_owned(scene,parent);assert ev['valid'];m=metrics(scene,parent,evidence=ev);row=dict(layout=parent.to_record(),metrics=m,sha256=layout_sha256(parent),sources=[])
    return CommonStart.from_record(dict(input_sha=scene.input_sha256,archive=[row],archive_sha=digest([row]),parent_sha=row['sha256'],primary_parent=row,slice_ref=row,slice_content_sha=digest(row),acquisition_cost_refs=[]))

def tiny():
    s=Scene([[50+i*.02,10] for i in range(8)],[12,2,3,4,2,1,1,0])
    p=Layout([[50+i*.04,10] for i in range(4)],[.5]*4,[0,0,1,1,2,2,3,3])
    return s,p,common_fixture(s,p)

class R2CoreTests(unittest.TestCase):
    def test_owned_equivalent_valid_invalid_empty_zero(self):
        s,p,c=tiny()
        layouts=[p,Layout(p.centers,p.widths,[-1]+p.assignment[1:].tolist()),Layout(p.centers,[.2]*4,p.assignment),Layout(p.centers,p.widths,[0]*8),Layout([[0,0]],[.5],[0]*8)]
        for layout in layouts:self.assertEqual(validate_layout(s,layout),validate_owned(s,layout))
        for s,p in [(Scene([],[]),Layout([],[],[])),(Scene([[50,10]],[0]),Layout([[50,10]],[.3],[0]))]:self.assertEqual(validate_layout(s,p),validate_owned(s,p))
    def test_boundary_exact_owner_logic(self):
        s=Scene([[50,10],[51,10]],[1,2]);angle=float(ray_angle_numpy(s.rays[0],s.rays[1]));g=GeometryConfig(width_min_deg=.0001,width_max_deg=2.,width_menu_deg=(.0001,.5,2.))
        for width in (2*angle-1e-7,2*angle,2*angle+1e-7):
            p=Layout([[50,10]],[width],[0,0]);self.assertEqual(validate_layout(s,p,g),validate_owned(s,p,g))
    def test_charge_before_exact_boundary(self):
        l=Ledger(TaskConfig(seconds=10,ray_pair_limit=4,geometry_limit=2,candidate_limit=1));l.geometry(4);l.geometry(0);l.attempt()
        with self.assertRaises(BudgetStop):l.geometry(0)
        self.assertEqual((l.ray_pair_checks,l.geometry_calls,l.attempts),(4,2,1))
        with self.assertRaises(BudgetStop):l.attempt()
        l=Ledger(TaskConfig(ray_pair_limit=3))
        with self.assertRaises(BudgetStop):l.geometry(4)
        self.assertEqual(l.geometry_calls,0);self.assertEqual(l.ray_pair_checks,0)
    def test_geometry_actual_n_not_nN(self):
        s,p,c=tiny();l=Ledger();self.assertTrue(validate_owned(s,p,budget=l)['valid']);self.assertEqual(l.ray_pair_checks,8)
    def test_common_frozen_bytes_and_task_roundtrip(self):
        s,p,c=tiny();original=c.frozen_bytes;x=c.to_record();x['archive'][0]['layout']['widths_deg'][0]=1.
        self.assertEqual(c.frozen_bytes,original)
        with self.assertRaises(ValueError):CommonStart.from_record(x)
        pool=make_pool(s,c);self.assertEqual(len(pool),8);self.assertEqual(len({t.task_id for t in pool}),8)
        for task in pool:self.assertEqual(Task.from_record(task.to_record()),task)
    def test_signed_five_source_freeze_and_bad_pin(self):
        s,p,c=tiny()
        with tempfile.TemporaryDirectory() as directory:
            refs=[]
            for method in ('07-CV','07-LEX','JOINT-CV','JOINT-LEX','POST_SHRINK'):
                path=Path(directory)/(method+'.json');a=Path(directory)/(method+'_audit.json')
                raw=dict(method=method,input_sha256=s.input_sha256,result=dict(layout=p.to_record(),slice_reference=c.slice_ref if method=='POST_SHRINK' else None))
                path.write_text(json.dumps(raw));sh=file_sha(path);a.write_text(json.dumps(dict(passed=True,issues=[],source_case_sha256=sh)))
                refs.append(dict(method=method,path=str(path),sha256=sh,audit_path=str(a),audit_sha256=file_sha(a)))
            frozen=freeze_common(s,refs);self.assertEqual(len(frozen.archive),5);self.assertEqual(frozen.parent_sha,c.parent_sha)
            refs[0]['sha256']='0'*64
            with self.assertRaises(ValueError):freeze_common(s,refs)
    def test_fixed_users_full_geometry_and_deleted_unsupported(self):
        s,p,c=tiny();pool=make_pool(s,c,TaskConfig(active_limit=1));t=pool[0];l=Ledger(TaskConfig(active_limit=1,seconds=30))
        r=solve_task(s,c,t,'D1',l)
        for trial in r['candidates']:
            q=Layout.from_record(trial['layout']);self.assertEqual(q.assignment[list(t.fixed_users)].tolist(),p.assignment[list(t.fixed_users)].tolist());self.assertTrue(validate_owned(s,q)['valid'])
        merge=pool[2];r=solve_task(s,c,merge,'D1',Ledger(TaskConfig(active_limit=1)))
        self.assertTrue(r['reason'].startswith('not_supported'));self.assertEqual(r['budget']['attempts'],0)
    def test_d0_d1_valid_n_and_width_candidates_budget(self):
        s,p,c=tiny();pool=make_pool(s,c);seen_N=False;seen_width=False
        for v in ('D0','D1'):
            for t in pool:
                r=solve_task(s,c,t,v,Ledger(TaskConfig(seconds=30)))
                self.assertLessEqual(r['budget']['attempts'],32);self.assertLessEqual(r['budget']['ray_pair_checks'],262144)
                if v=='D1':self.assertLessEqual(len(r['candidates']),8)
                for trial in r['candidates']:
                    q=Layout.from_record(trial['layout']);self.assertEqual(layout_sha256(q),trial['sha256'])
                    if trial['metrics']:
                        self.assertTrue(validate_layout(s,q)['valid']);seen_N|=len(q.centers)!=len(p.centers);seen_width|=any(abs(w-.5)>1e-10 for w in q.widths)
        self.assertTrue(seen_N);self.assertTrue(seen_width)
    def test_pool_no_parent_mutation_and_real_cost(self):
        s,p,c=tiny();before=c.frozen_bytes;r=evaluate_closed_pool(s,c,'D1',TaskConfig(seconds=30))
        self.assertEqual(len(r['results']),8);self.assertEqual(c.frozen_bytes,before);self.assertGreater(r['actual_cost']['ray_pair_checks'],0);self.assertEqual(r['training_updates'],0)
        self.assertEqual(r['actual_cost']['ray_pair_checks'],r['preparation_cost']['ray_pair_checks']+sum(t['budget']['ray_pair_checks'] for t in r['results']))
    def test_zero_request_not_quality_hit_and_stop(self):
        s,p,c=tiny();m=dict(c.record['primary_parent']['metrics']);m['CV']=None
        self.assertFalse(target_hit(m,m,m));r=evaluate_closed_pool(s,c,'D1',stop=lambda:True)
        self.assertFalse(r['closed'])
    def test_nonfinite_and_assignment_duplicate_input(self):
        with self.assertRaises(ValueError):Scene([[50,10],[50,10]],[1,2],('a','a'))
        with self.assertRaises(ValueError):Layout([[50,10]],[.5],[.1])
    def test_shared_preparation_and_seed_descriptor(self):
        s,p,c=tiny();prep=prepare_pool(s,c)
        for t in prep['tasks']:self.assertEqual(len(t['center_seed_sets']),8)
        first=evaluate_closed_pool(s,c,'D0',prepared_pool=prep);second=evaluate_closed_pool(s,c,'D1',prepared_pool=prep)
        self.assertEqual(first['pool_sha'],second['pool_sha']);self.assertEqual(first['tasks'],second['tasks'])
        for r in (first,second):
            self.assertFalse(r['preparation_executed_here'])
            self.assertEqual(r['actual_cost']['ray_pair_checks']-r['actual_compute_cost']['ray_pair_checks'],prep['preparation_cost']['ray_pair_checks'])
    def test_active_quota_and_mandatory_seed_and_overflow(self):
        positions=[[50+i*.0001,10] for i in range(160)];s=Scene(positions,[100]*80+[1]*80)
        p=Layout([[50,10],[50.01,10]],[.5,.5],[0]*80+[1]*80);c=common_fixture(s,p);pool=make_pool(s,c)
        reassign=pool[0];self.assertEqual(len(reassign.active_users),64)
        hot=sorted(range(80),key=lambda u:(-s.demands[u],s.aircraft_ids[u]))[:32]
        boundary=sorted((u for u in range(160) if u not in hot),key=lambda u:(reassign.features['parent_margin_by_user'][str(u)],s.aircraft_ids[u]))[:32]
        self.assertEqual(set(reassign.active_users),set(hot+boundary))
        merge=pool[2];self.assertEqual(merge.status,'unsupported_active_set');r=solve_task(s,c,merge,'D1',Ledger());self.assertEqual(r['status'],'terminal_unsupported_active_set');self.assertEqual(r['predecode_proof']['mandatory_count'],80)
        split=pool[3];seed_id=split.features['active_mandatory_aircraft_ids'][0];self.assertIn(seed_id,[s.aircraft_ids[u] for u in split.active_users])
    def test_late_metric_completion_cannot_publish(self):
        from unittest.mock import patch
        import satellite_rl.coverage.ai_search_r2.decoder as module
        s,p,c=tiny();task=make_pool(s,c)[0];clock=[0.];l=Ledger(TaskConfig(seconds=3.),clock=lambda:clock[0]);original=module.metrics
        def slow(*args,**kwargs):
            value=original(*args,**kwargs);clock[0]=4.;return value
        with patch.object(module,'metrics',slow):r=solve_task(s,c,task,'D1',l)
        self.assertTrue(r['candidates']);self.assertTrue(all(t['late'] for t in r['candidates']));self.assertFalse(r['closed']);self.assertGreater(r['budget']['overrun_seconds'],0)
    def test_delta_prefix_and_geometry_event_replay(self):
        from satellite_rl.coverage.ai_search.search import apply_delta
        s,p,c=tiny();r=solve_task(s,c,make_pool(s,c)[0],'D1',Ledger(TaskConfig(seconds=30)))
        for t in r['candidates']:
            self.assertEqual(layout_sha256(apply_delta(p,t['delta'])),t['sha256']);prefix=t['cost_prefix'];events=r['budget']['charge_events'][:prefix['event_count']]
            self.assertEqual(sum(e['pairs'] for e in events),prefix['ray_pair_checks']);self.assertEqual(len(events),prefix['geometry_calls']);self.assertLessEqual(prefix['seconds'],t['completed_seconds']);self.assertTrue(all(e['charge_before'] for e in events))
    def test_fixed_width_N_change_not_retained_width_change(self):
        p=Layout([[50,10],[50.01,10]],[.5,.5],[0,1],('a','b'));q=Layout([[50,10]],[.5],[0,0],('a',));m=mechanism(p,q,'menu');self.assertEqual(m['N_delta'],-1);self.assertEqual(m['retained_width_changed'],[]);self.assertEqual(m['deleted_beam_ids'],['b'])
    def test_empty_and_zero_pool_preserve_geometry_scope(self):
        for s,p in [(Scene([],[]),Layout([],[],[])),(Scene([[50,10]],[0]),Layout([[50,10]],[.5],[0]))]:
            c=common_fixture(s,p);r=evaluate_closed_pool(s,c,'D1');self.assertEqual(len(r['tasks']),8);self.assertFalse(any(t['first_hit'] for t in r['results']))
    def test_first_target_prefix_includes_preparation(self):
        s,p,c=tiny();r=evaluate_closed_pool(s,c,'D1');hits=[t['first_hit'] for t in r['results'] if t['first_hit']]
        self.assertTrue(hits)
        for hit in hits:
            for key in ('seconds','ray_pair_checks','geometry_calls','attempts'):self.assertEqual(hit['with_common_preparation'][key],hit['cost_prefix'][key]+r['preparation_cost'][key])
            self.assertGreater(hit['cost_prefix']['ray_pair_checks'],0)
    def test_split_at_actual_200_cap_has_closed_proof(self):
        s=Scene([[50,10]]*200,[1]*200);p=Layout([[50,10]]*200,[.5]*200,list(range(200)));c=common_fixture(s,p);task=make_pool(s,c)[3];r=solve_task(s,c,task,'D1',Ledger())
        self.assertEqual(r['status'],'terminal_structurally_unsupported');self.assertTrue(r['closed']);self.assertEqual(r['predecode_proof']['result_N'],201);self.assertEqual(r['budget']['ray_pair_checks'],0)
    def test_more_than64_nondominated_points_not_crowded_away(self):
        from satellite_rl.coverage.ai_search_r2.probe import archive_config
        from satellite_rl.coverage.ai_search.search import add_archive
        cfg=archive_config(TaskConfig(),5);self.assertGreaterEqual(cfg.archive_limit,261);archive=[]
        for i in range(1,101):
            row=dict(sha256=str(i).zfill(64),metrics=dict(N=i,solid_angle_sum_sr=float(201-i),maxload=1.,top10_mean=1.,CV=1.))
            archive,retained=add_archive(archive,row,cfg);self.assertTrue(retained)
        self.assertEqual(len(archive),100)
    def test_parent_metrics_and_slice_tampering_rejected(self):
        s,p,c=tiny();record=c.to_record();record['primary_parent']['metrics']['maxload']-=1.
        with self.assertRaises(ValueError):CommonStart.from_record(record)
        record=c.to_record();record['primary_parent']['metrics']['evidence']['valid']=False
        with self.assertRaises(ValueError):CommonStart.from_record(record)
        record=c.to_record();record['slice_ref']['layout']['widths_deg'][0]=.3
        with self.assertRaises(ValueError):CommonStart.from_record(record)
        record=c.to_record();record['slice_ref']['metrics']['N']=1
        with self.assertRaises(ValueError):CommonStart.from_record(record)
    def test_provisional_assignment_sum_squares_not_final_tail_rank(self):
        from satellite_rl.coverage.ai_search_r2.decoder import assignment_key
        first=np.array([10,9,5,5,1]+[0]*15,dtype=float);second=np.array([10,8,8,2,2]+[0]*15,dtype=float)
        self.assertEqual(first.sum(),second.sum());self.assertEqual(first.max(),second.max())
        self.assertLess(assignment_key(first,'a'),assignment_key(second,'b'))
        # Final peak/tail would instead prefer second's top-two mean 9 over 9.5.
        self.assertLess(np.sort(second)[-2:].mean(),np.sort(first)[-2:].mean())
        self.assertLess(assignment_key(first,'a'),assignment_key(first,'b'))
    def test_incomplete_preparation_both_decoders_preserve_cost_no_solve(self):
        from unittest.mock import patch
        s,p,c=tiny();config=TaskConfig(ray_pair_limit=1);prepared=prepare_pool(s,c,config)
        self.assertEqual(prepared['status'],'incomplete_preparation');self.assertGreater(prepared['preparation_cost']['seconds'],0)
        self.assertGreater(prepared['preparation_cost']['ray_pair_checks'],0)
        with patch('satellite_rl.coverage.ai_search_r2.probe.solve_task',side_effect=AssertionError('must not decode')):
            rows=[evaluate_closed_pool(s,c,v,config,prepared_pool=prepared) for v in ('D0','D1')]
        for row in rows:
            self.assertFalse(row['closed']);self.assertEqual(row['preparation_cost'],prepared['preparation_cost']);self.assertEqual(row['actual_cost']['ray_pair_checks'],prepared['preparation_cost']['ray_pair_checks']);self.assertEqual(row['results'],[])
        stopped=prepare_pool(s,c,stop=lambda:True);self.assertEqual(stopped['reason'],'stop_requested')
        ticks=[0.]
        def clock():ticks[0]+=.1;return ticks[0]
        expired=prepare_pool(s,c,clock=clock);self.assertEqual(expired['status'],'incomplete_preparation');self.assertEqual(expired['reason'],'time_budget');self.assertGreater(expired['preparation_cost']['ray_pair_checks'],0);self.assertGreaterEqual(expired['preparation_cost']['seconds'],3.)
    def test_envelope_exact_R1_and_visibility_work(self):
        from satellite_rl.coverage.ai_search_r2.envelope import minimum_enclosing_width as new
        from satellite_rl.coverage.ai_search.core import minimum_enclosing_width as old
        s,p,c=tiny()
        for members,center in [([0,1],p.centers[0]),([0,7],p.centers[0]),([0],[0,0]),([],p.centers[0])]:
            ledger=Ledger();self.assertEqual(new(s,members,center,budget=ledger),old(s,members,center))
            expected_pairs=len(members) if members and center is not None and np.asarray(center).tolist()!=[0,0] else 0
            self.assertEqual(ledger.ray_pair_checks,expected_pairs)
            self.assertEqual(ledger.geometry_calls,2 if expected_pairs else 1 if members else 0)
        invalid=Layout([[0,0]],[.5],[0]*8);ledger=Ledger();self.assertFalse(validate_owned(s,invalid,budget=ledger)['valid']);self.assertEqual((ledger.geometry_calls,ledger.ray_pair_checks),(1,0))
    def test_visibility_angle_budget_reject_before_query(self):
        from unittest.mock import patch
        import satellite_rl.coverage.ai_search_r2.envelope as envelope
        s,p,c=tiny()
        with patch('satellite_rl.coverage.ai_search_r2.validation.visible_numpy',side_effect=AssertionError('query executed before charge')):
            with self.assertRaises(BudgetStop):validate_owned(s,p,budget=Ledger(TaskConfig(geometry_limit=0)))
        with patch.object(envelope,'visible_numpy',side_effect=AssertionError('query executed before charge')):
            with self.assertRaises(BudgetStop):envelope.minimum_enclosing_width(s,[0],p.centers[0],budget=Ledger(TaskConfig(geometry_limit=0)))
        ledger=Ledger(TaskConfig(ray_pair_limit=0))
        with patch.object(envelope,'ray_angle_numpy',side_effect=AssertionError('angle query executed before charge')):
            with self.assertRaises(BudgetStop):envelope.minimum_enclosing_width(s,[0],p.centers[0],budget=ledger)
        self.assertEqual((ledger.geometry_calls,ledger.ray_pair_checks),(1,0))
    def test_late_same_sha_not_marked_as_selected_candidate(self):
        from unittest.mock import patch
        s,p,c=tiny()
        def late_row(scene,common,task,variant,ledger):
            parent=common.record['primary_parent'];trial=dict(layout=p.to_record(),sha256=parent['sha256'],metrics=parent['metrics'],late=True,completed_seconds=4.,attempt=1,cost_prefix=dict(seconds=4.,attempts=1,ray_pair_checks=8,geometry_calls=2,event_count=2),shrink=False)
            return dict(task=task.to_record(),reason='time_budget',closed=False,candidates=[trial],budget=dict(seconds=4.,attempts=1,ray_pair_checks=8,geometry_calls=2))
        with patch('satellite_rl.coverage.ai_search_r2.probe.solve_task',side_effect=late_row):r=evaluate_closed_pool(s,c,'D1')
        for result in r['results']:
            trial=result['candidates'][0];self.assertTrue(trial['layout_in_final_archive']);self.assertFalse(trial['archive_retained_final']);self.assertFalse(trial['primary_slice_selected']);self.assertFalse(trial['final_published']);self.assertIsNone(result['first_hit'])

if __name__=='__main__':unittest.main()
