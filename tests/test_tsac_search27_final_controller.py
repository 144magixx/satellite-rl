import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.tsac_search27.final_controller import submit_matrix_once,LATEST_START
from satellite_rl.tsac_search27.known_evaluation import close_final_committed_wal
import torch
import json
from types import SimpleNamespace
from datetime import timedelta
import scripts.tsac_search27.final_controller as driver

class FinalDriverTests(unittest.TestCase):
    def test_resume_requires_pinned_explicit_bounded_authority(self):
        from satellite_rl.t0_width.logging import content_sha
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);state=out/'science.json';state.write_text('{}')
            instant=driver.datetime.fromisoformat('2026-10-15T01:00:00+00:00')
            authority=dict(schema='W27KnownResumeOpsAuthorityV1',root_approved=True,
                root_attested_user_continue=True,experiment_id='S2-W27',science_state_SHA=driver.r.sha(state),
                controller_SHA=driver.r.sha(driver.__file__),evaluation_source_commit='a'*40,
                training_updates=0,allowed_episode_budgets=[560,680],epochs={'20418':'new'},
                user_continue_UTC=instant.isoformat(),latest_matrix_start_UTC=(instant+timedelta(minutes=30)).isoformat(),
                soft_UTC=(instant+timedelta(minutes=110)).isoformat(),hard_UTC=(instant+timedelta(hours=2)).isoformat())
            args=SimpleNamespace(resume_authority_ref='ref',science_state=str(state),evaluation_commit='a'*40)
            with patch.object(driver,'read',return_value={}),patch.object(driver,'read_json_ref',return_value=authority),patch.object(driver,'now',return_value=instant),patch.object(driver,'LATEST_START'),patch.object(driver,'SOFT'),patch.object(driver,'HARD'):
                driver.apply_resume_window(args,{'epochs':{'20418':'new'}})
                self.assertEqual(driver.HARD,instant+timedelta(hours=2))
                authority['root_attested_user_continue']=False
                with self.assertRaises(ValueError):driver.apply_resume_window(args,{'epochs':{'20418':'new'}})
    def test_unknown_post_is_not_answer_and_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            with patch('scripts.tsac_search27.final_controller.now',return_value=LATEST_START.replace(hour=21)),patch('scripts.tsac_search27.final_controller.guard'),patch('scripts.tsac_search27.final_controller.eval_api',side_effect=TimeoutError('unknown')) as post:
                answer=submit_matrix_once('run',{'path':'plan'},out)
                self.assertEqual(answer['stage'],'unknown_observe_only');self.assertFalse((out/'POST_answer.json').exists())
                self.assertEqual(submit_matrix_once('run',{'path':'plan'},out)['stage'],'observe_existing_intent')
                self.assertEqual(post.call_count,1)
    def test_latest_matrix_start_defers_without_post_or_intent(self):
        with tempfile.TemporaryDirectory() as temp,patch('scripts.tsac_search27.final_controller.now',return_value=LATEST_START),patch('scripts.tsac_search27.final_controller.eval_api') as post:
            out=Path(temp);self.assertEqual(submit_matrix_once('r',{},out)['stage'],'deferred_until_explicit_user_resume')
            post.assert_not_called();self.assertFalse((out/'POST_intent.json').exists())
    def test_last_committed_done_wal_closes_without_any_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            wal=Path(temp)/'known_wal.pt';saved=dict(identity={'context':{'task_sha':'task','matrix_cursor':559}},stage='committed',done=True)
            torch.save(saved,wal);self.assertTrue(close_final_committed_wal(wal,'task',560,560));self.assertFalse(wal.exists())
            torch.save(dict(saved,stage='intent'),wal);self.assertFalse(close_final_committed_wal(wal,'task',560,560));self.assertTrue(wal.exists())

if __name__=='__main__':unittest.main()
