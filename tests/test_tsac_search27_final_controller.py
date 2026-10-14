import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.tsac_search27.final_controller import submit_matrix_once,LATEST_START
from satellite_rl.tsac_search27.known_evaluation import close_final_committed_wal
import torch

class FinalDriverTests(unittest.TestCase):
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
