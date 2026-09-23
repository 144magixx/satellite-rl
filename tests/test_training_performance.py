"""GPU 训练线程设置只影响调用期间，CPU 路径与异常退出均不泄漏设置。"""
import unittest
from unittest import mock

import torch

from satellite_rl.training.sac_runner import _with_cuda_cpu_threads


class TrainingThreadTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        torch.set_num_threads(2)

    def test_cuda_call_uses_one_thread_and_restores_caller(self):
        @_with_cuda_cpu_threads
        def observe(value):
            return value, torch.get_num_threads()

        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(observe("result"), ("result", 1))
        self.assertEqual(torch.get_num_threads(), 2)

    def test_cpu_call_keeps_caller_threads(self):
        @_with_cuda_cpu_threads
        def observe():
            return torch.get_num_threads()

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(observe(), 2)
        self.assertEqual(torch.get_num_threads(), 2)

    def test_exception_restores_caller_threads(self):
        @_with_cuda_cpu_threads
        def fail():
            self.assertEqual(torch.get_num_threads(), 1)
            raise RuntimeError("bounded test failure")

        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "bounded test failure"):
                fail()
        self.assertEqual(torch.get_num_threads(), 2)


if __name__ == "__main__":
    unittest.main()
