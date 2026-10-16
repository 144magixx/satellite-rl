"""动态ID、冻结区域、缓存计时和PPO终回合更新的小例子。"""
import tempfile
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch as mocked

import numpy as np
import torch

from satellite_rl.coverage.s1fix.data import Frame
from satellite_rl.coverage.s1fix.dynamic import (DynamicEpisode, FullCache, costs, evaluate,
    forecast_peak, heavy_region, inherit_ids, patch, predicted_frame, repair, with_members)
from satellite_rl.coverage.s1fix.geom import Layout, forecast, rays, validate_layout
from satellite_rl.coverage.s1fix.io import save_layout
from satellite_rl.coverage.s1fix.ppo_dyn import (ActorCritic, advantages, load_policy,
    merge_curves, train)


def toy_frames():
    positions = np.array([[45., -10.], [45.01, -10.], [45., 20.], [45.01, 20.]])
    return [Frame("2024-01-08", 1704693600. + 60 * i, i,
                  np.arange(4, dtype=np.uint32), positions + [i * .001, 0.],
                  np.full(4, 400.), np.full(4, 90.)) for i in range(3)]


def toy_layout(frame):
    return with_members(frame.positions[[0, 2]], np.array([0, 0, 1, 1]),
                        np.array([11, 22]), rays(frame.positions))


class DynamicTests(unittest.TestCase):
    def test_identity_matching_and_costs(self):
        frame = toy_frames()[0]
        old = toy_layout(frame)
        swapped = Layout(old.centers_latlon[::-1], old.widths[::-1],
                         1 - old.owner, np.array([0, 1]))
        matched, next_id = inherit_ids(old, swapped, 23)
        self.assertEqual(next_id, 23)
        np.testing.assert_array_equal(matched.beam_ids, [22, 11])
        self.assertEqual(costs(old, frame, matched, frame), (0., 0))

    def test_patch_departures_new_ids_and_forecast_shape(self):
        frames = toy_frames()
        old = toy_layout(frames[0])
        # 两个旧束都已空；预算2必须腾出原位置给两束新飞机。
        frame = Frame("2024-01-08", frames[1].time, 1, np.array([8, 9], np.uint32),
                      np.array([[30., -30.], [65., 40.]]), np.array([np.nan, 400.]),
                      np.array([90., 90.]))
        current, next_id = patch(old, frames[0], frame, N_budget=2, next_id=23)
        self.assertTrue(validate_layout(current, rays(frame.positions), 2))
        np.testing.assert_array_equal(current.beam_ids, [23, 24])
        self.assertEqual(next_id, 25)
        future = predicted_frame(frame)
        self.assertEqual(future.positions.shape, (2, 2))
        np.testing.assert_array_equal(future.positions[0], frame.positions[0])
        np.testing.assert_allclose(future.positions,
            forecast(frame.positions[:, 0], frame.positions[:, 1], frame.gs_kt, frame.track_deg, 60))

    def test_heavy_freezes_distant_beams(self):
        centers = np.array([(lat, lon) for lat in (30., 50., 73.) for lon in (-35., 5., 45.)])
        positions = np.repeat(centers, 2, axis=0)
        frame = Frame("2024-01-08", 1704693600., 0, np.arange(18, dtype=np.uint32),
                      positions, np.zeros(18), np.zeros(18))
        layout = with_members(centers, np.repeat(np.arange(9), 2), np.arange(9), rays(positions))
        active = heavy_region(layout)
        self.assertTrue(np.any(~active))
        result, _ = repair(layout, frame, "heavy", 1, 1., False)
        self.assertTrue(validate_layout(result, rays(positions), 200))
        for j in np.flatnonzero(~active):
            k = np.flatnonzero(result.beam_ids == layout.beam_ids[j])[0]
            np.testing.assert_array_equal(result.centers_latlon[k], layout.centers_latlon[j])
            np.testing.assert_array_equal(result.owner == k, layout.owner == j)
            self.assertEqual(result.widths[k], layout.widths[j])

    def test_forecast_eligibility_and_current_position_fallback(self):
        positions = np.array([[45., -10.], [45., -2.], [45., -6.], [45., -10.01]])
        frame = Frame("2024-01-08", 1704693600., 0, np.arange(4, dtype=np.uint32),
                      positions, np.zeros(4), np.zeros(4))
        layout = with_members(positions[:2], np.array([0, 1, 0, 0]), [5, 6], rays(positions))
        future = replace(frame, positions=np.array([[45., -10.], [45., -2.],
                                                    [45., -3.], [45., 35.]]))
        masks = []

        def capture(*args, **kwargs):
            masks.append(kwargs["eligible_mask"].copy())
            return None

        with mocked("satellite_rl.coverage.s1fix.dynamic.predicted_frame", return_value=future), \
                mocked("satellite_rl.coverage.s1fix.dynamic.assign", side_effect=capture):
            repair(layout, frame, "all", 1, 1., True)
        # 中间飞机现在两束均可，未来仅右束可；最后飞机未来无可用束，退回当前位置。
        np.testing.assert_array_equal(masks[0][2], [False, True])
        np.testing.assert_array_equal(masks[0][3], [True, False])

    def test_forecast_removes_empty_beams_before_opening_new_ones(self):
        # 200个旧束预测时全空；两簇未来飞机应新开两束，不挤同一旧束。
        positions = np.repeat([[45., -30.]], 200, axis=0)
        frame = Frame("2024-01-08", 1704693600., 0, np.arange(200, dtype=np.uint32),
                      positions, np.zeros(200), np.zeros(200))
        layout = Layout(positions.copy(), np.full(200, .3 + 1e-9),
                        np.arange(200), np.arange(200))
        future_positions = np.repeat([[45., 10.], [45., 20.]], 100, axis=0)
        future = replace(frame, positions=future_positions)
        with mocked("satellite_rl.coverage.s1fix.dynamic.predicted_frame", return_value=future):
            self.assertEqual(forecast_peak(layout, frame), 100)

    def test_cached_sequence_rollout_and_tiny_ppo_update(self):
        frames = toy_frames()
        with tempfile.TemporaryDirectory() as directory:
            cache = FullCache(Path(directory) / "B/full_cache")
            for frame in frames:
                save_layout(cache.path(frame), toy_layout(frame), frame, t_total=3.25)
            episode = DynamicEpisode(frames, cache)
            self.assertEqual(episode.observation().shape, (16,))
            self.assertEqual(DynamicEpisode(frames, cache, use_forecast=False).observation().shape, (15,))
            rows = evaluate([frames], cache, "B0")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(3.25 <= row["t_frame"] < 3.35 for row in rows))
            with mocked("satellite_rl.coverage.s1fix.dynamic.patch", return_value=(None, 99)):
                forced = evaluate([frames], cache, "B1")
            self.assertTrue(all(row["forced_full"] == 1 for row in forced))
            path = train([frames], cache, directory, episodes=1, episodes_per_update=1, epochs=1)
            policy, config = load_policy(path)
            self.assertEqual(config["episode"], 1)
            self.assertIn(policy(episode.observation()), range(5))
            merge_curves(directory)
            self.assertTrue((Path(directory) / "B/train_curves.csv").exists())
            trained = ActorCritic()
            trained.load_state_dict(torch.load(path, weights_only=False)["model"])
            torch.manual_seed(51001)
            initial = ActorCritic()
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(trained.parameters(), initial.parameters())))

    def test_gae_terminal_bootstrap(self):
        gae, returns = advantages([1., 2.], [.5, .7], gamma=.9, gae_lambda=.8)
        np.testing.assert_allclose(gae, [1.13 + .72 * 1.3, 1.3], rtol=1e-6)
        np.testing.assert_allclose(returns, gae + [.5, .7], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
