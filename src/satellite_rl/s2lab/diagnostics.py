"""机制诊断：用包装器改动已训练策略的观测或动作，看模型到底用没用干扰信息、起点选得好不好。

- random_start：模型照常定组、槽数、功率档，只把起点换成同组内随机的合法空闲槽。
- intf_zero：把干扰通道（intf、intf_abs、每束令牌里两列耦合）全部置 0 再给模型。
- intf_shuffle：在每个样本内把干扰通道沿槽方向随机打乱（保留数值分布、打乱位置）。
"""
import torch

from .agents import head_masks


class DiagnosticPolicy:
    off_policy = True                       # 让 run_policy 以 act(obs, explore=False) 调用

    def __init__(self, agent, env, mode, seed=0):
        self.agent, self.env, self.mode = agent, env, mode
        self.gen = torch.Generator(device=env.device)
        self.gen.manual_seed(seed)

    def _edit_obs(self, obs):
        obs = dict(obs)
        if self.mode == 'intf_zero':
            for k in ('intf', 'intf_abs'):
                if k in obs:
                    obs[k] = torch.zeros_like(obs[k])
            if 'beam' in obs:
                beam = obs['beam'].clone()
                beam[..., 8:10] = 0.
                obs['beam'] = beam
        elif self.mode == 'intf_shuffle':
            for k in ('intf', 'intf_abs'):
                if k in obs:
                    x = obs[k]
                    perm = torch.argsort(torch.rand(x.shape[0], x.shape[-1], device=x.device, generator=self.gen), -1)
                    obs[k] = x.gather(-1, perm[:, None, :].expand_as(x))
        return obs

    @torch.no_grad()
    def act(self, obs, explore=False):
        action = self.agent.act(self._edit_obs(obs), explore=False)
        if self.mode == 'random_start':
            env = self.env
            T = env.T
            g = action[:, 0] // T
            mask = head_masks(obs)[0].reshape(env.B, env.G, T)[env.arange_b, g].float()   # 同组合法起点
            mask[mask.sum(1) == 0, 0] = 1.
            s = torch.multinomial(mask, 1, generator=self.gen).squeeze(1)
            action = action.clone()
            action[:, 0] = g * T + s
        return action
