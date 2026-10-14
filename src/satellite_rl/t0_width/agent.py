"""W26 input-only adapters; SAC mathematics remains WorkbenchAgent's T0 path."""
from copy import deepcopy
import hashlib
import json
import numpy as np

import torch
from torch import nn

from satellite_rl.agents.workbench_algorithms import WorkbenchAgent
from satellite_rl.workbench.config import default_config
from .schema import GROUPS, observation_schema, state_width



def state_dimension(group):
    return state_width(group)


def build_config(group, device="cpu", phase="scientific"):
    if group not in GROUPS or phase not in ("scientific", "engineering", "correctness"):
        raise ValueError("Unknown W26 group/phase")
    c = default_config()
    c.update(name="S2-W26 " + group, algorithm="sac", group=group, phase=phase)
    c["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                         embedding_dim=128, num_heads=8, num_layers=2, feedforward_dim=512,
                         activation="relu", dropout=0.0, sac_policy_structure="factorized",
                         sac_critic_structure="additive", temporal_mode="none", graph_mode="none",
                         context_beam_count="fixed_100", readout_mode="first_token",
                         beam_count_conditioning="none")
    c["state"].update(features=["beam", "power", "interference", "occupancy", "natural_global_context"],
                       demand_visibility="known", context_level="compact")
    c["reward"].update(preset="legacy_repaired")
    c["environment"].update(profile="europe_v1", beam_count_max=200, power_budget_w=3000,
                             action_mask="joint_feasible", power_levels_w=list(range(5, 51, 5)))
    base_seed = 73026 if phase == "engineering" else 51001
    c["training"].update(seed=base_seed, scene_order="shuffle_each_cycle", scene_order_seed=base_seed,
                          total_steps=150000 if phase == "scientific" else 2048,
                          initial_alpha=.01, fixed_alpha=.01, alpha_mode="fixed",
                          gamma=.99, tau=.005, batch_size=256, replay_capacity=20000,
                          warmup_steps=2000, actor_lr=1e-5, critic_lr=1e-4,
                          update_every=1, gradient_steps=1, gradient_clip=10,
                          checkpoint_interval=10000, device=device)
    c["evaluation"].update(validation_count=20, test_count=40, action_mode="stochastic", repeats=3,
                           interval_steps=10000, extra_steps=[0, 2000, 5000],
                           validation_seed=274000, test_seed=275000)
    return c


def schema_identity(group):
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
    width, relation = GROUPS[group]
    value = dict(observation=observation_schema(group), group=group, dimension=state_dimension(group),
                 fields=["T0_compact_207"] + (["full_width_deg/1deg"] if width else [])
                 + (["log1p_out_noise_ratio_100", "log1p_future_in_noise_ratio_100"] if relation else []),
                 tokens=14, readout="first_geo", action_dims=[100, 10, 10],
                 physics_SHA=PHYSICS_SHA, reward="legacy_repaired_T0_threshold_50Mbps_v1")
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


class WidthEncoder(nn.Module):
    """Only selected fields have projection columns; common weights are copied exactly."""
    def __init__(self, base, group, seed):
        super().__init__()
        self.group = group
        self.width_visible, self.relation_visible = GROUPS[group]
        self.state_dim = state_dimension(group)
        self.output_dim = base.output_dim
        # Preserve registration/gradient-reduction order as well as common numerical values.
        for name in ("geo_embed", "rate_embed", "power_embed", "slot_embed", "net", "global_embed"):
            setattr(self, name, deepcopy(getattr(base, name)))
        self.pos_embed = nn.Parameter(base.pos_embed.detach().clone())
        self.extra_pos_embed = nn.Parameter(base.extra_pos_embed.detach().clone())
        # fork_rng ensures input-column additions do not shift common initialization or sampling RNG.
        with torch.random.fork_rng(devices=[]):
            if self.width_visible:
                torch.manual_seed(seed + 17011)
                self.geo_embed = self._extend(self.geo_embed, 3)
            if self.relation_visible:
                torch.manual_seed(seed + 29009)
                self.slot_embed = self._extend(self.slot_embed, 40)
        self.last_token_count = 14

    @staticmethod
    def _extend(old, inputs):
        new = nn.Linear(inputs, old.out_features, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:, :old.in_features].copy_(old.weight)
            new.bias.copy_(old.bias)
        return new

    def forward(self, states, lengths=None):
        if lengths is not None or states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("W26 fixed-width state schema mismatch")
        if not bool(torch.isfinite(states).all()):
            raise ValueError("Nonfinite W26 state")
        batch = states.shape[0]
        geo = states[:, :2]
        offset = 207
        if self.width_visible:
            geo = torch.cat((geo, states[:, offset:offset + 1]), dim=-1)
            offset += 1
        channels = [states[:, 4:104].reshape(batch, 10, 10),
                    states[:, 104:204].reshape(batch, 10, 10)]
        if self.relation_visible:
            channels.extend([states[:, offset:offset + 100].reshape(batch, 10, 10),
                             states[:, offset + 100:offset + 200].reshape(batch, 10, 10)])
        tokens = torch.cat((self.geo_embed(geo).unsqueeze(1),
                            self.rate_embed(states[:, 2:3]).unsqueeze(1),
                            self.power_embed(states[:, 3:4]).unsqueeze(1),
                            self.slot_embed(torch.cat(channels, dim=-1))), dim=1) + self.pos_embed
        global_token = self.global_embed(states[:, 204:207]).unsqueeze(1) + self.extra_pos_embed
        sequence = torch.cat((tokens, global_token), dim=1).permute(1, 0, 2)
        return self.net(sequence).permute(1, 0, 2)[:, 0]


class WidthAgent(WorkbenchAgent):
    def __init__(self, config, device=None):
        group = config["group"]
        expected = build_config(group, device or config["training"]["device"], config.get("phase", "scientific"))
        for section, keys in {"network": ("kind", "embedding_dim", "num_heads", "num_layers", "feedforward_dim",
                                             "activation", "dropout", "sac_policy_structure", "sac_critic_structure",
                                             "temporal_mode", "graph_mode", "context_beam_count", "readout_mode",
                                             "transformer_layout", "beam_count_conditioning"),
                              "training": ("seed", "gamma", "tau", "alpha_mode", "fixed_alpha", "actor_lr", "critic_lr",
                                             "alpha_lr", "initial_alpha", "target_entropy", "progress_strata"),
                              "state": ("features", "demand_visibility", "context_level"),
                              "reward": ("preset",), "environment": ("action_mask", "power_levels_w", "power_budget_w")}.items():
            if any(config[section][key] != expected[section][key] for key in keys):
                raise ValueError("W26 T0 contract differs in " + section)
        if config.get("algorithm") != "sac":
            raise ValueError("W26 is T0 SAC")
        if any(config["reward"].get("shaping", {}).get(k, 0) != 0
               for k in ("w_throughput", "w_spectrum", "w_shape")):
            raise ValueError("W26 does not add reward shaping")
        super().__init__(config, 207, (100, 10, 10), device=device or config["training"]["device"])
        for index, name in enumerate(("actor", "critic1", "critic2")):
            network = getattr(self, name)
            network.encoder = WidthEncoder(network.encoder, group,
                           self.train["seed"] + 100003 * (index + 1)).to(self.device)
        self.target1 = deepcopy(self.critic1).eval()
        self.target2 = deepcopy(self.critic2).eval()
        # Rebind optimizers to the selected, real input projections.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.train["actor_lr"])
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.train["critic_lr"])
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.train["critic_lr"])
        self.schema = schema_identity(group)

    def state_dict(self):
        data = super().state_dict()
        data["w26_schema"] = self.schema
        return data

    def observe(self, state, action, reward, next_state, done, action_mask=None,
                next_action_mask=None, **kwargs):
        for value in (state, next_state):
            array = np.asarray(value)
            if array.shape != (self.schema["dimension"],) or not np.isfinite(array).all():
                raise ValueError("W26 replay state schema mismatch")
        return super().observe(state, action, reward, next_state, done,
                               action_mask, next_action_mask, **kwargs)

    def load_state_dict(self, data):
        if data.get("w26_schema") != self.schema:
            raise ValueError("W26 checkpoint schema mismatch; old T0 is not a resume checkpoint")
        for row in data.get("replay", {}).get("rows", []):
            if any(np.asarray(row[index]).shape != (self.schema["dimension"],)
                   for index in (0, 3)):
                raise ValueError("W26 checkpoint replay schema mismatch")
        super().load_state_dict(data)

    def policy_state(self):
        return dict(schema=self.schema, actor=self.actor.state_dict(), update_step=self.update_step)

    def parameter_counts(self):
        return {name: sum(p.numel() for p in getattr(self, name).parameters())
                for name in ("actor", "critic1", "critic2", "target1", "target2")}
