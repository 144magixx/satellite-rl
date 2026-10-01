"""Beam-token encoders for the fixed 2946-field full observation.

The optional directed edge is j -> i: an already transmitting beam j may
interfere with receiver i. No action proposed for i or any future beam is read.
"""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

BEAM_OFFSET = 214
RESOURCE_OFFSET = 1314
TOKEN_COUNT = 122
PHYSICAL_PROFILES = {
    "china_legacy": ((4.0, 54.0), (73.0, 135.0), 122.2, 6371, 42164),
    "europe_v1": ((25.0, 80.0), (-40.0, 50.0), 13.0, 6371, 42164),
}
PHYSICAL_CONSTANTS = {
    "transmit_gain_dbi": 50.0,
    "receive_gain_dbi": 40.0,
    "beam_width_deg": 0.5,
    "carrier_start_hz": 1.77e10,
    "slot_bandwidth_hz": 2.5e7,
    "light_speed_m_s": 2.99792458e8,
}


def validate_physical_contract(scene, config, environment_constants=None):
    """Check that G2's inferred fields match a real scene and environment.

    ``environment_constants`` is the runner's mapping of actual physics
    constants; the agent does not import the environment at runtime. It must
    contain the six PHYSICAL_CONSTANTS keys plus latitude/longitude bounds,
    satellite longitude, Earth radius, and orbit radius from the live profile.
    """
    import numpy as np

    profile_id = config["environment"]["profile"]
    if profile_id not in PHYSICAL_PROFILES:
        raise ValueError(f"G2 has no frozen physical contract for {profile_id!r}")
    expected_lat, expected_lon, expected_sat, expected_earth, expected_orbit = PHYSICAL_PROFILES[profile_id]
    required = dict(PHYSICAL_CONSTANTS, latitude_bounds_deg=expected_lat,
                    longitude_bounds_deg=expected_lon,
                    satellite_longitude_deg=expected_sat,
                    earth_radius_km=expected_earth, orbit_radius_km=expected_orbit)
    if environment_constants is None or set(required) - set(environment_constants):
        raise ValueError("G2 requires the live environment physics constants at reset")
    for key, expected in required.items():
        actual = environment_constants[key]
        if not np.allclose(actual, expected, rtol=0, atol=1e-9):
            raise ValueError(f"G2 physical constant mismatch: {key}")
    for key, expected in (("beam_G_Tx_max", 50.0), ("UE_G_Rx_max", 40.0)):
        values = np.asarray(scene[key])
        if values.shape != (100, 1) or not np.isfinite(values).all() or not np.allclose(values, expected, rtol=0, atol=1e-9):
            raise ValueError(f"G2 requires every scene beam to have {expected:g} dBi {key}")
    return required


class _AttentionLayer(nn.Module):
    """Post-norm Transformer layer using fused scaled dot product attention."""

    def __init__(self, embed_dim, heads, feedforward_dim, dropout, activation):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(nn.Linear(embed_dim, feedforward_dim),
                                         {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation](),
                                         nn.Dropout(dropout), nn.Linear(feedforward_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout

    def forward(self, x, bias=None):
        batch, tokens, width = x.shape
        q, k, v = (part.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
                   for part in self.qkv(x).chunk(3, dim=-1))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.attention_dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        x = self.norm1(x + self.dropout(self.out(attended)))
        return self.norm2(x + self.dropout(self.feedforward(x)))


class GraphEncoder(nn.Module):
    """G1/G2 encoder; states are [batch,2946], output [batch,embedding_dim]."""

    def __init__(self, config, state_dim=2946, graph_mode=None, reference_encoder=None):
        super().__init__()
        if state_dim != 2946:
            raise ValueError("GraphEncoder requires the full ordered 2946-field observation")
        spec = config["network"]
        self.graph_mode = graph_mode or spec["graph_mode"]
        if self.graph_mode not in ("beam_tokens", "interference_bias"):
            raise ValueError("graph_mode must be beam_tokens or interference_bias")
        if (spec.get("kind") != "transformer" or
                spec.get("transformer_layout") != "tsac_natural_context" or
                config["state"].get("context_level") != "full"):
            raise ValueError("GraphEncoder requires the natural full 2946-field Transformer")
        self.output_dim = int(spec["embedding_dim"])
        heads = int(spec["num_heads"])
        if self.output_dim % heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if reference_encoder is None:
            # Consume precisely the G0 Encoder RNG stream before constructing
            # G1/G2's new beam projection. The outer SAC heads then start from
            # the same RNG state as G0 under a common training seed.
            from satellite_rl.agents.workbench_networks import Encoder

            reference_config = deepcopy(config)
            reference_config["network"]["graph_mode"] = "none"
            reference_encoder = Encoder(reference_config, state_dim)
        with torch.random.fork_rng(devices=[]):
            self.geo_embed = nn.Linear(2, self.output_dim)
            self.rate_embed = nn.Linear(1, self.output_dim)
            self.power_embed = nn.Linear(1, self.output_dim)
            self.slot_embed = nn.Linear(20, self.output_dim)
            self.global_embed = nn.Linear(10, self.output_dim)
            self.beam_embed = nn.Linear(11, self.output_dim)
            self.resource_embed = nn.Linear(204, self.output_dim)
            # Only the 13 original positions and 8 fixed group identities need
            # positions. Each beam carries its true processing index in field 4.
            self.base_position = nn.Parameter(torch.zeros(1, 13, self.output_dim))
            self.global_position = nn.Parameter(torch.zeros(1, 1, self.output_dim))
            self.resource_position = nn.Parameter(torch.zeros(1, 8, self.output_dim))
            self.layers = nn.ModuleList([
                _AttentionLayer(self.output_dim, heads, int(spec["feedforward_dim"]),
                                float(spec["dropout"]), spec["activation"])
                for _ in range(int(spec["num_layers"]))
            ])
            self._copy_common_initialization(reference_encoder)
        if self.graph_mode == "interference_bias":
            profile_id = config["environment"]["profile"]
            if profile_id not in PHYSICAL_PROFILES:
                raise ValueError(f"G2 has no frozen physical contract for {profile_id!r}")
            (self.latitude_bounds, self.longitude_bounds, self.satellite_lon,
             self.earth_radius_km, self.orbit_radius_km) = PHYSICAL_PROFILES[profile_id]
            self.max_power_w = max(float(p) for p in config["environment"]["power_levels_w"])
            # G1 and G2 have identical common initialization under one seed.
            with torch.random.fork_rng(devices=[]):
                self.edge_weight = nn.Parameter(torch.zeros(2))

    def _copy_common_initialization(self, reference):
        """Match G0's common spatial weights without sharing live parameters.

        The caller first constructs the real 32-token G0 Encoder to consume
        its original RNG stream, then constructs this module in fork_rng. The
        only unmatched projection is G0's 110-value beam section versus G1's
        11-value beam token; their shapes and represented units differ.
        """
        if (getattr(reference, "transformer_layout", None) != "tsac_natural_context" or
                getattr(reference, "natural_level", None) != "full" or
                len(reference.net.layers) != len(self.layers) or
                reference.output_dim != self.output_dim or
                reference.extra_pos_embed.shape != (1, 19, self.output_dim)):
            raise ValueError("G0 initialization reference must be the full 32-token Encoder")
        with torch.no_grad():
            for name in ("geo_embed", "rate_embed", "power_embed", "slot_embed",
                         "global_embed", "resource_embed"):
                getattr(self, name).load_state_dict(getattr(reference, name).state_dict(), strict=True)
            self.base_position.copy_(reference.pos_embed)
            self.global_position.copy_(reference.extra_pos_embed[:, :1])
            self.resource_position.copy_(reference.extra_pos_embed[:, 11:19])
            for layer, original in zip(self.layers, reference.net.layers):
                layer.qkv.weight.copy_(original.self_attn.in_proj_weight)
                layer.qkv.bias.copy_(original.self_attn.in_proj_bias)
                layer.out.load_state_dict(original.self_attn.out_proj.state_dict(), strict=True)
                layer.norm1.load_state_dict(original.norm1.state_dict(), strict=True)
                layer.norm2.load_state_dict(original.norm2.state_dict(), strict=True)
                layer.feedforward[0].load_state_dict(original.linear1.state_dict(), strict=True)
                layer.feedforward[3].load_state_dict(original.linear2.state_dict(), strict=True)

    def _tokens(self, states):
        batch = states.shape[0]
        slots = torch.cat((states[:, 4:104].reshape(batch, 10, 10),
                           states[:, 104:204].reshape(batch, 10, 10)), dim=-1)
        base = torch.cat((self.geo_embed(states[:, :2]).unsqueeze(1),
                          self.rate_embed(states[:, 2:3]).unsqueeze(1),
                          self.power_embed(states[:, 3:4]).unsqueeze(1),
                          self.slot_embed(slots)), dim=1) + self.base_position
        global_token = self.global_embed(states[:, 204:214]).unsqueeze(1) + self.global_position
        beams = self.beam_embed(states[:, BEAM_OFFSET:RESOURCE_OFFSET].reshape(batch, 100, 11))
        resources = (self.resource_embed(states[:, RESOURCE_OFFSET:].reshape(batch, 8, 204))
                     + self.resource_position)
        return torch.cat((base, global_token, beams, resources), dim=1)

    def edge_features(self, states):
        """Return [B,100,100,2], axis 1 receiver i, axis 2 sender j.

        Feature 0 is potential received power (normalized dBW) from a real
        transmitting sender, using its occupied-slot mean frequency. Feature 1
        is actual overlap / sender slots; it is zero for an unallocated receiver.
        The potential feature is not an assertion of future slot overlap.
        """
        rows = states[:, BEAM_OFFSET:RESOURCE_OFFSET].reshape(-1, 100, 11)
        device = states.device
        dtype = states.dtype
        # Near-identical GEO directions differ at the sub-degree scale. Use
        # float64 for geometry so acos does not quantize them to zero/NaN.
        rows = rows.to(torch.float64)
        lat_min, lat_max = self.latitude_bounds
        lon_min, lon_max = self.longitude_bounds
        lat = ((rows[..., 0] + 1) * (lat_max - lat_min) / 2 + lat_min) * (math.pi / 180)
        lon = ((rows[..., 1] + 1) * (lon_max - lon_min) / 2 + lon_min) * (math.pi / 180)
        cos_lat = lat.cos()
        earth = self.earth_radius_km
        ground = torch.stack((earth * cos_lat * lon.cos(), earth * cos_lat * lon.sin(),
                              earth * lat.sin()), dim=-1)
        sat_lon = self.satellite_lon * math.pi / 180
        satellite = torch.tensor([self.orbit_radius_km * math.cos(sat_lon),
                                  self.orbit_radius_km * math.sin(sat_lon), 0.0],
                                 dtype=torch.float64, device=device)
        direction = satellite - ground
        distance_m = torch.linalg.vector_norm(direction, dim=-1) * 1000
        unit = F.normalize(direction, dim=-1)
        # The environment uses satellite-to-ground vectors; reversing both
        # vectors leaves their mutual angle unchanged.
        cosine = (unit[:, :, None, :] * unit[:, None, :, :]).sum(-1).clamp(-1, 1)
        angle = torch.acos(cosine) * (180 / math.pi)
        sigma = 0.5 / (2 * math.sqrt(2 * math.log10(2)))
        edge = 0.25
        attenuation_db = torch.where(angle <= edge,
                                     3 * (angle / edge).square(),
                                     3 + 10 / math.log(10) * (angle.square() - edge ** 2)
                                     / (2 * sigma ** 2))
        sender_slots = (rows[..., 7] * 10).round().clamp(0, 10)
        sender_start = (rows[..., 6] * 99).round().clamp(0, 99)
        sender_power_w = (rows[..., 8] * self.max_power_w).clamp_min(0)
        group = (rows[..., 3] * 7).round().clamp(0, 7).long()
        allocated = rows[..., 5] > 0.5
        sender_active = allocated & (sender_slots > 0) & (sender_power_w > 0)
        # 50 dBi transmitter and 40 dBi receiver are fixed by the scene
        # generator; no unobserved per-scene channel or noise is inserted.
        # Sum 1/f_slot² over the sender's *actual* occupied slots. This is the
        # same per-slot free-space loss used by the environment; evaluating
        # only the center frequency would silently change the power scale.
        slot = torch.arange(100, device=device, dtype=torch.float64)
        relative_gain = (1.77e10 / (1.77e10 + 2.5e7 * (slot + 0.5))).square()
        prefix = F.pad(relative_gain.cumsum(0), (1, 0))
        source_end = (sender_start + sender_slots).clamp(max=100).long()
        source_begin = sender_start.long()
        mean_frequency_gain = ((prefix[source_end] - prefix[source_begin])
                               / sender_slots.clamp_min(1))
        path_loss_db = (20 * torch.log10(4 * math.pi * distance_m.unsqueeze(2)
                                          * 1.77e10 / 2.99792458e8)
                        - 10 * torch.log10(mean_frequency_gain.clamp_min(1e-30)).unsqueeze(1))
        power_db = (10 * torch.log10(sender_power_w.clamp_min(1e-30)).unsqueeze(1)
                    - 10 * torch.log10(sender_slots.clamp_min(1)).unsqueeze(1)
                    + 50 - attenuation_db + 40 - path_loss_db)
        same_polarization = (group[:, :, None] % 2 == group[:, None, :] % 2)
        identity = torch.eye(100, dtype=torch.bool, device=device).unsqueeze(0)
        valid = sender_active[:, None, :] & same_polarization & ~identity
        potential = ((power_db.clamp(-300, -100) + 300) / 200) * valid
        receiver_start = sender_start.unsqueeze(2)
        receiver_end = receiver_start + sender_slots.unsqueeze(2)
        source_start = sender_start.unsqueeze(1)
        source_end = source_start + sender_slots.unsqueeze(1)
        overlap = (torch.minimum(receiver_end, source_end)
                   - torch.maximum(receiver_start, source_start)).clamp_min(0)
        actual = (overlap / sender_slots.unsqueeze(1).clamp_min(1)
                  * (allocated & (sender_slots > 0))[:, :, None] * valid)
        return torch.stack((potential, actual), dim=-1).to(dtype)

    def _attention_bias(self, states):
        edges = self.edge_features(states)
        batch = states.shape[0]
        bias = states.new_zeros(batch, TOKEN_COUNT, TOKEN_COUNT)
        # No head-specific tensor: SDPA broadcasts [B,1,T,T] over all heads.
        bias[:, 14:114, 14:114] = (edges * self.edge_weight).sum(-1)
        # Token 0 is the current-beam query. Use its physical receiver row,
        # whose index comes from progress already present in the observation.
        current = (states[:, 205] * 100).round().long()
        valid = current < 100
        rows = states[:, BEAM_OFFSET:RESOURCE_OFFSET].reshape(batch, 100, 11)
        # Storage permutations move rows but retain the physical processing
        # index in field 4; select by that identity, not array offset.
        current_row = (rows[..., 4] - current.clamp(0, 99)[:, None] / 99).abs().argmin(1)
        selected = edges.gather(1, current_row[:, None, None, None]
                                .expand(batch, 1, 100, 2)).squeeze(1)
        bias[:, 0, 14:114] = (selected * self.edge_weight).sum(-1) * valid[:, None]
        return bias.unsqueeze(1)

    def forward(self, states):
        if states.ndim != 2 or states.shape[1] != 2946 or not torch.is_floating_point(states):
            raise ValueError("State shape must be floating [batch,2946]")
        tokens = self._tokens(states)
        bias = self._attention_bias(states) if self.graph_mode == "interference_bias" else None
        for layer in self.layers:
            tokens = layer(tokens, bias)
        return tokens[:, 0]
