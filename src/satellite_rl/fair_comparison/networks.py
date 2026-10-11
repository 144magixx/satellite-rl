"""Exactly matched pre-LN Transformer action representations."""
import hashlib
import numpy as np
import torch
from torch import nn

BEAM_WIDTH = 24
RESOURCE_WIDTH = 803


def tensor_observations(observations, device):
    if isinstance(observations, dict):
        observations = [observations]
    return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=device,
                               dtype=torch.bool if k == "padding" else torch.long if k == "current" else torch.float32)
            for k in ("beams", "resources", "padding", "current")}


class TransformerNetwork(nn.Module):
    def __init__(self, output=1000):
        super().__init__()
        self.beam_embedding = nn.Linear(BEAM_WIDTH, 128)
        self.resource_embedding = nn.Linear(RESOURCE_WIDTH, 128)
        positions = torch.arange(200).float()[:, None]
        frequencies = torch.exp(torch.arange(0, 128, 2).float()*(-np.log(10000.)/128))
        encoding = torch.zeros(200, 128)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions*frequencies), torch.cos(positions*frequencies)
        self.register_buffer("position_encoding", encoding)
        layer = nn.TransformerEncoderLayer(128, 4, 512, 0., "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(128), enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, output))

    def forward(self, obs):
        beams = self.beam_embedding(obs["beams"]) + self.position_encoding
        resource = self.resource_embedding(obs["resources"])[:, None]
        padding = torch.cat((torch.zeros((beams.shape[0], 1), device=beams.device, dtype=torch.bool), obs["padding"]), dim=1)
        encoded = self.encoder(torch.cat((resource, beams), dim=1), src_key_padding_mask=padding)
        current = encoded[torch.arange(len(encoded), device=beams.device), obs["current"]+1]
        return self.head(torch.cat((encoded[:, 0], current), dim=-1))


def masked_distribution(logits, mask):
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("Cannot normalize an empty legal action set")
    logp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    probability = logp.exp()
    # Mask before multiplication: avoid 0 * -inf and ensure zero illegal gradients.
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    return probability, logp


def parameter_sha(network):
    h = hashlib.sha256()
    for key, value in network.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
