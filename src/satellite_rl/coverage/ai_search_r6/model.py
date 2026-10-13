"""Exact preregistered R6 graph and CPU initialization order."""
import math
import torch
from torch import nn
from .features import array_sha, record_sha


class GraphLayer(nn.Module):
    def __init__(self):
        super().__init__(); self.w1 = nn.Linear(256, 128); self.w2 = nn.Linear(128, 128)

    def forward(self, h, neighbors, valid):
        safe = neighbors.clamp(min=0)
        batch = torch.arange(h.shape[0], device=h.device)[:, None, None]
        neighbor_valid = (neighbors >= 0) & valid[batch, safe] & valid[:, :, None]
        gathered = h[batch, safe] * neighbor_valid[..., None]
        mean = gathered.sum(dim=2) / neighbor_valid.sum(dim=2).clamp(min=1)[..., None]
        return torch.relu(h + self.w2(torch.relu(self.w1(torch.cat((h, mean), dim=-1))))) * valid[..., None]


class CrossLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(128,128); self.k = nn.Linear(128,128); self.v = nn.Linear(128,128)
        self.out = nn.Linear(128,128); self.ff1 = nn.Linear(128,256); self.ff2 = nn.Linear(256,128)
        self.ln1 = nn.LayerNorm(128, eps=1e-5); self.ln2 = nn.LayerNorm(128, eps=1e-5)

    def forward(self, query, memory, valid):
        b, n, _ = memory.shape
        q = self.q(query).reshape(b,200,4,32).transpose(1,2)
        k = self.k(memory).reshape(b,n,4,32).transpose(1,2)
        v = self.v(memory).reshape(b,n,4,32).transpose(1,2)
        scores = (q @ k.transpose(-1,-2)) / math.sqrt(32)
        scores = scores.masked_fill(~valid[:,None,None,:], -torch.inf)
        attention = (scores.softmax(dim=-1) @ v).transpose(1,2).reshape(b,200,128)
        z = self.ln1(query + self.out(attention))
        return self.ln2(z + self.ff2(torch.relu(self.ff1(z))))


class R6Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(10,128); self.encoder2 = nn.Linear(128,128)
        self.graph = nn.ModuleList([GraphLayer() for _ in range(3)])
        self.query = nn.Embedding(200,128)
        self.cross = nn.ModuleList([CrossLayer() for _ in range(2)])
        self.enable = nn.Linear(128,1)
        self.pointer_query = nn.Linear(128,128,bias=False); self.pointer_aircraft = nn.Linear(128,128,bias=False)
        self.owner_aircraft = nn.Linear(128,128,bias=False); self.owner_query = nn.Linear(128,128,bias=False)

    def forward(self, features, neighbors, valid):
        if features.dtype != torch.float32 or features.shape[-1] != 10 or not valid.any(dim=1).all(): raise ValueError('FP32 actual aircraft required')
        h = torch.relu(self.encoder2(torch.relu(self.encoder1(features)))) * valid[...,None]
        for layer in self.graph: h = layer(h,neighbors,valid)
        q = self.query.weight[None].expand(features.shape[0],-1,-1)
        for layer in self.cross: q = layer(q,h,valid)
        pointer = self.pointer_query(q) @ self.pointer_aircraft(h).transpose(-1,-2) / math.sqrt(128)
        owner = self.owner_aircraft(h) @ self.owner_query(q).transpose(-1,-2) / math.sqrt(128)
        return dict(enable=self.enable(q).squeeze(-1), pointer=pointer.masked_fill(~valid[:,None,:],-torch.inf),
                    owner=owner, valid=valid)


def initialize_model():
    torch.manual_seed(51001)
    with torch.device('cpu'):
        return R6Model()


def tensor_manifest(model):
    return {name: dict(dtype=str(t.dtype),shape=list(t.shape),sha256=array_sha(t.detach().cpu().contiguous().numpy()))
            for name,t in model.state_dict().items()}


def model_tensor_sha(model): return record_sha(tensor_manifest(model))
