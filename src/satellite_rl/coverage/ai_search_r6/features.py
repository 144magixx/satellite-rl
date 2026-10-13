"""R6 actual-aircraft inputs; no labels or decoder results enter features."""
from dataclasses import dataclass
import hashlib
import json
import time
import numpy as np
import torch


def record_sha(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class FeatureCase:
    values: np.ndarray
    neighbors: np.ndarray
    aircraft_ids: tuple
    input_sha: str
    cost: dict
    identity: dict = None


def build_features(scene, *, identity=None):
    started = time.monotonic()
    rays = np.asarray(scene.rays, np.float64)
    demands = np.asarray(scene.demands, np.float64)
    ids = tuple(scene.aircraft_ids)
    n = len(ids)
    if not n or rays.shape != (n, 3) or demands.shape != (n,) or len(set(ids)) != n:
        raise ValueError('nonempty original aircraft order required')
    if not np.isfinite(rays).all() or not np.isfinite(demands).all() or (demands < 0).any():
        raise ValueError('nonfinite or negative feature input')
    # Each unordered pair is evaluated once, in triangular blocks <=256 rows.
    id_rank = np.empty(n,np.int64)
    id_rank[np.argsort(np.asarray(ids),kind='stable')] = np.arange(n)
    best_indices=np.full((n,16),-1,np.int64); best_angles=np.full((n,16),np.inf)
    for begin in range(0, n, 256):
        end=min(begin+256,n)
        rows,cols=np.nonzero(np.arange(begin,end)[:,None] < np.arange(n)[None,:])
        first=rows+begin
        # Norm and angle evaluation occur only for this unique upper-triangle list.
        chord=np.linalg.norm(rays[cols]-rays[first],axis=1)
        angles=2*np.arcsin(np.minimum(chord/2,1))
        block=np.full((end-begin,n),np.inf)
        block[rows,cols]=angles
        for i in range(begin,n):
            if i<end:
                candidate_ids=np.concatenate((np.arange(begin,i),np.arange(i+1,n)))
                candidate_angles=np.concatenate((block[:i-begin,i],block[i-begin,i+1:]))
            else:
                candidate_ids=np.arange(begin,end);candidate_angles=block[:,i]
            keep=best_indices[i]>=0
            candidate_ids=np.concatenate((best_indices[i,keep],candidate_ids))
            candidate_angles=np.concatenate((best_angles[i,keep],candidate_angles))
            order=np.lexsort((id_rank[candidate_ids],candidate_angles))[:16]
            size=len(order);best_indices[i,:size]=candidate_ids[order];best_angles[i,:size]=candidate_angles[order]
    neighbors = np.full((n, 16), -1, np.int64)
    total = demands.sum(); shares = demands / total if total else np.zeros(n)
    angle_mean = np.zeros(n); angle_max = np.zeros(n); neighbor_share = np.zeros(n)
    for i in range(n):
        keep=best_indices[i]>=0; size=int(keep.sum())
        if size:
            neighbors[i,:size]=best_indices[i,keep]
            angle_mean[i]=best_angles[i,keep].mean()/np.deg2rad(1)
            angle_max[i]=best_angles[i,keep].max()/np.deg2rad(1)
            neighbor_share[i]=shares[neighbors[i,:size]].sum()
    values = np.column_stack(((rays - rays.mean(axis=0)) / np.deg2rad(1), shares,
        np.log1p(demands / 1e6), angle_mean, angle_max, neighbor_share,
        np.full(n, shares.max()), np.full(n, np.log1p(n))))
    if values.shape != (n, 10) or not np.isfinite(values).all(): raise ValueError('invalid10features')
    if identity is not None and identity.get('input_sha')!=scene.input_sha256:raise ValueError('feature identity input mismatch')
    return FeatureCase(values, neighbors, ids, scene.input_sha256,
        dict(seconds=time.monotonic()-started, NN_ray_pairs=n*(n-1)//2, block_rows_max=256),
        None if identity is None else dict(identity))


@dataclass
class FeatureStats:
    mean: np.ndarray
    scale: np.ndarray
    qualified_table_sha: str

    @classmethod
    def fit(cls, cases, *, role, qualified_table_sha):
        if role != 'train' or not qualified_table_sha: raise ValueError('qualified training inputs only')
        if any(c.identity is not None and c.identity.get('role')!='train' for c in cases):raise ValueError('normalization contains held-out role')
        x = np.concatenate([c.values for c in cases], axis=0)
        if x.shape[1:] != (10,) or not len(x) or not np.isfinite(x).all(): raise ValueError('invalid fit inputs')
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std == 0, 1, std), qualified_table_sha)

    def to_record(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), fitted_role='train',
                    qualified_table_sha=self.qualified_table_sha, dimension=10)

    @classmethod
    def from_record(cls, value):
        if value.get('fitted_role') != 'train' or value.get('dimension') != 10: raise ValueError('stats role/schema')
        mean, scale = np.asarray(value['mean']), np.asarray(value['scale'])
        if mean.shape != (10,) or scale.shape != (10,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any(): raise ValueError('stats invalid')
        return cls(mean, scale, value['qualified_table_sha'])

    @property
    def sha(self): return record_sha(self.to_record())


def collate_features(cases, stats, device='cpu'):
    if not cases:raise ValueError('nonempty actual feature batch')
    nmax = max(len(c.aircraft_ids) for c in cases)
    x = np.zeros((len(cases), nmax, 10), np.float32)
    neighbors = np.full((len(cases), nmax, 16), -1, np.int64)
    valid = np.zeros((len(cases), nmax), bool)
    for b, c in enumerate(cases):
        n = len(c.aircraft_ids); valid[b, :n] = True
        if c.values.shape!=(n,10) or c.neighbors.shape!=(n,16) or (c.neighbors<-1).any() or (c.neighbors>=n).any() or not np.isfinite(c.values).all():raise ValueError('feature/neighbors shape/range')
        x[b, :n] = (c.values - stats.mean) / stats.scale
        neighbors[b, :n] = c.neighbors
    return dict(features=torch.tensor(x, device=device), neighbors=torch.tensor(neighbors, device=device),
                valid=torch.tensor(valid, device=device))
