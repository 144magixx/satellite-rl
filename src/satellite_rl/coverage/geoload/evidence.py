"""Pure positional/requested-demand features and immutable pre-teacher pools."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import legal_matrix, layout_sha256

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def pool_features(scene, current, pool):
    centers = []
    for layout in [current] + [entry[1] for entry in pool]:
        for center in layout.centers:
            if not any(np.array_equal(center, prior) for prior in centers): centers.append(center.copy())
    centers = np.asarray(centers)
    def mapping(layout):
        ids = [next(i for i,c in enumerate(centers) if np.array_equal(c,x)) for x in layout.centers]
        return np.asarray(ids, dtype=np.int64)[layout.assignment]
    initial = mapping(current)
    legal = legal_matrix(scene, centers)
    total = scene.demands.sum(); mean = total / max(1,len(current.centers))
    normalized = scene.demands / mean if mean else np.zeros_like(scene.demands)
    loads = np.bincount(initial, weights=normalized, minlength=len(centers))
    counts = np.bincount(initial, minlength=len(centers))
    # Eight inputs, all from position, requested load or current geometry.
    air = np.column_stack((scene.positions[:,0]/90,scene.positions[:,1]/180,
          normalized, np.log1p(normalized), legal.sum(1)/max(1,len(centers)),
          loads[initial], counts[initial]/max(1,len(initial)), np.ones(len(initial))))
    cand = np.column_stack((centers[:,0]/90,centers[:,1]/180, loads,
          counts/max(1,len(initial)),legal.sum(0)/max(1,len(initial)),
          np.log1p(loads), (counts>0).astype(float),np.ones(len(centers))))
    return dict(aircraft=air.astype(np.float32), candidates=cand.astype(np.float32),
                legal=legal, initial=initial, centers=centers), mapping

def write_sample(path, sample):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez(temp, **sample); temp.replace(path)
    return {'path':path.name,'sha256':digest(path)}

class EvidenceDataset:
    def __init__(self,path):
        self.path=Path(path); self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest['status']!='completed': raise ValueError('incomplete teacher evidence')
        self.sha256=digest(self.path); self.rows=[]; self.mothers={}
        for record in self.manifest['samples']:
            p=(self.path.parent/record['path']).resolve()
            if not p.is_relative_to(self.path.parent.resolve()) or digest(p)!=record['sha256']: raise ValueError('evidence hash/path')
            with np.load(p,allow_pickle=False) as data: sample={k:data[k].copy() for k in data.files}
            labels=sample['labels']; legal=sample['legal']
            if not legal[np.arange(len(labels)),labels].all(): raise ValueError('teacher illegal label')
            self.mothers.setdefault(record['mother_id'],[]).append(len(self.rows)); self.rows.append(sample)
        self.keys=sorted(self.mothers)
        if not self.keys: raise ValueError('no teacher samples')
    def draw(self,rng,size):
        ids=[int(rng.choice(self.mothers[self.keys[int(rng.integers(len(self.keys)))]])) for _ in range(size)]
        return [self.rows[i] for i in ids],ids
