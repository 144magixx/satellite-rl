"""Legal discrete allocation moves shared by the literature searches."""
from typing import NamedTuple
import numpy as np


class Alloc(NamedTuple):
    groups: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    powers: np.ndarray


KINDS = ('count', 'power', 'start', 'group')


def legal(alloc):
    groups, starts, counts, _ = alloc
    if np.any(counts < 1) or np.any(counts > 10) or np.any(starts < 0) or np.any(starts + counts > 100):
        return False
    for group in range(8):
        members = np.flatnonzero(groups == group)
        order = members[np.argsort(starts[members])]
        if np.any(starts[order[1:]] < (starts + counts)[order[:-1]]):
            return False
    return True


def propose(alloc, rng, kinds=KINDS):
    n = len(alloc.groups)
    for _ in range(20):
        candidate = Alloc(*(value.copy() for value in alloc))
        g, st, k, p = candidate
        kind = rng.choice(kinds)
        i = int(rng.integers(n))
        if kind == 'power':
            i, j = rng.choice(n, 2, replace=False)
            amount = min(rng.uniform(0, 5), p[i], 50 - p[j])
            p[i] -= amount
            p[j] += amount
        elif kind == 'count':
            delta = rng.choice([-1, 1])
            k[i] += delta
            if legal(candidate):
                return candidate
            if delta > 0:
                for offset in (-1, 1):
                    st[i] = alloc.starts[i] + offset
                    if legal(candidate):
                        return candidate
        else:
            if kind == 'group':
                g[i] = rng.choice(np.delete(np.arange(8), g[i]))
            occupied = np.zeros(100, bool)
            for j in np.flatnonzero((g == g[i]) & (np.arange(n) != i)):
                occupied[st[j]:st[j] + k[j]] = True
            empty = np.flatnonzero(np.convolve(occupied.astype(int), np.ones(k[i], int), 'valid') == 0)
            if len(empty):
                st[i] = rng.choice(empty)
                return candidate
        if legal(candidate):
            return candidate
    return alloc


def pack(groups, counts, demand):
    counts = np.asarray(counts, int).copy()
    starts = np.zeros(len(counts), int)
    for group in range(8):
        members = np.flatnonzero(groups == group)
        order = members[np.argsort(-np.asarray(demand)[members], kind='stable')]
        if counts[order].sum() > 100:
            counts[order] = np.maximum(1, np.floor(counts[order] * 100 / counts[order].sum()).astype(int))
            while counts[order].sum() > 100:
                for i in order[::-1]:
                    if counts[i] > 1 and counts[order].sum() > 100:
                        counts[i] -= 1
            while counts[order].sum() < 100:
                for i in order:
                    if counts[i] < 10 and counts[order].sum() < 100:
                        counts[i] += 1
        starts[order] = np.cumsum(counts[order]) - counts[order]
    return starts, counts


def uniform_start(stage_input):
    n = len(stage_input.layout['beam_ids'])
    groups = np.arange(n) % 8
    counts = np.full(n, min(10, 100 // int(np.ceil(n / 8))), int)
    starts = (np.arange(n) // 8) * counts
    return Alloc(groups, starts, counts, np.full(n, min(50., 3000 / n)))
