"""R5 anchor-domain kernel; preparation and teacher math copied from frozen R4. All requests stay whole; no native optimizer.

Only the three offline diagnostic proposal builders accept teacher metadata. A decoder sees
the same discriminated proposal interface for every method. Pair and proposal
charges precede the corresponding computation, including rejected trials.
"""
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import time
import numpy as np

from ..ai_search.core import Scene, Layout, metrics, ray_location, layout_sha256
from ..ai_search_r2.validation import validate_owned
from ..geometry import satellite_rays_numpy, ray_angle_numpy, visible_numpy


class InterfaceError(ValueError):
    """Hard input/provenance error: dispatch must stop rather than hide it."""


class Exhausted(RuntimeError):
    pass


def digest(*items):
    return hashlib.sha256(json.dumps([51001, *items], ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).digest()


def omega(widths):
    return float(np.sum(2*np.pi*(1-np.cos(np.deg2rad(np.asarray(widths)/2)))))


class Ledger:
    def __init__(self, n, pair_factor, seconds, proposals=0, refits=0, reserve=0):
        self.started = time.monotonic()
        self.seconds_limit = float(seconds)
        self.ray_limit = int(pair_factor*n)
        self.proposal_limit = proposals
        self.refit_limit = refits
        self.reserve = int(reserve*n)
        self.terminal = False
        self.data = dict(raypairs=0, proposals=0, refits=0, mean_input_count=0,
                        sort_input_count=0, scan_input_count=0, hash_calls=0, cache_hits=0, rank_geometry_pairs=0,
                        skipped_obsolete=0, phases=[], accepted=[], geometry_calls=0,
                        center_constructions=0, truncation_reason=None)
        self._rays = {}

    def check(self):
        if not self.terminal and time.monotonic()-self.started >= self.seconds_limit:
            raise Exhausted('nominal_time_limit')

    def geometry(self, count, label='geometry'):
        self.check()
        limit = self.ray_limit if self.terminal else self.ray_limit-self.reserve
        if self.data['raypairs']+count > limit:
            raise Exhausted('raypair_limit')
        self.data['raypairs'] += int(count)
        self.data['geometry_calls'] += 1

    def center_rays(self, centers, profile):
        result = []
        for center in np.asarray(centers, dtype=np.float64).reshape(-1, 2):
            self.check()
            key = (profile, center.tobytes())
            if key not in self._rays:
                self.data['center_constructions'] += 1
                self._rays[key] = satellite_rays_numpy(center[None], profile)[0]
            result.append(self._rays[key])
        return np.asarray(result).reshape(-1, 3)

    def proposal(self, refit=False):
        self.check()
        if self.data['proposals'] >= self.proposal_limit:
            raise Exhausted('proposal_limit')
        if refit and self.data['refits'] >= self.refit_limit:
            raise Exhausted('refit_limit')
        self.data['proposals'] += 1
        self.data['refits'] += int(refit)

    def record(self):
        elapsed = time.monotonic()-self.started
        return {**self.data, 'seconds': elapsed, 'seconds_nominal': self.seconds_limit,
                'nominal_overrun_seconds': max(0., elapsed-self.seconds_limit),
                'raypairs_max': self.ray_limit, 'proposals_max': self.proposal_limit,
                'refits_max': self.refit_limit, 'terminal_reserve_raypairs': self.reserve}


def _angles_and_chord(first, second):
    # One paid raypair batch; outward boundary validation reuses its exact
    # squared chords instead of computing another distance batch.
    squared = np.sum((np.asarray(first)-np.asarray(second))**2, axis=-1)
    angles = np.rad2deg(2.*np.arcsin(np.minimum(np.sqrt(squared)/2.,1.)))
    return angles, squared


def envelope(scene, members, center, ledger):
    """Fixed-center true-ray envelope; an oversized group is never clipped."""
    members = np.asarray(members, dtype=int)
    ledger.geometry(0, 'envelope_visibility')
    center = np.asarray(center, dtype=np.float64)
    if not len(members) or not visible_numpy(center[None], scene.profile).all():
        return None
    ray = ledger.center_rays(center[None], scene.profile)[0]
    ledger.geometry(len(members), 'envelope_member_angles')
    angles, chord_squared = _angles_and_chord(scene.rays[members], ray)
    required = 2*float(angles.max())
    if required > 1.:
        return None
    width = max(.3, float(np.nextafter(required, np.inf)))
    if width > 1.:
        # At an exact 1-degree boundary the independent chord predicate decides.
        width = 1.
        limit = 4*np.sin(np.deg2rad(.5)/2)**2
        ledger.data['cache_hits'] += len(members)
        if not (chord_squared <= limit+1e-14).all():
            return None
    return width


@dataclass
class PreparedCase:
    scene: Scene
    identity: dict
    source_sha256: str
    raw_w: Layout = None
    Wenv: Layout = None
    anchors: tuple = ()
    Ncap: int = 0
    Omega_cap: float = 0.
    ledger: dict = None
    status: str = 'available'
    reason: str = None

    def to_dict(self):
        return dict(schema='PreparedCaseV1', status=self.status, reason=self.reason,
            identity=self.identity, source_sha256=self.source_sha256,
            raw_w=None if self.raw_w is None else self.raw_w.to_record(),
            Wenv=None if self.Wenv is None else self.Wenv.to_record(),
            anchors=list(self.anchors), Ncap=self.Ncap, Omega_cap=self.Omega_cap,
            ledger=self.ledger, input_sha256=self.scene.input_sha256)

    @classmethod
    def from_dict(cls, scene, record):
        if record['input_sha256'] != scene.input_sha256:
            raise InterfaceError('prepared_input_sha_mismatch')
        _identity(scene,record['identity'])
        if record.get('schema')!='PreparedCaseV1' or record.get('status') not in ('available','unavailable'):
            raise InterfaceError('prepared_schema_or_status_invalid')
        return cls(scene, record['identity'], record['source_sha256'],
            None if record['raw_w'] is None else Layout.from_record(record['raw_w']),
            None if record['Wenv'] is None else Layout.from_record(record['Wenv']),
            tuple(record['anchors']), int(record['Ncap']), float(record['Omega_cap']),
            record['ledger'], record['status'], record.get('reason'))


def _identity(scene, identity):
    needed = ('case_id', 'mother_id', 'profile', 'input_sha256', 'identity_sha256', 'role')
    if not isinstance(identity, dict) or any(k not in identity for k in needed):
        raise InterfaceError('identity_fields_missing')
    if identity['role'] not in ('train','development','validation','test') or identity['input_sha256'] != scene.input_sha256:
        raise InterfaceError('SHA_or_role_mismatch')
    if any(not isinstance(identity[k], str) or not identity[k] for k in needed):
        raise InterfaceError('identity_must_be_original_nonempty_strings')


def prepare_case(scene, identity, source_sha256, *, phase_manifest, time_limit=10.):
    validate_phase(scene, identity, phase_manifest)
    if not 0<=time_limit<=10.:raise InterfaceError('preparation_nominal_limit_exceeds_contract')
    _identity(scene, identity)
    if not len(scene.demands):
        raise InterfaceError('empty_scene')
    if not isinstance(source_sha256, str) or not source_sha256:
        raise InterfaceError('source_sha_missing')
    n = len(scene.demands)
    ledger = Ledger(n, 1024, time_limit, reserve=1)
    result = PreparedCase(scene, dict(identity), source_sha256)
    started = time.monotonic()
    try:
        order = sorted(range(n), key=lambda i: scene.aircraft_ids[i])
        ledger.data['sort_input_count'] += n
        witness = None
        for width in (.5, 1.):
            remaining = np.ones(n, bool)
            owner = np.full(n, -1, dtype=int)
            anchors = []
            for i in order:
                ledger.check()
                if not remaining[i]:
                    continue
                if len(anchors) >= 200:
                    break
                ledger.geometry(n, 'streaming_witness')
                chord = np.sum((scene.rays-scene.rays[i])**2, axis=1)
                member = remaining & (chord <= 4*np.sin(np.deg2rad(width/2)/2)**2+1e-14)
                owner[member] = len(anchors)
                anchors.append(i)
                remaining[member] = False
            if not remaining.any():
                witness = Layout(scene.positions[anchors], np.full(len(anchors), width), owner)
                break
        if witness is None:
            raise Exhausted('typed_witness_unavailable_not_infeasibility')
        result.raw_w = witness
        result.anchors = tuple(scene.aircraft_ids[i] for i in anchors)
        result.Ncap = min(200, n, math.ceil(1.25*len(anchors)))
        result.Omega_cap = omega(witness.widths)
        widths = [envelope(scene, np.flatnonzero(owner==b), witness.centers[b], ledger)
                  for b in range(len(anchors))]
        if any(w is None for w in widths):
            raise Exhausted('witness_envelope_invalid')
        result.Wenv = Layout(witness.centers, widths, owner, witness.beam_ids)
        ledger.terminal = True
        audit = validate_owned(scene, result.Wenv, budget=ledger)
        if not audit['valid'] or omega(result.Wenv.widths)>result.Omega_cap+1e-12:
            raise Exhausted('witness_independent_audit_failed')
        ledger.data['audit'] = audit
        ledger.data['raw_W_sha256'] = layout_sha256(witness)
        ledger.data['Wenv_sha256'] = layout_sha256(result.Wenv)
        ledger.data['hash_calls'] += 2
    except Exhausted as exc:
        result.status, result.reason = 'unavailable', str(exc)
        result.raw_w = result.Wenv = None
        result.anchors = ()
    result.ledger = ledger.record()
    result.ledger['full_publication_seconds'] = time.monotonic()-started
    return result


def _state(layout, anchors):
    return dict(ids=list(layout.beam_ids), centers=layout.centers.copy(),
                widths=layout.widths.copy(), owners=layout.assignment.copy(), anchors=list(anchors))


def _copy(state):
    return {k: v.copy() for k, v in state.items()}


def _layout(state):
    return Layout(state['centers'], state['widths'], state['owners'], tuple(state['ids']))


def _compact(state):
    present = [b for b in range(len(state['ids'])) if np.any(state['owners']==b)]
    mapping = {b:i for i,b in enumerate(present)}
    return dict(ids=[state['ids'][b] for b in present], centers=state['centers'][present].copy(),
        widths=state['widths'][present].copy(), anchors=[state['anchors'][b] for b in present],
        owners=np.array([mapping.get(int(x), -1) for x in state['owners']], dtype=int))


def _loads(scene, state):
    active = state['owners'] >= 0
    return np.bincount(state['owners'][active], weights=scene.demands[active], minlength=len(state['ids']))


def _key(scene, state, ledger=None):
    loads = _loads(scene, state)
    if ledger is not None:ledger.data['scan_input_count'] += len(scene.demands)+len(state['ids'])
    total = float(loads.sum())
    if ledger is not None and len(loads):
        ledger.data['sort_input_count'] += len(loads)
    return (float(loads.max()) if len(loads) else 0.,
        float(np.sort(loads)[-max(1, math.ceil(.1*len(loads))):].mean()) if len(loads) else 0.,
        float(loads.std()/loads.mean()) if total else 0., len(loads), omega(state['widths']))


def lex_better(new, old, tolerances=(1e-6, 1e-6, 1e-9, 0., 1e-12)):
    for a, b, absolute in zip(new, old, tolerances):
        tol = absolute+1e-12*max(abs(a), abs(b))
        if a < b-tol:
            return True
        if a > b+tol:
            return False
    return False


def _legal(p, state, ledger, *, partial=False, resources=True):
    scene = p.scene
    ledger.data['scan_input_count'] += len(scene.demands)+len(state['ids'])
    if (len(state['ids'])>min(200,len(scene.demands),p.Ncap) or len(state['owners'])!=len(scene.demands)
        or np.any(state['owners'] < -1) or np.any(state['owners']>=len(state['ids']))):
        return False
    if resources and (len(state['ids']) > p.Ncap or omega(state['widths']) > p.Omega_cap+1e-12):
        return False
    if not partial:
        return validate_owned(scene, _layout(state), budget=ledger)['valid']
    active = state['owners']>=0
    if not active.any():
        return not len(state['ids'])
    # Audit every assigned owner in an incomplete state without dropping input
    # from publication: orphans remain explicit and prevent final success.
    ledger.geometry(int(active.sum()), 'partial_owner_validation')
    rays = ledger.center_rays(state['centers'], scene.profile)
    limits = 4*np.sin(np.deg2rad(state['widths'][state['owners'][active]]/2)/2)**2
    return (visible_numpy(state['centers'], scene.profile).all()
        and bool((state['widths']>=.3).all() and (state['widths']<=1.).all())
        and bool((np.sum((scene.rays[active]-rays[state['owners'][active]])**2, axis=1)<=limits+1e-14).all())
        and bool((np.bincount(state['owners'][active], minlength=len(state['ids']))>0).all()))


def _fit(p, state, beams, ledger):
    for b in beams:
        ledger.data['scan_input_count'] += len(state['owners'])
        width = envelope(p.scene, np.flatnonzero(state['owners']==b), state['centers'][b], ledger)
        if width is None:
            return False
        state['widths'][b] = width
    return True


def _accepted(p, state, ledger, operator):
    ledger.data['accepted'].append(dict(operator=operator, layout=_layout(state).to_record(),
        anchor_aircraft_ids=state['anchors'].copy(), optimization_key=list(_key(p.scene, state, ledger)),
        spend=dict(raypairs=ledger.data['raypairs'], proposals=ledger.data['proposals'],
                   refits=ledger.data['refits'], seconds=time.monotonic()-ledger.started)))
    # Transport only: persistence time remains inside the same nominal ledger.
    callback = getattr(ledger, 'on_accepted', None)
    if callback is not None:
        callback(ledger.data['accepted'][-1])


def _phase(p, state, operator, cap, ledger, *, partial=False):
    """Snapshot identities; skip obsolete entries without candidate geometry."""
    try:
        ledger.check()
    except Exhausted as exc:
        ledger.data['truncation_reason']=str(exc)
        ledger.data['phases'].append(dict(operator=operator,proposal_cap=cap,proposals=0,
                                         seconds=0.,stop_reason=str(exc)))
        return state
    scene = p.scene
    phase_start = ledger.data['proposals']
    started = time.monotonic()
    ledger.data['scan_input_count'] += len(scene.demands)*(len(state['ids'])+1)
    load = _loads(scene, state)
    groups = {bid: tuple(np.flatnonzero(state['owners']==b)) for b,bid in enumerate(state['ids'])}
    beam_order = sorted(state['ids'], key=lambda bid: (-load[state['ids'].index(bid)], bid))
    ledger.data['sort_input_count'] += len(beam_order)

    def live(bid):
        ledger.data['scan_input_count'] += len(state['ids'])
        return state['ids'].index(bid) if bid in state['ids'] else None

    def try_state(candidate, affected, *, repair=False):
        nonlocal state
        if not _fit(p, candidate, affected, ledger):
            return
        if not _legal(p, candidate, ledger, partial=partial, resources=not repair):
            return
        if repair:
            old = (max(0,len(state['ids'])-p.Ncap), max(0.,omega(state['widths'])-p.Omega_cap))
            new = (max(0,len(candidate['ids'])-p.Ncap), max(0.,omega(candidate['widths'])-p.Omega_cap))
            accept = lex_better(new, old, (0., 1e-12))
        else:
            new_key = _key(scene, candidate, ledger)
            old_key = _key(scene, state, ledger)
            accept = lex_better(new_key, old_key)
            q0 = getattr(ledger, 'q0', None)
            if q0 is not None:
                protected = q0_protected(new_key, q0)
                ledger.data.setdefault('post_trials', []).append(dict(operator=operator,
                    proposal_index=ledger.data['proposals'],candidate_key=list(new_key),
                    previous_key=list(old_key),fixed_Q0_key=list(q0),
                    strict_lex_better=bool(accept),Q0_protected=bool(protected)))
                accept = accept and protected
        if accept:
            state = candidate
            _accepted(p, state, ledger, operator)

    def debit(refit=False):
        if ledger.data['proposals']-phase_start >= cap:
            raise Exhausted('phase_proposal_cap')
        ledger.proposal(refit=refit)

    reason = None
    try:
        if operator == 'MOVE':
            dest = sorted(state['ids'], key=lambda bid: (load[state['ids'].index(bid)], bid))
            ledger.data['sort_input_count'] += len(dest)
            for source in beam_order:
                ledger.check()
                member_order = sorted(groups[source], key=lambda i:(-scene.demands[i],scene.aircraft_ids[i]))
                ledger.data['sort_input_count'] += len(member_order)
                for i in member_order:
                    ledger.check()
                    for target in dest:
                        ledger.check()
                        a,b = live(source), live(target)
                        if a is None or b is None or a==b or state['owners'][i]!=a or np.count_nonzero(state['owners']==a)<=1:
                            ledger.data['skipped_obsolete'] += 1
                            continue
                        debit()
                        candidate = _copy(state)
                        candidate['owners'][i] = b
                        try_state(candidate, (a,b))
        elif operator == 'REFIT':
            for bid in beam_order:
                ledger.check()
                b = live(bid)
                if b is None:
                    ledger.data['skipped_obsolete'] += 1
                    continue
                debit(refit=True)
                members = np.flatnonzero(state['owners']==b)
                ledger.data['mean_input_count'] += len(members)
                mean = scene.rays[members].sum(0)
                norm = np.linalg.norm(mean)
                if not np.isfinite(norm) or norm<1e-12:
                    continue
                mean /= norm
                ledger.geometry(len(members), 'refit_nearest_mean')
                distance = np.sum((scene.rays[members]-mean)**2, axis=1)
                anchor = min(range(len(members)), key=lambda j:(distance[j],scene.aircraft_ids[members[j]]))
                i = int(members[anchor])
                candidate = _copy(state)
                candidate['centers'][b] = scene.positions[i]
                candidate['anchors'][b] = scene.aircraft_ids[i]
                try_state(candidate, (b,))
        elif operator == 'SPLIT':
            for bid in beam_order:
                ledger.check()
                b = live(bid)
                if b is None or len(state['ids'])>=p.Ncap:
                    ledger.data['skipped_obsolete'] += 1
                    continue
                members = np.flatnonzero(state['owners']==b)
                if len(members)<2:
                    continue
                # Anchor construction is geometry, charged even when the later
                # partition is rejected. No refit/mean is hidden in splitting.
                ledger.geometry(len(members), 'split_anchor_a')
                ray = ledger.center_rays(state['centers'][b:b+1], scene.profile)[0]
                distances = np.sum((scene.rays[members]-ray)**2, axis=1)
                a = min(range(len(members)), key=lambda j:(-distances[j],scene.aircraft_ids[members[j]]))
                ai = int(members[a])
                ledger.geometry(len(members)-1, 'split_anchor_b')
                others = members[members!=ai]
                ds = np.sum((scene.rays[others]-scene.rays[ai])**2, axis=1)
                bi = int(others[min(range(len(others)),key=lambda j:(-ds[j],scene.aircraft_ids[others[j]]))])
                for partition in (1,2):
                    ledger.check()
                    if b!=live(bid):
                        ledger.data['skipped_obsolete'] += 1
                        continue
                    debit()
                    ledger.geometry(2*len(members), 'split_partition_distance')
                    da=np.sum((scene.rays[members]-scene.rays[ai])**2,axis=1)
                    db=np.sum((scene.rays[members]-scene.rays[bi])**2,axis=1)
                    choices = np.zeros(len(members),dtype=int)
                    choices[members==bi]=1
                    if partition==1:
                        choices=(db<da).astype(int)
                        choices[members==ai]=0
                        choices[members==bi]=1
                    else:
                        accum=[float(scene.demands[ai]),float(scene.demands[bi])]
                        order=sorted(range(len(members)),key=lambda j:(-scene.demands[members[j]],scene.aircraft_ids[members[j]]))
                        ledger.data['sort_input_count'] += len(order)
                        for j in order:
                            if members[j] in (ai,bi):continue
                            choice=min((0,1),key=lambda c:(accum[c],da[j] if c==0 else db[j],c))
                            choices[j]=choice;accum[choice]+=float(scene.demands[members[j]])
                    candidate=_copy(state)
                    k=len(candidate['ids'])
                    old=bid
                    trial=ledger.data['proposals']
                    candidate['ids'][b]=f'{old}/s{trial}/a'
                    candidate['ids'].append(f'{old}/s{trial}/b')
                    candidate['anchors'][b]=scene.aircraft_ids[ai]
                    candidate['anchors'].append(scene.aircraft_ids[bi])
                    candidate['centers'][b]=scene.positions[ai]
                    candidate['centers']=np.vstack((candidate['centers'],scene.positions[bi]))
                    candidate['widths']=np.append(candidate['widths'],.3)
                    candidate['owners'][members]=np.where(choices==0,b,k)
                    try_state(candidate,(b,k))
        elif operator in ('MERGE','RESOURCE_MERGE'):
            # Retain only cap nearest pairs in a bounded heap, not a B^2 table.
            heap=[]
            for a in range(len(state['ids'])):
                for b in range(a+1,len(state['ids'])):
                    ledger.geometry(1, 'merge_pair_ranking')
                    rays=ledger.center_rays(state['centers'][[a,b]],scene.profile)
                    angle=float(ray_angle_numpy(rays[0],rays[1]))
                    pair=tuple(sorted((state['ids'][a],state['ids'][b])))
                    ledger.data['sort_input_count'] += 2
                    item=(angle,pair)
                    # nsmallest over current cap+one keeps memory bounded; ties
                    # use the actual stable IDs, never hash/array ordering.
                    heap.append(item)
                    if len(heap)>cap:
                        ledger.data['sort_input_count'] += len(heap)
                        heap=heapq.nsmallest(cap,heap)
            pairs=sorted(heap)
            ledger.data['sort_input_count'] += len(pairs)
            for _,pair in pairs:
                ledger.check()
                for anchor_bid in pair:
                    a,b=live(pair[0]),live(pair[1])
                    if a is None or b is None:
                        ledger.data['skipped_obsolete'] += 1
                        continue
                    if operator=='RESOURCE_MERGE' and len(state['ids'])<=p.Ncap and omega(state['widths'])<=p.Omega_cap+1e-12:
                        break
                    debit()
                    chosen=live(anchor_bid)
                    candidate=_copy(state)
                    candidate['owners'][candidate['owners']==b]=a
                    candidate['centers'][a]=state['centers'][chosen]
                    candidate['anchors'][a]=state['anchors'][chosen]
                    candidate['ids'][a]=min(pair)
                    candidate=_compact(candidate)
                    target=candidate['ids'].index(min(pair))
                    try_state(candidate,(target,),repair=operator=='RESOURCE_MERGE')
        else:
            raise InterfaceError('unknown_phase')
    except Exhausted as exc:
        reason=str(exc)
        if reason!='phase_proposal_cap':ledger.data['truncation_reason']=reason
    ledger.data['phases'].append(dict(operator=operator,proposal_cap=cap,
        proposals=ledger.data['proposals']-phase_start,seconds=time.monotonic()-started,stop_reason=reason))
    return state


def _record_layout(p,state,audit,ledger):
    layout=_layout(state)
    record=layout.to_record()
    record.update(anchor_aircraft_ids=list(state['anchors']),layout_sha256=layout_sha256(layout))
    ledger.data['hash_calls'] += 1
    ledger.data['sort_input_count'] += len(state['ids'])
    return record,metrics(p.scene,layout,evidence=audit)


def build_teacher(p, *, phase_manifest, time_limit=30., on_accepted=None):
    validate_phase(p.scene, p.identity, phase_manifest)
    if not 0<=time_limit<=30.:raise InterfaceError('teacher_nominal_limit_exceeds_contract')
    n=len(p.scene.demands)
    ledger=Ledger(n,4096,time_limit,2048,16,reserve=1)
    ledger.on_accepted=on_accepted
    if p.status!='available':
        return dict(schema='TeacherResultV1',status='unavailable',reason='preparation_unavailable',
                    cost=ledger.record(),evidence_refs=[],provenance={**p.identity,'source_sha256':p.source_sha256})
    state=_state(p.Wenv,p.anchors)
    phases=(('MOVE',512),('SPLIT',512),('MOVE',512),('MERGE',240),('REFIT',16),('MOVE',256))
    for operator,cap in phases:
        if ledger.data['truncation_reason']:break
        state=_phase(p,state,operator,cap,ledger)
    ledger.terminal=True
    try:
        audit=validate_owned(p.scene,_layout(state),budget=ledger)
        if not audit['valid'] or len(state['ids'])>p.Ncap or omega(state['widths'])>p.Omega_cap+1e-12:
            raise Exhausted('teacher_final_audit_invalid')
        if not p.scene.demands.sum():
            raise Exhausted('zero_total_request_learning_ineligible')
        layout,quality=_record_layout(p,state,audit,ledger)
        result=dict(schema='TeacherResultV1',status='available',layout=layout,
            metrics=quality,audit=audit,**p.identity,source_sha256=p.source_sha256,
            source_method='T_bounded_geometric_request_constructor',evidence_refs=[])
    except Exhausted as exc:
        result=dict(schema='TeacherResultV1',status='unavailable',reason=str(exc),
                    provenance={**p.identity,'source_sha256':p.source_sha256},evidence_refs=[])
    result['cost']=ledger.record()
    return result


def physical_equivalent(scene,left,right,ledger=None):
    def groups(layout):
        if ledger is not None:
            ledger.data['sort_input_count'] += len(layout.centers)+len(layout.assignment)
        return sorted((tuple(float(x) for x in layout.centers[b]),float(layout.widths[b]),
            tuple(sorted(scene.aircraft_ids[i] for i in np.flatnonzero(layout.assignment==b))))
            for b in range(len(layout.centers)))
    a,b=groups(left),groups(right)
    return len(a)==len(b) and all(x[2]==y[2] and np.allclose(x[0],y[0],rtol=0,atol=1e-12)
        and abs(x[1]-y[1])<=1e-12 for x,y in zip(a,b))



def validate_phase(scene, identity, manifest):
    """A phase capability must bind the actual role and exact input identity."""
    _identity(scene, identity)
    if not isinstance(manifest, dict) or manifest.get('role') != identity['role']:
        raise InterfaceError('phase_role_mismatch')
    if manifest.get('stage') not in ('P0','labels','development','validation','test'):
        raise InterfaceError('phase_stage_invalid')
    expected={'P0':'development','labels':'train','development':'development','validation':'validation','test':'test'}
    if expected[manifest['stage']] != identity['role'] or not manifest.get('admitted',False):
        raise InterfaceError('phase_not_admitted')
    if manifest.get('identity_sha256') != identity['identity_sha256']:
        raise InterfaceError('phase_identity_sha_mismatch')
    cases=manifest.get('cases',{})
    entry=cases.get(identity['case_id'])
    if not isinstance(entry,dict) or any(entry.get(k)!=identity[k] for k in identity):
        raise InterfaceError('phase_case_binding_mismatch')
    if not isinstance(manifest.get('dataset_sha256'),str) or len(manifest['dataset_sha256'])!=64:
        raise InterfaceError('phase_dataset_sha_missing')


def q0_protected(key, q0):
    return all(key[j] <= q0[j]+absolute+1e-12*max(abs(key[j]),abs(q0[j]))
               for j,absolute in ((1,1e-6),(2,1e-9)))


def _sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf8')).hexdigest()


def _provenance(p, method, producer_sha256, evidence_refs):
    if not isinstance(producer_sha256,str) or not producer_sha256:
        raise InterfaceError('producer_sha_missing')
    return dict(producer_sha256=producer_sha256,source_method=method,
        prepared_source_sha256=p.source_sha256,case_identity=dict(p.identity),
        input_sha=p.scene.input_sha256,role_sha=p.identity['identity_sha256'],
        evidence_refs=list(evidence_refs or []))


def build_proposal(p, method, *, teacher=None, producer_sha256, evidence_refs=()):
    """Independent R5 builders. Never invokes any old proposer or optimizer."""
    from ..ai_search_r4.core import _teacher_checked
    started=time.monotonic()
    if method not in ('RULE','RANDOM','EXACT','OWNER10','JOINT10'):
        raise InterfaceError('source_method_invalid')
    if method in ('RULE','RANDOM') and teacher is not None:
        raise InterfaceError('teacher_forbidden_for_rule_random')
    provenance=_provenance(p,method,producer_sha256,evidence_refs)
    def unavailable(reason):
        return dict(schema='ProposalV2',status='unavailable',reason=reason,
            provenance=provenance,construction_cost=dict(seconds=time.monotonic()-started,raypairs=0))
    if p.status!='available':return unavailable('preparation_unavailable')
    scene=p.scene;n=len(scene.demands);pairs=hashes=sorts=0
    selected=[];anchor_changes=[]
    if method in ('EXACT','OWNER10','JOINT10'):
        if not _teacher_checked(p,teacher):return unavailable('unavailable_teacher')
        layout=teacher['layout'];slots=list(layout['beam_ids']);anchors=list(layout['anchor_aircraft_ids'])
        owners=np.asarray(layout['assignment'],int)
        scores=np.zeros((n,len(slots)),np.float32);scores[np.arange(n),owners]=1.
    elif method=='RULE':
        slots=list(p.raw_w.beam_ids);anchors=list(p.anchors)
        index={aid:i for i,aid in enumerate(scene.aircraft_ids)}
        rays=scene.rays[[index[aid] for aid in anchors]]
        pairs=n*len(anchors)
        scores=(-np.sum((scene.rays[:,None]-rays[None])**2,axis=-1)).astype(np.float32)
    else:
        anchors=sorted(scene.aircraft_ids,key=lambda aid:(digest(p.identity['case_id'],'anchor',aid),aid))[:len(p.anchors)]
        sorts+=n;hashes+=n
        slots=[f'random:{aid}' for aid in anchors]
        scores=np.asarray([[int.from_bytes(digest(p.identity['case_id'],'score',aid,slot)[:8],'big')/2**64
             for slot in slots] for aid in scene.aircraft_ids],np.float32);hashes+=n*len(slots)
    mask=np.ones(scores.shape,bool)
    if method in ('OWNER10','JOINT10'):
        selected=sorted(range(n),key=lambda i:(digest(p.identity['case_id'],'mask',scene.aircraft_ids[i]),scene.aircraft_ids[i]))[:math.floor(.1*n)]
        hashes+=n;sorts+=n
        for i in selected:mask[i,owners[i]]=False
    if method=='JOINT10':
        chosen=sorted(range(len(slots)),key=lambda j:(digest(p.identity['case_id'],'r5_anchor_corrupt',slots[j]),slots[j]))[:math.floor(.1*len(slots))]
        hashes+=len(slots);sorts+=len(slots)
        index={aid:i for i,aid in enumerate(scene.aircraft_ids)}
        for j in chosen:
            old=anchors[j];ai=index[old];other=[i for i in range(n) if i!=ai]
            if not other:
                anchor_changes.append(dict(slot_id=slots[j],old_anchor=old,new_anchor=old,reason='one_aircraft_no_distinct_anchor'));continue
            pairs+=len(other);sorts+=len(other)
            angles=ray_angle_numpy(scene.rays[other],scene.rays[ai])
            z=min(range(len(other)),key=lambda z:(angles[z],scene.aircraft_ids[other[z]]));ni=other[z]
            anchors[j]=scene.aircraft_ids[ni]
            anchor_changes.append(dict(slot_id=slots[j],old_anchor=old,new_anchor=anchors[j],angle_deg=float(angles[z])))
    return dict(schema='ProposalV2',status='ready',case_identity=dict(p.identity),
        input_sha=scene.input_sha256,role_sha=p.identity['identity_sha256'],provenance=provenance,
        slot_ids=slots,anchor_aircraft_ids=anchors,scores_n_by_K=scores.tolist(),mask_n_by_K=mask.tolist(),
        construction_cost=dict(seconds=time.monotonic()-started,raypairs=pairs,hash_calls=hashes,
            sort_input_count=sorts,scan_input_count=n,mask_selected_aircraft_ids=[scene.aircraft_ids[i] for i in selected],
            anchor_changes=anchor_changes))


def validate_proposal(p, proposal):
    if proposal is None:return 'typed_unavailable_missing_payload',None
    if not isinstance(proposal,dict) or proposal.get('schema')!='ProposalV2' or proposal.get('status') not in ('ready','unavailable'):
        raise InterfaceError('invalid_union_tag')
    prov=proposal.get('provenance')
    if not isinstance(prov,dict) or prov.get('case_identity')!=p.identity or prov.get('input_sha')!=p.scene.input_sha256 or prov.get('role_sha')!=p.identity['identity_sha256'] or prov.get('prepared_source_sha256')!=p.source_sha256 or not prov.get('producer_sha256'):
        raise InterfaceError('proposal_provenance_mismatch')
    if prov.get('source_method') not in ('RULE','RANDOM','EXACT','OWNER10','JOINT10','AI'):
        raise InterfaceError('source_method_invalid')
    if any('offset' in str(k).lower() for k in proposal):raise InterfaceError('offset_field_forbidden')
    if proposal['status']=='unavailable':
        if any(k in proposal for k in ('slot_ids','anchor_aircraft_ids','scores_n_by_K','mask_n_by_K','layout','case_identity','input_sha','role_sha')):
            raise InterfaceError('unavailable_has_payload')
        return proposal.get('reason','typed_unavailable'),None
    if proposal.get('case_identity')!=p.identity or proposal.get('input_sha')!=p.scene.input_sha256 or proposal.get('role_sha')!=p.identity['identity_sha256']:
        raise InterfaceError('SHA_or_role_mismatch')
    try:
        slots=proposal['slot_ids'];anchors=proposal['anchor_aircraft_ids'];k=len(slots)
        scores=np.asarray(proposal['scores_n_by_K'],np.float32);mask=np.asarray(proposal['mask_n_by_K'])
    except (KeyError,TypeError,ValueError) as exc:raise InterfaceError('array_field_invalid') from exc
    if any(not isinstance(x,str) or not x for x in slots) or len(set(slots))!=k:raise InterfaceError('duplicate_or_invalid_slot_ID')
    if len(anchors)!=k or any(a not in p.scene.aircraft_ids for a in anchors):raise InterfaceError('anchor_not_in_scene')
    n=len(p.scene.demands)
    if not k and scores.size==mask.size==0:scores=scores.reshape(n,0);mask=mask.astype(bool).reshape(n,0)
    if scores.shape!=(n,k) or mask.shape!=(n,k) or mask.dtype!=np.bool_:raise InterfaceError('array_shape_or_mask_type_mismatch')
    if not np.isfinite(scores).all():raise InterfaceError('NaN_or_Inf')
    if k>min(p.Ncap,200,n):return 'typed_proposal_count_exceeds_cap',None
    if k==0:return 'typed_no_active_slots',None
    if not mask.any():return 'typed_all_scores_unavailable',None
    return None,(list(slots),list(anchors),scores,mask)


def _phase_record(ledger, name, cap, start, started, reason):
    ledger.data['phases'].append(dict(operator=name,proposal_cap=cap,
        proposals=ledger.data['proposals']-start,seconds=time.monotonic()-started,stop_reason=reason))


def _evict(p,state,scores,mask,slots,ledger):
    start=ledger.data['proposals'];started=time.monotonic();reason=None;version=0;cache={}
    ledger.data['evictions']=[]
    try:
        while omega(state['widths'])>p.Omega_cap+1e-12:
            ledger.check();ranking=[]
            for b,bid in enumerate(state['ids']):
                members=np.flatnonzero(state['owners']==b);ledger.data['scan_input_count']+=len(state['owners'])
                key=(bid,state['anchors'][b],tuple(int(x) for x in members))
                if key in cache:
                    ledger.data['cache_hits']+=1;distance=cache[key]
                else:
                    ledger.geometry(len(members),'EVICT_ranking_angles')
                    ledger.data['rank_geometry_pairs']+=len(members)
                    ray=ledger.center_rays(state['centers'][b:b+1],p.scene.profile)[0]
                    distance=ray_angle_numpy(p.scene.rays[members],ray);cache[key]=distance
                ledger.data['sort_input_count']+=len(members)
                order=sorted(range(len(members)),key=lambda z:(-distance[z],p.scene.aircraft_ids[members[z]]))
                i=int(members[order[0]])
                if len(members)==1:new_omega=0.
                else:
                    required=2*float(distance[order[1]])
                    width=max(.3,float(np.nextafter(required,np.inf)))
                    if width>1. and required<=1.:width=1.
                    new_omega=omega([width])
                old_omega=omega([state['widths'][b]]);gain=old_omega-new_omega
                if gain<=1e-12+1e-12*max(abs(old_omega),abs(new_omega)):continue
                # Only an available score may participate in the ranking.
                j=slots.index(bid) if bid in slots else None
                score=float(scores[i,j]) if j is not None and mask[i,j] else float('inf')
                ranking.append(((-gain,score,bid,p.scene.aircraft_ids[i]),bid,i,gain))
            ledger.data['sort_input_count']+=len(ranking);ranking.sort(key=lambda x:x[0])
            accepted=False
            for _,bid,i,gain in ranking:
                if ledger.data['proposals']-start>=128:raise Exhausted('EVICT_phase_cap')
                ledger.proposal();b=state['ids'].index(bid);candidate=_copy(state);candidate['owners'][i]=-1
                candidate=_compact(candidate)
                fit=True if bid not in candidate['ids'] else _fit(p,candidate,(candidate['ids'].index(bid),),ledger)
                new_area=omega(candidate['widths']);old_area=omega(state['widths'])
                legal=fit and _legal(p,candidate,ledger,partial=True,resources=False)
                accept=legal and old_area-new_area>1e-12+1e-12*max(abs(old_area),abs(new_area))
                ledger.data['evictions'].append(dict(state_version=version,slot_id=bid,aircraft_id=p.scene.aircraft_ids[i],
                    estimated_gain=gain,actual_gain=old_area-new_area,accepted=bool(accept),proposal_index=ledger.data['proposals']))
                if accept:
                    state=candidate;version+=1;accepted=True;_accepted(p,state,ledger,'EVICT');break
            if not accepted:raise Exhausted('eviction_no_positive_legal_gain')
        reason='resources_repaired' if version else 'not_needed_no_resource_violation'
    except Exhausted as exc:
        reason=str(exc);ledger.data['truncation_reason']=reason
    _phase_record(ledger,'EVICT',128,start,started,reason)
    return state


def _insert(p,state,scores,mask,slots,ledger):
    start=ledger.data['proposals'];started=time.monotonic();reason=None;cache={}
    ledger.data['insertions']=[]
    try:
        orphans=sorted(np.flatnonzero(state['owners']<0),key=lambda i:(-p.scene.demands[i],p.scene.aircraft_ids[i]))
        ledger.data['sort_input_count']+=len(orphans);ledger.data['scan_input_count']+=len(state['owners'])
        for i in orphans:
            ledger.check();ranking=[];loads=_loads(p.scene,state)
            for b,bid in enumerate(state['ids']):
                key=(int(i),state['anchors'][b])
                if key in cache:ledger.data['cache_hits']+=1;angle=cache[key]
                else:
                    ledger.geometry(1,'INSERT_ranking_angle');ledger.data['rank_geometry_pairs']+=1
                    ray=ledger.center_rays(state['centers'][b:b+1],p.scene.profile)[0]
                    angle=float(ray_angle_numpy(p.scene.rays[i],ray));cache[key]=angle
                if angle>.5:continue
                required=max(.3,float(state['widths'][b]),float(np.nextafter(2*angle,np.inf)))
                if required>1. and 2*angle<=1.:required=1.
                if required>1.:continue
                delta=omega([required])-omega([state['widths'][b]])
                j=slots.index(bid) if bid in slots else None
                usable=j is not None and bool(mask[i,j])
                score=float(scores[i,j]) if usable else 0.
                ranking.append(((not usable,-score if usable else 0.,delta,float(loads[b]),bid),b))
            ledger.data['sort_input_count']+=len(ranking);ranking.sort(key=lambda x:x[0]);inserted=False
            for _,b in ranking:
                if ledger.data['proposals']-start>=256:raise Exhausted('INSERT_phase_cap')
                ledger.proposal();candidate=_copy(state);candidate['owners'][i]=b
                accept=_fit(p,candidate,(b,),ledger) and _legal(p,candidate,ledger,partial=True)
                ledger.data['insertions'].append(dict(aircraft_id=p.scene.aircraft_ids[i],slot_id=state['ids'][b],accepted=bool(accept),proposal_index=ledger.data['proposals']))
                if accept:state=candidate;inserted=True;_accepted(p,state,ledger,'INSERT');break
            if not inserted:
                if ledger.data['proposals']-start>=256:raise Exhausted('INSERT_phase_cap')
                ledger.proposal();candidate=_copy(state);b=len(candidate['ids']);bid=f'insert:{p.scene.aircraft_ids[i]}:{ledger.data["proposals"]}'
                candidate['ids'].append(bid);candidate['anchors'].append(p.scene.aircraft_ids[i])
                candidate['centers']=np.vstack((candidate['centers'],p.scene.positions[i]));candidate['widths']=np.append(candidate['widths'],.3);candidate['owners'][i]=b
                accept=_fit(p,candidate,(b,),ledger) and _legal(p,candidate,ledger,partial=True)
                ledger.data['insertions'].append(dict(aircraft_id=p.scene.aircraft_ids[i],slot_id=bid,accepted=bool(accept),proposal_index=ledger.data['proposals']))
                if not accept:raise Exhausted('unrepaired_orphan')
                state=candidate;_accepted(p,state,ledger,'INSERT_SINGLETON')
        reason='complete' if orphans else 'not_needed_no_orphans'
    except Exhausted as exc:
        reason=str(exc);ledger.data['truncation_reason']=reason
    _phase_record(ledger,'INSERT',256,start,started,reason)
    return state


def decode_groups(p,proposal,*,time_limit=5.,on_accepted=None):
    if not 0<=time_limit<=5.:raise InterfaceError('decoder_nominal_limit_exceeds_contract')
    ledger=Ledger(len(p.scene.demands),2048,time_limit,512,8,reserve=3);ledger.on_accepted=on_accepted
    reason,parsed=validate_proposal(p,proposal)
    if proposal is None or proposal.get('status')=='unavailable' or p.status!='available':
        return dict(schema='DecoderResultV2',status='unavailable',reason=reason or 'preparation_unavailable',
            cost=ledger.record(),**p.identity,evidence_refs=[])
    try:
        if reason:raise Exhausted(reason)
        slots,anchors,scores,mask=parsed;scene=p.scene;index={aid:i for i,aid in enumerate(scene.aircraft_ids)}
        centers=scene.positions[[index[a] for a in anchors]].copy();rays=ledger.center_rays(centers,scene.profile)
        ledger.geometry(len(scene.demands)*len(slots),'INITIAL_angles')
        angle=ray_angle_numpy(scene.rays[:,None,:],rays[None,:,:]);owners=np.full(len(scene.demands),-1,int)
        for i in range(len(scene.demands)):
            ledger.check();available=np.flatnonzero(mask[i] & (angle[i]<=.5));ledger.data['scan_input_count']+=len(slots)
            ledger.data['sort_input_count']+=len(available)
            if len(available):owners[i]=min(available,key=lambda b:(-float(scores[i,b]),angle[i,b],slots[b]))
        state=_compact(dict(ids=slots,anchors=anchors,centers=centers,widths=np.full(len(slots),.3),owners=owners))
        if not _fit(p,state,range(len(state['ids'])),ledger):raise Exhausted('initial_envelope_invalid')
        ledger.data['phases'].append(dict(operator='INITIAL',proposal_cap=0,proposals=0,seconds=time.monotonic()-ledger.started,stop_reason=None))
        ledger.data['initial']=dict(owners=state['owners'].tolist(),slot_ids=list(state['ids']),
            orphan_aircraft_ids=[scene.aircraft_ids[i] for i in np.flatnonzero(state['owners']<0)],N=len(state['ids']),Omega=omega(state['widths']))
        state=_evict(p,state,scores,mask,slots,ledger)
        if omega(state['widths'])>p.Omega_cap+1e-12:raise Exhausted('eviction_resource_repair_failed')
        state=_insert(p,state,scores,mask,slots,ledger)
        if np.any(state['owners']<0):raise Exhausted('unrepaired_orphans')
        if not _legal(p,state,ledger):raise Exhausted('Q0_audit_invalid')
        ledger.q0=tuple(_key(scene,state,ledger));ledger.data['q0']=dict(optimization_key=list(ledger.q0),layout=_layout(state).to_record(),anchor_aircraft_ids=list(state['anchors']))
        _accepted(p,state,ledger,'Q0')
        for op,cap in (('REFIT',8),('SPLIT',64),('MOVE',56)):
            if ledger.data['truncation_reason']:break
            state=_phase(p,state,op,cap,ledger)
        ledger.terminal=True
        if not _fit(p,state,range(len(state['ids'])),ledger):raise Exhausted('terminal_envelope_invalid')
        audit=validate_owned(scene,_layout(state),budget=ledger)
        if not audit['valid'] or len(state['ids'])>p.Ncap or omega(state['widths'])>p.Omega_cap+1e-12:
            raise Exhausted('terminal_audit_invalid')
        ledger.data['terminal_audit']=audit
        ledger.data['phases'].append(dict(operator='AUDIT',proposal_cap=0,proposals=0,seconds=0.,stop_reason=None))
        layout,quality=_record_layout(p,state,audit,ledger)
        result=dict(schema='DecoderResultV2',status='success',proposal_success=True,layout=layout,
            metrics=quality,audit=audit,reason=None,physical_equivalent_Wenv=physical_equivalent(scene,_layout(state),p.Wenv,ledger),**p.identity)
    except Exhausted as exc:
        reason=str(exc);ledger.data['truncation_reason']=reason;ledger.terminal=True
        try:
            audit=validate_owned(p.scene,p.Wenv,budget=ledger)
            if not audit['valid']:raise Exhausted('fallback_audit_invalid')
            layout,quality=_record_layout(p,_state(p.Wenv,p.anchors),audit,ledger)
            result=dict(schema='DecoderResultV2',status='fallback',proposal_success=False,reason=reason,
                layout=layout,metrics=quality,audit=audit,physical_equivalent_Wenv=True,**p.identity)
        except Exhausted as terminal:
            result=dict(schema='DecoderResultV2',status='unpublished',reason=str(terminal),original_failure=reason,proposal_success=False,**p.identity)
    result['cost']=ledger.record()
    return result


def decode_d4(p,proposal,*,contract_sha256,adapter_source_sha256,time_limit=5.):
    """V2 diagnostic to frozen V1 compatibility; old source remains untouched."""
    from ..ai_search_r4 import core as frozen
    started=time.monotonic();reason,parsed=validate_proposal(p,proposal)
    method=proposal['provenance']['source_method']
    if method not in ('OWNER10','JOINT10'):raise InterfaceError('D4_diagnostic_method_invalid')
    if reason:raise InterfaceError('D4_requires_ready_diagnostic')
    slots,anchors,scores,mask=parsed
    inner=dict(schema='ProposalV1_discriminated_union',status='ready',**p.identity,
        source_method='TEACHER_EXACT',source_sha256=p.source_sha256,slot_ids=slots,
        anchor_aircraft_ids=anchors,direction_offset_Kx3=np.zeros((len(slots),3)).tolist(),
        scores_nxK_float32_finite=scores.tolist(),score_dtype='float32',score_mask_nxK_bool=mask.tolist(),
        proposal_cost_ledger=proposal['construction_cost'])
    derivation=dict(schema='R5D4DerivationV1',diagnostic_method=method,contract_sha256=contract_sha256,
        producer_sha256=proposal['provenance']['producer_sha256'],adapter_source_sha256=adapter_source_sha256,
        old_prepare_teacher_refs=proposal['provenance']['evidence_refs'],prepared_source_sha256=p.source_sha256,
        proposal_before=proposal,proposal_after=inner,proposal_before_sha256=_sha(proposal),proposal_after_sha256=_sha(inner),
        conversion_reason='V1_compatibility_tag_only_actual_R5_diagnostic_arrays_zero_offset',
        conversion_cost=dict(seconds=time.monotonic()-started,raypairs=0,hash_calls=2))
    result=frozen.decode_groups(p,inner,time_limit=time_limit)
    result['derivation']=derivation
    return result
