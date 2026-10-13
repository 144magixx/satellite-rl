"""Frozen B0 geometry kernel. All requests stay whole; no native optimizer.

Only the two offline proposal builders accept teacher metadata. A decoder sees
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
                        sort_input_count=0, hash_calls=0, cache_hits=0, rank_geometry_pairs=0,
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


def envelope(scene, members, center, ledger):
    """Fixed-center true-ray envelope; an oversized group is never clipped."""
    members = np.asarray(members, dtype=int)
    ledger.geometry(0, 'envelope_visibility')
    center = np.asarray(center, dtype=np.float64)
    if not len(members) or not visible_numpy(center[None], scene.profile).all():
        return None
    ray = ledger.center_rays(center[None], scene.profile)[0]
    ledger.geometry(len(members), 'envelope_member_angles')
    angles = ray_angle_numpy(scene.rays[members], ray)
    required = 2*float(angles.max())
    if required > 1.:
        return None
    width = max(.3, float(np.nextafter(required, np.inf)))
    if width > 1.:
        # At an exact 1-degree boundary the independent chord predicate decides.
        width = 1.
        limit = 4*np.sin(np.deg2rad(.5)/2)**2
        if not (np.sum((scene.rays[members]-ray)**2, axis=1) <= limit+1e-14).all():
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
    if identity['role'] != 'development' or identity['input_sha256'] != scene.input_sha256:
        raise InterfaceError('SHA_or_role_mismatch')
    if any(not isinstance(identity[k], str) or not identity[k] for k in needed):
        raise InterfaceError('identity_must_be_original_nonempty_strings')


def prepare_case(scene, identity, source_sha256, *, time_limit=10.):
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
    load = _loads(scene, state)
    groups = {bid: tuple(np.flatnonzero(state['owners']==b)) for b,bid in enumerate(state['ids'])}
    beam_order = sorted(state['ids'], key=lambda bid: (-load[state['ids'].index(bid)], bid))
    ledger.data['sort_input_count'] += len(beam_order)

    def live(bid):
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
            accept = lex_better(_key(scene, candidate, ledger), _key(scene, state, ledger))
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


def build_teacher(p, *, time_limit=30., on_accepted=None):
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


def _unavailable_proposal(p,method,reason,cost=None):
    return dict(schema='ProposalV1_discriminated_union',status='unavailable',reason=reason,
        provenance={**p.identity,'source_method':method,'source_sha256':p.source_sha256},
        cost=cost or dict(seconds=0.,raypairs=0),evidence_refs=[])


def _teacher_checked(p,teacher):
    if teacher is None:return False
    if not isinstance(teacher,dict) or teacher.get('status') not in ('available','unavailable'):
        raise InterfaceError('invalid_teacher_status_tag')
    if teacher['status']=='unavailable':
        if any(k in teacher for k in ('layout','metrics','anchor_aircraft_ids','assignment','centers','widths_deg')):
            raise InterfaceError('unavailable_teacher_has_layout')
        _check_unavailable_provenance(p,teacher)
        return False
    for key,value in p.identity.items():
        if teacher.get(key)!=value:raise InterfaceError('teacher_SHA_or_role_mismatch')
    if teacher.get('source_sha256')!=p.source_sha256:raise InterfaceError('teacher_source_sha_mismatch')
    if 'layout' not in teacher:raise InterfaceError('available_teacher_layout_missing')
    try:
        layout=Layout.from_record(teacher['layout'])
        anchors=teacher['layout']['anchor_aircraft_ids']
    except (ValueError,KeyError,TypeError) as exc:raise InterfaceError('teacher_layout_fields_invalid') from exc
    if (len(layout.assignment)!=len(p.scene.demands) or not len(layout.centers)
        or len(layout.centers)>p.Ncap or len(anchors)!=len(layout.centers)
        or layout.widths.shape!=(len(layout.centers),)
        or not np.isfinite(layout.centers).all() or not np.isfinite(layout.widths).all()
        or any(a not in p.scene.aircraft_ids for a in anchors)
        or not ((layout.assignment>=0)&(layout.assignment<len(layout.centers))).all()
        or not teacher.get('audit',{}).get('valid')
        or not teacher.get('metrics',{}).get('total_demand',0)>0):
        raise InterfaceError('teacher_layout_or_audit_invalid')
    return True


def _check_unavailable_provenance(p,value):
    provenance=value.get('provenance')
    if not isinstance(provenance,dict):
        raise InterfaceError('unavailable_provenance_missing')
    for key,expected in {**p.identity,'source_sha256':p.source_sha256}.items():
        if provenance.get(key)!=expected:
            raise InterfaceError('unavailable_SHA_or_role_mismatch')


def build_proposal(p,method,*,teacher=None):
    started=time.monotonic()
    if method not in ('RULE','RANDOM','TEACHER_EXACT','TEACHER_PERTURBED'):
        raise InterfaceError('source_method_invalid')
    if method in ('RULE','RANDOM') and teacher is not None:
        raise InterfaceError('teacher_forbidden_for_rule_random')
    if p.status!='available':return _unavailable_proposal(p,method,'preparation_unavailable')
    scene=p.scene;n=len(scene.demands)
    pair_count=0
    if method.startswith('TEACHER'):
        if not _teacher_checked(p,teacher):
            return _unavailable_proposal(p,method,'unavailable_teacher',dict(seconds=time.monotonic()-started,raypairs=0))
        layout=teacher['layout']
        slots=list(layout['beam_ids']);anchors=list(layout['anchor_aircraft_ids'])
        owners=np.asarray(layout['assignment'],int)
        scores=np.zeros((n,len(slots)),dtype=np.float32)
        scores[np.arange(n),owners]=1.
    elif method=='RULE':
        slots=list(p.raw_w.beam_ids);anchors=list(p.anchors)
        index={aid:i for i,aid in enumerate(scene.aircraft_ids)}
        rays=scene.rays[[index[aid] for aid in anchors]]
        pair_count=n*len(anchors)
        scores=(-np.sum((scene.rays[:,None]-rays[None])**2,axis=-1)).astype(np.float32)
    else:
        anchors=sorted(scene.aircraft_ids,key=lambda aid:(digest(p.identity['case_id'],'anchor',aid),aid))[:len(p.anchors)]
        slots=[f'random:{aid}' for aid in anchors]
        scores=np.array([[int.from_bytes(digest(p.identity['case_id'],'score',aid,slot)[:8],'big')/2**64
                         for slot in slots] for aid in scene.aircraft_ids],dtype=np.float32)
    mask=np.ones(scores.shape,dtype=bool)
    offset=np.zeros((len(slots),3),dtype=float)
    selected=[]
    if method=='TEACHER_PERTURBED':
        index={aid:i for i,aid in enumerate(scene.aircraft_ids)}
        for b,aid in enumerate(anchors):
            r=scene.rays[index[aid]]
            axis=np.eye(3)[int(np.argmin(np.abs(r)))]
            u=axis-float(axis@r)*r;u/=np.linalg.norm(u)
            v=np.cross(r,u)
            direction=digest(p.identity['case_id'],'direction',slots[b])[0]%4
            offset[b]=(u,v,-u,-v)[direction]
        selected=sorted(range(n),key=lambda i:(digest(p.identity['case_id'],'mask',scene.aircraft_ids[i]),scene.aircraft_ids[i]))[:math.floor(.1*n)]
        for i in selected:mask[i,owners[i]]=False
    return dict(schema='ProposalV1_discriminated_union',status='ready',**p.identity,
        source_method=method,source_sha256=p.source_sha256,slot_ids=slots,
        anchor_aircraft_ids=anchors,direction_offset_Kx3=offset.tolist(),
        scores_nxK_float32_finite=scores.tolist(),score_dtype='float32',score_mask_nxK_bool=mask.tolist(),
        proposal_cost_ledger=dict(seconds=time.monotonic()-started,raypairs=pair_count,
            sort_input_count=n if method in ('RANDOM','TEACHER_PERTURBED') else 0,
            hash_calls=n+n*len(slots) if method=='RANDOM' else n+len(slots) if method=='TEACHER_PERTURBED' else 0,
            mask_selected_aircraft_ids=[scene.aircraft_ids[i] for i in selected]))


def validate_proposal(p,proposal):
    """Inspect union tag first. Hard schema errors never become W fallbacks."""
    if proposal is None:return 'typed_unavailable_missing_payload',None
    if not isinstance(proposal,dict) or proposal.get('status') not in ('ready','unavailable'):
        raise InterfaceError('invalid_status_tag')
    if proposal['status']=='unavailable':
        if any(k in proposal for k in ('slot_ids','anchor_aircraft_ids','scores_nxK_float32_finite',
                                      'score_mask_nxK_bool','direction_offset_Kx3','layout')):
            raise InterfaceError('unavailable_has_layout_arrays')
        _check_unavailable_provenance(p,proposal)
        if proposal['provenance'].get('source_method') not in ('RULE','RANDOM','TEACHER_EXACT','TEACHER_PERTURBED'):
            raise InterfaceError('source_method_invalid')
        return proposal.get('reason','typed_unavailable'),None
    for key,value in p.identity.items():
        if proposal.get(key)!=value:raise InterfaceError('SHA_or_role_mismatch')
    if proposal.get('source_sha256')!=p.source_sha256:raise InterfaceError('source_sha_mismatch')
    method=proposal.get('source_method')
    if method not in ('RULE','RANDOM','TEACHER_EXACT','TEACHER_PERTURBED'):
        raise InterfaceError('source_method_invalid')
    try:
        slots=proposal['slot_ids'];anchors=proposal['anchor_aircraft_ids'];k=len(slots)
        scores=np.asarray(proposal['scores_nxK_float32_finite'],dtype=np.float32)
        rawmask=np.asarray(proposal['score_mask_nxK_bool'])
        offset=np.asarray(proposal['direction_offset_Kx3'],dtype=np.float64)
    except (KeyError,TypeError,ValueError) as exc:raise InterfaceError('array_field_invalid') from exc
    if any(not isinstance(x,str) or not x for x in slots) or len(set(slots))!=k:
        raise InterfaceError('duplicate_or_invalid_slot_ID')
    if len(anchors)!=k or any(a not in p.scene.aircraft_ids for a in anchors):
        raise InterfaceError('anchor_not_in_scene')
    n=len(p.scene.demands)
    # Empty JSON arrays are reshaped only for the explicitly allowed K=0 union.
    if k==0 and scores.size==rawmask.size==offset.size==0:
        scores=scores.reshape(n,0);rawmask=rawmask.astype(bool).reshape(n,0);offset=offset.reshape(0,3)
    if scores.shape!=(n,k) or rawmask.shape!=(n,k) or offset.shape!=(k,3) or rawmask.dtype!=np.bool_:
        raise InterfaceError('array_shape_or_mask_type_mismatch')
    if not np.isfinite(scores).all() or not np.isfinite(offset).all():raise InterfaceError('NaN_or_Inf')
    if method!='TEACHER_PERTURBED' and np.any(offset):raise InterfaceError('direction_offset_forbidden')
    if k>p.Ncap or k>200:return 'typed_proposal_count_exceeds_cap',None
    if k==0:return 'typed_no_active_slots',None
    if not rawmask.any():return 'typed_all_scores_unavailable',None
    return None,(slots,anchors,scores,rawmask,offset)


def _insert(p,state,scores,mask,original_slots,ledger):
    scene=p.scene
    cache={}
    start=ledger.data['proposals'];started=time.monotonic()
    try:
        ledger.check()
    except Exhausted as exc:
        ledger.data['truncation_reason']=str(exc)
        ledger.data['phases'].append(dict(operator='INSERT',proposal_cap=256,proposals=0,
                                         seconds=0.,stop_reason=str(exc)))
        return state
    orphans=sorted(np.flatnonzero(state['owners']<0),key=lambda i:(-scene.demands[i],scene.aircraft_ids[i]))
    ledger.data['sort_input_count']+=len(orphans)
    reason=None
    try:
        for i in orphans:
            ledger.check()
            loads=_loads(scene,state)
            ranking=[]
            for b,bid in enumerate(state['ids']):
                key=(scene.aircraft_ids[i],np.asarray(state['centers'][b],dtype=np.float64).tobytes())
                if key in cache:
                    ledger.data['cache_hits']+=1;angle=cache[key]
                else:
                    ledger.geometry(1,'INSERT_ranking_distance')
                    ledger.data['rank_geometry_pairs']+=1
                    ray=ledger.center_rays(state['centers'][b:b+1],scene.profile)[0]
                    angle=float(ray_angle_numpy(scene.rays[i],ray));cache[key]=angle
                required=max(float(state['widths'][b]),2*angle,.3)
                delta=omega([required])-omega([state['widths'][b]]) if required<=1 else float('inf')
                if bid in original_slots:
                    j=original_slots.index(bid);masked=not mask[i,j];score=float(scores[i,j])
                else:masked=True;score=0.
                ranking.append(((masked,-score if not masked else 0.,delta,float(loads[b]),bid),b))
            ledger.data['sort_input_count']+=len(ranking)
            ranking.sort(key=lambda item:item[0])
            inserted=False
            for _,b in ranking:
                if ledger.data['proposals']-start>=256:raise Exhausted('INSERT_phase_cap')
                ledger.proposal()
                candidate=_copy(state);candidate['owners'][i]=b
                if _fit(p,candidate,(b,),ledger) and _legal(p,candidate,ledger,partial=True):
                    state=candidate;inserted=True;_accepted(p,state,ledger,'INSERT');break
            if not inserted:
                if ledger.data['proposals']-start>=256:raise Exhausted('INSERT_phase_cap')
                ledger.proposal()
                candidate=_copy(state);b=len(candidate['ids'])
                candidate['ids'].append(f'insert:{scene.aircraft_ids[i]}:{ledger.data["proposals"]}')
                candidate['anchors'].append(scene.aircraft_ids[i])
                candidate['centers']=np.vstack((candidate['centers'],scene.positions[i]))
                candidate['widths']=np.append(candidate['widths'],.3)
                candidate['owners'][i]=b
                if _fit(p,candidate,(b,),ledger) and _legal(p,candidate,ledger,partial=True):
                    state=candidate;_accepted(p,state,ledger,'INSERT_SINGLETON')
                else:raise Exhausted('unrepaired_orphan')
    except Exhausted as exc:
        reason=str(exc);ledger.data['truncation_reason']=reason
    ledger.data['phases'].append(dict(operator='INSERT',proposal_cap=256,
        proposals=ledger.data['proposals']-start,seconds=time.monotonic()-started,stop_reason=reason))
    return state


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


def decode_groups(p,proposal,*,time_limit=5.):
    if not 0<=time_limit<=5.:raise InterfaceError('decoder_nominal_limit_exceeds_contract')
    n=len(p.scene.demands)
    ledger=Ledger(n,2048,time_limit,512,8,reserve=2)
    reason,parsed=validate_proposal(p,proposal)
    # Missing offline labels retain their slot and cost, and do zero decoding.
    if reason in ('unavailable_teacher','typed_unavailable_missing_payload') or (proposal is not None and proposal.get('status')=='unavailable'):
        return dict(schema='DecoderResultV1',status='unavailable',reason=reason,
                    cost=ledger.record(),**p.identity,evidence_refs=[])
    if p.status!='available':
        return dict(schema='DecoderResultV1',status='unavailable',reason='preparation_unavailable',cost=ledger.record(),**p.identity)
    state=None;projection=None
    try:
        if reason:raise Exhausted(reason)
        slots,anchors,scores,mask,offset=parsed
        scene=p.scene;index={aid:i for i,aid in enumerate(scene.aircraft_ids)}
        centers=[]
        for b,aid in enumerate(anchors):
            ledger.check();i=index[aid]
            if np.any(offset[b]):
                direction=offset[b]
                if not np.isclose(np.linalg.norm(direction),1.,rtol=0,atol=1e-10) or abs(float(direction@scene.rays[i]))>1e-10:
                    raise InterfaceError('direction_not_unit_tangent')
                delta=np.deg2rad(.01)
                center=ray_location(scene,np.cos(delta)*scene.rays[i]+np.sin(delta)*direction)
                if center is None:raise Exhausted('typed_proposal_invisible')
                centers.append(center)
            else:centers.append(scene.positions[i])
        centers=np.asarray(centers,dtype=float)
        rays=ledger.center_rays(centers,scene.profile)
        ledger.geometry(n*len(slots),'INITIAL_distances')
        chord=np.sum((scene.rays[:,None]-rays[None])**2,axis=-1)
        owners=np.full(n,-1,dtype=int)
        for i in range(n):
            ledger.check()
            available=np.flatnonzero(mask[i])
            ledger.data['sort_input_count']+=len(available)
            if len(available):
                b=min(available,key=lambda b:(-float(scores[i,b]),chord[i,b],slots[b]))
                if chord[i,b]<=4*np.sin(np.deg2rad(.5)/2)**2+1e-14:owners[i]=b
        state=_compact(dict(ids=list(slots),anchors=list(anchors),centers=centers,
                            widths=np.full(len(slots),.3),owners=owners))
        if not _fit(p,state,range(len(state['ids'])),ledger):raise Exhausted('initial_envelope_invalid')
        ledger.data['initial']=dict(orphan_count=int(np.count_nonzero(state['owners']<0)),N=len(state['ids']),Omega=omega(state['widths']))
        if len(state['ids'])>p.Ncap or omega(state['widths'])>p.Omega_cap+1e-12:
            state=_phase(p,state,'RESOURCE_MERGE',64,ledger,partial=True)
            if len(state['ids'])>p.Ncap or omega(state['widths'])>p.Omega_cap+1e-12:
                raise Exhausted('resource_repair_failed')
        else:
            ledger.data['phases'].append(dict(operator='RESOURCE_MERGE',proposal_cap=64,
                proposals=0,seconds=0.,stop_reason='not_needed_no_resource_violation'))
        if np.any(state['owners']<0):state=_insert(p,state,scores,mask,list(slots),ledger)
        else:
            ledger.data['phases'].append(dict(operator='INSERT',proposal_cap=256,
                proposals=0,seconds=0.,stop_reason='not_needed_no_orphans'))
        if np.any(state['owners']<0):raise Exhausted('unrepaired_orphans')
        for operator,cap in (('REFIT',8),('SPLIT',64),('MOVE',120)):
            if ledger.data['truncation_reason']:break
            state=_phase(p,state,operator,cap,ledger)
        # The reserved fixed-anchor batch is permitted after a nominal cutoff;
        # no new search operation/refit is started in this terminal section.
        ledger.terminal=True
        projection_started=time.monotonic()
        projection_pairs=ledger.data['raypairs']
        pre=_layout(state).to_record()
        pre_key=list(_key(scene,state,ledger))
        state['centers']=scene.positions[[index[aid] for aid in state['anchors']]].copy()
        if not _fit(p,state,range(len(state['ids'])),ledger):
            raise Exhausted('typed_final_anchor_projection_failure')
        if len(state['ids'])>p.Ncap or omega(state['widths'])>p.Omega_cap+1e-12:
            projection=dict(pre=pre,post=_layout(state).to_record(),pre_key=pre_key,
                post_key=list(_key(scene,state,ledger)),status='invalid_resources',
                seconds=time.monotonic()-projection_started,
                raypairs=ledger.data['raypairs']-projection_pairs,Ncap=p.Ncap,Omega_cap=p.Omega_cap,
                proposals_added=0,refits_added=0)
            raise Exhausted('typed_final_anchor_projection_failure')
        audit=validate_owned(scene,_layout(state),budget=ledger)
        projection=dict(pre=pre,post=_layout(state).to_record(),audit=audit,
            pre_key=pre_key,post_key=list(_key(scene,state,ledger)),status='valid' if audit['valid'] else 'invalid_geometry',
            seconds=time.monotonic()-projection_started,raypairs=ledger.data['raypairs']-projection_pairs,
            Ncap=p.Ncap,Omega_cap=p.Omega_cap,proposals_added=0,refits_added=0)
        if not audit['valid'] or len(state['ids'])>p.Ncap or omega(state['widths'])>p.Omega_cap+1e-12:
            raise Exhausted('typed_final_anchor_projection_failure')
        layout,quality=_record_layout(p,state,audit,ledger)
        result=dict(schema='DecoderResultV1',status='success',proposal_success=True,
            layout=layout,metrics=quality,audit=audit,reason=None,
            physical_equivalent_Wenv=physical_equivalent(scene,_layout(state),p.Wenv,ledger),
            final_anchor_projection=projection,**p.identity)
    except Exhausted as exc:
        reason=str(exc)
        ledger.data['truncation_reason']=reason
        # Cached Wenv publication still needs a new independent owner audit.
        # The remaining terminal allowance covers this audit even if projection
        # itself consumed n envelope pairs but failed before its owner audit.
        ledger.terminal=True
        fallback=_state(p.Wenv,p.anchors)
        try:
            audit=validate_owned(p.scene,p.Wenv,budget=ledger)
            if not audit['valid']:raise Exhausted('fallback_audit_invalid')
            layout,quality=_record_layout(p,fallback,audit,ledger)
            result=dict(schema='DecoderResultV1',status='fallback',proposal_success=False,
                reason=reason,layout=layout,metrics=quality,audit=audit,
                physical_equivalent_Wenv=True,final_anchor_projection=projection,**p.identity)
        except Exhausted as terminal:
            result=dict(schema='DecoderResultV1',status='unpublished',reason=str(terminal),
                original_failure=reason,proposal_success=False,**p.identity)
    result['cost']=ledger.record()
    return result
