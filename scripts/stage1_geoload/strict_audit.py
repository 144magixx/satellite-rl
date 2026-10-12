"""Read-only, independent geometry and immutable ZIP evidence audits for G16.

No worker, network, controller or downstream service code is imported here.
Unknown evidence is never converted to zero or a scientific failure.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import zipfile
import numpy as np


class AuditError(ValueError):
    pass


def _ijson():
    try:import ijson
    except ImportError as error:raise AuditError('streaming audit requires pinned local ijson==3.3.0') from error
    if ijson.__version__!='3.3.0':raise AuditError('streaming audit requires pinned ijson==3.3.0')
    return ijson


class _Journal:
    def __init__(self,archive,name,path):self.archive=archive;self.name=name;self.path=path
    def __iter__(self):
        parser=_ijson()
        with self.archive.zip.open(self.name) as raw:
            if self.path==('journal',):
                yield from parser.items(raw,'journal.item',use_float=True)
                return
            target='entries.item.journal.item';entry=-1;builder=None;depth=0
            for prefix,event,value in parser.parse(raw,use_float=True):
                if prefix=='entries.item' and event=='start_map':entry+=1
                if entry!=self.path[1]:continue
                if prefix==target and event=='start_map':builder=parser.ObjectBuilder();depth=0
                if builder is not None:
                    builder.event(event,value)
                    if event in ('start_map','start_array'):depth+=1
                    elif event in ('end_map','end_array'):depth-=1
                    if depth==0:yield builder.value;builder=None


def _read_case(archive,name):
    parser=_ijson();builder=parser.ObjectBuilder();skip=None
    with archive.zip.open(name) as raw:
        for prefix,event,value in parser.parse(raw,use_float=True):
            if skip is not None:
                if prefix==skip and event=='end_array':skip=None
                continue
            if event=='start_array' and prefix.rsplit('.',1)[-1] in ('journal','ranking_journal','pools','samples'):
                builder.event('start_array',None);builder.event('end_array',None);skip=prefix
            else:builder.event(event,value)
    record=builder.value
    if 'journal' in record:record['journal']=_Journal(archive,name,('journal',))
    for i,entry in enumerate(record.get('entries',[])):
        if 'journal' in entry:entry['journal']=_Journal(archive,name,('entries',i,'journal'))
    return record


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class JobZip:
    """Verify all members by bounded streaming before exposing any evidence."""
    def __init__(self, receipt, scientific_commit):
        self.receipt = receipt
        self.path = Path(receipt['local_path'])
        if receipt.get('verified') is not True or digest(self.path) != receipt['sha256']:
            raise AuditError('ZIP receipt/digest differs')
        if self.path.stat().st_size != receipt['bytes']:
            raise AuditError('ZIP byte count differs')
        self.zip = zipfile.ZipFile(self.path)
        try:
            names = self.zip.namelist()
            if len(names) != len(set(names)) or any(n.startswith('/') or '..' in n.split('/') for n in names):
                raise AuditError('unsafe or duplicate ZIP members')
            self.manifest = self.read('archive_manifest.json')
            m = self.manifest
            if m['group'] != 'S1-G16' or m['run_id'] != receipt['run_id'] or m['scientific_commit'] != scientific_commit:
                raise AuditError('archive identity/source differs')
            if set(names) != set(m['members']) | {'archive_manifest.json'}:
                raise AuditError('archive membership not closed')
            for name, ref in m['members'].items():
                h = hashlib.sha256(); count = 0
                with self.zip.open(name) as f:
                    for b in iter(lambda: f.read(1 << 20), b''):
                        h.update(b); count += len(b)
                if h.hexdigest() != ref['sha256'] or count != ref['bytes']:
                    raise AuditError('member SHA/size differs: ' + name)
            self.job = self.read('job/job.json')
            self.status = self.read('job/status.json')
            if self.job['run_id'] != receipt['run_id'] or self.job['scientific_commit'] != scientific_commit:
                raise AuditError('job source/identity differs')
        except Exception:
            self.zip.close(); raise

    def read(self, name):
        with self.zip.open(name) as f:
            return json.load(f)

    def read_case(self,name):
        """Only small fields remain resident; every journal event is replayed."""
        return _read_case(self,name)

    def ref(self, name):
        return dict(zip=str(self.path), member=name, **self.manifest['members'][name])

    def close(self):
        self.zip.close()


def load_metrics(loads):
    a = np.asarray(loads, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all() or (a < 0).any():
        raise AuditError('invalid beam demand')
    n = len(a); total = float(a.sum())
    if not n or total == 0:
        return dict(N=n, total_demand=total, P=None, T=None, Q=None, CV=None)
    z = a / a.mean()
    return dict(N=n, total_demand=total, P=float(z.max()),
                T=float(np.sort(z)[-max(1, math.ceil(.1*n)):].mean()),
                Q=float(np.mean(z*z)), CV=float(a.std()/a.mean()))


def compare_metrics(actual, claimed):
    for key in ('N', 'total_demand', 'P', 'T', 'Q', 'CV'):
        x, y = actual[key], claimed.get(key)
        if x is None or y is None:
            if x is not y: raise AuditError('null metric differs: ' + key)
        elif not math.isfinite(float(y)) or not math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-9):
            raise AuditError('metric differs: ' + key)


def _ground(locations):
    a = np.asarray(locations, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise AuditError('invalid coordinates')
    lat, lon = np.deg2rad(a.T)
    return 6371*np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def _visible(locations, surface, sat):
    a = np.asarray(locations)
    line = sat - surface
    sine = np.sum(line*surface, axis=1)/(np.linalg.norm(line, axis=1)*6371)
    return ((a[:,0] >= 25) & (a[:,0] <= 80) & (a[:,1] >= -40) & (a[:,1] <= 50)
            & (np.rad2deg(np.arcsin(np.clip(sine,-1,1))) >= 10))


def audit_layout(case, layout, claimed=None):
    """Recompute assigned-ray geometry algebraically, without production core."""
    if case.get('profile', 'europe_v1') != 'europe_v1': raise AuditError('unfrozen geometry profile')
    d = np.asarray(case['demands'], dtype=np.float64)
    p = np.asarray(case['positions'], dtype=np.float64)
    c = np.asarray(layout['centers'], dtype=np.float64).reshape(-1,2)
    raw = np.asarray(layout['assignment'])
    if d.ndim != 1 or len(p) != len(d) or not np.isfinite(d).all() or (d < 0).any():
        raise AuditError('invalid aircraft demands')
    n = len(c)
    if n > 200 or raw.shape != d.shape or not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():
        raise AuditError('N/unique assignment invalid')
    a = raw.astype(np.int64)
    if ((a < 0) | (a >= n)).any(): raise AuditError('unassigned aircraft')
    if n and (np.bincount(a,minlength=n)==0).any(): raise AuditError('empty beam')
    sat = 42164*np.array([np.cos(np.deg2rad(13)),np.sin(np.deg2rad(13)),0])
    ground, centers = _ground(p), _ground(c)
    if not _visible(p,ground,sat).all() or not _visible(c,centers,sat).all(): raise AuditError('visibility')
    rays, brays = ground-sat, centers-sat
    rays /= np.linalg.norm(rays,axis=1,keepdims=True)
    brays /= np.linalg.norm(brays,axis=1,keepdims=True)
    limit = 4*np.sin(np.deg2rad(.25)/2)**2
    if len(a) and (np.sum((rays-brays[a])**2,axis=1)>limit+1e-14).any(): raise AuditError('geometric coverage')
    loads = np.bincount(a,weights=d,minlength=n)
    if not np.isclose(loads.sum(),d.sum(),rtol=1e-12,atol=1e-6): raise AuditError('demand conservation')
    values = load_metrics(loads)
    if claimed is not None: compare_metrics(values,claimed)
    return values


def better(after, before, method='JOINT', tolerance=1e-9):
    keys = ('CV',) if method == '09-CV' else ('P','T','Q')
    for k in keys:
        x, y = after[k], before[k]
        if x is None or y is None: return False
        if x < y-tolerance: return True
        if x > y+tolerance: return False
    return False


def audit_search(case, row, *, compact=False,progress=None):
    result = audit_layout(case,row['layout'],row['metrics'])
    budget = row.get('budgets', row.get('budget', {}))
    journal = row.get('journal', [])
    current = row.get('initial_metrics')
    teacher_layout = row.get('input',{}).get('initial')
    teacher_stage=None;pending_teacher_layout=None
    if teacher_layout is not None:
        current = audit_layout(case,teacher_layout)
    limit = budget.get('seconds_limit',budget.get('limit_seconds'))
    previous = 0.; accepted = late = 0
    out = [];checkpoint_values={};seen=0
    started_audit=time.monotonic();last_progress=started_audit
    for i, event in enumerate(journal,1):
        seen=i
        if progress is not None and time.monotonic()-last_progress>=5:
            last_progress=time.monotonic();progress(dict(journal_count=seen,audit_elapsed_seconds=last_progress-started_audit))
        if teacher_layout is not None:
            stage=event.get('stage')
            if teacher_stage is not None and stage!=teacher_stage and pending_teacher_layout is not None:
                teacher_layout=pending_teacher_layout;pending_teacher_layout=None
            teacher_stage=stage
        seconds = event.get('elapsed_seconds',event.get('ended'))
        if seconds is None or not math.isfinite(seconds) or seconds < previous:
            raise AuditError('candidate chronology incomplete')
        previous = seconds
        values = event.get('candidate_metrics',event.get('metrics'))
        evidence = event.get('coverage_evidence',event.get('geometry',{}))
        if teacher_layout is not None and evidence.get('valid'):
            candidate = event.get('layout')
            if candidate is None:
                delta=event['delta']
                centers=np.asarray(teacher_layout['centers']).copy()
                assignment=np.asarray(teacher_layout['assignment']).copy()
                assignment[delta['users']]=delta['owners']
                if delta['beams']: centers[delta['beams']]=delta['centers']
                candidate=dict(centers=centers.tolist(),assignment=assignment.tolist())
            audit_layout(case,candidate,values)
        if values is not None:
            compare_metrics(load_metrics(evidence['loads']),values)
            if evidence.get('valid') is not True: raise AuditError('metrics on invalid candidate')
        islate = event.get('late',False) or (limit is not None and seconds > limit)
        commit = event.get('reason')=='accepted' or event.get('pool_committed',False)
        if islate:
            late += 1
            if commit or event.get('temporary_best'): raise AuditError('late candidate published')
        if commit:
            if values is None: raise AuditError('accepted missing metrics')
            if current is not None and row.get('method') != 'VARIABLE':
                if values['N'] != current['N'] or not better(values,current,row['method']):
                    raise AuditError('acceptance objective/N differs')
            current={key:values[key] for key in ('N','total_demand','P','T','Q','CV')};accepted += 1
            # Retain only the latest eligible state of each frozen reference node,
            # not all accepted metrics/evidence arrays across the full trajectory.
            for node_index,node in enumerate(row.get('reference_nodes',[])):
                sl=node.get('seconds_limit',node.get('limit') if node['kind']=='seconds' else None)
                cl=node.get('candidate_limit',node.get('limit') if node['kind']=='attempts' else None)
                if (sl is None or seconds<=sl) and (cl is None or event.get('attempt',i)<=cl):
                    checkpoint_values[node_index]=current
            # All trial deltas in a pool are relative to the pre-pool layout.
            # A best trial may occur before later rejected trials in that pool.
            if teacher_layout is not None: pending_teacher_layout=candidate
        if limit is not None:
            started=event.get('started_seconds',event.get('started'))
            if started is None or started>=limit:raise AuditError('candidate started beyond deadline or unknown start')
        if not compact:out.append(dict(attempt=i, seconds=seconds, accepted=commit, late=islate,
                        geometry_scope='per_aircraft' if event.get('candidate_layout') else 'worker_evidence_load_algebra'))
        if event.get('candidate_layout') and evidence.get('valid'):
            audit_layout(case,event['candidate_layout'],values if evidence.get('valid') else None)
    count = budget.get('attempts',budget.get('attempted'))
    if count is not None and count != seen: raise AuditError('candidate count differs')
    if count is not None and budget.get('candidate_limit') is not None and count > budget['candidate_limit']:
        raise AuditError('completed candidate cap exceeded')
    if budget.get('late_candidates',budget.get('late_completed_candidates',late)) != late:
        raise AuditError('late candidate ledger differs')
    if budget.get('accepted',accepted) != accepted: raise AuditError('accepted count differs')
    if current is not None: compare_metrics(result,current)
    for node_index,node in enumerate(row.get('reference_nodes',[])):
        audit_layout(case,node['layout'],node['metrics'])
        sl = node.get('seconds_limit', node.get('limit') if node['kind']=='seconds' else None)
        cl = node.get('candidate_limit', node.get('limit') if node['kind']=='attempts' else None)
        if sl is not None and node['state_seconds'] > sl: raise AuditError('future reference state')
        if cl is not None and node['state_attempts'] > cl: raise AuditError('future reference candidate')
        eligible=checkpoint_values.get(node_index)
        if eligible is not None: compare_metrics(node['metrics'],eligible)
        elif row.get('initial_metrics'): compare_metrics(node['metrics'],row['initial_metrics'])
    return result, dict(known_completed_candidates=seen,accepted=accepted,completed_late=late) if compact else out


def paired_bootstrap(rows, method, baseline, metric='P', draws=5000, seed=51001):
    """Four profiles stay together; resampling remains within mother-date strata."""
    groups = {}
    for row in rows:
        if row.get('outcome') != 'completed' or row.get('method') not in (method,baseline): continue
        key = (row['mother_id'],row['date'])
        profile=groups.setdefault(key,{}).setdefault(row['demand_profile'],{})
        if row['method'] in profile:raise AuditError('duplicate paired mother/profile/method')
        profile[row['method']]=row
    strata = {}
    for (mother,date), profiles in groups.items():
        if set(profiles) != {'base','one_hotspot','two_hotspots','edge_heavy'}: raise AuditError('paired profiles incomplete')
        diffs = []
        for values in profiles.values():
            if set(values) != {method,baseline}:raise AuditError('paired methods incomplete')
            a,b=values[method],values[baseline]
            if a.get('N') is None or a.get('N')!=b.get('N') or a.get('case_id') is None or a.get('case_id')!=b.get('case_id'):
                raise AuditError('paired methods case identity/N differs')
            if b[metric] in (None,0) or a[metric] is None:
                raise AuditError('paired methods/denominator incomplete')
            diffs.append((b[metric]-a[metric])/b[metric])
        strata.setdefault(date,[]).append(float(np.mean(diffs)))
    if not strata: raise AuditError('no complete mother pairs')
    rng=np.random.default_rng(seed); samples=np.zeros(draws); n=sum(map(len,strata.values()))
    for values in strata.values():
        a=np.asarray(values); samples += a[rng.integers(len(a),size=(draws,len(a)))].sum(axis=1)/n
    return dict(method=method,baseline=baseline,metric=metric,mothers=n,draws=draws,
                mean=float(sum(map(sum,strata.values()))/n),lower95=float(np.quantile(samples,.025)),
                upper95=float(np.quantile(samples,.975)),comparison_family='MLM_vs_JOINT_rule_random',
                multiplicity='frozen_plan_has_no_correction; pairwise_exploratory_only',
                broad_significance_claim_allowed=False,single_training_seed=51001)
