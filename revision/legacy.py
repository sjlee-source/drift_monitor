"""Bridge to the original (frozen) simulation/run.py for bit-exact reproduction tests.

``legacy_batch`` regenerates the archived E2 streams with the original numpy seeding
(SeedSequence([role_seed, 20260922, 2, condition, replicate])) and wraps them as a
StreamBatch, so the new monitors can be checked against archived operating-map counts.
The original module is imported read-only from simulation/run.py (not modified).
"""
import csv
import importlib.util
from pathlib import Path
import numpy as np

from .generators import StreamBatch

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / 'supplementary' / 'evidence' / 'simulation'
_run = None


def original_module():
    global _run
    if _run is None:
        spec = importlib.util.spec_from_file_location('drift_original_run', REPO / 'simulation' / 'run.py')
        _run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_run)
    return _run


def conditions():
    rows = list(csv.DictReader((EVIDENCE / 'E2_production_conditions.csv').open()))
    for c in rows:
        for k, v in list(c.items()):
            try:
                c[k] = float(v) if '.' in v else int(v)
            except (ValueError, TypeError):
                pass
    return {c['condition_id']: c for c in rows}


def archived_thresholds():
    import json
    return json.loads((EVIDENCE / 'E2' / 'thresholds.json').read_text())


def cal_key(c):
    return f"pi{c['prevalence']}_q{c['audit_rate']}_d{c['label_delay']}_H{c['horizon']}"


def legacy_batch(c, role='final', n=2000, start=0):
    run = original_module()
    H = int(c['horizon'])
    org, arr, loss, nd, nm, na = [], [], [], [], [], []
    for i in range(start, start + n):
        x, origin, arrival, request, harm, all_def, totmiss, pm = run.generate(2, c, role, i)
        org.append(origin.astype(np.int64)); arr.append(arrival.astype(np.int64)); loss.append(x.astype(np.int8))
        nd.append(all_def); nm.append(totmiss); na.append(len(request))
    offs = np.zeros(n + 1, np.int64); offs[1:] = np.cumsum([len(o) for o in org])
    cat = lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt)
    lo = cat(loss, np.int8)
    return StreamBatch(H=H, offsets=offs, origin=cat(org, np.int64), arrival=cat(arr, np.int64), loss=lo,
                       true_loss=lo.copy(), prob=np.full(len(lo), float(c['audit_rate'])), tiebreak=np.full(n, 0.5),
                       n_audits=np.array(na, np.int64), n_defects=np.array(nd, np.int64), n_misses=np.array(nm, np.int64),
                       n_informative=np.diff(offs), checkpoints=np.zeros(0, np.int64),
                       counts_at=np.zeros((n, 0, 3), np.int64), meta=dict(role=role, legacy=True))


def original_trajectories(x):
    """Original vectorized statistics [betting, cusum.12, cusum.20, msr] for one stream."""
    return original_module().trajectory(x)


def original_periodic(x, arrival, H):
    return original_module().periodic(x, arrival, H)
