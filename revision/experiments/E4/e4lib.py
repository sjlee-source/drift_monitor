"""Shared helpers for E4 real-data replay (public Pfab AOI data).  New module; library files untouched."""
import numpy as np
from revision import generators as G, monitors as M, calibration as C

MASTER = 20261111
EXP = 14
ROLE_REPLAY = 6
MON_NAMES = ['cusum_2x', 'cusum_4x', 'betting', 'msr', 'esr', 'ewma', 'winbin']


def monitors(r0):
    """protocol.json 'monitors' (parameters scaled with r0; r0=0.10 -> library defaults)."""
    v = r0 * (1 - r0)
    return {
        'cusum_2x': M.CUSUM(p1=2 * r0, r0=r0),
        'cusum_4x': M.CUSUM(p1=4 * r0, r0=r0),
        'betting': M.Betting(r0, gain=0.36 / v, cap=0.9 / r0, window=50),
        'msr': M.MixtureSR(r0, stakes=np.array([.02, .05, .10, .20, .40]) / v),
        'esr': M.EDetector(r0, kind='SR', alpha=1e-3, min_gap=r0, increment='betting'),
        'ewma': M.EWMA(r0, lam=0.05),
        'winbin': M.WindowBinomial(r0, window=200),
    }


def cond_id(site, ti, qi, di, mode, var):
    return site * 100000 + ti * 10000 + qi * 1000 + di * 100 + mode * 10 + var


def replay_rng(cond, rep):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=MASTER, spawn_key=(ROLE_REPLAY, EXP, cond, rep))))


def make_batch(streams, H, tiebreak):
    """streams: list of (origin, loss) with arrival = origin + d already applied by caller as (origin, arrival, loss)."""
    n = len(streams)
    lens = np.array([len(s[0]) for s in streams], np.int64)
    off = np.zeros(n + 1, np.int64); off[1:] = np.cumsum(lens)
    cat = lambda k, dt: (np.concatenate([s[k] for s in streams]).astype(dt) if off[-1] else np.zeros(0, dt))
    org, arr, loss = cat(0, np.int64), cat(1, np.int64), cat(2, np.int8)
    z = np.zeros(n, np.int64)
    return G.StreamBatch(H=int(H), offsets=off, origin=org, arrival=arr, loss=loss, true_loss=loss.copy(),
                         prob=np.ones(len(org)), tiebreak=np.asarray(tiebreak, float), n_audits=z, n_defects=z,
                         n_misses=z, n_informative=lens, checkpoints=np.zeros(0, np.int64),
                         counts_at=np.zeros((n, 0, 3), np.int64))


def null_cfg(pi_arr, r, q, d):
    H = len(pi_arr)
    return G.StreamConfig(H=H, pi=pi_arr, r=r, q=q, delay=G.Delay.constant(d))


def synth_maxima(cfg, role, cond, n, mons, chunk=5000):
    out = {k: [] for k in mons}; tie = []
    for b in G.generate_chunks(cfg, role, EXP, cond, n, chunk):
        for k, m in mons.items():
            out[k].append(m.max_stat(b))
        tie.append(b.tiebreak)
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(tie)


def calibrate(pi_arr, r0, q, d, cond, mons, n=20000, target=0.02):
    mx, _ = synth_maxima(null_cfg(pi_arr, r0, q, d), 'calibration', cond, n, mons)
    return {k: C.calibrate_threshold(v, target, method='randomized') for k, v in mx.items()}


def fa_synth(cfg, role, cond, n, mons, thr):
    mx, tie = synth_maxima(cfg, role, cond, n, mons)
    return {k: C.fa_from_maxima(mx[k], thr[k], tie) for k in mons}


def alarms(mons, thr, batch):
    return {k: C.alarm_times(m, batch, thr[k]) for k, m in mons.items()}


_NP = {}
def oracle(L, r0, r1, alpha=0.02):
    import sys
    sys.path.insert(0, '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory')
    from converse import np_power
    if r1 <= r0:
        return np.nan
    key = (int(L), r0, r1, alpha)
    if key not in _NP:
        _NP[key] = float(np_power(int(L), 1.0, r0, r1, alpha)) if L > 0 else alpha
    return _NP[key]


def cp(k, n, conf=0.95):
    return C.clopper_pearson(k, n, conf)
