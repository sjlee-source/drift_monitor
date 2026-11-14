"""E4 Pfab step 2: synthetic calibration/validation, chronological replay with item- and board-level audits,
calibration-board block-resampling FA.  Usage: ... pfab_02_replay.py <target_idx>
"""
import sys, csv, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4lib as L
from revision import calibration as C

OUT = Path('/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E4/pfab')
TARGETS = [0.01, 0.05, 0.10]; QS = [0.1, 0.2, 0.5, 1.0]; DS = [0, 3000]
import os
SITE = 2; NREP = 2000; NCAL = 20000; NVAL = 10000; NBLK = 10000
if os.environ.get('E4_SMOKE'):
    NREP, NCAL, NVAL, NBLK = 20, 200, 100, 50; QS = [0.2]; DS = [3000]
    OUT_W = Path(os.environ['E4_SMOKE'])
else:
    OUT_W = OUT
ti = int(sys.argv[1]); a = TARGETS[ti]; r0 = a
FAMILY = len(L.MON_NAMES) * len(TARGETS) * len(QS) * len(DS)

df = pd.read_csv(OUT / 'stream_calib_eval.csv.gz')
rk = f'release_{a:g}'
E = df[df.split == 'eval'].sort_values('row', kind='stable').reset_index(drop=True)
Cb = df[df.split == 'calib'].sort_values('row', kind='stable').reset_index(drop=True)
y = E['class'].values.astype(np.int8); l = E[rk].values.astype(np.int8)
bid = E.board.values; wk = E.week_start.values
Hs = len(E)
_, binv = np.unique(bid, return_inverse=True); nb = binv.max() + 1
first = lambda w: int(np.nonzero(wk == w)[0][0]) + 1   # 1-based item index
tau = first('10-19'); t_late = first('10-26'); t_end = first('11-02')   # stable: < tau; excursion: [tau, t_late)
pi_hat = y.sum() / Hs
def rate(lo, hi):
    m = (np.arange(1, Hs + 1) >= lo) & (np.arange(1, Hs + 1) < hi) & (y == 1)
    return l[m].mean(), int(m.sum()), int(l[m].sum())
r_stable, n_stable, k_stable = rate(1, tau); r_exc, n_exc, k_exc = rate(tau, t_late); r_post, n_post, k_post = rate(tau, Hs + 1)
info = dict(target=a, Hs=Hs, tau=tau, t_late=t_late, t_end=t_end, pi_hat=pi_hat, r_stable=r_stable, n_def_stable=n_stable,
            miss_stable=k_stable, r_exc=r_exc, n_def_exc=n_exc, miss_exc=k_exc, r_post=r_post, n_def_post=n_post,
            miss_post=k_post)
print(info, flush=True)
# calibration-period boards
cy = Cb['class'].values.astype(np.int8); cl = Cb[rk].values.astype(np.int8)
_, cst, clen = np.unique(Cb.board.values, return_index=True, return_counts=True)
r_cal = cl[cy == 1].mean()

def real_stream(rng, q, d, board):
    if board:
        aud = (y == 1) & (rng.random(nb)[binv] < q)
    else:
        aud = (y == 1) & (rng.random(Hs) < q)
    o = np.nonzero(aud)[0] + 1
    return (o, o + d, l[aud]), rng.random()

def block_stream(rng, q, d, board):
    k = int(Hs / clen.mean() * 1.3) + 10
    while True:
        pick = rng.integers(len(cst), size=k)
        if clen[pick].sum() >= Hs: break
        k *= 2
    ln = clen[pick]; cs = np.cumsum(ln); m = int(np.searchsorted(cs, Hs)) + 1
    pick, ln = pick[:m], ln[:m]
    idx = np.repeat(cst[pick] - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum())
    idx = idx[:Hs]; yy = cy[idx]; ll = cl[idx]
    if board:
        bix = np.repeat(np.arange(m), ln)[:Hs]
        aud = (yy == 1) & (rng.random(m)[bix] < q)
    else:
        aud = (yy == 1) & (rng.random(Hs) < q)
    o = np.nonzero(aud)[0] + 1
    return (o, o + d, ll[aud]), rng.random()

t0 = time.time()
thr_rows, val_rows, rep_rows, wk_rows, blk_rows = [], [], [], [], []
store = {}
wk_of = np.asarray(wk)
for qi, q in enumerate(QS):
    for di, d in enumerate(DS):
        H = Hs + d
        pi_arr = np.concatenate([np.full(Hs, pi_hat), np.zeros(d)])
        mons = L.monitors(r0)
        syn = L.cond_id(SITE, ti, qi, di, 0, 0)
        thr = L.calibrate(pi_arr, r0, q, d, syn, mons, n=NCAL)
        for k, t in thr.items():
            thr_rows.append(dict(alpha_target=a, q=q, d=d, monitor=k, H=H, pi_hat=pi_hat, **t.as_dict()))
        paths = {'const_r0': r0, 'const_r0/2': r0 / 2, 'path_r0/2->r0': np.where(np.arange(1, H + 1) >= tau, r0, r0 / 2)}
        for pj, (pn, rr) in enumerate(paths.items()):
            fa = L.fa_synth(L.null_cfg(pi_arr, rr, q, d), 'validation', syn + pj, NVAL, mons, thr)
            for k in mons:
                val_rows.append(dict(target=a, q=q, d=d, path=pn, monitor=k, **C.validate(fa[k], cap=0.03, family=FAMILY)))
        print(f'[t{ti}] q={q} d={d} calibrated {time.time() - t0:.0f}s', flush=True)
        for mode in [0, 1]:
            cond = L.cond_id(SITE, ti, qi, di, mode, 0)
            ss, tbs, Lexc, Lpost = [], [], [], []
            for rep in range(NREP):
                s, tb = real_stream(L.replay_rng(cond, rep), q, d, mode == 1)
                ss.append(s); tbs.append(tb); o = s[0]
                Lexc.append(int(((o >= tau) & (s[1] < t_late)).sum())); Lpost.append(int((o >= tau).sum()))
            b = L.make_batch(ss, H, tbs); al = L.alarms(mons, thr, b)
            Lexc = np.array(Lexc); Lpost = np.array(Lpost)
            o_exc = np.nanmean([L.oracle(x, r0, r_exc) for x in Lexc]) if (mode == 0 and r_exc > r0) else np.nan
            o_all = np.nanmean([L.oracle(x, r0, r_post) for x in Lpost]) if (mode == 0 and r_post > r0) else np.nan
            for k, T in al.items():
                store[f'q{q}_d{d}_m{mode}_{k}'] = T
                ev = dict(stable=T < tau, excursion=(T >= tau) & (T < t_late), by_end_1026=(T >= tau) & (T < t_end),
                          after_tau_any=(T >= tau) & (T <= H))
                row = dict(target=a, q=q, d=d, audit=['item', 'board'][mode], monitor=k, H=H, tau=tau,
                           mean_L_exc=Lexc.mean(), mean_L_post=Lpost.mean(), P_none=float((~np.isfinite(T)).mean()),
                           oracle_exc_week=o_exc, oracle_end=o_all)
                for nm, e in ev.items():
                    lo, hi = C.clopper_pearson(int(e.sum()), NREP)
                    row[f'P_{nm}'] = e.mean(); row[f'{nm}_lo'] = lo; row[f'{nm}_hi'] = hi
                fin = T[np.isfinite(T)]
                for qq in [0.1, 0.5, 0.9]:
                    row[f'T_q{qq}'] = float(np.quantile(fin, qq)) if len(fin) else np.nan
                    row[f'week_q{qq}'] = (wk_of[min(int(np.quantile(fin, qq, method='inverted_cdf')), Hs) - 1]
                                          if len(fin) else '')
                rep_rows.append(row)
                ws, cnt = np.unique([('trailing' if t > Hs else wk_of[int(t) - 1]) for t in fin], return_counts=True)
                for w_, c_ in zip(ws, cnt):
                    wk_rows.append(dict(target=a, q=q, d=d, audit=['item', 'board'][mode], monitor=k, first_alarm_week=w_,
                                        n=int(c_), frac=c_ / NREP))
        for mode in [2, 3]:
            cond = L.cond_id(SITE, ti, qi, di, mode, 0); ss, tbs = [], []
            for rep in range(NBLK):
                s, tb = block_stream(L.replay_rng(cond, rep), q, d, mode == 3); ss.append(s); tbs.append(tb)
            b = L.make_batch(ss, H, tbs); al = L.alarms(mons, thr, b)
            nl = np.diff(b.offsets)
            for k, T in al.items():
                fa = np.isfinite(T) & (T <= H); lo, hi = C.clopper_pearson(int(fa.sum()), NBLK)
                blk_rows.append(dict(target=a, q=q, d=d, audit=['item', 'board'][mode - 2], monitor=k, r_cal=r_cal,
                                     n_cal_boards=len(cst), mean_labels=nl.mean(), FA=fa.mean(), lo=lo, hi=hi, n=NBLK))
            del ss, b
        print(f'[t{ti}] q={q} d={d} done {time.time() - t0:.0f}s', flush=True)

def wr(name, rr):
    with (OUT_W / f'{name}_t{ti}.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
wr('thresholds', thr_rows); wr('validation', val_rows); wr('replay', rep_rows); wr('replay_alarm_weeks', wk_rows)
wr('blockfa', blk_rows); wr('periods', [info])
np.savez_compressed(OUT_W / f'alarms_t{ti}.npz', **store)
print('finished', time.time() - t0, flush=True)
