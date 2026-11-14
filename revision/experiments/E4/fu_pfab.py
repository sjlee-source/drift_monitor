"""E4 follow-up 2026-11-13b (sensitivity / post hoc).  Pfab items 1-3.
Usage: fu_pfab.py boardcal <ti> | influence <ti> | recentred <ti>
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import betaln, gammaln
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4lib as L
from revision import calibration as C, generators as G
from revision.rng import role_code

E4 = Path('/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E4'); FU = E4 / 'followup'; FU.mkdir(exist_ok=True)
TARGETS = [0.01, 0.05, 0.10]; QS = [0.1, 0.2, 0.5, 1.0]; ICCS = [0.0, 0.1, 0.3, 0.5]
cmd = sys.argv[1]; ti = int(sys.argv[2]); a = TARGETS[ti]; r0 = a
NREP, NCAL, NVAL, NBLK = 2000, 20000, 10000, 10000
t0 = time.time()
def log(*x): print(f'[{cmd} t{ti} {time.time() - t0:.0f}s]', *x, flush=True)
def grng(role, cond, chunk):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=L.MASTER, spawn_key=(role_code(role), L.EXP, cond, chunk))))
def thr_from(df):
    return {r.monitor: C.Threshold(h=r.h, gamma=r.gamma, h_hi=r.h_hi, h_lo=r.h_lo, method='randomized') for r in df.itertuples()}
def batch_from_mask(A, M, pos, d, H, tb):
    rows, cols = np.nonzero(A)
    cnt = np.bincount(rows, minlength=A.shape[0])
    off = np.zeros(A.shape[0] + 1, np.int64); off[1:] = np.cumsum(cnt)
    org = pos[cols].astype(np.int64); loss = M[rows, cols].astype(np.int8)
    z = np.zeros(A.shape[0], np.int64)
    return G.StreamBatch(H=int(H), offsets=off, origin=org, arrival=org + d, loss=loss, true_loss=loss.copy(), prob=np.ones(len(org)),
                         tiebreak=tb, n_audits=z, n_defects=z, n_misses=z, n_informative=cnt.astype(np.int64),
                         checkpoints=np.zeros(0, np.int64), counts_at=np.zeros((A.shape[0], 0, 3), np.int64))
def metrics_row(T, tau, t_late, t_end, H, NR):
    ev = dict(stable=T < tau, excursion=(T >= tau) & (T < t_late), by_end_1026=(T >= tau) & (T < t_end))
    out = {f'P_{k}': float(v.mean()) for k, v in ev.items()}; out['P_none'] = float((~np.isfinite(T)).mean())
    return out

if True:
    df = pd.read_csv(E4 / 'pfab/stream_calib_eval.csv.gz')
    rk = f'release_{a:g}'
    E = df[df.split == 'eval'].sort_values('row', kind='stable').reset_index(drop=True)
    Cb = df[df.split == 'calib'].sort_values('row', kind='stable').reset_index(drop=True)
    PTH = pd.read_csv(E4 / 'pfab_thresholds.csv')

def layout(Edf):
    y = Edf['class'].values.astype(np.int8); l = Edf[rk].values.astype(np.int8); wk = Edf.week_start.values
    Hs = len(Edf); pos = np.nonzero(y == 1)[0] + 1
    _, bdef = np.unique(Edf.board.values[y == 1], return_inverse=True)
    _, binv = np.unique(Edf.board.values, return_inverse=True)
    f = lambda w: int(np.nonzero(wk == w)[0][0]) + 1
    return dict(y=y, l=l, Hs=Hs, pos=pos, bdef=bdef, binv=binv, nb=binv.max() + 1, tau=f('10-19'), t_late=f('10-26'), t_end=f('11-02'))

def boardunit_batch(LY, q, d, icc, rng, R, r=None):
    r = r0 if r is None else r
    nbd = LY['bdef'].max() + 1
    A = (rng.random((R, nbd)) < q)[:, LY['bdef']]
    if icc > 0:
        k = 1 / icc - 1; p = rng.beta(r * k, (1 - r) * k, (R, nbd))[:, LY['bdef']]
    else:
        p = r
    M = rng.random((R, len(LY['pos']))) < p
    return batch_from_mask(A, M, LY['pos'], d, LY['Hs'] + d, rng.random(R))

def real_batch(LY, q, d, board, cond, R=NREP):
    ss, tbs = [], []
    for rep in range(R):
        rng = L.replay_rng(cond, rep)
        aud = (LY['y'] == 1) & ((rng.random(LY['nb'])[LY['binv']] < q) if board else (rng.random(LY['Hs']) < q))
        o = np.nonzero(aud)[0] + 1; ss.append((o, o + d, LY['l'][aud])); tbs.append(rng.random())
    return L.make_batch(ss, LY['Hs'] + d, tbs)

def logbb(k, n, r, icc):
    if icc == 0:
        return k * np.log(r) + (n - k) * np.log1p(-r)
    s = 1 / icc - 1; al, be = r * s, (1 - r) * s
    return betaln(k + al, n - k + be) - betaln(al, be)

def bb_oracle(ns, q, r1, icc, rng, R=20000, alpha=0.02):
    """MC Neyman-Pearson power: audited boards (sizes ns) each w.p. q; beta-binomial counts, same ICC under H0/H1."""
    def draw(r):
        A = rng.random((R, len(ns))) < q
        p = rng.beta(r * (1 / icc - 1), (1 - r) * (1 / icc - 1), (R, len(ns))) if icc > 0 else np.full((R, len(ns)), r)
        k = rng.binomial(np.broadcast_to(ns, (R, len(ns))), p)
        llr = (logbb(k, ns, r1, icc) - logbb(k, ns, r0, icc)) * A
        return np.round(llr.sum(1), 10)
    l0, l1 = draw(r0), draw(r1)
    c = np.quantile(l0, 1 - alpha, method='higher')
    p_gt = np.mean(l0 > c); p_eq = np.mean(l0 == c); g = (alpha - p_gt) / p_eq if p_eq > 0 else 0
    return float(np.mean(l1 > c) + g * np.mean(l1 == c))

# ------------------------------------------------------------------ item 2
if cmd == 'boardcal':
    LY = layout(E); rows_t, rows_v, rows_r, rows_o = [], [], [], []
    r_exc = LY['l'][(LY['y'] == 1) & (np.arange(1, LY['Hs'] + 1) >= LY['tau']) & (np.arange(1, LY['Hs'] + 1) < LY['t_late'])].mean()
    ex = (E['class'] == 1) & (np.arange(1, LY['Hs'] + 1) >= LY['tau']) & (np.arange(1, LY['Hs'] + 1) < LY['t_late'])
    ns_exc = E[ex].groupby('board').size().values.astype(np.int64)
    for qi, q in enumerate(QS):
        for di, d in enumerate([0, 3000]):
            mons = L.monitors(r0)
            for ii, icc in enumerate(ICCS):
                cond = L.cond_id(2, ti, qi, di, 5, ii)
                mx = {k: [] for k in mons}; tie = []
                for ch in range(NCAL // 5000):
                    b = boardunit_batch(LY, q, d, icc, grng('calibration', cond, ch), 5000)
                    for k, m in mons.items(): mx[k].append(m.max_stat(b))
                    tie.append(b.tiebreak)
                thr = {k: C.calibrate_threshold(np.concatenate(v), 0.02, 'randomized') for k, v in mx.items()}
                for k, t in thr.items(): rows_t.append(dict(alpha_target=a, q=q, d=d, icc=icc, monitor=k, **t.as_dict()))
                mx = {k: [] for k in mons}; tie = []
                for ch in range(NVAL // 5000):
                    b = boardunit_batch(LY, q, d, icc, grng('validation', cond, ch), 5000)
                    for k, m in mons.items(): mx[k].append(m.max_stat(b))
                    tie.append(b.tiebreak)
                tie = np.concatenate(tie)
                for k in mons:
                    v = C.validate(C.fa_from_maxima(np.concatenate(mx[k]), thr[k], tie), cap=0.03, family=1)
                    rows_v.append(dict(target=a, q=q, d=d, icc=icc, monitor=k, **v))
                # re-evaluate board-audit replay (original mode-1 seeds)
                b = real_batch(LY, q, d, True, L.cond_id(2, ti, qi, di, 1, 0))
                for k, m in mons.items():
                    T = C.alarm_times(m, b, thr[k])
                    rows_r.append(dict(target=a, q=q, d=d, audit='board', calibration=f'board-unit ICC={icc}', monitor=k,
                                       **metrics_row(T, LY['tau'], LY['t_late'], LY['t_end'], LY['Hs'] + d, NREP)))
                if d == 0 and r_exc > r0:
                    rows_o.append(dict(target=a, q=q, icc=icc, r1_exc=r_exc, n_exc_defect_boards=len(ns_exc),
                                       oracle_board_unit=bb_oracle(ns_exc, q, r_exc, icc, grng('evaluation', cond, 99))))
                log(q, d, icc)
    pd.DataFrame(rows_t).to_csv(FU / f'pfab_boardunit_thresholds_t{ti}.csv', index=False)
    pd.DataFrame(rows_v).to_csv(FU / f'pfab_boardunit_validation_t{ti}.csv', index=False)
    pd.DataFrame(rows_r).to_csv(FU / f'pfab_boardaudit_replay_boardunit_t{ti}.csv', index=False)
    if rows_o: pd.DataFrame(rows_o).to_csv(FU / f'pfab_boardunit_oracle_t{ti}.csv', index=False)

# ------------------------------------------------------------------ item 1
if cmd == 'influence':
    BU = pd.read_csv(FU / f'pfab_boardunit_thresholds_t{ti}.csv')
    D = E[E['class'] == 1]; top = D.groupby('board')[rk].sum().sort_values(ascending=False)
    top = [b for b in top.index[:5] if top[b] > 0]
    rows = []
    for qi, q in enumerate(QS):
        for di, d in enumerate([0, 3000]):
            LY = layout(E[E.board != 35957].reset_index(drop=True)); mons = L.monitors(r0)
            iid = thr_from(PTH[(PTH.alpha_target == a) & (PTH.q == q) & (PTH.d == d)])
            bu = thr_from(BU[(BU.q == q) & (BU.d == d) & (BU.icc == 0.1)])
            for mode, board in [(0, False), (1, True)]:
                b = real_batch(LY, q, d, board, L.cond_id(2, ti, qi, di, 7, mode))
                for cal, thr in ([('iid', iid)] + ([('board-unit ICC=0.1', bu)] if board else [])):
                    for k, m in mons.items():
                        rows.append(dict(analysis='without board 35957', target=a, q=q, d=d, audit=['item', 'board'][mode], calibration=cal,
                                         monitor=k, **metrics_row(C.alarm_times(m, b, thr[k]), LY['tau'], LY['t_late'], LY['t_end'], 0, 0)))
            log('drop35957', q, d)
    for bi, bd in enumerate(top):
        LY = layout(E[E.board != bd].reset_index(drop=True)); mons = L.monitors(r0)
        for (q, board) in [(0.2, False), (0.5, False), (0.2, True)]:
            qi = QS.index(q)
            iid = thr_from(PTH[(PTH.alpha_target == a) & (PTH.q == q) & (PTH.d == 0)])
            b = real_batch(LY, q, 0, board, L.cond_id(2, ti, qi, 1 if board else 0, 8, bi))
            for k, m in mons.items():
                rows.append(dict(analysis=f'LOBO rank{bi + 1} board {bd} (misses {int(D[D.board == bd][rk].sum())}/{int((D.board == bd).sum())})',
                                 target=a, q=q, d=0, audit=['item', 'board'][board], calibration='iid', monitor=k,
                                 **metrics_row(C.alarm_times(m, b, iid[k]), LY['tau'], LY['t_late'], LY['t_end'], 0, 0)))
        log('lobo', bd)
    pd.DataFrame(rows).to_csv(FU / f'pfab_influence_t{ti}.csv', index=False)

# ------------------------------------------------------------------ item 3 (Pfab)
if cmd == 'recentred':
    BU = pd.read_csv(FU / f'pfab_boardunit_thresholds_t{ti}.csv')
    cy = Cb['class'].values.astype(np.int8); cl = Cb[rk].values.astype(np.int8)
    _, cst, clen = np.unique(Cb.board.values, return_index=True, return_counts=True)
    rhat = cl[cy == 1].mean(); lift = (r0 - rhat) / (1 - rhat); assert lift >= 0
    Hs = int((df.split == 'eval').sum()); rows = []
    for qi, q in enumerate(QS):
        for di, d in enumerate([0, 3000]):
            mons = L.monitors(r0)
            iid = thr_from(PTH[(PTH.alpha_target == a) & (PTH.q == q) & (PTH.d == d)])
            bus = {icc: thr_from(BU[(BU.q == q) & (BU.d == d) & (BU.icc == icc)]) for icc in ICCS}
            for mode, board in [(0, False), (1, True)]:
                ss, tbs, rr = [], [], []
                for rep in range(NBLK):
                    rng = L.replay_rng(L.cond_id(2, ti, qi, di, 6, mode), rep)
                    k = int(Hs / clen.mean() * 1.3) + 10
                    while True:
                        pick = rng.integers(len(cst), size=k)
                        if clen[pick].sum() >= Hs: break
                        k *= 2
                    ln = clen[pick]; m_ = int(np.searchsorted(np.cumsum(ln), Hs)) + 1; pick, ln = pick[:m_], ln[:m_]
                    idx = (np.repeat(cst[pick] - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum()))[:Hs]
                    yy = cy[idx]; ll = cl[idx].copy()
                    ll[(yy == 1) & (ll == 0) & (rng.random(Hs) < lift)] = 1
                    if board:
                        aud = (yy == 1) & (rng.random(m_)[np.repeat(np.arange(m_), ln)[:Hs]] < q)
                    else:
                        aud = (yy == 1) & (rng.random(Hs) < q)
                    o = np.nonzero(aud)[0] + 1; ss.append((o, o + d, ll[aud])); tbs.append(rng.random())
                    rr.append(ll[yy == 1].mean())
                b = L.make_batch(ss, Hs + d, tbs)
                for cal, thr in [('iid', iid)] + ([(f'board-unit ICC={i}', bus[i]) for i in ICCS] if board else []):
                    for k_, m in mons.items():
                        T = C.alarm_times(m, b, thr[k_]); fa = np.isfinite(T); lo, hi = C.clopper_pearson(int(fa.sum()), NBLK)
                        rows.append(dict(site='Pfab', null='recentred calibration-board blocks (marginal r0)', target=a, q=q, d=d,
                                         audit=['item', 'board'][mode], calibration=cal, monitor=k_, rhat_pool=rhat, lift=lift,
                                         mean_realized_r=float(np.mean(rr)), FA=fa.mean(), lo=lo, hi=hi, n=NBLK))
                del ss, b
            log(q, d)
    pd.DataFrame(rows).to_csv(FU / f'pfab_recentred_null_t{ti}.csv', index=False)

log('finished')
