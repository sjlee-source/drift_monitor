"""E2b re-analysis (2026-11-13, CODEX_REVIEW E2b): history-cluster-aware inference for the unconditional FA of
estimated (primary), PH1 and PH2 thresholds.  Validation streams are regenerated exactly as in E2b.

X[k, j] = FA indicator of history-k threshold on validation stream j (K=100 x N=10,000, crossed design).
Unconditional FA  theta = E_hist E_stream X.  Estimator: grand mean of X.
Two-way (history x stream) bootstrap, linearised: theta* = mean_k* rowmean_k + (mean_j* colmean_j - grand);
history part by exact resampling of the K row means (B=200,000), stream part by its normal approximation
(sum of 10,000 terms).  Gate: (1 - 0.05/112) percentile <= 0.03.  A full (non-linearised) Poisson two-way
bootstrap (B=4,000; tail percentile is rough) is computed as a check.
"""
import numpy as np, pandas as pd
from revision import calibration as C
from revision.experiments.E2 import e2b, sim

OUT = e2b.OUT
FAM = 112
A = 0.05 / FAM
BH = 200000
rng = np.random.default_rng(20261113)


def thr_from(df):
    """per (variant,q,monitor): list over histories of primary (max over L) Threshold"""
    out = {}
    for (v, q, m), g in df.groupby(['variant', 'q', 'monitor']):
        lst = []
        for k, gk in g.groupby('history'):
            best = None
            for _, r in gk.iterrows():
                tol = 1e-9 * max(1.0, abs(r.h))
                t = C.Threshold(h=r.h, gamma=r.gamma, h_hi=r.get('h_hi', r.h + tol), h_lo=r.get('h_lo', r.h - tol))
                if 'h_hi' not in r or pd.isna(r.get('h_hi', np.nan)):
                    t.h_hi, t.h_lo = r.h + tol, r.h - tol
                best = t if best is None else sim.conservative(best, t)
            lst.append(best)
        out[(v, q, m)] = lst
    return out


def bounds(X):
    K, N = X.shape
    rm = X.mean(1); cm = X.mean(0); g = X.mean()
    idx = rng.integers(0, K, size=(BH, K))
    hist = rm[idx].mean(1)
    sj = cm.std(ddof=1) / np.sqrt(N)
    lin = hist + rng.normal(0, sj, BH)
    up_lin = float(np.quantile(lin, 1 - A))
    up_hist_only = float(np.quantile(hist, 1 - A))
    se = np.sqrt(rm.var(ddof=1) / K + sj ** 2)
    up_norm = g + 3.3183 * se            # z_{1-0.05/112}
    # full Poisson two-way bootstrap check
    Xf = X.astype(np.float32); vals = []
    for _ in range(4):
        w = rng.poisson(1.0, (1000, K)).astype(np.float32); v = rng.poisson(1.0, (1000, N)).astype(np.float32)
        num = ((w @ Xf) * v).sum(1); den = w.sum(1) * v.sum(1)
        vals.append(num / den)
    full = np.concatenate(vals)
    up_full = float(np.quantile(full, 1 - A))
    return dict(fa_uncond=float(g), sd_hist=float(rm.std(ddof=1)), se_cluster=float(se),
                upper_twoway_lin=up_lin, upper_hist_only=up_hist_only, upper_normal=float(up_norm),
                upper_twoway_full_poisson=up_full, gate_twoway=bool(up_lin <= 0.03),
                P_hist_fa_gt_003=float((rm > 0.03).mean()))


def main():
    est = pd.read_csv(f'{OUT}/thresholds_est.csv')
    methods = {'estimated': thr_from(est)}
    for tag in ('PH1', 'PH2'):
        methods[f'posthoc_{tag}'] = thr_from(pd.read_csv(f'{OUT}/posthoc_{tag}_thresholds.csv'))
    old = pd.read_csv(f'{OUT}/summary_fa_all_methods.csv')
    mons = sim.monitors()
    rows = []
    for qi, q in enumerate(e2b.QS):
        for v in e2b.VARS:
            for path in ('null_const010', 'null_path'):
                mx = {m: [] for m in e2b.MON}; tb = []
                for b in e2b.eval_batches(v, qi, path, e2b.N_VAL):
                    tb.append(b.tiebreak)
                    for m in e2b.MON:
                        mx[m].append(mons[m].max_stat(b))
                tb = np.concatenate(tb)
                for m in e2b.MON:
                    M = np.concatenate(mx[m])
                    for meth, T in methods.items():
                        X = np.stack([C.fa_from_maxima(M, t, tb) for t in T[(v, q, m)]]).astype(np.int8)
                        r = dict(method=meth, q=q, variant=v, monitor=m, path=path, K=X.shape[0], N=X.shape[1])
                        r.update(bounds(X))
                        o = old[(old.method == meth) & (old.q == q) & (old.variant == v) & (old.monitor == m) &
                                (old.path == path)]
                        r['old_fa_diag'] = float(o.fa_diag.iloc[0]); r['old_upper_sim112_conditional'] = float(o.upper_sim112.iloc[0])
                        r['old_gate_conditional'] = bool(o.pass_gate.iloc[0])
                        rows.append(r)
                print(q, v, path, flush=True)
                pd.DataFrame(rows).to_csv(f'{OUT}/cluster_inference.csv', index=False)
    d = pd.DataFrame(rows)
    s = d.groupby('method').agg(old_conditional=('old_gate_conditional', 'sum'), new_twoway=('gate_twoway', 'sum'),
                                n=('gate_twoway', 'size'))
    s['new_full_poisson'] = d.assign(x=d.upper_twoway_full_poisson <= .03).groupby('method').x.sum()
    s['new_normal'] = d.assign(x=d.upper_normal <= .03).groupby('method').x.sum()
    s.to_csv(f'{OUT}/cluster_inference_gate_counts.csv'); print(s)


if __name__ == '__main__':
    main()
