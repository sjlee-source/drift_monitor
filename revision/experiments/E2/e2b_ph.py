"""E2b POST HOC (amendment 2026-11-13): recentred block bootstrap (PH1, H_hist=30k) and longer history (PH2, 150k).
Usage: python -m revision.experiments.E2.e2b_ph PH1|PH2
"""
import sys, time
import numpy as np, pandas as pd
from revision import calibration as C
from revision.experiments.E2 import e2b, sim

TAG = sys.argv[1]
e2b.H_HIST = 30000 if TAG == 'PH1' else 150000
OFF = 50000 if TAG == 'PH1' else 60000
R0 = 0.10
OUT = e2b.OUT


def estimate(variant, qi, mons):
    vi = e2b.VARS.index(variant)
    rows = []; prim = {m: [] for m in mons}
    for k in range(e2b.K_HIST):
        idx, loss = e2b.history(variant, qi, k)
        rhat = float(loss.mean())
        per = {}
        for li, L in enumerate(e2b.BLOCKS):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                20261111, spawn_key=(8, e2b.EXP, OFF + 3000 + 100 * vi + 10 * qi, k, li))))
            b = e2b.boot_batch(idx, loss, L, e2b.B_BOOT, rng, e2b.QS[qi])
            u = rng.random(len(b.loss))
            if rhat < R0:
                b.loss = np.where((b.loss == 0) & (u < (R0 - rhat) / (1 - rhat)), 1, b.loss).astype(np.int8)
            else:
                b.loss = np.where((b.loss == 1) & (u < 1 - R0 / rhat), 0, b.loss).astype(np.int8)
            for m, mo in mons.items():
                t = C.calibrate_threshold(mo.max_stat(b), e2b.TARGET, 'randomized')
                per[(m, L)] = t
                rows.append(dict(variant=variant, q=e2b.QS[qi], history=k, hist_miss_rate=rhat, monitor=m, L=L,
                                 h=t.h, gamma=t.gamma))
        for m in mons:
            best = per[(m, e2b.BLOCKS[0])]
            for L in e2b.BLOCKS[1:]:
                best = sim.conservative(best, per[(m, L)])
            prim[m].append(best)
    return rows, prim


def main():
    mons = {m: sim.monitors()[m] for m in e2b.MON}
    thr_rows, fa_rows, det_rows = [], [], []
    K, N = e2b.K_HIST, e2b.N_VAL
    for qi, q in enumerate(e2b.QS):
        for variant in e2b.VARS:
            t1 = time.time()
            rows, prim = estimate(variant, qi, mons); thr_rows += rows
            for path in ('null_const010', 'null_path', 'null_const005', 'harm'):
                acc = {m: [] for m in mons}
                for bt in e2b.eval_batches(variant, qi, path, N):
                    for m, mo in mons.items():
                        if path == 'harm':
                            acc[m].append(e2b.alarms_many(mo, bt, prim[m]))
                        else:
                            mx = mo.max_stat(bt)
                            acc[m].append(np.stack([C.fa_from_maxima(mx, t, bt.tiebreak) for t in prim[m]], axis=1))
                for m in mons:
                    a = np.concatenate(acc[m])
                    if path == 'harm':
                        for D in sim.DEADLINES:
                            s = (a >= e2b.TAU) & (a <= min(e2b.H, e2b.TAU + D - 1)); ph = s.mean(0)
                            det_rows.append(dict(variant_ph=TAG, q=q, variant=variant, monitor=m, D=D, pD=float(s.mean()),
                                                 pD_hist_p05=float(np.quantile(ph, .05)), pD_hist_p50=float(np.median(ph)),
                                                 pD_hist_p95=float(np.quantile(ph, .95))))
                    else:
                        x = a[np.arange(N), np.arange(N) % K]; kk = int(x.sum()); fk = a.mean(0)
                        up = e2b.cp_up(kk, N, e2b.FAMILY)
                        fa_rows.append(dict(variant_ph=TAG, q=q, variant=variant, monitor=m, path=path, n=N,
                                            alarms_diag=kk, fa_diag=kk / N, upper_sim112=up,
                                            pass_gate=(up <= 0.03) if path != 'null_const005' else None,
                                            fa_cross_mean=float(fk.mean()), fa_hist_p50=float(np.median(fk)),
                                            fa_hist_p90=float(np.quantile(fk, .9)), fa_hist_max=float(fk.max()),
                                            P_hist_fa_gt_003=float((fk > 0.03).mean())))
            print(f'{TAG} q={q} {variant} {time.time()-t1:.0f}s', flush=True)
            pd.DataFrame(thr_rows).to_csv(f'{OUT}/posthoc_{TAG}_thresholds.csv', index=False)
            pd.DataFrame(fa_rows).to_csv(f'{OUT}/posthoc_{TAG}_fa.csv', index=False)
            pd.DataFrame(det_rows).to_csv(f'{OUT}/posthoc_{TAG}_detection.csv', index=False)
    print('total done', flush=True)


if __name__ == '__main__':
    main()
