"""E2 stressor 7: planning misspecification and pilot-estimation uncertainty (exact DP, no simulation).

Matched integer CUSUM (p1 = r1 = 0.20, r0 = 0.10, scale B = 40, as theory/qstar.py), pi=0.1, d=500,
D=10,000, tau=10,000, H=29,999, target p_D = 0.80, FA alpha = 0.02.  The CUSUM's exact probabilities
depend on (pi, q) only through the label rate rho = pi*q, so the planned rate q*(pi') = rho*/pi'.
Threshold variants:
 (a) 'true'    : c calibrated (exact DP, constant r0 null) at the TRUE realized label rate pi*q.
                 This is what a calibration on null streams of the actual process would give.
 (b) 'assumed' : c calibrated at the ASSUMED label rate pi'*q (= rho* unless q is clipped at 1);
                 the realized FA then deviates from 0.02 and is reported.
"""
import sys, json, math, time
import numpy as np, pandas as pd
from scipy.stats import binom, beta
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory')
from exact_cusum import scores, detection_prob, calibrate, false_alarm
from converse import np_power

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E2'
R0, RPRE, R1, PI, D0, DL, TAU, H, ALPHA, TARGET, SB = 0.10, 0.05, 0.20, 0.10, 500, 10000, 10000, 29999, 0.02, 0.80, 40
A, B, S = scores(R1, R0, SB)
Q_STAR = 0.18836490894898006   # theory/qstar_grid.csv (pi .1, r1 .2, d 500, D 10k)
RHO_STAR = PI * Q_STAR
PILOT_N = (1000, 5000, 20000)
TAIL = 1e-7


def pd_true(rho):
    c, fa = calibrate(H, D0, rho, A, B, ALPHA, R0)
    p, early = detection_prob(H, D0, rho, TAU, RPRE, R1, DL, A, B, c)
    return dict(rho=rho, c=c, pD=p, early=early, fa_const=fa)


def pd_fixed_c(args):
    rho, c = args
    p, early = detection_prob(H, D0, rho, TAU, RPRE, R1, DL, A, B, c)
    fa = false_alarm(H, D0, rho, lambda s: R0, A, B, c)
    fap = false_alarm(H, D0, rho, lambda s: RPRE if s < TAU else R0, A, B, c)
    return dict(rho=rho, c=c, pD=p, early=early, fa_const=fa, fa_path=fap)


def job(args):
    kind, rho, c = args
    return pd_true(rho) if kind == 'true' else pd_fixed_c((rho, c))


def run(tasks, nproc=4):
    with ProcessPoolExecutor(nproc) as ex:
        return list(ex.map(job, tasks, chunksize=4))


def main():
    t0 = time.time()
    star = pd_true(RHO_STAR)
    c_star = star['c']
    star.update(pd_fixed_c((RHO_STAR, c_star)))
    print('rho*', RHO_STAR, star, flush=True)
    # ---------------- (i) misspecified pi'
    rows = []
    facs = (0.5, 0.75, 1.0, 1.25, 2.0)
    tasks = []
    for f in facs:
        pip = f * PI; q = min(1.0, RHO_STAR / pip); rho = PI * q
        c_ass = calibrate(H, D0, pip * q, A, B, ALPHA, R0)[0]
        tasks += [('true', rho, None), ('fixed', rho, c_ass)]
    res = run(tasks)
    for i, f in enumerate(facs):
        pip = f * PI; q = min(1.0, RHO_STAR / pip)
        a, b = res[2 * i], res[2 * i + 1]
        rows.append(dict(pi_assumed=pip, factor=f, q_planned=q, q_ratio_to_qstar=q / Q_STAR, rho_real=PI * q,
                         pD_cal_true=a['pD'], c_true=a['c'], fa_cal_true=a['fa_const'],
                         pD_cal_assumed=b['pD'], c_assumed=b['c'], fa_cal_assumed_const010=b['fa_const'],
                         fa_cal_assumed_path=b['fa_path'],
                         oracle_NP_bound=np_power(DL - D0, PI * q, R0, R1, ALPHA)))
    pd.DataFrame(rows).to_csv(f'{OUT}/planning_misspec.csv', index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)
    # ---------------- (ii) pilot estimation
    dist_rows, summ = [], []
    for n in PILOT_N:
        lo = int(binom.ppf(TAIL / 2, n, PI)); hi = int(binom.isf(TAIL / 2, n, PI))
        X = np.arange(max(1, lo), hi + 1); w = binom.pmf(X, n, PI)
        omitted = 1 - w.sum(); w = w / w.sum()
        pihat = X / n
        pilo = beta.ppf(0.05, X, n - X + 1)          # one-sided 95% CP lower bound
        rules = {'plugin': pihat, 'lower95': pilo}
        for rule, pe in rules.items():
            q = np.minimum(1.0, RHO_STAR / pe); rho = PI * q; rho_plan = pe * q
            tasks = [('true', r, None) for r in rho]
            ca = {}
            for rp in np.unique(np.round(rho_plan, 15)):
                ca[rp] = c_star if abs(rp - RHO_STAR) < 1e-13 else calibrate(H, D0, rp, A, B, ALPHA, R0)[0]
            tasks += [('fixed', r, ca[np.round(rp, 15)]) for r, rp in zip(rho, rho_plan)]
            t1 = time.time(); res = run(tasks)
            ra, rb = res[:len(X)], res[len(X):]
            pa = np.array([x['pD'] for x in ra]); pb = np.array([x['pD'] for x in rb])
            fb = np.array([x['fa_const'] for x in rb]); fbp = np.array([x['fa_path'] for x in rb])
            for j in range(len(X)):
                dist_rows.append(dict(n=n, rule=rule, X=int(X[j]), weight=w[j], pi_est=pe[j], q=q[j], rho=rho[j],
                                      pD_cal_true=pa[j], pD_cal_assumed=pb[j], fa_cal_assumed=fb[j],
                                      fa_path_cal_assumed=fbp[j]))
            def wq(v, p):
                o = np.argsort(v); cw = np.cumsum(w[o]); return float(v[o][np.searchsorted(cw, p)])
            summ.append(dict(n=n, rule=rule, omitted_mass=omitted, E_q=float(w @ q), E_q_over_qstar=float(w @ q) / Q_STAR,
                             q_p05=wq(q, .05), q_p95=wq(q, .95),
                             pD_true_mean=float(w @ pa), pD_true_p05=wq(pa, .05), pD_true_p50=wq(pa, .5),
                             pD_true_p95=wq(pa, .95), P_pD_true_lt_080=float(w @ (pa < TARGET)),
                             P_pD_true_lt_075=float(w @ (pa < 0.75)),
                             pD_ass_mean=float(w @ pb), P_pD_ass_lt_080=float(w @ (pb < TARGET)),
                             fa_ass_mean=float(w @ fb), fa_ass_p95=wq(fb, .95), P_fa_ass_gt_003=float(w @ (fb > 0.03)),
                             fa_path_ass_mean=float(w @ fbp)))
            print(n, rule, f'{time.time()-t1:.0f}s', summ[-1], flush=True)
    pd.DataFrame(dist_rows).to_csv(f'{OUT}/pilot_distribution.csv', index=False)
    pd.DataFrame(summ).to_csv(f'{OUT}/pilot_summary.csv', index=False)
    json.dump(dict(A=A, B=B, scale=S, q_star=Q_STAR, rho_star=RHO_STAR, c_star=c_star, star=star,
                   seconds=time.time() - t0), open(f'{OUT}/planning_meta.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
