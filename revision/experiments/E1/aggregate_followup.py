"""Aggregate E1 follow-up (amendment 2026-11-13, post hoc): repair, fine-q, ranking +/- DDM, restart incidence,
oracle curves for figures."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
from revision import calibration as C
from revision.experiments.E1 import e1_common as E
from revision.experiments.E1.run_followup import REPAIR, FINE_Q

ROOT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1'
FU = os.path.join(ROOT, 'followup')
f = lambda x: float(x) if isinstance(x, str) else x


def metric_rows(r, label):
    rows = []
    for r1s, e in r['evaluation'].items():
        for k, m in e.items():
            for D in E.DEADLINES:
                rows.append(dict(source=label, variant=k, pi=r['pi'], r1=float(r1s), q=r['q'], d=r['d'], D=D,
                                 pD=m[f'pD_{D}'], pD_lo=m[f'pD_{D}_lo'], pD_hi=m[f'pD_{D}_hi'], early_rate=m['early_rate'],
                                 rmd=m[f'rmd_{D}'], timely_q50=f(m[f'timely_q50_{D}']), timely_q90=f(m[f'timely_q90_{D}']),
                                 oracle_beta=m[f'oracle_{D}']))
    return rows


def val_rows(r, label):
    rows = []
    for k, vv in r['validation'].items():
        for p, v in vv.items():
            rows.append(dict(source=label, variant=k, pi=r['pi'], q=r['q'], d=r['d'], path=p, n=v['n'], alarms=v['alarms'],
                             rate=v['rate'], upper_one_sided=v['upper_simultaneous'], family=v['family'],
                             pass_gate=v['pass_gate'] if p != 'const05' else None,
                             h=f(r['thresholds'][k]['comp']['h']), gamma=r['thresholds'][k]['comp']['gamma'],
                             binding=r['thresholds'][k]['binding'], n_cal=r['n_cal']))
    return rows


def interp_cross(q, y, level=0.8):
    """first upward crossing of level by y(q) with linear interpolation in log q; inf if never."""
    lq = np.log(q)
    for i in range(len(q)):
        if y[i] >= level:
            if i == 0:
                return q[0] if y[0] >= level else np.inf
            t = (level - y[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(lq[i - 1] + t * (lq[i] - lq[i - 1])))
    return np.inf


def main():
    ev = pd.read_csv(os.path.join(ROOT, 'metrics_long.csv'))
    val0 = pd.read_csv(os.path.join(ROOT, 'fa_validation.csv'))
    # ---------------- A1 repair
    rep_v, rep_m = [], []
    for p in sorted(glob.glob(os.path.join(FU, 'repair_*.json'))):
        r = json.load(open(p)); rep_v += val_rows(r, 'repair_100k'); rep_m += metric_rows(r, 'repair_100k')
    rv = pd.DataFrame(rep_v); rm = pd.DataFrame(rep_m)
    if len(rv):
        orig_v = val0[val0.apply(lambda x: any(x.variant == k and np.isclose(x.pi, s[0]) and np.isclose(x.q, s[1]) and x.d == s[2]
                                               for k, s in REPAIR), axis=1)].assign(source='original_20k')
        pd.concat([orig_v, rv], ignore_index=True).to_csv(os.path.join(ROOT, 'fa_validation_repair.csv'), index=False)
        orig_m = ev[ev.apply(lambda x: any(x.variant == k and np.isclose(x.pi, s[0]) and np.isclose(x.q, s[1]) and x.d == s[2]
                                           for k, s in REPAIR), axis=1)].assign(source='original_20k')
        pd.concat([orig_m, rm], ignore_index=True).to_csv(os.path.join(ROOT, 'metrics_repair.csv'), index=False)
    # repaired long table for confirmatory rankings: replace the 4 cells
    evr = ev.copy()
    for _, x in rm.iterrows():
        sel = (evr.variant == x.variant) & np.isclose(evr.pi, x.pi) & np.isclose(evr.q, x.q) & (evr.d == x.d) & \
              np.isclose(evr.r1, x.r1) & (evr.D == x.D)
        for c in ('pD', 'pD_lo', 'pD_hi', 'early_rate', 'rmd', 'timely_q50', 'timely_q90'):
            evr.loc[sel, c] = x[c]
    # ---------------- ranking with / without DDM (on repaired table)
    rk = []
    for drop_ddm in (False, True):
        mons = [m for m in E.CORE if not (drop_ddm and m == 'ddm')]
        for D in E.DEADLINES:
            P = evr[(evr.family == 'core') & (evr.D == D)].pivot_table(index=['pi', 'r1', 'q', 'd'], columns='variant', values='pD')[mons]
            R = P.apply(lambda row: pd.Series(rankdata(-row.values, method='average'), index=row.index), axis=1)
            best = P.max(axis=1); within = P.ge(best - 0.01, axis=0)
            inf_ = (P.max(axis=1) - P.min(axis=1) > 0); hard = inf_ & (best < 0.95)
            k = len(mons)

            def W(Rx):
                n = len(Rx)
                return 12 * ((Rx.sum(axis=0) - Rx.sum(axis=0).mean()) ** 2).sum() / (n ** 2 * (k ** 3 - k)) if n else np.nan
            for m in mons:
                rk.append(dict(ddm_included=not drop_ddm, D=D, monitor=m, mean_rank_all=R[m].mean(), mean_rank_hard=R[m][hard].mean(),
                               n_best_or_within_001=int(within[m].sum()), mean_pD=P[m].mean(), n_conditions=len(P), n_hard=int(hard.sum()),
                               kendall_W_informative=W(R[inf_]), kendall_W_hard=W(R[hard])))
    pd.DataFrame(rk).to_csv(os.path.join(ROOT, 'ranking_repaired_with_without_ddm.csv'), index=False)
    # ---------------- restart false-alarm incidence (renamed)
    a = pd.read_csv(os.path.join(ROOT, 'arl_restart.csv')).rename(columns={
        'alarms_per_10k': 'restart_fa_incidence_per_10k', 'alarms_per_10k_lo': 'restart_fa_incidence_per_10k_lo',
        'alarms_per_10k_hi': 'restart_fa_incidence_per_10k_hi', 'arl_items': 'items_per_false_alarm_renewal_estimate'})
    a['note'] = 'restart after each alarm on 2,500 horizon-length (H=29,999) streams at constant 0.10, threshold h_hi; not an ARL on long streams'
    a.to_csv(os.path.join(ROOT, 'restart_fa_incidence.csv'), index=False)
    # ---------------- A3 fine q
    fq_v, fq_m = [], []
    for p in sorted(glob.glob(os.path.join(FU, 'fineq_*.json'))):
        r = json.load(open(p)); fq_v += val_rows(r, 'fineq'); fq_m += metric_rows(r, 'fineq')
    fv = pd.DataFrame(fq_v); fm = pd.DataFrame(fq_m)
    if len(fv):
        fv.to_csv(os.path.join(ROOT, 'fa_validation_fineq.csv'), index=False)
        fm.to_csv(os.path.join(ROOT, 'metrics_fineq.csv'), index=False)
        base = evr[(evr.family == 'core') & np.isclose(evr.pi, 0.1) & np.isclose(evr.r1, 0.2)].assign(source='primary')
        allq = pd.concat([base[fm.columns.intersection(base.columns)], fm], ignore_index=True)
        qs = pd.read_csv('/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory/qstar_grid.csv')
        out = []
        for (m, d, D), g in allq.groupby(['variant', 'd', 'D']):
            g = g.sort_values(['q', 'source']).groupby('q', as_index=False).agg(pD=('pD', 'mean'), pD_lo=('pD_lo', 'min'), pD_hi=('pD_hi', 'max'))
            qq = g.q.values
            pf = (g.pD.values >= 0.8).astype(int)
            qo = qs[np.isclose(qs.prevalence, 0.1) & np.isclose(qs.post_risk, 0.2) & (qs.delay == d) & (qs.deadline == D)]
            out.append(dict(variant=m, d=d, D=D, q_min_interp=interp_cross(qq, g.pD.values),
                            q_ci_lo=interp_cross(qq, g.pD_hi.values), q_ci_hi=interp_cross(qq, g.pD_lo.values),
                            n_q=len(qq), nonmonotone_pass_fail=int(np.sum(np.diff(pf) < 0)),
                            q_oracle=float(qo.q_oracle.iloc[0]) if len(qo) else np.nan,
                            q_cusum_exact=float(qo.q_cusum.iloc[0]) if len(qo) else np.nan))
        pd.DataFrame(out).to_csv(os.path.join(ROOT, 'qmin_interp_fineq.csv'), index=False)
        allq.to_csv(os.path.join(ROOT, 'pD_vs_q_pi0.1_r0.2_allq.csv'), index=False)
    # ---------------- oracle curves for figures
    oc = []
    for (pi, r1, q, d) in E.TRADEOFF:
        for al in np.geomspace(1e-3, 0.5, 60):
            oc.append(dict(kind='fa', pi=pi, r1=r1, q=q, d=d, D=10000, alpha=al, beta=E.oracle_beta(pi, q, d, r1, 10000, al)))
    for pi in E.PIS:
        for r1 in E.R1S:
            for d in E.DS:
                for q in np.geomspace(0.05, 1.0, 60):
                    oc.append(dict(kind='q', pi=pi, r1=r1, q=q, d=d, D=10000, alpha=0.02, beta=E.oracle_beta(pi, q, d, r1, 10000)))
    pd.DataFrame(oc).to_csv(os.path.join(ROOT, 'oracle_curves.csv'), index=False)
    # export ECDF data for delay figure
    rows = []
    for (pi, r1, q, d) in E.TRADEOFF:
        idx = E.SETTINGS.index((pi, q, d))
        z = np.load(os.path.join(ROOT, 'raw', f'setting_{idx:02d}_alarms.npz'))
        for m in E.CORE + E.SUPP:
            x = z[f'r{r1:g}__{m}'].astype(float) - E.TAU
            for t in np.arange(0, 20001, 250):
                rows.append(dict(pi=pi, r1=r1, q=q, d=d, monitor=m, t=t, cdf=float(np.mean((x >= 0) & (x <= t)))))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'delay_ecdf.csv'), index=False)
    print('done; repair rows', len(rv), 'fineq rows', len(fv))


if __name__ == '__main__':
    main()
