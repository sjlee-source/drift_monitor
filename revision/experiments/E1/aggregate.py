"""Aggregate E1 raw outputs into CSV tables + quick diagnostic PNGs."""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from revision import calibration as C
from revision.experiments.E1 import e1_common as E

ROOT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1'
RAW = os.path.join(ROOT, 'raw')
FAMILY = len(E.CORE) * len(E.SETTINGS) * 2


def f(x):
    if isinstance(x, str):
        return float(x)
    return x


def load():
    rs = []
    for p in sorted(glob.glob(os.path.join(RAW, 'setting_??.json'))):
        rs.append(json.load(open(p)))
    return rs


def main():
    rs = load()
    print('settings loaded:', len(rs))
    fam_of = {}
    # ---------------- thresholds + validation
    thr_rows, val_rows, arl_rows = [], [], []
    for r in rs:
        key = dict(pi=r['pi'], q=r['q'], d=r['d'])
        for v, x in r['variants'].items():
            fam_of[v] = x['family']
            t = x['thr']
            b = r['thresholds'].get(v, {}).get('binding') if x['family'] in ('core', 'supp') else None
            thr_rows.append(dict(variant=v, monitor=x['monitor'], family=x['family'], **key, h=f(t['h']), gamma=t['gamma'],
                                 binding=b, n_tie=t.get('n_tie')))
            for p in E.PATHS:
                vv = r['validation'][v][p]
                fam = FAMILY if (x['family'] == 'core' and p != 'const05') else 1
                up = C.clopper_pearson(vv['alarms'], vv['n'], 0.95, 'upper', fam)[1]
                val_rows.append(dict(variant=v, family=x['family'], **key, path=p, n=vv['n'], alarms=vv['alarms'], rate=vv['rate'],
                                     cp_lo=vv['cp_lo'], cp_hi=vv['cp_hi'], bonf_family=fam, upper_one_sided=up,
                                     gated=(x['family'] == 'core' and p != 'const05'), pass_gate=up <= 0.03))
        for v, a in r['arl_restart'].items():
            arl_rows.append(dict(variant=v, family=r['variants'][v]['family'], **key,
                                 **{k: f(a[k]) for k in ('streams', 'alarms', 'alarms_per_10k', 'alarms_per_10k_lo',
                                                         'alarms_per_10k_hi', 'arl_items', 'any_alarm_rate')}))
    thr = pd.DataFrame(thr_rows); val = pd.DataFrame(val_rows); arl = pd.DataFrame(arl_rows)
    thr.to_csv(os.path.join(ROOT, 'thresholds.csv'), index=False)
    val.to_csv(os.path.join(ROOT, 'fa_validation.csv'), index=False)
    arl.to_csv(os.path.join(ROOT, 'arl_restart.csv'), index=False)
    vw = val.pivot_table(index=['variant', 'pi', 'q', 'd'], columns='path', values=['rate', 'upper_one_sided'])
    vw.columns = [f'fa_{b}' if a == 'rate' else f'faup_{b}' for a, b in vw.columns]
    vw = vw.reset_index()

    # ---------------- evaluation long table
    rows = []
    for r in rs:
        for r1s, e in r['evaluation'].items():
            r1 = float(r1s)
            for v, m in e['metrics'].items():
                for D in E.DEADLINES:
                    beta = e['oracle'][str(D)]
                    rows.append(dict(variant=v, family=r['variants'][v]['family'], pi=r['pi'], r1=r1, q=r['q'], d=r['d'], D=D,
                                     pD=m[f'pD_{D}'], pD_k=m[f'pD_{D}_k'], pD_lo=m[f'pD_{D}_lo'], pD_hi=m[f'pD_{D}_hi'],
                                     early_rate=m['early_rate'], no_alarm_rate=m['no_alarm_rate'], rmd=m[f'rmd_{D}'],
                                     timely_q50=f(m[f'timely_q{50}_{D}'] if f'timely_q50_{D}' in m else np.nan),
                                     timely_q90=f(m[f'timely_q90_{D}']), oracle_beta=beta,
                                     ratio_to_oracle=m[f'pD_{D}'] / beta if beta > 0 else np.nan,
                                     labels_per_stream=e['labels_per_stream']))
    ev = pd.DataFrame(rows).merge(vw, on=['variant', 'pi', 'q', 'd'], how='left')
    ev.to_csv(os.path.join(ROOT, 'metrics_long.csv'), index=False)
    core = ev[ev.family == 'core']
    # main table at D=10k
    main10 = ev[(ev.D == 10000) & ev.family.isin(['core', 'supp'])].pivot_table(
        index=['pi', 'r1', 'q', 'd'], columns='variant', values='pD').reset_index()
    o = ev[(ev.D == 10000) & (ev.variant == 'cusum_0.20')][['pi', 'r1', 'q', 'd', 'oracle_beta']]
    main10 = main10.merge(o, on=['pi', 'r1', 'q', 'd'])
    main10.to_csv(os.path.join(ROOT, 'main_pD_D10000.csv'), index=False)

    # ---------------- oracle exceedance
    exc = ev[(ev.pD_lo > ev.oracle_beta + 1e-12)]
    exc.to_csv(os.path.join(ROOT, 'oracle_exceedance.csv'), index=False)
    exc_point = ev[ev.pD > ev.oracle_beta + 1e-12]

    # ---------------- q_min
    qs = pd.read_csv('/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory/qstar_grid.csv')
    qrows = []
    for (v, pi, r1, d), g in ev[(ev.D == 10000) & ev.family.isin(['core', 'supp', 'native', 'constcal'])].groupby(
            ['variant', 'pi', 'r1', 'd']):
        g = g.sort_values('q')
        ok = g[g.pD >= 0.80]; okl = g[g.pD_lo >= 0.80]
        qq = qs[(np.isclose(qs.prevalence, pi)) & (np.isclose(qs.post_risk, r1)) & (qs.delay == d) & (qs.deadline == 10000)]
        qrows.append(dict(variant=v, family=g.family.iloc[0], pi=pi, r1=r1, d=d,
                          q_min_point=ok.q.min() if len(ok) else np.inf, q_min_lower=okl.q.min() if len(okl) else np.inf,
                          q_oracle=float(qq.q_oracle.iloc[0]) if len(qq) else np.nan,
                          q_cusum_exact=float(qq.q_cusum.iloc[0]) if len(qq) else np.nan,
                          nonmonotone_pass_fail=int(np.sum(np.diff((g.pD.values >= 0.8).astype(int)) < 0))))
    qm = pd.DataFrame(qrows)
    qm.to_csv(os.path.join(ROOT, 'qmin_summary.csv'), index=False)

    # ---------------- ranking (core matched, D = 10k)
    rk = []
    for D in E.DEADLINES:
        sub = core[core.D == D]
        P = sub.pivot_table(index=['pi', 'r1', 'q', 'd'], columns='variant', values='pD')[E.CORE]
        R = P.apply(lambda row: pd.Series(rankdata(-row.values, method='average'), index=row.index), axis=1)
        best = P.max(axis=1)
        within = P.ge(best - 0.01, axis=0)
        inform = ~(P.isin([0.0, 1.0]).all(axis=1)) & (P.max(axis=1) - P.min(axis=1) > 0)
        Ri = R[inform]; n, k = Ri.shape
        Rs = Ri.sum(axis=0); S = ((Rs - Rs.mean()) ** 2).sum()
        W = 12 * S / (n ** 2 * (k ** 3 - k)) if n else np.nan
        # separate W for "hard" conditions (best p_D < 0.95)
        hard = inform & (best < 0.95)
        Rh = R[hard]; nh = len(Rh)
        Wh = 12 * ((Rh.sum(axis=0) - Rh.sum(axis=0).mean()) ** 2).sum() / (nh ** 2 * (k ** 3 - k)) if nh else np.nan
        for mname in E.CORE:
            rk.append(dict(D=D, monitor=mname, mean_rank_all=R[mname].mean(), mean_rank_informative=Ri[mname].mean(),
                           mean_rank_hard=Rh[mname].mean() if nh else np.nan,
                           n_best_or_within_001=int(within[mname].sum()), n_conditions=len(P),
                           n_informative=n, n_hard=nh, kendall_W_informative=W, kendall_W_hard=Wh,
                           mean_pD=P[mname].mean(), min_ratio_to_best_hard=(P[mname][hard] / best[hard]).min() if nh else np.nan))
    rk = pd.DataFrame(rk)
    rk.to_csv(os.path.join(ROOT, 'ranking.csv'), index=False)

    # ---------------- CUSUM exact DP sanity
    dp_path = os.path.join(ROOT, 'cusum_exact_dp.csv')
    if os.path.exists(dp_path):
        dp = pd.read_csv(dp_path)
        mc = ev[ev.variant == 'cusum_0.20'][['pi', 'r1', 'q', 'd', 'D', 'pD', 'early_rate', 'fa_const10', 'fa_path']]
        s = dp.merge(mc, on=['pi', 'r1', 'q', 'd', 'D'], suffixes=('_exact', '_mc'))
        n = E.N_EVAL
        s['z_pD'] = (s.pD - s.pD_exact) / np.sqrt(np.maximum(s.pD_exact * (1 - s.pD_exact), 1e-12) / n)
        s['diff_pD'] = s.pD - s.pD_exact
        s.to_csv(os.path.join(ROOT, 'cusum_mc_vs_exact.csv'), index=False)

    # ---------------- trade-off summaries
    to_rows = []
    for (pi, r1, q, d) in E.TRADEOFF:
        p = os.path.join(RAW, f'tradeoff_pi{pi:g}_r{r1:g}_q{q:g}_d{d}.csv')
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        for mname, g in t.groupby('monitor'):
            for lev in (0.005, 0.01, 0.02, 0.05):
                ok = g[g.fa_comp <= lev]
                for D in E.DEADLINES:
                    to_rows.append(dict(pi=pi, r1=r1, q=q, d=d, monitor=mname, fa_level=lev, D=D,
                                        pD=ok[f'pD_{D}'].max() if len(ok) else np.nan,
                                        oracle_beta=E.oracle_beta(pi, q, d, r1, D, lev)))
    tos = pd.DataFrame(to_rows)
    tos.to_csv(os.path.join(ROOT, 'tradeoff_summary.csv'), index=False)

    # ---------------- quick plots
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        for (pi, r1, q, d) in E.TRADEOFF:
            p = os.path.join(RAW, f'tradeoff_pi{pi:g}_r{r1:g}_q{q:g}_d{d}.csv')
            if not os.path.exists(p):
                continue
            t = pd.read_csv(p)
            fig, ax = plt.subplots(figsize=(7, 5))
            for mname, g in t.groupby('monitor'):
                g = g.sort_values('fa_comp')
                ax.plot(g.fa_comp, g.pD_10000, label=mname, lw=1)
            fl = np.geomspace(1e-3, 0.5, 40)
            ax.plot(fl, [E.oracle_beta(pi, q, d, r1, 10000, a) for a in fl], 'k--', lw=2, label='oracle beta*')
            ax.set_xscale('log'); ax.set_xlabel('composite FA (max of const 0.10, path)'); ax.set_ylabel('p_D, D=10k')
            ax.set_title(f'pi={pi}, r1={r1}, q={q}, d={d}'); ax.legend(fontsize=6, ncol=2); ax.axvline(0.02, color='grey', lw=0.5)
            fig.tight_layout(); fig.savefig(os.path.join(ROOT, f'tradeoff_pi{pi:g}_r{r1:g}_q{q:g}_d{d}.png'), dpi=110); plt.close(fig)
    except Exception as ex:
        print('plot failed', ex)

    # ---------------- console digest
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
    g = val[val.gated]
    print('gated FA cells', len(g), 'pass', int(g.pass_gate.sum()), 'max rate', g.rate.max(), 'max upper', g.upper_one_sided.max())
    print(g.sort_values('rate', ascending=False).head(8)[['variant', 'pi', 'q', 'd', 'path', 'rate', 'upper_one_sided']])
    print('oracle exceedance (CP lower > beta*):', len(exc), '; point > beta*:', len(exc_point))
    if len(exc_point):
        print(exc_point[['variant', 'pi', 'r1', 'q', 'd', 'D', 'pD', 'pD_lo', 'oracle_beta']].head(30))
    print(rk[rk.D == 10000])


if __name__ == '__main__':
    main()
