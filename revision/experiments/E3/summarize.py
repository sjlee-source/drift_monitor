"""Summarize E3 raw results into CSV/JSON.  python -m revision.experiments.E3.summarize"""
import glob, os, pickle, json, sys
import numpy as np, pandas as pd
from revision import calibration as C
from . import design as Dz
from .kernel import ST
sys.path.insert(0, '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory')
from converse import np_power

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E3'
RAW = os.path.join(OUT, 'raw')
IDX = {s: i for i, s in enumerate(ST)}
NBOOT = 10000


def pD(a, D):
    return (a >= Dz.TAU) & (a <= min(Dz.H, Dz.TAU + D - 1))


def boot(x, y, seed):
    d = x.astype(float) - y.astype(float); n = len(d)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(NBOOT // 500):
        idx = rng.integers(0, n, (500, n))
        bs.append(d[idx].mean(1))
    bs = np.concatenate(bs)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    R = {}
    for fn in sorted(glob.glob(RAW + '/*.pkl')):
        r = pickle.load(open(fn, 'rb')); R[(r['cond'], r['wf'], r['policy'])] = r
    fam = 2 * len(R)
    thr_rows, val_rows, ev_rows, con_rows = [], [], [], []
    for (cond, wf, name), r in R.items():
        cl = Dz.CONDITIONS[cond]['label']; wfl = 'W-A' if wf == 0 else 'W-B'
        base = dict(condition=cl, workflow=wfl, policy=name)
        tm, tb = r['thr']['msr'][0], r['thr']['bet'][0]
        thr_rows.append(dict(base, h_msr=tm.h, h_msr_const=r['thr']['msr'][1], h_msr_path=r['thr']['msr'][2], gamma_msr=tm.gamma,
                             h_bet=tb.h, h_bet_const=r['thr']['bet'][1], h_bet_path=r['thr']['bet'][2], h_warn=r['h_warn'],
                             fixed_point_iters=len(r['iters']),
                             fixed_point_last=(None if not r['iters'] else '%.4f->%.4f' % r['iters'][-1])))
        for path, pn in ((0, 'const_0.10'), (1, 'path_0.05to0.10'), (2, 'const_0.05')):
            v = r['val'][path]
            row = dict(base, null=pn)
            for mon in ('msr', 'bet'):
                vv = C.validate(v['fa_' + mon], cap=0.03, family=fam)
                row[f'FA_{mon}'] = vv['rate']; row[f'FA_{mon}_k'] = vv['alarms']
                row[f'FA_{mon}_cp_lo'] = vv['cp_lo']; row[f'FA_{mon}_cp_hi'] = vv['cp_hi']
                row[f'FA_{mon}_upper_bonf'] = vv['upper_simultaneous']
                row[f'gate_{mon}'] = bool(vv['pass_gate']) if path < 2 else None
            row['FA_bet_native_log50'] = float(v['native_bet'].mean())
            sm = v['stats_mean']
            row['HT_rhat_null'] = sm[IDX['sum_wx']] / sm[IDX['sum_w']]
            row['warn_occ_pre'] = sm[IDX['warn_pre']] / (Dz.TAU - 1)
            val_rows.append(row)
        st = r['ev']['stats']; n = st.shape[0]
        row = dict(base)
        d = Dz.CONDITIONS[cond]['d']; pi = Dz.CONDITIONS[cond]['pi']
        for mon in ('msr', 'bet'):
            a = r['ev']['alarm_' + mon]
            row[f'early_{mon}'] = float(np.mean(a < Dz.TAU))
            for D in Dz.DEADLINES:
                k = int(pD(a, D).sum()); lo, hi = C.clopper_pearson(k, n)
                row[f'pD{D//1000}k_{mon}'] = k / n; row[f'pD{D//1000}k_{mon}_lo'] = lo; row[f'pD{D//1000}k_{mon}_hi'] = hi
        g = lambda s: st[:, IDX[s]]
        row.update(audits_total=g('n_audits').mean(), audits_total_min=g('n_audits').min(),
                   audits_pre=g('aud_pre').mean(), audits_window=g('aud_win').mean(),
                   audits_window_arrived=g('aud_win_arrived').mean(),
                   audit_labels_window_arrived=g('inf_win_arr').mean(), free_labels_window_arrived=g('free_lab_win').mean(),
                   miss_labels_window_arrived=g('miss_lab_win').mean(), true_misses_window=g('true_miss_win').mean(),
                   yield_overall=float(g('n_audit_inf').sum() / g('n_audits').sum()),
                   yield_window=float(g('aud_inf_win').sum() / max(1, g('aud_win').sum())),
                   ess_window=float(np.mean(np.where(g('sw2_win') > 0, g('sw_win') ** 2 / np.maximum(g('sw2_win'), 1e-300), 0))),
                   forced=g('n_forced').mean(), floor_violations=g('n_floor_viol').mean(),
                   warn_occ_pre=g('warn_pre').mean() / (Dz.TAU - 1), warn_occ_window=g('warn_win').mean() / Dz.D_PRIMARY)
        L = Dz.D_PRIMARY - d
        row['beta_star_at_realized_audit_labels_10k'] = np_power(L, row['audit_labels_window_arrived'] / L, Dz.R0, Dz.R1, Dz.TARGET_FA) if wf == 0 else np.nan
        ev_rows.append(row)
    for (cond, wf, name), r in R.items():
        ref = 'P1_horizon_uniform' if wf == 0 else 'B1_passed_uniform'
        if name == ref or (cond, wf, ref) not in R:
            continue
        rr = R[(cond, wf, ref)]
        for mon in ('msr', 'bet'):
            for D in Dz.DEADLINES:
                x = pD(r['ev']['alarm_' + mon], D); y = pD(rr['ev']['alarm_' + mon], D)
                dm, lo, hi = boot(x, y, seed=13_000 + cond * 100 + wf * 10 + D // 5000)
                prim = (cond == 0 and wf == 0 and name == 'P5_score_stratified' and mon == 'msr' and D == 10000)
                con_rows.append(dict(condition=Dz.CONDITIONS[cond]['label'], workflow='W-A' if wf == 0 else 'W-B', policy=name,
                                     reference=ref, monitor=mon, D=D, diff=dm, boot_lo=lo, boot_hi=hi, primary=prim,
                                     note='PRIMARY (prespecified)' if prim else 'descriptive, unadjusted'))
    beta = []
    for cond, c in Dz.CONDITIONS.items():
        for D in Dz.DEADLINES:
            L = min(D, Dz.H - Dz.TAU + 1) - c['d']
            beta.append(dict(condition=c['label'], D=D, L=L, rho_uniform=Dz.QBAR * c['pi'],
                             beta_star_uniform=np_power(L, Dz.QBAR * c['pi'], Dz.R0, Dz.R1, Dz.TARGET_FA),
                             beta_star_full_information=np_power(L, c['pi'], Dz.R0, Dz.R1, Dz.TARGET_FA)))
    for nm, rows in (('thresholds', thr_rows), ('validation', val_rows), ('evaluation', ev_rows),
                     ('contrasts', con_rows), ('beta_star', beta)):
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'{nm}.csv'), index=False)
    v = pd.DataFrame(val_rows)
    gate = dict(family=fam, cap=0.03,
                msr_all_pass=bool(v[v.null != 'const_0.05'].gate_msr.all()), bet_all_pass=bool(v[v.null != 'const_0.05'].gate_bet.all()),
                msr_max_upper=float(v[v.null != 'const_0.05'].FA_msr_upper_bonf.max()),
                bet_max_upper=float(v[v.null != 'const_0.05'].FA_bet_upper_bonf.max()),
                native_bet_max=float(v.FA_bet_native_log50.max()))
    prim = [c for c in con_rows if c['primary']]
    json.dump(dict(gate=gate, primary=prim), open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print(json.dumps(gate, indent=1)); print(prim)


if __name__ == '__main__':
    main()
