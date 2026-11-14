"""E2 summary tables from fa.csv / detection.csv (+ analytic noisy-label risks)."""
import numpy as np, pandas as pd
OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E2'
fa = pd.read_csv(f'{OUT}/fa.csv'); det = pd.read_csv(f'{OUT}/detection.csv')

# ---- nominal-threshold FA classification (family = 364 cells: 13 variants x 7 mon x 2 q x 2 null paths)
n = fa[(fa.threshold == 'nominal') & (fa.path != 'null_const005')].copy()
n['broken_prespec'] = (n.upper_sim364 > 0.03) & (n.fa > 0.03)          # prespecified rule
n['inflated_prespec'] = (~n.broken_prespec) & (n.fa > 0.02) & (n.lower_sim364 > 0.02)
n['broken_conf_posthoc'] = n.lower_sim364 > 0.03                        # POST HOC: simultaneous lower bound > 0.03
w = n.pivot_table(index=['variant', 'q', 'monitor'], columns='path',
                  values=['fa', 'lower_sim364', 'upper_sim364'], aggfunc='first')
w.columns = [f'{a}_{b}' for a, b in w.columns]
g = n.groupby(['variant', 'q', 'monitor'])
w['worst_fa'] = g.fa.max()
w['n_cells_broken_prespec'] = g.broken_prespec.sum()
w['n_cells_inflated_prespec'] = g.inflated_prespec.sum()
w['n_cells_broken_conf_posthoc'] = g.broken_conf_posthoc.sum()
w['status'] = np.where(w.n_cells_broken_conf_posthoc > 0, 'broken(lower>.03)',
               np.where(w.n_cells_broken_prespec > 0, 'broken(point>.03)',
               np.where(w.n_cells_inflated_prespec > 0, 'inflated(>.02 sig)', 'controlled')))
c5 = fa[(fa.threshold == 'nominal') & (fa.path == 'null_const005')].set_index(['variant', 'q', 'monitor']).fa
w['fa_const005'] = c5
w.reset_index().to_csv(f'{OUT}/summary_fa_nominal.csv', index=False)

# ---- recalibrated FA (secondary)
r = fa[(fa.threshold == 'recalibrated') & (fa.path != 'null_const005')]
r.pivot_table(index=['variant', 'q', 'monitor'], columns='path', values='fa').reset_index() \
    .to_csv(f'{OUT}/summary_fa_recalibrated.csv', index=False)

# ---- detection: pD at D in {5k,10k,20k}, nominal & recalibrated, difference vs base (CRN-paired, point only)
cols = ['pD_5000', 'pD_10000', 'pD_20000', 'early_rate']
d = det.set_index(['variant', 'q', 'monitor', 'threshold'])[cols + ['pD_10000_lo', 'pD_10000_hi']]
base = det[det.variant == 'base_iid'].set_index(['q', 'monitor', 'threshold'])[cols]
dd = d.reset_index()
for cname in ['pD_10000', 'pD_5000', 'pD_20000']:
    dd[f'd_{cname}_vs_base'] = dd.apply(lambda x: x[cname] - base.loc[(x.q, x.monitor, x.threshold), cname], axis=1)
dd.to_csv(f'{OUT}/summary_detection.csv', index=False)

# ---- analytic noisy-label risks (stressor 6)
e10, e01 = 0.05, 0.01
f = lambda x: x * (1 - e10) + (1 - x) * e01
kl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
rows = [dict(true_r=x, noisy_r=f(x)) for x in (0.05, 0.10, 0.20)]
inv = (0.10 - e01) / (1 - e10 - e01)
rows.append(dict(true_r=inv, noisy_r=0.10, note='true risk whose noisy-label risk equals r0=0.10'))
an = pd.DataFrame(rows)
an.to_csv(f'{OUT}/labelnoise_analytic.csv', index=False)
print(an.to_string())
print('KL(r1||r0) true', kl(0.2, 0.1), 'noisy vs r0', kl(f(0.2), 0.10), 'noisy r1 vs noisy r0', kl(f(0.2), f(0.1)))
print(w.status.groupby(level=['variant']).value_counts().unstack(fill_value=0).to_string())
