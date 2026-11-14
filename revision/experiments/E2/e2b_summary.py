"""E2b summary: gate counts and compact tables (primary + post hoc)."""
import os
import pandas as pd
OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E2/E2b'
f = pd.read_csv(f'{OUT}/fa.csv'); d = pd.read_csv(f'{OUT}/detection.csv')
fs = [f.assign(method=f.threshold)[['method', 'q', 'variant', 'monitor', 'path', 'fa_diag', 'upper_sim112', 'pass_gate',
                                     'fa_cross_mean', 'fa_hist_p50', 'fa_hist_p90', 'P_hist_fa_gt_003']]]
ds = [d.assign(method=d.threshold)[['method', 'q', 'variant', 'monitor', 'D', 'pD', 'pD_hist_p05', 'pD_hist_p95']]]
for tag in ('PH1', 'PH2'):
    if os.path.exists(f'{OUT}/posthoc_{tag}_fa.csv'):
        x = pd.read_csv(f'{OUT}/posthoc_{tag}_fa.csv'); x['method'] = f'posthoc_{tag}'
        fs.append(x[fs[0].columns]); y = pd.read_csv(f'{OUT}/posthoc_{tag}_detection.csv'); y['method'] = f'posthoc_{tag}'
        ds.append(y[ds[0].columns])
F = pd.concat(fs); Dd = pd.concat(ds)
F.to_csv(f'{OUT}/summary_fa_all_methods.csv', index=False); Dd.to_csv(f'{OUT}/summary_detection_all_methods.csv', index=False)
g = F[F.path != 'null_const005']
gate = g.groupby(['method', 'variant']).pass_gate.agg(lambda s: f'{int(s.sum())}/{len(s)}').unstack()
gate['total'] = g.groupby('method').pass_gate.agg(lambda s: f'{int(s.sum())}/{len(s)}')
gate.to_csv(f'{OUT}/summary_gate_counts.csv')
pd.set_option('display.width', 250)
print(gate.to_string())
print(g.groupby(['method', 'monitor']).pass_gate.sum().unstack().to_string())
w = g.groupby(['method', 'variant', 'q', 'monitor']).fa_diag.max().unstack('monitor').round(3)
w.to_csv(f'{OUT}/summary_worstFA_by_method.csv'); print(w.to_string())
p = Dd[Dd.D == 10000].pivot_table(index=['method', 'variant', 'q'], columns='monitor', values='pD').round(3)
p.to_csv(f'{OUT}/summary_pD10k_by_method.csv'); print(p.to_string())
