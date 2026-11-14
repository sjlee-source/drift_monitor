"""E2c summary tables."""
import os
import pandas as pd
O = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E2/E2c'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
fa = pd.concat([pd.read_csv(f'{O}/fa_{a}.csv') for a in ('primary', 'curve1H', 'curve5H', 'incontrol')
                if os.path.exists(f'{O}/fa_{a}.csv')])
de = pd.concat([pd.read_csv(f'{O}/detection_{a}.csv') for a in ('primary', 'curve1H', 'curve5H', 'incontrol')
                if os.path.exists(f'{O}/detection_{a}.csv')])
fa.to_csv(f'{O}/summary_fa_all.csv', index=False); de.to_csv(f'{O}/summary_detection_all.csv', index=False)
fam = fa[fa.path != 'null_const005']
g = fam.groupby(['arm', 'proc']).agg(primary_pass=('pass_primary', 'sum'), n=('pass_primary', 'size'),
                                     secondary_pass=('pass_secondary', lambda s: s.dropna().sum()),
                                     worst_fa=('fa_uncond', 'max'), worst_upper=('upper_cluster', 'max'))
c5 = fa[fa.path == 'null_const005'].groupby(['arm', 'proc']).agg(c005_pass=('pass_primary', 'sum'),
                                                                   c005_n=('pass_primary', 'size'),
                                                                   c005_worst=('fa_uncond', 'max'))
g = g.join(c5); g.to_csv(f'{O}/summary_gate_counts.csv'); print(g.to_string())
print(fam[fam.proc.isin(['RC', 'NR'])].groupby(['arm', 'proc', 'variant']).pass_primary.sum().unstack().to_string())
print(fam[fam.proc.isin(['RC', 'NR'])].groupby(['arm', 'proc', 'variant']).pass_secondary.sum().unstack().to_string())
bad = fam[(fam.proc.isin(['RC', 'NR'])) & ((~fam.pass_primary) | (fam.pass_secondary == False))]
print(bad[['arm', 'proc', 'q', 'variant', 'monitor', 'path', 'fa_uncond', 'upper_cluster', 'P_hist_fa_gt_003',
           'upper_P_hist_gt_003', 'pass_primary', 'pass_secondary']].round(4).to_string())
w = fa.pivot_table(index=['arm', 'proc', 'variant', 'q'], columns=['path', 'monitor'], values='fa_uncond').round(4)
w.to_csv(f'{O}/summary_fa_uncond_wide.csv')
p = de[de.D == 10000].pivot_table(index=['arm', 'proc', 'variant', 'q'], columns='monitor',
                                    values=['pD', 'pD_hist_p05', 'pD_hist_p95', 'median_delay_timely', 'h_sd']).round(3)
p.to_csv(f'{O}/summary_detection_D10k_wide.csv')
print(de[(de.D == 10000) & (de.arm == 'primary')].pivot_table(index=['variant', 'q'], columns=['monitor', 'proc'],
                                                               values='pD').round(3).to_string())
