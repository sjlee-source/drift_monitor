"""A2 (exploratory): CRN paired differences of timely-detection indicators."""
import itertools, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from revision.experiments.E1 import e1_common as E
ROOT = os.environ.get('E1_ROOT', '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1')
MS = ['msr', 'cusum_0.20', 'eSR', 'winlr_D10000']
rows = []
for (pi, r1, q, d) in E.TRADEOFF:
    idx = E.SETTINGS.index((pi, q, d))
    z = np.load(os.path.join(ROOT, 'raw', f'setting_{idx:02d}_alarms.npz'))
    for D in E.DEADLINES:
        dl = min(E.H, E.TAU + D - 1)
        I = {m: ((z[f'r{r1:g}__{m}'] >= E.TAU) & (z[f'r{r1:g}__{m}'] <= dl)).astype(float) for m in MS}
        for a, b in itertools.combinations(MS, 2):
            x = I[a] - I[b]; n = len(x); mu = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
            n10 = int((x > 0).sum()); n01 = int((x < 0).sum())
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            rows.append(dict(pi=pi, r1=r1, q=q, d=d, D=D, a=a, b=b, pD_a=I[a].mean(), pD_b=I[b].mean(), diff=mu,
                             ci_lo=mu - 1.96 * se, ci_hi=mu + 1.96 * se, discordant_a_only=n10, discordant_b_only=n01,
                             mcnemar_p=p, label='exploratory, no multiplicity adjustment'))
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, 'crn_paired_diff.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
print(df[['pi', 'r1', 'q', 'd', 'D', 'a', 'b', 'diff', 'ci_lo', 'ci_hi', 'mcnemar_p']].round(4).to_string(index=False))
