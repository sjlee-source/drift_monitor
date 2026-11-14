"""Oracle Neyman-Pearson converse: upper bound on timely detection of ANY monitor.

Null path P0: risk r_pre before tau, r_adm (admissible, e.g. 0.10) from tau on.
Alternative P1: risk r_pre before tau, r1 from tau on.  The two paths share the prefix, audits
are noninformative, so the only distinguishing data available by the deadline are the
post-change-origin labels arrived by tau+D-1:  N ~ Bin(L, rho), L = max(0, D-d), K | N ~ Bin(N, r).
For any monitor whose false-alarm probability over the horizon under P0 is <= alpha, the event
{tau <= T <= tau+D-1} is a level-alpha test of P0 vs P1 measurable w.r.t. that data, so by the
Neyman-Pearson lemma its P1-probability is at most the most-powerful (randomized) test's power.
"""
import numpy as np
from scipy.stats import binom


def np_power(L, rho, r0, r1, alpha, nmax_tail=1e-14):
    if L <= 0:
        return alpha  # no informative data: power cannot exceed size
    n = np.arange(L + 1)
    pn = binom.pmf(n, L, rho)
    keep = pn > nmax_tail
    cells_llr, cells_p0, cells_p1 = [], [], []
    la, lb = np.log(r1 / r0), np.log((1 - r1) / (1 - r0))
    for nn, w in zip(n[keep], pn[keep]):
        k = np.arange(nn + 1)
        cells_llr.append(k * la + (nn - k) * lb)
        cells_p0.append(w * binom.pmf(k, nn, r0))
        cells_p1.append(w * binom.pmf(k, nn, r1))
    llr = np.concatenate(cells_llr); p0 = np.concatenate(cells_p0); p1 = np.concatenate(cells_p1)
    o = np.argsort(-llr, kind='stable')
    llr, p0, p1 = llr[o], p0[o], p1[o]
    # group equal LLR values (ties) for correct randomization
    c0 = np.cumsum(p0); c1 = np.cumsum(p1)
    j = np.searchsorted(c0, alpha, side='right')  # cells fully included: first j
    full0 = c0[j - 1] if j > 0 else 0.0
    full1 = c1[j - 1] if j > 0 else 0.0
    if j >= len(p0):
        return float(c1[-1])
    frac = (alpha - full0) / p0[j] if p0[j] > 0 else 0.0
    return float(full1 + frac * p1[j])


def min_labels(r0, r1, alpha, target, rho_small=1e-3):
    """Smallest expected informative-label count lambda=rho*L at which the oracle power reaches
    target (Poisson-thinning regime, rho small)."""
    lo, hi = 1, 10
    while np_power(int(hi / rho_small), rho_small, r0, r1, alpha) < target:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if np_power(int(mid / rho_small), rho_small, r0, r1, alpha) >= target:
            hi = mid
        else:
            lo = mid
    return hi
