"""Exact finite-horizon CUSUM probabilities under sparse, delayed audits.

Model: production items s = 1..H. Each item is informative (audited AND defective) with
probability rho = pi*q, independently. An informative item is a miss with probability r(s).
Its label arrives at s + d (constant delay preserves origin order). The Bernoulli CUSUM uses
integer-rounded log-likelihood scores (Reynolds & Stoumbos-style integer CUSUM):
    miss -> +A, nonmiss -> -B, reflected at 0, alarm when statistic > c  (strict).
For fine rounding scale this converges to the log-likelihood CUSUM; the integer chart is itself
an implementable monitor whose probabilities are computed exactly (up to float rounding).
"""
import numpy as np


def scores(p, r0, scale_B):
    a = np.log(p / r0); b = np.log((1 - p) / (1 - r0))
    s = scale_B / (-b)
    return int(round(a * s)), int(scale_B), s


def alarm_by_origin(H, d, rho, risk_fn, A, B, c):
    """Return array m[s] = P(first alarm triggered by the label of origin s), s=1..H-d.
    Alarm production time = s + d."""
    if H <= d:
        raise ValueError("delay must be shorter than horizon")
    S = H - d
    v = np.zeros(c + 1); v[0] = 1.0
    out = np.zeros(S + 1)
    for s in range(1, S + 1):
        r = risk_fn(s)
        stay = (1 - rho) * v
        miss = rho * r * v
        non = rho * (1 - r) * v
        new = stay
        # miss: shift up by A; overflow beyond c alarms
        if A <= c:
            new[A:] += miss[:c + 1 - A]
            out[s] = miss[c + 1 - A:].sum()
        else:
            out[s] = miss.sum()
        # nonmiss: shift down by B, floor at 0
        if B <= c:
            new[:c + 1 - B] += non[B:]
            new[0] += non[:B].sum()
        else:
            new[0] += non.sum()
        v = new
    return out


def detection_prob(H, d, rho, tau, r_pre, r_post, D, A, B, c):
    m = alarm_by_origin(H, d, rho, lambda s: r_pre if s < tau else r_post, A, B, c)
    lo = max(1, tau - d); hi = tau + D - 1 - d
    if hi > H - d:
        raise ValueError("deadline exceeds simulated alarm horizon")
    if D <= 0 or hi < lo:
        return 0.0, m[1:lo].sum()
    early = m[1:lo].sum() if lo > 1 else 0.0
    return m[lo:hi + 1].sum(), early


def false_alarm(H, d, rho, risk_fn, A, B, c):
    return alarm_by_origin(H, d, rho, risk_fn, A, B, c).sum()


def calibrate(H, d, rho, A, B, alpha, null_risk):
    """Smallest integer c >= 0 with exact FA(c) <= alpha under constant null_risk
    (FA is nonincreasing in c pathwise)."""
    lo, hi = 0, 2 * A
    while false_alarm(H, d, rho, lambda s: null_risk, A, B, hi) > alpha:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if false_alarm(H, d, rho, lambda s: null_risk, A, B, mid) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo, false_alarm(H, d, rho, lambda s: null_risk, A, B, lo)
