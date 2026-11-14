"""Threshold calibration, null validation, threshold sweeps and detection metrics.

Conventions
-----------
* FA (false alarm) = P(any alarm in production items 1..H) under the null.
* p_D = P(tau <= T <= tau + D - 1) (deadline clipped at H); early alarms (T < tau) and
  non-detections are failures; all streams stay in the denominator.
* Restricted mean delay at D: mean of (T - tau if detected by the deadline else D).
* Ties: several statistics are lattice-valued (CUSUM, binomial tests, DDM), so exact ties
  of the max statistic occur, and mathematically equal lattice values can differ by ~1e-13
  after floating point.  The randomized rule therefore groups values within
  tol*max(1,|h|) (default tol=1e-9) of h into one tie class:
      alarm at first S > h_hi = h + tol_abs;  streams with tiebreak V < gamma instead alarm at
      first S > h_lo = h - tol_abs,
  so the attained calibration FA = (#{M > h_hi} + gamma #{h_lo < M <= h_hi}) / N = target exactly.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.stats import beta, chi2

BIG = np.finfo(float).max


# ----------------------------------------------------------------------------- intervals

def clopper_pearson(k, n, conf=0.95, sided='two', family=1):
    """Exact CP interval.  sided='two' -> (lo, hi) with alpha/(2*family) per tail;
    sided='upper' -> (0, hi) one-sided with alpha/family."""
    a = (1 - conf) / family
    if sided == 'upper':
        hi = 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))
        return 0.0, hi
    lo = 0.0 if k <= 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def poisson_ci(k, exposure, conf=0.95):
    a = 1 - conf
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


# ----------------------------------------------------------------------------- thresholds

@dataclass
class Threshold:
    h: float
    gamma: float = 0.0
    h_hi: float = None
    h_lo: float = None
    method: str = 'strict'
    target: float = None
    attained_cal: float = None
    n_cal: int = None
    n_above: int = None
    n_tie: int = None

    def __post_init__(self):
        if self.h_hi is None:
            self.h_hi = self.h
        if self.h_lo is None:
            self.h_lo = self.h

    def as_dict(self):
        return asdict(self)


def strict(h):
    """Plain strict threshold S > h (e.g. native or archived thresholds)."""
    return Threshold(h=float(h))


def calibrate_threshold(maxima, target=0.02, method='randomized', tol=1e-9):
    """Calibrate on per-stream maxima of S over 1..H under the calibration null.

    method='randomized': attained calibration FA equals target exactly (in expectation over
        the tiebreak uniforms), ties grouped within tol*max(1,|h|).
    method='order_statistic': original rule h = sorted(M)[ceil((N+1)(1-target)) - 1], S > h.
    """
    x = np.asarray(maxima, float)
    N = len(x)
    if method == 'order_statistic':
        rank = int(np.ceil((N + 1) * (1 - target))) - 1
        h = float(np.sort(x)[min(rank, N - 1)])
        return Threshold(h=h, method=method, target=target, attained_cal=float(np.mean(x > h)), n_cal=N,
                         n_above=int(np.sum(x > h)), n_tie=0)
    assert method == 'randomized'
    kstar = target * N
    d = np.sort(x)[::-1]
    k = int(np.floor(kstar))
    if k >= N:
        return Threshold(h=-np.inf, h_hi=-BIG, h_lo=-BIG, method=method, target=target, attained_cal=1.0, n_cal=N)
    h = float(d[k])
    if np.isposinf(h):
        hi, lo = np.inf, BIG
    elif np.isneginf(h):
        hi, lo = -BIG, -np.inf
    else:
        ta = tol * max(1.0, abs(h))
        hi, lo = h + ta, h - ta
    n_gt = int(np.sum(x > hi))
    n_tie = int(np.sum((x > lo) & (x <= hi)))
    gamma = 0.0 if n_tie == 0 else float(np.clip((kstar - n_gt) / n_tie, 0.0, 1.0))
    att = (n_gt + gamma * n_tie) / N
    return Threshold(h=h, gamma=gamma, h_hi=hi, h_lo=lo, method=method, target=target, attained_cal=att, n_cal=N,
                     n_above=n_gt, n_tie=n_tie)


def alarm_times(monitor, batch, thr, weights=None):
    """First alarm items (float; inf = none) under a Threshold (randomized rule if gamma>0)."""
    if not isinstance(thr, Threshold):
        thr = strict(thr)
    if thr.gamma <= 0:
        return monitor.evaluate(batch, (thr.h_hi,), (), weights)['alarm'][:, 0]
    e = monitor.evaluate(batch, (thr.h_hi, thr.h_lo), (), weights)['alarm']
    return np.where(batch.tiebreak < thr.gamma, e[:, 1], e[:, 0])


def any_alarm(monitor, batch, thr, horizon=None, weights=None):
    """Bool per stream: alarm within 1..horizon (default H)."""
    a = alarm_times(monitor, batch, thr, weights)
    return a <= (batch.H if horizon is None else horizon)


def fa_from_maxima(maxima, thr, tiebreak):
    """FA indicator from per-stream maxima (cheap; equivalent to any_alarm within the horizon)."""
    m = np.asarray(maxima, float)
    if not isinstance(thr, Threshold):
        thr = strict(thr)
    return np.where(tiebreak < thr.gamma, m > thr.h_lo, m > thr.h_hi)


# ----------------------------------------------------------------------------- validation

def validate(alarmed, cap=0.05, family=1, conf=0.95):
    """Null validation with exact CP bounds (one-sided upper Bonferroni over `family`)."""
    k = int(np.sum(alarmed)); n = len(alarmed)
    lo, hi = clopper_pearson(k, n, conf)
    _, up = clopper_pearson(k, n, conf, 'upper', family)
    return dict(n=n, alarms=k, rate=k / n, cp_lo=lo, cp_hi=hi, upper_simultaneous=up, cap=cap,
                pass_gate=bool(up <= cap), family=family)


# ----------------------------------------------------------------------------- metrics

def detection_metrics(alarm, tau, H, deadlines=(5000, 10000, 20000), quantiles=(0.5, 0.9), conf=0.95, family=1):
    """Metrics for one harmful condition from first-alarm items (float, inf = none)."""
    a = np.asarray(alarm, float)
    n = len(a)
    early = a < tau
    out = dict(n=n, early=int(early.sum()), early_rate=float(early.mean()),
               no_alarm_H=int(np.isinf(a).sum()), no_alarm_rate=float(np.isinf(a).mean()))
    for D in deadlines:
        dl = min(H, tau + D - 1)
        s = (a >= tau) & (a <= dl)
        k = int(s.sum())
        lo, hi = clopper_pearson(k, n, conf, family=family)
        out[f'pD_{D}'] = k / n; out[f'pD_{D}_k'] = k; out[f'pD_{D}_lo'] = lo; out[f'pD_{D}_hi'] = hi
        out[f'rmd_{D}'] = float(np.mean(np.where(s, a - tau, D)))
    det = (a >= tau) & np.isfinite(a)
    out['cond_mean_delay'] = float(np.mean(a[det] - tau)) if det.any() else np.nan
    ne = a[~early] - tau
    for q in quantiles:
        out[f'delay_q{q:g}'] = float(np.quantile(ne, q, method='inverted_cdf')) if len(ne) else np.nan
    return out


def arl0(monitor, batch, thr, weights=None, conf=0.95):
    """ARL0 / false-alarm frequency from null streams with restart after each alarm.

    Use long null streams (e.g. H = 1e6) for an ARL in production items; on horizon-length
    streams this gives false alarms per 10,000 items within H.  ARL estimate = total
    production items / total alarms (renewal estimator).
    """
    h = thr.h_hi if isinstance(thr, Threshold) else float(thr)
    r = monitor.restart_run(batch, h, weights)
    tot = int(r['n_alarms'].sum())
    exposure = batch.n * batch.H
    lo, hi = poisson_ci(tot, exposure, conf)
    return dict(streams=batch.n, H=batch.H, alarms=tot, items=exposure,
                alarms_per_10k=1e4 * tot / exposure, alarms_per_10k_lo=1e4 * lo, alarms_per_10k_hi=1e4 * hi,
                arl_items=exposure / tot if tot else np.inf,
                arl_items_lo=1 / hi if hi > 0 else np.inf, arl_items_hi=1 / lo if lo > 0 else np.inf,
                any_alarm_rate=float(np.isfinite(r['first_alarm']).mean()))


# ----------------------------------------------------------------------------- sweeps

def threshold_grid(null_maxima, fa_levels=None, n=120):
    """Thresholds hitting a log-spaced set of FA levels (from null maxima quantiles)."""
    if fa_levels is None:
        fa_levels = np.geomspace(1e-3, 0.5, n)
    x = np.asarray(null_maxima, float)
    x = x[np.isfinite(x)] if np.isfinite(x).any() else x
    g = np.quantile(x, 1 - np.asarray(fa_levels), method='lower')
    return np.unique(g)


def tradeoff_curve(monitor, null_batch, alt_batch, tau, grid=None, deadlines=(5000, 10000, 20000), weights_null=None,
                   weights_alt=None):
    """Detection-vs-FA trade-off: for every grid threshold h (strict S > h) the null FA within H
    and p_D on the harmful streams.  Returns a DataFrame (one row per threshold)."""
    nm = monitor.max_stat(null_batch, weights=weights_null)
    if grid is None:
        grid = threshold_grid(nm)
    grid = np.asarray(grid, float)
    fa = (nm[:, None] > grid[None, :]).mean(0)
    A = monitor.alarm_grid(alt_batch, grid, weights_alt)
    H = alt_batch.H
    df = pd.DataFrame(dict(h=grid, fa=fa, early=(A < tau).mean(0)))
    for D in deadlines:
        dl = min(H, tau + D - 1)
        df[f'pD_{D}'] = ((A >= tau) & (A <= dl)).mean(0)
    return df


def pd_at_fa(curve, fa_target, D):
    """p_D at the largest grid threshold-index whose FA <= fa_target (from tradeoff_curve)."""
    ok = curve[curve.fa <= fa_target]
    return float(ok[f'pD_{D}'].max()) if len(ok) else np.nan
