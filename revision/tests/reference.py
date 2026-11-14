"""Deliberately naive pure-Python reference implementations (independent of the numba code)."""
import math
import numpy as np
from scipy.stats import binom


def betting(x, r0=0.1, gain=4, cap=9, win=50):
    out, hist, lw = [], [], 0.0
    for v in x:
        prev = np.mean(hist[-win:]) if hist else 0.0
        lam = min(cap, gain * max(prev - r0, 0))
        lw += math.log1p(lam * (v - r0)); hist.append(v); out.append(lw)
    return np.array(out)


def cusum(x, p1, r0=0.1):
    s, out = 0.0, []
    for v in x:
        s = max(0.0, s + (math.log(p1 / r0) if v else math.log((1 - p1) / (1 - r0)))); out.append(s)
    return np.array(out)


def edetector_direct(x, stakes, weights, r0=0.1, kind='SR'):
    """Literal Eqs. (24)/(25): M_n = sum_k w_k [sum_j | max_j] prod_{i=j}^n (1 + stake_k (X_i - r0))."""
    x = np.asarray(x, float)
    out = []
    for n in range(1, len(x) + 1):
        tot = 0.0
        for st, w in zip(stakes, weights):
            prods = [np.prod(1 + st * (x[j:n] - r0)) for j in range(n)]
            tot += w * (sum(prods) if kind == 'SR' else max(prods))
        out.append(math.log(tot))
    return np.array(out)


def periodic(x, arrival, H, r0=0.1, interval=3000):
    checks = np.unique(np.r_[np.arange(interval, H + 1, interval), H])
    out, prev = [], 0
    for c in checks:
        sel = (arrival > prev) & (arrival <= c)
        m, s = int(sel.sum()), int(np.asarray(x)[sel].sum())
        out.append(0.0 if m == 0 else -binom.logsf(s - 1, m, r0)); prev = c
    return checks, np.array(out)


def ewma(x, r0=0.1, lam=0.05):
    z, out = r0, []
    for i, v in enumerate(x, 1):
        z = (1 - lam) * z + lam * v
        var = r0 * (1 - r0) * lam / (2 - lam) * (1 - (1 - lam) ** (2 * i))
        out.append((z - r0) / math.sqrt(var))
    return np.array(out)


def glr(x, r0=0.1, W=500):
    x = np.asarray(x, float); out = []
    for n in range(1, len(x) + 1):
        best = 0.0
        for k in range(max(0, n - W), n):        # all starts, not only misses
            seg = x[k:n]; m = len(seg); s = seg.sum(); p = max(s / m, r0)
            if p > r0:
                llr = s * math.log(p / r0) + ((m - s) * math.log((1 - p) / (1 - r0)) if m > s else 0.0)
                best = max(best, llr)
        out.append(best)
    return np.array(out)


def window_binom(x, r0=0.1, W=200):
    x = np.asarray(x); out = []
    for n in range(1, len(x) + 1):
        seg = x[max(0, n - W):n]
        s = int(seg.sum()); m = len(seg)
        out.append(0.0 if s == 0 else -binom.logsf(s - 1, m, r0))
    return np.array(out)


def adwin_first_alarm(x, delta, min_len=5, M=5):
    """Direct ADWIN2 epsilon test on the same split grid; returns first alarm index (0-based) or -1."""
    x = np.asarray(x, float)
    for i in range(len(x)):
        n = i + 1
        if n < 2 * min_len or n < 3:
            continue
        w = x[:n]; var = w.mean() * (1 - w.mean())
        dd = math.log(2 * math.log(n) / delta)
        n1 = min_len
        while n - n1 >= min_len:
            n0 = n - n1
            mu1, mu0 = w[n0:].mean(), w[:n0].mean()
            m = 1 / (1 / n0 + 1 / n1)
            eps = math.sqrt(2 * var * dd / m) + 2 * dd / (3 * m)
            if mu1 - mu0 > eps:
                return i
            nn = int(n1 * (1 + 1 / M)); n1 = nn if nn > n1 else n1 + 1
    return -1


def ddm_first_alarm(x, n_min=30, level=3.0):
    err, pmin, smin = 0, math.inf, math.inf
    for i, v in enumerate(x):
        err += v; k = i + 1; p = err / k; s = math.sqrt(p * (1 - p) / k)
        if k >= n_min:
            if p + s <= pmin + smin:
                pmin, smin = p, s
            if p + s > pmin + level * smin:
                return i
    return -1


def cs_detector(x, r0=0.1, alpha=0.02, G=25, K=40, c=0.5):
    x = np.asarray(x, float); out = []
    for n in range(1, len(x) + 1):
        starts = [j for j in range(0, n, G)][-K:]
        best = -math.inf
        for j in starts:
            lk, sx, ssq, sig2 = 0.0, 0.0, 0.0, 0.25
            for t, v in enumerate(x[j:n], 1):
                lam = min(math.sqrt(2 * math.log(1 / alpha) / (sig2 * t * math.log(1 + t))), c / r0)
                lk += math.log1p(lam * (v - r0)); sx += v; mu = (0.5 + sx) / (t + 1)
                ssq += (v - mu) ** 2; sig2 = (0.25 + ssq) / (t + 1)
            best = max(best, lk)
        out.append(best)
    return np.array(out)
