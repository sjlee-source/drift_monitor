"""Independent Markov-chain evaluation of a delayed Bernoulli CUSUM.

Grid: delta = |b|/M so b is exact; a rounded to nearest multiple of delta.
State k means S = k*delta, alive states k=0..n with n = floor(h/delta).
Per production item s: w.p. 1-rho nothing; rho*r(s): S+=a; rho*(1-r(s)): S=max(0,S+b).
The label of origin s arrives at item s+d; alarm time T = s+d.
"""
import numpy as np


def run_chain(h, p, rho, r_of_s, s_max, M=1000, r0=0.10):
    a = np.log(p / r0)
    b = np.log((1 - p) / (1 - r0))
    delta = -b / M
    A = int(round(a / delta))
    B = M
    n = int(np.floor(h / delta + 1e-12))
    v = np.zeros(n + 1)
    v[0] = 1.0
    absorbed = np.zeros(s_max + 1)  # absorbed[s] = P(first crossing at origin s)
    cur_r = None
    for s in range(1, s_max + 1):
        r = r_of_s(s)
        pu, pd, ps = rho * r, rho * (1 - r), 1 - rho
        new = ps * v
        # down moves
        new[:-B] += pd * v[B:] if B <= n else 0
        new[0] += pd * v[:min(B, n + 1)].sum()
        # up moves
        if A <= n:
            new[A:] += pu * v[:n + 1 - A]
            absorbed[s] = pu * v[n + 1 - A:].sum()
        else:
            absorbed[s] = pu * v.sum()
        v = new
    return absorbed, dict(A=A, B=B, n=n, a=a, b=b, a_err=A * delta - a, delta=delta)


def timely_fa(h, p, pi, q, d, tau=10000, D=10000, H=29999, r_pre=0.05, r1=0.20, M=1000):
    rho = pi * q
    smax_t = tau + D - 1 - d
    rf = lambda s: r_pre if s < tau else r1
    ab, info = run_chain(h, p, rho, rf, smax_t, M=M)
    early = ab[1:tau - d].sum()            # origins < tau-d => T < tau
    timely = ab[max(tau - d, 1):smax_t + 1].sum()
    abfa, _ = run_chain(h, p, rho, lambda s: 0.10, H - d, M=M)
    fa = abfa.sum()
    return dict(timely=timely, early=early, fa=fa, **info)


def fa_only(h, p, pi, q, d, H=29999, M=1000):
    ab, _ = run_chain(h, p, pi * q, lambda s: 0.10, H - d, M=M)
    return ab.sum()
