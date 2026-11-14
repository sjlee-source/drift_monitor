"""Exact (no discretization) evaluation of delayed Bernoulli CUSUM on the
lattice S = i*a + j*b, where (i, j) = (#misses, #non-misses) since last reset.
Since a/|b| is irrational, distinct (i,j) give distinct S. Only truncation:
i <= Imax (mass escaping is tracked and reported as 'lost').
"""
import numpy as np


def build(h, p, r0=0.10, Imax=400):
    a = np.log(p / r0)
    b = np.log((1 - p) / (1 - r0))  # negative
    nb = -b
    rows = []
    idx = {}
    k = 0
    margin = np.inf
    for i in range(Imax + 1):
        jmax = int(np.floor(i * a / nb + 1e-15))
        jmin = max(0, int(np.ceil((i * a - h) / nb - 1e-15)))
        for j in range(jmin, jmax + 1):
            val = i * a + j * b
            if val < -1e-12 or val > h + 1e-12:
                continue
            margin = min(margin, abs(val - h), abs(val) if (i, j) != (0, 0) else np.inf)
            idx[(i, j)] = k
            k += 1
    n = k
    up_src, up_dst, up_abs, up_lost = [], [], [], []
    dn_src, dn_dst, dn_reset = [], [], []
    for (i, j), s in idx.items():
        val = i * a + j * b
        # miss
        if val + a > h:
            up_abs.append(s)
        elif (i + 1, j) in idx:
            up_src.append(s); up_dst.append(idx[(i + 1, j)])
        else:
            up_lost.append(s)  # i+1 > Imax
        # non-miss
        if val + b < 0:
            dn_reset.append(s)
        else:
            dn_src.append(s); dn_dst.append(idx[(i, j + 1)])
    A = lambda x: np.array(x, dtype=np.int64)
    return dict(n=n, a=a, b=b, margin=margin,
                up_src=A(up_src), up_dst=A(up_dst), up_abs=A(up_abs), up_lost=A(up_lost),
                dn_src=A(dn_src), dn_dst=A(dn_dst), dn_reset=A(dn_reset))


def run(G, rho, r_of_s, s_max):
    v = np.zeros(G['n']); v[0] = 1.0
    absorbed = np.zeros(s_max + 1)
    lost = 0.0
    us, ud, ua, ul = G['up_src'], G['up_dst'], G['up_abs'], G['up_lost']
    ds, dd, dr = G['dn_src'], G['dn_dst'], G['dn_reset']
    for s in range(1, s_max + 1):
        r = r_of_s(s)
        pu, pd = rho * r, rho * (1 - r)
        new = (1 - rho) * v
        new[ud] += pu * v[us]
        new[dd] += pd * v[ds]
        new[0] += pd * v[dr].sum()
        absorbed[s] = pu * v[ua].sum()
        lost += pu * v[ul].sum()
        v = new
    return absorbed, lost


def timely_fa(h, p, pi, q, d, tau=10000, D=10000, H=29999, r_pre=0.05, r1=0.20, Imax=400, G=None):
    G = G or build(h, p, Imax=Imax)
    rho = pi * q
    smax_t = tau + D - 1 - d
    ab, lost_t = run(G, rho, lambda s: r_pre if s < tau else r1, smax_t)
    early = ab[1:tau - d].sum()
    timely = ab[max(tau - d, 1):smax_t + 1].sum()
    abfa, lost_f = run(G, rho, lambda s: 0.10, H - d)
    return dict(timely=timely, early=early, fa=abfa.sum(), lost=max(lost_t, lost_f),
                nstates=G['n'], margin=G['margin'])


def fa_only(h, p, pi, q, d, H=29999, G=None, Imax=400):
    G = G or build(h, p, Imax=Imax)
    ab, lost = run(G, pi * q, lambda s: 0.10, H - d)
    return ab.sum(), lost
