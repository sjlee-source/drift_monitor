"""Exact DP p_D for the integer CUSUM(p1=0.20) (theory/exact_cusum.py, scale_B=80) with exact randomized
composite calibration at FA 0.02 (stricter of constant-0.10 and 0.05->0.10 path), all E1 conditions.
Output: exp_E1/cusum_exact_dp.csv  (MC comparison is done in aggregate.py)."""
import sys, csv, time
import numpy as np
sys.path.insert(0, '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory')
from exact_cusum import scores, alarm_by_origin
from revision.experiments.E1 import e1_common as E

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1/cusum_exact_dp.csv'
SB = 80


def fa(rho, d, A, B, c, path):
    rf = (lambda s: 0.10) if path == 'const' else (lambda s: E.RPRE if s < E.TAU else 0.10)
    return alarm_by_origin(E.H, d, rho, rf, A, B, c).sum()


def rand_cal(rho, d, A, B, path, alpha=E.FA_TARGET):
    """(c, g): use threshold c-1 with prob g, else c; exact FA = alpha (FA nonincreasing in c)."""
    lo, hi = 0, 2 * A
    while fa(rho, d, A, B, hi, path) > alpha:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if fa(rho, d, A, B, mid, path) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    c = lo; f_c = fa(rho, d, A, B, c, path)
    if c == 0:
        return c, 0.0, f_c, f_c
    f_m = fa(rho, d, A, B, c - 1, path)
    g = (alpha - f_c) / (f_m - f_c) if f_m > f_c else 0.0
    return c, float(np.clip(g, 0, 1)), f_c, f_m


def pd_all(m, d, D):
    lo = max(1, E.TAU - d); hi = E.TAU + D - 1 - d
    return m[lo:hi + 1].sum(), m[1:lo].sum()


def main():
    A, B, s = scores(0.20, E.R0, SB)
    rows = []
    T0 = time.time()
    for idx, (pi, q, d) in enumerate(E.SETTINGS):
        rho = pi * q
        cc = rand_cal(rho, d, A, B, 'const'); cp = rand_cal(rho, d, A, B, 'path')
        (c, g) = (cc[0], cc[1]) if (cc[0] - cc[1]) >= (cp[0] - cp[1]) else (cp[0], cp[1])
        binding = 'const' if (cc[0] - cc[1]) >= (cp[0] - cp[1]) else 'path'
        fac = g * fa(rho, d, A, B, c - 1, 'const') + (1 - g) * fa(rho, d, A, B, c, 'const') if c > 0 else fa(rho, d, A, B, c, 'const')
        fap = g * fa(rho, d, A, B, c - 1, 'path') + (1 - g) * fa(rho, d, A, B, c, 'path') if c > 0 else fa(rho, d, A, B, c, 'path')
        for r1 in E.R1S:
            rf = lambda t, r1=r1: E.RPRE if t < E.TAU else r1
            m_c = alarm_by_origin(E.H, d, rho, rf, A, B, c)
            m_m = alarm_by_origin(E.H, d, rho, rf, A, B, c - 1) if (c > 0 and g > 0) else m_c
            for D in E.DEADLINES:
                p1, e1 = pd_all(m_c, d, D); p0, e0 = pd_all(m_m, d, D)
                rows.append(dict(pi=pi, q=q, d=d, r1=r1, D=D, A=A, B=B, scale=s, c=c, gamma=g, logh=c / s, binding=binding,
                                 fa_const10=fac, fa_path=fap, pD_exact=g * p0 + (1 - g) * p1, early_exact=g * e0 + (1 - g) * e1))
        print(f'[{time.time() - T0:6.0f}s] setting {idx} c={c} g={g:.3f} {binding} FAc={fac:.4f} FAp={fap:.4f}', flush=True)
        with open(OUT, 'w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
