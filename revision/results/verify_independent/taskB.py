import numpy as np
from scipy import stats
from scipy.optimize import brentq

def np_power(L, rho, r1, r0=0.10, alpha=0.02, poisson=False):
    lam = L * rho
    sd = np.sqrt(lam)
    lo = max(0, int(lam - 14 * sd - 10)); hi = int(lam + 14 * sd + 10)
    if not poisson: hi = min(hi, L)
    N = np.arange(lo, hi + 1)
    lpN = stats.poisson.logpmf(N, lam) if poisson else stats.binom.logpmf(N, L, rho)
    Ns, Ks, l0s, l1s = [], [], [], []
    for n, lp in zip(N, lpN):
        if lp < -700: continue
        K = np.arange(0, n + 1)
        Ns.append(np.full(K.size, n)); Ks.append(K)
        l0s.append(lp + stats.binom.logpmf(K, n, r0)); l1s.append(lp + stats.binom.logpmf(K, n, r1))
    N = np.concatenate(Ns); K = np.concatenate(Ks)
    p0 = np.exp(np.concatenate(l0s)); p1 = np.exp(np.concatenate(l1s))
    llr = K * np.log(r1 / r0) + (N - K) * np.log((1 - r1) / (1 - r0))
    o = np.argsort(-llr, kind='stable')
    c0 = np.cumsum(p0[o]); c1 = np.cumsum(p1[o])
    j = np.searchsorted(c0, alpha)  # first index with c0 >= alpha
    before0 = c0[j - 1] if j > 0 else 0.0
    before1 = c1[j - 1] if j > 0 else 0.0
    frac = (alpha - before0) / p0[o[j]]
    pw = before1 + frac * p1[o[j]]
    return pw, dict(mass0=p0.sum(), mass1=p1.sum(), gap=float(llr[o[j]] - llr[o[j + 1]]))

def min_param(f, lo, hi, target=0.80):
    return brentq(lambda x: f(x) - target, lo, hi, xtol=1e-7)

if __name__ == '__main__':
    pi, d, D = 0.1, 500, 10000; L = D - d
    print('(i)')
    for q in [0.1, 0.2]:
        pw, inf = np_power(L, pi * q, 0.2); print(f' q={q} power={pw:.6f}', inf)
    qmin = min_param(lambda q: np_power(L, pi * q, 0.2)[0], 0.01, 1.0)
    print(f'(ii) q_min={qmin:.6f}  power check={np_power(L, pi*qmin, 0.2)[0]:.6f}  lambda={pi*qmin*L:.4f}')
    print('(iii) lambda_min')
    for r1 in [0.30, 0.20, 0.15, 0.12]:
        lp = min_param(lambda lam: np_power(10**7, lam / 10**7, r1, poisson=True)[0], 1, 20000)
        row = [f'poisson={lp:.4f}']
        for (pi_, Ld) in [(0.01, 5000), (0.1, 5000), (0.01, 20000), (0.1, 20000), (0.1, 9500)]:
            # lambda = pi*q*L; q must be <= 1 -> lambda <= pi*L
            f = lambda lam: np_power(Ld, lam / Ld, r1)[0]
            if pi_ * Ld < lp * 0.9:
                row.append(f'pi={pi_},L={Ld}: infeasible (max lambda={pi_*Ld:.0f})'); continue
            try:
                lb = min_param(f, 1, pi_ * Ld)
                row.append(f'pi={pi_},L={Ld}: {lb:.4f} (q={lb/(pi_*Ld):.4f})')
            except ValueError:
                row.append(f'pi={pi_},L={Ld}: power<0.8 at q=1 ({f(pi_*Ld):.3f})')
        print(f' r1={r1}:', '; '.join(row), flush=True)
