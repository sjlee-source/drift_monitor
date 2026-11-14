"""E3 design constants: score model, strata, policy definitions (all prespecified)."""
import numpy as np
from scipy.stats import norm

EXP = 13
H, TAU, R0, RPRE, R1 = 29999, 10000, 0.10, 0.05, 0.20
B = int(round(0.2 * H))            # 6000 audits, pathwise hard budget
QBAR = B / H
EPS = 0.02                          # positivity floor on every eligible item's audit probability
BLK = 3000
AUC_REF = 0.90                      # defect vs non-defect AUC at r = r0
MU_REF = np.sqrt(2) * norm.ppf(AUC_REF)          # 1.8124
C_THR = MU_REF + norm.ppf(R0)                     # pass iff S < c ; 0.5308
EDGES = C_THR + np.array([-2.0, -1.0, -0.5, 0.0, 1.0, 2.0])   # 7 score strata; strata 0..3 passed, 4..6 flagged
NSTRATA = len(EDGES) + 1
FLAG_STRATA = np.arange(4, NSTRATA)
CONDITIONS = {0: dict(pi=0.10, d=500, label='pi0.10_d500'),     # primary
              1: dict(pi=0.10, d=0, label='pi0.10_d0'),
              2: dict(pi=0.01, d=500, label='pi0.01_d500')}       # original continuity (floor)
PATHS = {'null_const010': 0, 'null_path': 1, 'null_const005': 2, 'harm': 3}
DEADLINES = (5000, 10000, 20000)
D_PRIMARY = 10000
TARGET_FA = 0.02
N_CAL, N_VAL, N_EVAL = 20000, 10000, 10000
# monitor constants (original mixture SR and Betting)
LAM = np.array([.02, .05, .10, .20, .40]) / .09
GAIN, BCAP, WIN, CAPF = 4.0, 9.0, 50, 0.99
WARN_FRAC = 0.25


def warn_level(h, mode):
    """linear: SR evidence exp(S) >= exp(h)/4  <=>  S >= h - ln 4 (original reading);  log: S >= h/4."""
    return h - np.log(4.0) if mode == 'linear' else h * WARN_FRAC
BOOST = 0.5


def mu1_of_r(r):
    return C_THR - norm.ppf(r)


def risk_arr(path, r1=R1):
    r = np.full(H, R0)
    t = np.arange(1, H + 1)
    if path == 1:
        r = np.where(t >= TAU, R0, RPRE)
    elif path == 2:
        r = np.full(H, RPRE)
    elif path == 3:
        r = np.where(t >= TAU, r1, RPRE)
    return r


def strata_stats(pi):
    e = np.concatenate([[-np.inf], EDGES, [np.inf]])
    P0 = np.diff(norm.cdf(e)); P1 = np.diff(norm.cdf(e - MU_REF))
    N = (1 - pi) * P0 + pi * P1
    pz = pi * P1 / N
    return N, pz


def _solve_scale(v, mass, target):
    lo, hi = 0.0, 1e6
    for _ in range(200):
        a = 0.5 * (lo + hi)
        m = np.sum(mass * np.clip(a * v, EPS, 1.0) * (v >= 0))
        if m > target:
            hi = a
        else:
            lo = a
    return 0.5 * (lo + hi)


def _post(s, pi):
    lr = MU_REF * s - 0.5 * MU_REF ** 2
    return 1 / (1 + (1 - pi) / pi * np.exp(-lr))


def policies(pi, wf):
    """Return {name: dict(kernel params)} for one workflow (0 = W-A, 1 = W-B)."""
    N, pz = strata_stats(pi)
    elig = np.ones(NSTRATA, bool) if wf == 0 else (np.arange(NSTRATA) < 4)
    p_pass = float(np.sum(N[:4]))
    base_pol = dict(ptype=0, scope_block=False, base=0.0, boost=0.0, warn=False, warn_mode='linear', gtype=0,
                    G=np.zeros(NSTRATA), a_unc=0.0, gmin_flag=1.0, ipw=0)
    out = {}

    def add(name, **kw):
        d = dict(base_pol); d.update(kw); out[name] = d

    # continuous grid for the uncertainty policy
    s = np.linspace(-9, 13, 220001); ds = s[1] - s[0]
    f = ((1 - pi) * norm.pdf(s) + pi * norm.pdf(s - MU_REF)) * ds
    u = _post(s, pi); vu = u * (1 - u)
    if wf == 1:
        f = f * (s < C_THR)

    def scale_strata(v):
        v = np.where(elig, v, -1.0)
        a = _solve_scale(v, N, QBAR)
        return np.where(elig, a * v, 0.0)

    a_unc = _solve_scale(vu, f, QBAR)
    coef = np.where(np.arange(NSTRATA) < 4, 1 - R0, R0)
    if wf == 0:
        add('P1_horizon_uniform', ptype=0)
        add('P2_block_uniform', ptype=0, scope_block=True)
        for sc in (False, True):
            for b in (0.05, 0.20):
                add(f'P3_{"block" if sc else "horizon"}_warn_q{int(b * 100):02d}', ptype=1, scope_block=sc,
                    base=b, boost=BOOST, warn=True)
        # amendment (after smoke run, before final runs): log-scale quarter warning = frequent warnings
        for sc in (False, True):
            for b in (0.05, 0.20):
                add(f'P3x_{"block" if sc else "horizon"}_logwarn_q{int(b * 100):02d}', ptype=1, scope_block=sc,
                    base=b, boost=BOOST, warn=True, warn_mode='log')
        add('P4_uncertainty', ptype=2, gtype=2, a_unc=a_unc, gmin_flag=0.0, ipw=1)
        G5 = scale_strata(pz)
        add('P5_score_stratified', ptype=2, gtype=1, G=G5, gmin_flag=float(G5[FLAG_STRATA].min()), ipw=1)
        G6 = scale_strata(coef * np.sqrt(pz * (1 - pz)))
        add('P6_neyman', ptype=2, gtype=1, G=G6, gmin_flag=float(G6[FLAG_STRATA].min()), ipw=1)
    else:
        rho = QBAR / p_pass
        add('B1_passed_uniform', ptype=2, gtype=0, G=np.full(NSTRATA, rho), ipw=1)
        add('B2_passed_block_uniform', ptype=2, gtype=0, G=np.full(NSTRATA, rho), scope_block=True, ipw=1)
        add('B3_passed_block_warn', ptype=1, scope_block=True, base=rho, boost=min(1.0, rho * BOOST / 0.2),
            warn=True, ipw=1)
        add('B4_passed_uncertainty', ptype=2, gtype=2, a_unc=a_unc, ipw=1)
        add('B5_passed_score_stratified', ptype=2, gtype=1, G=scale_strata(pz), ipw=1)
        add('B6_passed_neyman', ptype=2, gtype=1, G=scale_strata(np.sqrt(pz * (1 - pz))), ipw=1)
    for v in out.values():
        v['G'] = np.asarray(v['G'], float)
    return out, dict(N=N.tolist(), pz=pz.tolist(), p_pass=p_pass)


def expected_yield(pi, pol):
    """Expected informative audit labels per audit at kappa=1 under the nominal null (design check)."""
    N, pz = strata_stats(pi)
    if pol['gtype'] == 1:
        p = np.clip(pol['G'], EPS, 1) * (pol['G'] > 0)
        return float(np.sum(N * pz * p) / np.sum(N * p))
    return None
