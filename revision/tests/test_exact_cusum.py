"""(a) Nominal generator + CUSUM vs exact integer-CUSUM DP (theory/exact_cusum.py)."""
import json
from pathlib import Path
import numpy as np
import pytest

from revision import generators as G, monitors as M, legacy as L

X = pytest.importorskip('exact_cusum')
OUT = Path(__file__).resolve().parent / '_results'
OUT.mkdir(exist_ok=True)
N = 20000
CONDS = [  # (pi, q, d, tau, H, p1, r_post, D)
    (0.1, 0.1, 500, 10000, 29999, 0.20, 0.20, 10000),
    (0.1, 0.2, 500, 10000, 29999, 0.20, 0.20, 10000),
    (0.1, 0.5, 500, 10000, 29999, 0.20, 0.20, 10000),
    (0.1, 0.2, 0, 10000, 29999, 0.12, 0.12, 20000),
    (0.01, 1.0, 500, 20000, 39999, 0.20, 0.20, 20000),
]


@pytest.mark.parametrize('cond', CONDS)
def test_integer_and_continuous_cusum_vs_exact(cond):
    pi, q, d, tau, H, p1, r1, D = cond
    T = L.archived_thresholds()
    h = T[f'pi{pi}_q{q}_d{d}_H{H}'][1 if p1 == 0.12 else 2]
    A, B, s = X.scores(p1, 0.1, 80)
    c = int(np.floor(h * s))
    pD, early = X.detection_prob(H, d, pi * q, tau, 0.05, r1, D, A, B, c)
    fa10 = X.false_alarm(H, d, pi * q, lambda _: 0.10, A, B, c)
    fa05 = X.false_alarm(H, d, pi * q, lambda _: 0.05, A, B, c)
    faP = X.false_alarm(H, d, pi * q, lambda t: 0.05 if t < tau else 0.10, A, B, c)
    cid = int(pi * 1000 + q * 100 + d + p1 * 10)
    mi = M.CUSUM(p1, 0.1, integer_scale=80)
    assert mi.params[2] == A and mi.params[3] == B
    res = dict(cond=cond, c=c, A=A, B=B, h=h)
    rows = []
    for lab, cfg, exact in [('fa_r0.10', G.null_config(pi, q, d, H, 0.10), fa10),
                            ('fa_r0.05', G.null_config(pi, q, d, H, 0.05), fa05),
                            ('fa_path', G.StreamConfig(H=H, pi=pi, q=q, r=G.risk_path(H, 0.05, 0.10, tau),
                                                       delay=G.Delay.constant(d)), faP),
                            ('pD', G.nominal_config(pi, q, d, H, tau, 0.05, r1), pD)]:
        b = G.generate(cfg, 'test', 100 + len(rows), cid, N)
        a = mi.first_alarm(b, c)
        est = float(np.isfinite(a).mean()) if lab != 'pD' else float(((a >= tau) & (a <= min(H, tau + D - 1))).mean())
        se_ = np.sqrt(max(exact * (1 - exact), 1 / N) / N)
        z = (est - exact) / se_
        rows.append(dict(quantity=lab, exact=exact, mc=est, z=z))
        assert abs(z) < 4.0, (lab, exact, est, z)
        if lab == 'pD':
            ac = M.CUSUM(p1, 0.1).first_alarm(b, h)
            cont = float(((ac >= tau) & (ac <= min(H, tau + D - 1))).mean())
            rows.append(dict(quantity='pD_continuous_chart', exact=exact, mc=cont, z=(cont - exact) / se_))
            assert abs(cont - exact) < 0.02    # rounding-scale B=80 integer chart approximates continuous
            em = float((a < tau).mean())
            rows.append(dict(quantity='early', exact=early, mc=em, z=(em - early) / np.sqrt(max(early, 1 / N) / N)))
    res['rows'] = rows
    (OUT / f'exact_cusum_{pi}_{q}_{d}_{p1}.json').write_text(json.dumps(res, indent=1, default=float))
