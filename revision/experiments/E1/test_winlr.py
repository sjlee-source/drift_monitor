"""Validate WindowLR kernel vs the naive O(n^2) reference (paths, maxima, first crossing, restart)."""
import json, sys
import numpy as np
from revision import generators as G
from revision.experiments.E1.winlr import WindowLR, naive_path

out = {}
worst = 0.0; ncmp = 0
for name, cfg, D in [
    ('const_d500_D10000', G.nominal_config(pi=0.1, q=0.3, d=500, r_post=0.2), 10000),
    ('const_d0_D2000', G.nominal_config(pi=0.1, q=0.5, d=0, r_post=0.2), 2000),
    ('geom_mean300_D3000', G.StreamConfig(H=29999, pi=0.1, r=G.risk_path(29999, .05, .2, 10000), q=0.3,
                                           delay=G.Delay.geometric(300)), 3000),
]:
    b = G.generate(cfg, role='test', experiment=11, condition=900, replicates=12)
    m = WindowLR(D=D)
    mx = m.max_stat(b)
    for i in range(b.n):
        s = b.stream(i)
        t1, v1 = m.path(s['origin'], s['arrival'], s['loss'], H=b.H)
        t2, v2 = naive_path(s['origin'], s['arrival'], s['loss'], D)
        assert np.array_equal(t1, t2), name
        e = np.max(np.abs(v1 - v2)) if len(v1) else 0.0
        worst = max(worst, e); ncmp += len(v1)
        assert e < 1e-9
        assert abs(mx[i] - v2.max()) < 1e-9
        # first crossing for a few thresholds
        for h in np.quantile(v2, [0.5, 0.9, 0.99]):
            fa = m.evaluate(b, (h,))['alarm'][i, 0]
            k = np.nonzero(v2 > h)[0]
            ref = t2[k[0]] if len(k) else np.inf
            assert fa == ref
        # restart vs manual loop over the naive reference
        h = np.quantile(v2, 0.8)
        tl = 0; na = 0
        while True:
            tt, vv = naive_path(s['origin'], s['arrival'], s['loss'], D, t0=tl)
            k = np.nonzero(vv > h)[0]
            if not len(k):
                break
            tl = tt[k[0]]; na += 1
        r = m.restart_run(b, h)
        assert r['n_alarms'][i] == na, (name, i, r['n_alarms'][i], na)
    out[name] = dict(streams=b.n, labels=int(len(b.loss)))
out['max_abs_path_error'] = float(worst); out['values_compared'] = int(ncmp); out['pass'] = True
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
