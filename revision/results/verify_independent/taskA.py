import sys, json, time
import numpy as np
import lattice, mc
cells = [
 dict(name='q0.10_d500_p0.20', h=5.7766, p=0.20, pi=0.10, q=0.10, d=500),
 dict(name='q0.20_d500_p0.20', h=6.5240, p=0.20, pi=0.10, q=0.20, d=500),
 dict(name='q0.50_d500_p0.20', h=7.5394, p=0.20, pi=0.10, q=0.50, d=500),
 dict(name='q0.20_d0_p0.12', h=3.135759473072654, p=0.12, pi=0.10, q=0.20, d=0),
]
which = sys.argv[1]
out = {}
for i, c in enumerate(cells):
    t = time.time()
    if which == 'exact':
        r = lattice.timely_fa(c['h'], c['p'], c['pi'], c['q'], c['d'], Imax=300)
    else:
        r = mc.timely_fa(c['h'], c['p'], c['pi'], c['q'], c['d'], n_rep=200000, seed=100 + i)
    r = {k: float(v) for k, v in r.items()}
    r['sec'] = time.time() - t
    out[c['name']] = r
    print(c['name'], r, flush=True)
json.dump(out, open(f'taskA_{which}.json', 'w'), indent=1)
