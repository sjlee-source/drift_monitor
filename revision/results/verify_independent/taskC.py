import os, json
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from multiprocessing import Pool
import lattice

P, PI, D_, ALPHA = 0.20, 0.10, 500, 0.02
a = np.log(P / 0.1); b = np.log(0.8 / 0.9)

def lattice_vals(lo, hi, Imax=200):
    v = []
    for i in range(Imax + 1):
        js = np.arange(max(0, int(np.ceil((i * a - hi) / -b)) - 1), int(np.floor((i * a - lo) / -b)) + 2)
        x = i * a + js * b
        v.append(x[(x >= lo) & (x <= hi)])
    return np.unique(np.concatenate(v))

def fa_at(h, q):
    return lattice.fa_only(h, P, PI, q, D_, Imax=250)[0]

def calibrate(q):
    vals = lattice_vals(4.5, 9.0)
    lo, hi = 0, len(vals) - 1   # FA(vals[hi]+eps) <= alpha assumed; find smallest index with FA<=alpha
    assert fa_at(vals[hi] + 1e-9, q) <= ALPHA and fa_at(vals[lo] + 1e-9, q) > ALPHA
    while hi - lo > 1:
        m = (lo + hi) // 2
        if fa_at(vals[m] + 1e-9, q) <= ALPHA: hi = m
        else: lo = m
    h = vals[hi] + 1e-9
    r = lattice.timely_fa(h, P, PI, q, D_, Imax=250)
    fa_next_lower = fa_at(vals[lo] + 1e-9, q)
    return dict(q=q, h=float(h), fa=float(r['fa']), fa_at_next_lower_h=float(fa_next_lower),
                timely=float(r['timely']), early=float(r['early']), lost=float(r['lost']))

if __name__ == '__main__':
    qs = [round(0.15 + 0.005 * k, 3) for k in range(21)]
    with Pool(4) as pool:
        res = pool.map(calibrate, qs)
    json.dump(res, open('taskC.json', 'w'), indent=1)
    for r in res: print(r)
