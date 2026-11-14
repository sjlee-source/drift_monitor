"""POST HOC (2026-11-13): independent re-validation of the 4 recalibrated-threshold cells whose
simultaneous upper bound exceeded 0.03 in fa.csv.  Recalibrates exactly as sim.py (same seeds),
then validates on FRESH validation replicates 10000..19999 (disjoint from the original 0..9999).
"""
import numpy as np, pandas as pd
from revision import generators as G, calibration as C
from revision.experiments.E2 import sim

OUT = sim.OUT
CELLS = [(0.2, 'markov_phi0.3', 'ewma'), (0.2, 'board_icc_r0.2', 'ewma'),
         (0.5, 'markov_phi0.6', 'msr'), (0.5, 'board_icc_r0.2', 'ewma')]
rows = []
mons_all = sim.monitors()
for q, var, mon in CELLS:
    mons = {mon: mons_all[mon]}
    thr = sim.calibrate(var, q, mons)[mon]['thr']
    qi = sim.QS.index(q)
    for path in ('null_const010', 'null_path'):
        cfg = sim.config(var, path, q)
        al = []
        for s in range(10000, 20000, sim.CHUNK):
            b = G.generate(cfg, 'validation', sim.EXP, 10 * qi + sim.PATHS[path], np.arange(s, s + sim.CHUNK))
            al.append(C.alarm_times(mons[mon], b, thr) <= sim.H)
        a = np.concatenate(al)
        v = C.validate(a, cap=0.03, family=364)
        rows.append(dict(q=q, variant=var, monitor=mon, path=path, h=thr.h, gamma=thr.gamma, n=v['n'],
                         alarms=v['alarms'], fa=v['rate'], cp_lo=v['cp_lo'], cp_hi=v['cp_hi'],
                         upper_sim364=v['upper_simultaneous'], gate_pass=v['pass_gate']))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/posthoc_recal_revalidation.csv', index=False)
