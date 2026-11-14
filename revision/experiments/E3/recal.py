"""Uniform recalibration of all W-B auc0.95 procedures (prespecified in protocol.json recal_2026_11_13)."""
import os, pickle, json, numpy as np, pandas as pd
from revision import calibration as C
from . import followup as F, design as Dz
OUT = os.path.join(F.OUT, 'recal'); os.makedirs(OUT, exist_ok=True)
sc = [s for s, _, _ in F.SCENS if s.name == 'auc0.95_pi0.10_d500'][0]
names = F.WB4; pols, _ = F.policies(sc, 1, names)
rows, res = [], {}
for n in names:
    old = pickle.load(open(os.path.join(F.OUT, 'raw', f'{sc.name}_wf1_{n}.pkl'), 'rb'))
    mx = {p: F.sim(sc, pols[n], p, 1, 'calibration', 100000, start=300000)[0] for p in (0, 1)}
    thr = {}
    for mon, col in (('msr', 0), ('bet', 1)):
        t0 = C.calibrate_threshold(mx[0][:, col], 0.02); t1 = C.calibrate_threshold(mx[1][:, col], 0.02)
        thr[mon] = t0 if t0.h >= t1.h else t1
    th = (thr['msr'].h_hi, thr['msr'].h_lo, thr['bet'].h_hi, thr['bet'].h_lo)
    row = dict(policy=n)
    for mon in ('msr', 'bet'):
        row[f'h_{mon}_old'] = old['thr'][mon][0].h; row[f'h_{mon}_new'] = thr[mon].h
    for path, pn in ((0, 'c010'), (1, 'path'), (2, 'c005')):
        o, s, t = F.sim(sc, pols[n], path, 1, 'validation', 10000, th, start=300000)
        for mon in ('msr', 'bet'):
            fa = F.alarm(o, t, mon, thr[mon]) <= Dz.H; v = C.validate(fa, cap=0.03, family=112)
            fo = old['val'][path]['fa_' + mon]; vo = C.validate(fo, cap=0.03, family=112)
            row[f'FA_{mon}_{pn}_old'] = vo['rate']; row[f'up_{mon}_{pn}_old'] = vo['upper_simultaneous']
            row[f'FA_{mon}_{pn}_new'] = v['rate']; row[f'up_{mon}_{pn}_new'] = v['upper_simultaneous']
    o, s, t = F.sim(sc, pols[n], 3, 1, 'evaluation', 10000, th, start=200000)
    ev = {mon: F.alarm(o, t, mon, thr[mon]) for mon in ('msr', 'bet')}
    for mon in ('msr', 'bet'):
        for D in Dz.DEADLINES:
            f = lambda a: (a >= sc.tau) & (a <= min(Dz.H, sc.tau + D - 1))
            row[f'pD{D//1000}k_{mon}_old'] = f(old['ev']['alarm_' + mon]).mean(); row[f'pD{D//1000}k_{mon}_new'] = f(ev[mon]).mean()
    for mon in ('msr', 'bet'):
        row[f'certified_{mon}'] = bool(row[f'up_{mon}_c010_new'] <= 0.03 and row[f'up_{mon}_path_new'] <= 0.03)
    res[n] = ev; rows.append(row); print(row, flush=True)
K = []
for n in names[1:]:
    for mon in ('msr', 'bet'):
        for D in Dz.DEADLINES:
            f = lambda a: (a >= sc.tau) & (a <= min(Dz.H, sc.tau + D - 1))
            dm, lo, hi = F.boot3(f(res[n][mon]), f(res['B1_passed_uniform'][mon]), 555 + D)
            K.append(dict(policy=n, reference='B1_passed_uniform', monitor=mon, D=D, diff_pD=dm, boot_lo=lo, boot_hi=hi))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'recal_auc0.95_WB.csv'), index=False)
pd.DataFrame(K).to_csv(os.path.join(OUT, 'recal_contrasts.csv'), index=False)
pickle.dump(dict(rows=rows, ev=res), open(os.path.join(OUT, 'recal.pkl'), 'wb'))
print('done')
