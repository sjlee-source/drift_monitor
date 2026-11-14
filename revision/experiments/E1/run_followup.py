"""E1 follow-up (protocol amendment 2026-11-13, post hoc): A1 gate repair, A3 fine q grid.

usage: python -m revision.experiments.E1.run_followup repair K     (K = 0..3, failed cell)
       python -m revision.experiments.E1.run_followup fineq J      (J = 0..17, (q,d) index)
"""
import json, os, sys, time
import numpy as np
from revision import generators as G
from revision import calibration as C
from revision.experiments.E1 import e1_common as E
from revision.experiments.E1.run_setting import collect_maxima, timely_quantiles, jsonable

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1/followup'
FAMILY = 576  # amendment b uses 1008; upper bounds are recomputed in aggregation
REPAIR = [('adwin', (0.01, 0.20, 0)), ('cusum_0.12', (0.01, 0.30, 500)), ('cusum_0.20', (0.10, 0.50, 0)),
          ('cusum_0.12', (0.10, 0.50, 500))]
FINE_Q = (0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27)
FINE = [(q, d) for d in (0, 500) for q in FINE_Q]


def run_cell(pi, q, d, mons, cal_id, val_id, ev, ncal, tag, log):
    """cal_id/val_id: functions path -> condition id; ev: list of (r1, condition id)."""
    cfg = {'const10': G.null_config(pi, q, d, E.H, r=0.10), 'const05': G.null_config(pi, q, d, E.H, r=0.05),
           'path': G.nominal_config(pi, q, d, E.H, E.TAU, E.RPRE, 0.10)}
    timing = {}
    res = dict(tag=tag, pi=pi, q=q, d=d, n_cal=ncal, n_val=E.N_VAL, n_eval=E.N_EVAL)
    cal = {p: collect_maxima(mons, cfg[p], 'calibration', cal_id(p), ncal, timing)[0] for p in ('const10', 'path')}
    log('cal done')
    thr = {}
    for k in mons:
        tc = C.calibrate_threshold(cal['const10'][k], E.FA_TARGET); tp = C.calibrate_threshold(cal['path'][k], E.FA_TARGET)
        thr[k] = E.stricter(tc, tp)
        res.setdefault('thresholds', {})[k] = dict(comp=thr[k].as_dict(), binding='const' if thr[k] is tc else 'path')
    del cal
    res['validation'] = {k: {} for k in mons}
    for p in E.PATHS:
        mx, tb, _ = collect_maxima(mons, cfg[p], 'validation', val_id(p), E.N_VAL, timing)
        for k in mons:
            v = C.validate(C.fa_from_maxima(mx[k], thr[k], tb), cap=0.03, family=FAMILY if p != 'const05' else 1)
            res['validation'][k][p] = v
    log('val done')
    res['evaluation'] = {}; alarms = {}
    for r1, cid in ev:
        hcfg = G.nominal_config(pi, q, d, E.H, E.TAU, E.RPRE, r1)
        al = {k: [] for k in mons}
        for b in G.generate_chunks(hcfg, 'evaluation', E.EXP, cid, E.N_EVAL, chunk=E.CHUNK):
            for k, m in mons.items():
                al[k].append(C.alarm_times(m, b, thr[k]))
        res['evaluation'][f'{r1:g}'] = {}
        for k in mons:
            x = np.concatenate(al[k]); alarms[f'r{r1:g}__{k}'] = x.astype(np.float32)
            mt = C.detection_metrics(x, E.TAU, E.H, E.DEADLINES)
            for D in E.DEADLINES:
                mt[f'timely_q50_{D}'], mt[f'timely_q90_{D}'] = timely_quantiles(x, D)
                mt[f'oracle_{D}'] = E.oracle_beta(pi, q, d, r1, D)
            res['evaluation'][f'{r1:g}'][k] = mt
        log('eval', r1, 'done')
    np.savez_compressed(os.path.join(OUT, f'{tag}_alarms.npz'), **alarms)
    with open(os.path.join(OUT, f'{tag}.json'), 'w') as f:
        json.dump(jsonable(res), f, indent=1)
    log('wrote', tag)


def main():
    global OUT
    mode, j = sys.argv[1], int(sys.argv[2])
    if mode == 'fineq100k':   # amendment 2026-11-13b
        OUT = OUT + '_cal100k'
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time(); log = lambda *x: print(f'[{time.time() - T0:7.1f}s]', mode, j, *x, flush=True)
    allm = E.make_monitors()
    if mode == 'repair':
        k, (pi, q, d) = REPAIR[j]
        idx = E.SETTINGS.index((pi, q, d))
        run_cell(pi, q, d, {k: allm[k]}, lambda p: 3000 + 4 * idx + E.PATHS.index(p),
                 lambda p: 3100 + 4 * idx + E.PATHS.index(p), [(r1, E.harm_cond(idx, r1)) for r1 in E.R1S],
                 100000, f'repair_{j}_{k}_s{idx:02d}', log)
    elif mode == 'fineq':
        q, d = FINE[j]
        mons = {k: allm[k] for k in E.CORE}
        run_cell(0.10, q, d, mons, lambda p: 4000 + 4 * j + E.PATHS.index(p), lambda p: 4000 + 4 * j + E.PATHS.index(p),
                 [(0.20, 4500 + j)], E.N_CAL, f'fineq_{j:02d}_q{q:g}_d{d}', log)
    elif mode == 'fineq100k':
        q, d = FINE[j]
        mons = {k: allm[k] for k in E.CORE}
        run_cell(0.10, q, d, mons, lambda p: 7000 + 4 * j + E.PATHS.index(p), lambda p: 7100 + 4 * j + E.PATHS.index(p),
                 [(0.20, 4500 + j)], 100000, f'fineq_{j:02d}_q{q:g}_d{d}', log)


if __name__ == '__main__':
    main()
