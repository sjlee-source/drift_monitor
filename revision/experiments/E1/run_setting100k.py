"""[amendment 2026-11-13b: 100k calibration, fresh cal/val ids 6000/6100+, same eval ids]
E1 runner for one null setting (pi, q, d): calibration -> validation -> evaluation (3 r1) -> trade-off.

usage: python -m revision.experiments.E1.run_setting IDX [--pilot SCALE] [--out DIR]
Writes DIR/setting_IDX.json, DIR/setting_IDX_alarms.npz, DIR/setting_IDX_valmax.npz, DIR/tradeoff_*.csv.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from revision import generators as G
from revision import calibration as C
from revision.experiments.E1 import e1_common as E

OUTDEF = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1/raw_cal100k'


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if np.isfinite(o) else (str(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def chunks(cfg, role, cond, n):
    return G.generate_chunks(cfg, role, E.EXP, cond, n, chunk=E.CHUNK)


def collect_maxima(mons, cfg, role, cond, n, timing, first_chunk_hook=None):
    mx = {k: [] for k in mons}; tb = []; nlab = 0
    for ci, b in enumerate(chunks(cfg, role, cond, n)):
        tb.append(b.tiebreak.copy()); nlab += len(b.loss)
        for k, m in mons.items():
            t0 = time.time(); mx[k].append(m.max_stat(b)); timing[k] = timing.get(k, 0) + time.time() - t0
        if ci == 0 and first_chunk_hook is not None:
            first_chunk_hook(b)
        del b
    return {k: np.concatenate(v) for k, v in mx.items()}, np.concatenate(tb), nlab


def timely_quantiles(a, D):
    dl = min(E.H, E.TAU + D - 1)
    s = (a >= E.TAU) & (a <= dl)
    if not s.any():
        return np.nan, np.nan
    x = a[s] - E.TAU
    return float(np.quantile(x, 0.5, method='inverted_cdf')), float(np.quantile(x, 0.9, method='inverted_cdf'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('idx', type=int)
    ap.add_argument('--pilot', type=float, default=0.0, help='scale stream counts (pilot role)')
    ap.add_argument('--out', default=OUTDEF)
    ap.add_argument('--skip-supp', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pi, q, d = E.SETTINGS[a.idx]
    sc = a.pilot if a.pilot > 0 else 1.0
    rc, rv, re_, ra = (('pilot',) * 4) if a.pilot > 0 else ('calibration', 'validation', 'evaluation', 'arl')
    ncal, nval, nev = int(100000 * sc), int(E.N_VAL * sc), int(E.N_EVAL * sc)
    mons = E.make_monitors()
    if a.skip_supp:
        for k in E.SUPP:
            mons.pop(k)
    timing = {}; T0 = time.time(); log = lambda *x: print(f'[{time.time() - T0:7.1f}s]', *x, flush=True)
    # pilot role: shift condition ids so the four stages do not share streams
    off = {'cal': 0, 'val': 10000, 'ev': 20000} if a.pilot > 0 else {'cal': 0, 'val': 0, 'ev': 0}
    cfg = {'const10': G.null_config(pi, q, d, E.H, r=0.10), 'const05': G.null_config(pi, q, d, E.H, r=0.05),
           'path': G.nominal_config(pi, q, d, E.H, E.TAU, E.RPRE, 0.10)}
    res = dict(setting=a.idx, pi=pi, q=q, d=d, pilot=a.pilot, n_cal=ncal, n_val=nval, n_eval=nev)

    # ---------------- 1. calibration (20,000 streams per null path)
    cal = {}
    for p in ('const10', 'path'):
        mx, tb, nl = collect_maxima(mons, cfg[p], rc, 6000 + 4 * a.idx + E.PATHS.index(p) + off['cal'], ncal, timing)
        cal[p] = mx; res[f'labels_per_stream_cal_{p}'] = nl / ncal
        log('cal', p, 'done')
    thr_c = {k: C.calibrate_threshold(cal['const10'][k], E.FA_TARGET, 'randomized') for k in mons}
    thr_p = {k: C.calibrate_threshold(cal['path'][k], E.FA_TARGET, 'randomized') for k in mons}
    variants = {}   # vname -> (monitor key, Threshold, family)
    for k in mons:
        comp = E.stricter(thr_c[k], thr_p[k])
        variants[k] = (k, comp, 'core' if k in E.CORE else 'supp')
    for v, k in E.CONSTCAL.items():
        variants[v] = (k, thr_c[k], 'constcal')
    for v, (k, f) in E.NATIVE.items():
        if k in mons:
            variants[v] = (k, C.strict(f(mons[k])), 'native')
    res['thresholds'] = {k: dict(const=thr_c[k].as_dict(), path=thr_p[k].as_dict(),
                                 binding='const' if E.stricter(thr_c[k], thr_p[k]) is thr_c[k] else 'path') for k in mons}
    res['variants'] = {v: dict(monitor=k, family=fam, thr=t.as_dict()) for v, (k, t, fam) in variants.items()}

    # ---------------- 2. validation (10,000 independent streams per null path) + restart ARL
    arl = {}

    def arl_hook(b):
        for v, (k, t, fam) in variants.items():
            if fam in ('core', 'native') or v in ('eCUSUM',):
                t0 = time.time()
                r = C.arl0(mons[k], b, t)
                timing['arl_' + k] = timing.get('arl_' + k, 0) + time.time() - t0
                arl[v] = r
    val = {}; valtb = {}
    for p in E.PATHS:
        hook = arl_hook if p == 'const10' else None
        val[p], valtb[p], nl = collect_maxima(mons, cfg[p], rv, 6100 + 4 * a.idx + E.PATHS.index(p) + off['val'], nval, timing, hook)
        log('val', p, 'done')
    res['validation'] = {}
    for v, (k, t, fam) in variants.items():
        res['validation'][v] = {p: C.validate(C.fa_from_maxima(val[p][k], t, valtb[p]), cap=0.03) for p in E.PATHS}
    res['arl_restart'] = arl
    res['arl_note'] = f'restart after alarm on the first {E.CHUNK} validation streams (const 0.10), threshold h_hi (no randomization)'
    np.savez_compressed(os.path.join(a.out, f'setting_{a.idx:02d}_valmax.npz'),
                        **{f'{p}__{k}': val[p][k].astype(np.float32) for p in E.PATHS for k in mons},
                        **{f'tb__{p}': valtb[p] for p in E.PATHS})

    # ---------------- 3. evaluation (10,000 streams per harmful condition) + trade-off sweeps
    res['evaluation'] = {}; alarms_out = {}
    for r1 in E.R1S:
        hcfg = G.nominal_config(pi, q, d, E.H, E.TAU, E.RPRE, r1)
        do_to = (pi, r1, q, d) in [tuple(x) for x in E.TRADEOFF]
        grids = {}
        if do_to:
            for k in mons:
                g = np.unique(np.r_[C.threshold_grid(val['const10'][k]), C.threshold_grid(val['path'][k])])
                grids[k] = g[np.isfinite(g)]
            to_acc = {k: {'early': np.zeros(len(grids[k])), **{D: np.zeros(len(grids[k])) for D in E.DEADLINES}}
                      for k in mons}
        al = {v: [] for v in variants}; nl = 0
        for b in chunks(hcfg, re_, E.harm_cond(a.idx, r1) + off['ev'], nev):
            nl += len(b.loss)
            for k, m in mons.items():
                vs = [v for v in variants if variants[v][0] == k]
                ths = sorted({x for v in vs for x in (variants[v][1].h_hi, variants[v][1].h_lo)})
                t0 = time.time()
                A = m.evaluate(b, ths, ())['alarm']
                for v in vs:
                    t = variants[v][1]
                    hi = A[:, ths.index(t.h_hi)]; lo = A[:, ths.index(t.h_lo)]
                    al[v].append(np.where(b.tiebreak < t.gamma, lo, hi))
                if do_to and len(grids[k]):
                    Ag = m.alarm_grid(b, grids[k])
                    to_acc[k]['early'] += (Ag < E.TAU).sum(0)
                    for D in E.DEADLINES:
                        dl = min(E.H, E.TAU + D - 1)
                        to_acc[k][D] += ((Ag >= E.TAU) & (Ag <= dl)).sum(0)
                timing['eval_' + k] = timing.get('eval_' + k, 0) + time.time() - t0
            del b
        ev = {}
        for v in variants:
            x = np.concatenate(al[v]); alarms_out[f'r{r1:g}__{v}'] = x.astype(np.float32)
            mt = C.detection_metrics(x, E.TAU, E.H, E.DEADLINES)
            for D in E.DEADLINES:
                mt[f'timely_q50_{D}'], mt[f'timely_q90_{D}'] = timely_quantiles(x, D)
            ev[v] = mt
        res['evaluation'][f'{r1:g}'] = dict(labels_per_stream=nl / nev, metrics=ev,
                                            oracle={D: E.oracle_beta(pi, q, d, r1, D) for D in E.DEADLINES})
        if do_to:
            rows = []
            for k in mons:
                g = grids[k]
                fc = (val['const10'][k][:, None] > g[None, :]).mean(0)
                fp = (val['path'][k][:, None] > g[None, :]).mean(0)
                f5 = (val['const05'][k][:, None] > g[None, :]).mean(0)
                for j in range(len(g)):
                    rows.append(dict(monitor=k, h=g[j], fa_const10=fc[j], fa_path=fp[j], fa_const05=f5[j],
                                     fa_comp=max(fc[j], fp[j]), early=to_acc[k]['early'][j] / nev,
                                     **{f'pD_{D}': to_acc[k][D][j] / nev for D in E.DEADLINES}))
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f'tradeoff_pi{pi:g}_r{r1:g}_q{q:g}_d{d}.csv'), index=False)
        log('eval r1', r1, 'done')
    np.savez_compressed(os.path.join(a.out, f'setting_{a.idx:02d}_alarms.npz'), **alarms_out)
    res['timing_s'] = timing; res['wall_s'] = time.time() - T0
    with open(os.path.join(a.out, f'setting_{a.idx:02d}.json'), 'w') as f:
        json.dump(jsonable(res), f, indent=1)
    log('wrote setting', a.idx)


if __name__ == '__main__':
    main()
