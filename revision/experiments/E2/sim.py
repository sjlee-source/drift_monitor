"""E2 robustness simulation (stressors 1-6). Experiment id 12.

Usage: python -m revision.experiments.E2.sim [--smoke] [--protocol-only]
"""
import json, sys, time, os
import numpy as np, pandas as pd
from revision import generators as G, monitors as Mo, calibration as C

EXP = 12
OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E2'
SMOKE = '--smoke' in sys.argv
N_CAL, N_VAL, N_EVAL = (2000, 1000, 1000) if SMOKE else (20000, 10000, 10000)
CHUNK = 5000
H, TAU, PI, R0, RPRE, R1, D0 = 29999, 10000, 0.10, 0.10, 0.05, 0.20, 500
QS = (0.20, 0.50)
DEADLINES = (5000, 10000, 20000)
TARGET = 0.02
MON_NAMES = ['cusum_0.2', 'cusum_0.12', 'betting', 'msr', 'eSR', 'ewma', 'window_binomial']

# condition id = 10*q_index + path_index; identical across variants -> common random numbers
PATHS = {'null_const010': 0, 'null_path': 1, 'null_const005': 2, 'harm': 3}

# variant spec: dict of StreamConfig overrides + optional ramp / prevalence spec
VARIANTS = {
    'base_iid':        dict(),
    'markov_phi0.3':   dict(cfg=dict(markov_rho=0.3)),
    'markov_phi0.6':   dict(cfg=dict(markov_rho=0.6)),
    'board_icc_r0.2':  dict(cfg=dict(board_size=50, icc_r=0.2)),
    'board_icc_pi0.1': dict(cfg=dict(board_size=50, icc_pi=0.1)),
    'ramp_5000':       dict(ramp=5000),
    'ramp_2000':       dict(ramp=2000),   # supplement
    'prev_step_x2':    dict(prev=dict(kind='step', pi=PI, pi_post=2 * PI, tau=TAU)),
    'prev_sin_50pct':  dict(prev=dict(kind='sinusoid', pi=PI, amplitude=0.5, period=10000)),
    'delay_geom500':   dict(delay=('geometric', 500)),
    'delay_lognorm_med500_s1': dict(delay=('lognormal', 500 * np.exp(0.5), 1.0)),
    'delay_batch1000': dict(delay=('batch', 1000)),
    'labelnoise_e10_.05_e01_.01': dict(cfg=dict(e10=0.05, e01=0.01)),
}


def make_delay(spec):
    if spec is None:
        return G.Delay.constant(D0)
    k = spec[0]
    if k == 'geometric': return G.Delay.geometric(spec[1])
    if k == 'lognormal': return G.Delay.lognormal(spec[1], spec[2])
    if k == 'batch': return G.Delay.batch(spec[1])
    raise ValueError(k)


def config(variant, path, q):
    v = VARIANTS[variant]
    ramp = v.get('ramp', 0)
    if path == 'null_const010': r = G.risk_path(H, R0)
    elif path == 'null_const005': r = G.risk_path(H, RPRE)
    elif path == 'null_path': r = G.risk_path(H, RPRE, R0, TAU, ramp=ramp)
    elif path == 'harm': r = G.risk_path(H, RPRE, R1, TAU, ramp=ramp)
    pi = G.prevalence_path(H, **v['prev']) if 'prev' in v else PI
    return G.StreamConfig(H=H, pi=pi, r=r, q=q, delay=make_delay(v.get('delay')), **v.get('cfg', {}))


def monitors():
    s = Mo.default_suite(R0)
    return {k: s[k] for k in MON_NAMES}


def batches(variant, path, q, role, n):
    qi = QS.index(q)
    return G.generate_chunks(config(variant, path, q), role, EXP, 10 * qi + PATHS[path], n, CHUNK)


def maxima(variant, path, q, mons):
    out = {k: [] for k in mons}; tie = []
    for b in batches(variant, path, q, 'calibration', N_CAL):
        for k, m in mons.items():
            out[k].append(m.max_stat(b))
        tie.append(b.tiebreak)
    return {k: np.concatenate(v) for k, v in out.items()}


def conservative(ta, tb):
    """Pick the threshold with the lower FA (larger h; equal h -> smaller gamma)."""
    if ta.h != tb.h:
        return ta if ta.h > tb.h else tb
    return ta if ta.gamma <= tb.gamma else tb


def calibrate(variant, q, mons):
    mc = maxima(variant, 'null_const010', q, mons)
    mp = maxima(variant, 'null_path', q, mons)
    thr = {}
    for k in mons:
        tc = C.calibrate_threshold(mc[k], TARGET, 'randomized')
        tp = C.calibrate_threshold(mp[k], TARGET, 'randomized')
        t = conservative(tc, tp)
        thr[k] = dict(thr=t, h_const=tc.h, h_path=tp.h, binding='const' if t is tc else 'path')
    return thr


def alarms(variant, path, q, role, n, mons, thrsets):
    """thrsets: {label: {monitor: Threshold}} -> {label: {monitor: alarm array}}"""
    acc = {lab: {k: [] for k in mons} for lab in thrsets}
    nlab = []
    for b in batches(variant, path, q, role, n):
        nlab.append(np.diff(b.offsets))
        for lab, ts in thrsets.items():
            for k, m in mons.items():
                acc[lab][k].append(C.alarm_times(m, b, ts[k]))
    return {lab: {k: np.concatenate(v) for k, v in d.items()} for lab, d in acc.items()}, np.concatenate(nlab)


def protocol():
    return dict(
        experiment='E2 robustness', experiment_id=EXP, written=time.strftime('%Y-%m-%d %H:%M:%S'),
        base=dict(pi=PI, r_pre=RPRE, r0=R0, r1=R1, d=D0, H=H, tau=TAU, q=QS, deadlines=DEADLINES, primary_D=10000),
        monitors=MON_NAMES, monitor_settings='revision.monitors.default_suite defaults (betting gain4/cap9/window50; '
        'msr 5 stakes; eSR alpha_design=1e-3 min_gap .02; ewma lam .05; window_binomial W=200)',
        variants={k: {kk: (str(vv) if kk == 'delay' else vv) for kk, vv in v.items()} for k, v in VARIANTS.items()},
        variant_notes=dict(
            markov='markov_rho (paper: phi) = lag-1 corr of misses over successive DEFECTIVE items (markov_over=defect)',
            board='consecutive boards of 50 items; Beta random effect, ICC on r (0.2) or on pi (0.1) separately',
            ramp='linear r from 0.05 at tau-1 to r1 at tau+L-1; null path variant = same-length ramp 0.05->0.10',
            prevalence='applied to all paths (null and harmful); step pi 0.1->0.2 at tau; sinusoid pi(1+0.5 sin(2 pi t/1e4))',
            delay='geometric on {0,1,..} mean 500; lognormal median 500 sigma 1 (mean 824); batch release at ends of 1000-item shifts (mean ~500)',
            labelnoise='e10=P(label 0|true miss)=0.05, e01=P(label 1|true non-miss)=0.01; noisy-label rate r\'=r(1-e10)+(1-r)e01'),
        calibration=dict(rule='h*=max(h_const,h_path) two-point composite null, randomized ties, target 0.02, '
                         f'{N_CAL} streams per null path, role=calibration', nominal='calibrated under base_iid model (PRIMARY)',
                         recalibrated='same rule on null streams of the variant (SECONDARY)'),
        validation=dict(n=N_VAL, role='validation', paths=['null_const010', 'null_path(variant)', 'null_const005'],
                        ci='exact Clopper-Pearson; gate one-sided 95% upper <= 0.03 with Bonferroni over family',
                        family='nominal-threshold FA cells: 7 monitors x 2 q x 2 null paths (const010, path) x 13 variants = 364'),
        evaluation=dict(n=N_EVAL, role='evaluation', endpoint='P(tau<=T<=tau+D-1), unconditional'),
        seeding='rng.stream_states(role, 12, condition=10*q_index+path_index, replicate); condition ids shared across '
                'variants (common random numbers; paired with base)',
        classification_prespecified='FA control broken: nominal-threshold FA simultaneous one-sided upper bound > 0.03 AND '
                                    'point FA > 0.03; inflated-but-gated: point FA > 0.02 with sim. lower CP bound > 0.02',
        planning_part='stressor 7 by exact DP (theory/exact_cusum.py, converse.py); see plan_dp.py')


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = '_smoke' if SMOKE else ''
    if not SMOKE:
        json.dump(protocol(), open(f'{OUT}/protocol.json', 'w'), indent=1, default=str)
    if '--protocol-only' in sys.argv:
        return
    mons = monitors()
    fa_rows, det_rows, thr_rows = [], [], []
    variants = list(VARIANTS) if not os.environ.get('E2_VARIANTS') else os.environ['E2_VARIANTS'].split(',')
    for q in QS:
        t0 = time.time()
        nom = calibrate('base_iid', q, mons)
        print(f'q={q} nominal calibrated {time.time()-t0:.0f}s', flush=True)
        for variant in variants:
            t0 = time.time()
            rec = nom if variant == 'base_iid' else calibrate(variant, q, mons)
            ts = {'nominal': {k: nom[k]['thr'] for k in mons}, 'recalibrated': {k: rec[k]['thr'] for k in mons}}
            for k in mons:
                for lab, src in (('nominal', nom), ('recalibrated', rec)):
                    t = src[k]['thr']
                    thr_rows.append(dict(q=q, variant=variant, threshold=lab, monitor=k, h=t.h, gamma=t.gamma,
                                         h_const=src[k]['h_const'], h_path=src[k]['h_path'], binding=src[k]['binding']))
            for path in ('null_const010', 'null_path', 'null_const005'):
                al, nl = alarms(variant, path, q, 'validation', N_VAL, mons, ts)
                for lab in ts:
                    for k in mons:
                        a = al[lab][k] <= H
                        v = C.validate(a, cap=0.03, family=364)
                        _, up = C.clopper_pearson(v['alarms'], v['n'], 0.95, 'upper')
                        # simultaneous one-sided lower bound
                        lo_sim = 0.0 if v['alarms'] == 0 else float(C.beta.ppf(0.05 / 364, v['alarms'], v['n'] - v['alarms'] + 1))
                        fa_rows.append(dict(q=q, variant=variant, threshold=lab, monitor=k, path=path, n=v['n'],
                                            alarms=v['alarms'], fa=v['rate'], cp_lo=v['cp_lo'], cp_hi=v['cp_hi'],
                                            upper_sim364=v['upper_simultaneous'], lower_sim364=lo_sim,
                                            gate_pass=v['pass_gate'], mean_labels=float(nl.mean())))
            al, nl = alarms(variant, 'harm', q, 'evaluation', N_EVAL, mons, ts)
            for lab in ts:
                for k in mons:
                    m = C.detection_metrics(al[lab][k], TAU, H, DEADLINES)
                    det_rows.append(dict(q=q, variant=variant, threshold=lab, monitor=k, mean_labels=float(nl.mean()), **m))
            print(f'  q={q} {variant} done {time.time()-t0:.0f}s', flush=True)
            pd.DataFrame(fa_rows).to_csv(f'{OUT}/fa{tag}.csv', index=False)
            pd.DataFrame(det_rows).to_csv(f'{OUT}/detection{tag}.csv', index=False)
            pd.DataFrame(thr_rows).to_csv(f'{OUT}/thresholds{tag}.csv', index=False)


if __name__ == '__main__':
    main()
