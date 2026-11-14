"""E3 main run: calibrate -> validate -> evaluate for every (condition, workflow, policy).
python -m revision.experiments.E3.run [--smoke]"""
import json, os, sys, time, pickle
import numpy as np
from revision import calibration as C
from . import design as Dz, core as Co

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E3'
SMOKE = '--smoke' in sys.argv
NC, NV, NE = (2000, 1000, 1000) if SMOKE else (Dz.N_CAL, Dz.N_VAL, Dz.N_EVAL)
START = 0 if SMOKE else 100000   # fresh replicate ids (smoke used 0..1999)
RAW = os.path.join(OUT, 'raw_smoke' if SMOKE else 'raw'); os.makedirs(RAW, exist_ok=True)
INF = np.inf


def cal_pair(pol, cond, wf, h_warn):
    thr = {}; mx = {}
    for path in (0, 1):
        out, st, tb = Co.simulate(pol, cond, path, wf, 'calibration', NC, h_warn=h_warn, start=START)
        mx[path] = out
    for mon, col in (('msr', 0), ('bet', 1)):
        t0 = C.calibrate_threshold(mx[0][:, col], Dz.TARGET_FA, 'randomized')
        t1 = C.calibrate_threshold(mx[1][:, col], Dz.TARGET_FA, 'randomized')
        thr[mon] = (t0 if t0.h >= t1.h else t1, t0.h, t1.h)
    return thr


def main():
    log = open(os.path.join(OUT, 'run_smoke.log' if SMOKE else 'run.log'), 'a')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    for cond in (0, 1, 2):
        pi = Dz.CONDITIONS[cond]['pi']
        for wf in (0, 1):
            pols, _ = Dz.policies(pi, wf)
            ref_h = None
            for name, pol in pols.items():
                fn = os.path.join(RAW, f'c{cond}_wf{wf}_{name}.pkl')
                if os.path.exists(fn):
                    if name in ('P2_block_uniform', 'B2_passed_block_uniform'):
                        ref_h = pickle.load(open(fn, 'rb'))['thr']['msr'][0].h
                    continue
                t0 = time.time()
                iters = []
                if pol['warn']:
                    h = ref_h
                    for it in range(8):
                        thr = cal_pair(pol, cond, wf, Dz.warn_level(h, pol['warn_mode']))
                        hn = thr['msr'][0].h
                        iters.append((h, hn))
                        P(f'   iter {it}: warn from h={h:.4f} -> calibrated {hn:.4f}')
                        if abs(hn - h) < 0.02:
                            break
                        h = hn
                    h_warn = Dz.warn_level(h, pol['warn_mode'])
                else:
                    thr = cal_pair(pol, cond, wf, INF)
                    h_warn = INF
                if name in ('P2_block_uniform', 'B2_passed_block_uniform'):
                    ref_h = thr['msr'][0].h
                tm, tb_ = thr['msr'][0], thr['bet'][0]
                th = (tm.h_hi, tm.h_lo, tb_.h_hi, tb_.h_lo)
                val = {}
                for path in (0, 1, 2):
                    out, st, tbr = Co.simulate(pol, cond, path, wf, 'validation', NV, th=th, h_warn=h_warn, start=START)
                    am = Co.alarm(out, tbr, 'msr', tm); ab = Co.alarm(out, tbr, 'bet', tb_)
                    val[path] = dict(fa_msr=am <= Dz.H, fa_bet=ab <= Dz.H, native_bet=out[:, 1] > np.log(1 / 0.02),
                                     stats_mean=st.mean(0))
                out, st, tbr = Co.simulate(pol, cond, 3, wf, 'evaluation', NE, th=th, h_warn=h_warn, start=START)
                ev = dict(alarm_msr=Co.alarm(out, tbr, 'msr', tm), alarm_bet=Co.alarm(out, tbr, 'bet', tb_), stats=st)
                res = dict(cond=cond, wf=wf, policy=name, thr={k: v for k, v in thr.items()}, h_warn=h_warn,
                           iters=iters, val=val, ev=ev)
                pickle.dump(res, open(fn, 'wb'))
                pd10 = np.mean((ev['alarm_msr'] >= Dz.TAU) & (ev['alarm_msr'] <= Dz.TAU + 9999))
                P(f'c{cond} wf{wf} {name}: h_msr={tm.h:.3f} (const {thr["msr"][1]:.3f}, path {thr["msr"][2]:.3f}) '
                  f'h_bet={tb_.h:.3f} FAmsr={val[0]["fa_msr"].mean():.4f}/{val[1]["fa_msr"].mean():.4f} '
                  f'pD10k_msr={pd10:.4f} aud={st[:,0].mean():.0f} {time.time()-t0:.0f}s')
    P('done')


if __name__ == '__main__':
    main()
