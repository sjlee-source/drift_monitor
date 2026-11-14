"""E2c CONFIRMATORY: boundary-recentred moving-block bootstrap calibration from a finite null history.

Usage: python -m revision.experiments.E2.e2c [--protocol-only] [--smoke] [--arm primary|curve1H|curve5H|incontrol]
Experiment id 12; fresh namespaces (conditions >= 5000), disjoint from E2/E2b.
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import beta
from revision import generators as G, calibration as C
from revision.experiments.E2 import e2b, sim

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E2/E2c'
SMOKE = '--smoke' in sys.argv
EXP = 12
H, TAU, D0, R0, TARGET = sim.H, sim.TAU, sim.D0, 0.10, 0.02
QS = (0.2, 0.5)
VARS = ['base_iid', 'markov_phi0.3', 'markov_phi0.6', 'board_icc_r0.2']
MON = ['cusum_0.12', 'cusum_0.2', 'betting', 'window_binomial']
BLOCKS, GRID = (50, 250, 1000), 50
B_BOOT = 300 if SMOKE else 2000
M_VAL = 10                 # fresh validation streams per history and null path (history = sampling unit)
N_POOL = 1000 if SMOKE else 10000
ARMS = {  # name: (history length in items, K histories, history risk, procedures)
    'primary':   (300000, 20 if SMOKE else 1000, 0.10, ('RC',)),
    'curve1H':   (30000, 20 if SMOKE else 500, 0.10, ('RC',)),
    'curve5H':   (150000, 20 if SMOKE else 500, 0.10, ('RC',)),
    'incontrol': (300000, 20 if SMOKE else 500, 0.05, ('NR', 'RC')),
}
ARM_ID = {'primary': 0, 'curve1H': 1, 'curve5H': 2, 'incontrol': 3}
PATHS = {'null_const010': 0, 'null_path': 1, 'null_const005': 2, 'harm': 3}
FAM_PRIMARY = len(VARS) * len(QS) * len(MON) * 2   # 64
FAM_C005 = len(VARS) * len(QS) * len(MON)          # 32
NBOOT_CL = 200000


def protocol():
    return dict(
        experiment='E2c confirmatory: boundary-recentred block-bootstrap calibration from finite null history',
        experiment_id=EXP, written=time.strftime('%Y-%m-%d %H:%M:%S'),
        basis='CODEX_REVIEW of E2b (exp_E2/E2b/CODEX_REVIEW.md); E2b PH1/PH2 were exploratory; nothing below is tuned on E2c data',
        conditions=dict(variants=VARS, q=list(QS), other='pi .1, d 500 constant delay, H 29,999, tau 10,000, harmful r .05->.20'),
        monitors=MON,
        monitor_exclusion_rationale=('e-SR, mSR, EWMA excluded a priori: in E2b they had the largest residual FA failures '
                                     'even with recentring and 5H histories (PH2: 16 of 19 failing cells), i.e. the '
                                     'exploratory stage already gave no support for finite-history calibration of these '
                                     'monitors; E2c tests the restricted claim for the four monitors that the exploratory '
                                     'evidence did not contradict. Their E2/E2b results stand as reported (not controlled).'),
        procedure=dict(
            RC=('boundary-recentred moving-block bootstrap: one null history (length per arm) of the true variant, same q, '
                'd=500, labels with origin <= history length; blocks of L items with starts uniform on the 50-item board '
                'grid; concatenated to H items; arrival=origin+500; labels after H dropped; every bootstrap label flipped '
                'independently so that its marginal rate is r0=0.10 (0->1 w.p. (r0-rhat)/(1-rhat) if rhat<r0, else 1->0 '
                'w.p. 1-r0/rhat; rhat = history miss rate); B=2000 per L; randomized calibrate_threshold at 0.02 on '
                'bootstrap maxima (const null); threshold = max over L in {50,250,1000} (fixed rule carried over from '
                'E2b, not tuned).'),
            NR='in-control arm only: identical but without recentring (bootstrap at the history rate)'),
        arms={k: dict(history_items=v[0], history_in_H_units=round(v[0] / 30000, 2), K_histories=v[1],
                      history_risk=v[2], procedures=list(v[3])) for k, v in ARMS.items()},
        primary_arm='primary (10H history, r=0.10 boundary history, RC); 1H/5H = learning curve (secondary); '
                    'incontrol (10H history at r=0.05; NR vs RC) = secondary',
        comparators='iid-nominal and oracle thresholds = E2 sim.calibrate (20,000 calibration streams/null path, E2 seeds)',
        seeds=dict(history='role pilot, condition 5000+100*arm_id+10*q_index, replicate k (CRN across variants)',
                   bootstrap='SeedSequence(20261111, spawn_key=(8,12,6000+1000*arm_id+100*variant_index+10*q_index,k,L_index))',
                   per_history_validation='role validation, condition 7000+100*arm_id+10*q_index+path_index, replicates '
                                          'k*M..k*M+M-1, M=10 (independent across histories)',
                   pools='role validation (nulls) / evaluation (harm), condition 7900+10*q_index+path_index, 10,000 streams, '
                         'shared by all histories and arms'),
        inference=dict(
            sampling_unit='history',
            unconditional_FA=('each history k paired with its own M=10 fresh validation streams per null path; estimate = '
                              'mean of the K*M indicators; clusters (histories) i.i.d.; bound = cluster (history) '
                              'bootstrap percentile, B=200,000, one-sided at 1-0.05/F; exact CP on one stream per '
                              'history (first replicate) reported as a check'),
            conditional_reliability=('FA_k = FA of history-k threshold on the shared pool (10,000 streams); key secondary '
                                     '= one-sided 95% CP upper bound (Bonferroni over F=64) on P_history(FA_k > 0.03) '
                                     'from the count of histories with pooled FA_k > 0.03 (FA_k measured with sd ~0.0017)'),
            families=dict(primary=f'{FAM_PRIMARY} cells = 4 variants x 2 q x 4 monitors x 2 null paths (const .10, '
                                  '.05->.10 at tau) in the primary arm', const005=f'{FAM_C005} cells, separate family',
                          secondary_arms='same bounds per arm, reported descriptively with the same per-cell F')),
        success_criterion=('PRIMARY: in the primary arm every one of the 64 cells has cluster-bootstrap simultaneous one-'
                           'sided 95% upper bound on unconditional FA <= 0.03. KEY SECONDARY: 95% upper bound (F=64) on '
                           'P_history(FA_k > 0.03) <= 0.10 in every cell. Both reported per cell and as counts.'),
        price=('p_D at D=5k/10k/20k on the shared harmful pool (10,000 streams) averaged over histories, its 5/50/95% '
               'distribution over histories, median detection delay among timely detections, threshold dispersion '
               '(sd, 5/95% of h over histories) vs the oracle threshold; reported, not a criterion'),
        compute_plan=('<=4 threads, chunks of 5,000 streams, RAM < 2 GB; cost dominated by 6,000 bootstrap streams x 4 '
                      'monitors per history; histories total: primary 8x1000, curves 2x8x500, incontrol 8x500'),
        not_tuned='L rule, B, M, K and recentring fixed before any E2c data were generated')


def history(variant, qi, k, arm):
    HL, _, rh, _ = ARMS[arm]
    cfg = sim.config(variant, 'null_const010', QS[qi])
    cfg = G.StreamConfig(**{**cfg.__dict__, 'H': HL + D0, 'r': np.full(HL + D0, rh)})
    b = G.generate(cfg, 'pilot', EXP, 5000 + 100 * ARM_ID[arm] + 10 * qi, np.array([k]))
    s = b.stream(0); keep = s['origin'] <= HL
    return (s['origin'][keep] - 1).astype(np.int64), s['loss'][keep].astype(np.int8), HL


def boot_batch(idx, loss, HL, L, B, rng, q):
    old = e2b.H_HIST; e2b.H_HIST = HL
    try:
        return e2b.boot_batch(idx, loss, L, B, rng, q)
    finally:
        e2b.H_HIST = old


def thresholds(variant, qi, k, arm, mons):
    idx, loss, HL = history(variant, qi, k, arm)
    rhat = float(loss.mean())
    procs = ARMS[arm][3]
    best = {p: {} for p in procs}
    vi = VARS.index(variant)
    for li, L in enumerate(BLOCKS):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            20261111, spawn_key=(8, EXP, 6000 + 1000 * ARM_ID[arm] + 100 * vi + 10 * qi, k, li))))
        b = boot_batch(idx, loss, HL, L, B_BOOT, rng, QS[qi])
        u = rng.random(len(b.loss))
        raw = b.loss.copy()
        for p in procs:
            if p == 'RC':
                if rhat < R0:
                    b.loss = np.where((raw == 0) & (u < (R0 - rhat) / (1 - rhat)), 1, raw).astype(np.int8)
                else:
                    b.loss = np.where((raw == 1) & (u < 1 - R0 / rhat), 0, raw).astype(np.int8)
            else:
                b.loss = raw
            for m, mo in mons.items():
                t = C.calibrate_threshold(mo.max_stat(b), TARGET, 'randomized')
                best[p][m] = t if m not in best[p] else sim.conservative(best[p][m], t)
    return best, rhat, len(loss)


def cluster_upper(counts, M, F, rng):
    K = len(counts)
    idx = rng.integers(0, K, size=(NBOOT_CL, K))
    bs = counts[idx].sum(1) / (K * M)
    return float(np.quantile(bs, 1 - 0.05 / F))


def cp_up(k, n, F):
    return 1.0 if k >= n else float(beta.ppf(1 - 0.05 / F, k + 1, n - k))


def run_arm(arm):
    os.makedirs(OUT, exist_ok=True)
    tag = ('_smoke' if SMOKE else '') + f'_{arm}'
    HL, K, rh, procs = ARMS[arm]
    mons_all = sim.monitors(); mons = {m: mons_all[m] for m in MON}
    crng = np.random.default_rng(20261113 + ARM_ID[arm])
    fa_rows, det_rows, thr_rows = [], [], []
    for qi, q in enumerate(QS):
        nom = sim.calibrate('base_iid', q, mons)
        # shared pools (generated once per q and variant below)
        for variant in VARS:
            t0 = time.time()
            orc = nom if variant == 'base_iid' else sim.calibrate(variant, q, mons)
            T = {p: {m: [] for m in MON} for p in procs}
            for k in range(K):
                best, rhat, nl = thresholds(variant, qi, k, arm, mons)
                for p in procs:
                    for m in MON:
                        t = best[p][m]; T[p][m].append(t)
                        thr_rows.append(dict(arm=arm, proc=p, q=q, variant=variant, history=k, rhat=rhat, n_labels=nl,
                                             monitor=m, h=t.h, gamma=t.gamma, h_hi=t.h_hi, h_lo=t.h_lo))
            print(f'{arm} q={q} {variant} thresholds {time.time()-t0:.0f}s', flush=True)
            labs = [('nominal', None), ('oracle', None)] + [(p, p) for p in procs]
            getT = lambda lab, m: [nom[m]['thr']] if lab == 'nominal' else ([orc[m]['thr']] if lab == 'oracle' else T[lab][m])
            for path in ('null_const010', 'null_path', 'null_const005'):
                cfg = sim.config(variant, path, q)
                # per-history fresh streams: replicate k*M + i belongs to history k
                cond = 7000 + 100 * ARM_ID[arm] + 10 * qi + PATHS[path]
                mxv = {m: [] for m in MON}; tbv = []
                for b in G.generate_chunks(cfg, 'validation', EXP, cond, K * M_VAL, 5000):
                    tbv.append(b.tiebreak)
                    for m in MON: mxv[m].append(mons[m].max_stat(b))
                tbv = np.concatenate(tbv)
                # shared pool
                mxp = {m: [] for m in MON}; tbp = []
                for b in G.generate_chunks(cfg, 'validation', EXP, 7900 + 10 * qi + PATHS[path], N_POOL, 5000):
                    tbp.append(b.tiebreak)
                    for m in MON: mxp[m].append(mons[m].max_stat(b))
                tbp = np.concatenate(tbp)
                F = FAM_PRIMARY if path != 'null_const005' else FAM_C005
                for m in MON:
                    Mv = np.concatenate(mxv[m]).reshape(K, M_VAL); tv = tbv.reshape(K, M_VAL)
                    Mp = np.concatenate(mxp[m])
                    for lab, _ in labs:
                        ts = getT(lab, m)
                        if len(ts) == 1:
                            ts = ts * K
                        X = np.stack([C.fa_from_maxima(Mv[k], ts[k], tv[k]) for k in range(K)])   # (K, M)
                        cnt = X.sum(1)
                        fak = np.array([C.fa_from_maxima(Mp, t, tbp).mean() for t in (ts if lab in procs else ts[:1])])
                        n_gt = int((fak > 0.03).sum()); Kp = len(fak)
                        r = dict(arm=arm, proc=lab, q=q, variant=variant, monitor=m, path=path, K=K, M=M_VAL,
                                 fa_uncond=float(cnt.sum() / (K * M_VAL)),
                                 upper_cluster=cluster_upper(cnt, M_VAL, F, crng),
                                 fa_first_stream=float(X[:, 0].mean()),
                                 upper_cp_first_stream=cp_up(int(X[:, 0].sum()), K, F),
                                 icc_hint=float(np.var(cnt / M_VAL, ddof=1)),
                                 fa_pool_mean=float(fak.mean()), fa_pool_p50=float(np.median(fak)),
                                 fa_pool_p90=float(np.quantile(fak, .9)), fa_pool_max=float(fak.max()),
                                 n_hist_fa_gt_003=n_gt, P_hist_fa_gt_003=n_gt / Kp,
                                 upper_P_hist_gt_003=cp_up(n_gt, Kp, FAM_PRIMARY) if lab in procs else None, F=F)
                        r['pass_primary'] = r['upper_cluster'] <= 0.03
                        r['pass_secondary'] = (r['upper_P_hist_gt_003'] <= 0.10) if lab in procs else None
                        fa_rows.append(r)
            # harmful pool
            cfg = sim.config(variant, 'harm', q)
            acc = {(lab, m): [] for lab, _ in labs for m in MON}
            for b in G.generate_chunks(cfg, 'evaluation', EXP, 7900 + 10 * qi + 3, N_POOL, 2500):
                for m in MON:
                    for lab, _ in labs:
                        acc[(lab, m)].append(e2b.alarms_many(mons[m], b, getT(lab, m)))
            for m in MON:
                hs = np.array([t.h for t in T[procs[-1]][m]])
                for lab, _ in labs:
                    a = np.concatenate(acc[(lab, m)])           # (N, K or 1)
                    for D in sim.DEADLINES:
                        s = (a >= TAU) & (a <= min(H, TAU + D - 1)); ph = s.mean(0)
                        dl = np.where(s, a - TAU, np.nan)
                        det_rows.append(dict(arm=arm, proc=lab, q=q, variant=variant, monitor=m, D=D, pD=float(s.mean()),
                                             pD_hist_p05=float(np.quantile(ph, .05)), pD_hist_p50=float(np.median(ph)),
                                             pD_hist_p95=float(np.quantile(ph, .95)),
                                             early=float((a < TAU).mean()),
                                             median_delay_timely=float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
                                             h_mean=float(np.mean([t.h for t in getT(lab, m)])),
                                             h_sd=float(np.std([t.h for t in getT(lab, m)])),
                                             h_p05=float(np.quantile([t.h for t in getT(lab, m)], .05)),
                                             h_p95=float(np.quantile([t.h for t in getT(lab, m)], .95))))
            print(f'{arm} q={q} {variant} done {time.time()-t0:.0f}s', flush=True)
            pd.DataFrame(fa_rows).to_csv(f'{OUT}/fa{tag}.csv', index=False)
            pd.DataFrame(det_rows).to_csv(f'{OUT}/detection{tag}.csv', index=False)
            pd.DataFrame(thr_rows).to_csv(f'{OUT}/thresholds{tag}.csv', index=False)
    print(f'{arm} total done', flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not SMOKE and not os.path.exists(f'{OUT}/protocol.json'):
        json.dump(protocol(), open(f'{OUT}/protocol.json', 'w'), indent=1, default=str)
    if '--protocol-only' in sys.argv:
        return
    arm = sys.argv[sys.argv.index('--arm') + 1]
    run_arm(arm)


if __name__ == '__main__':
    main()
