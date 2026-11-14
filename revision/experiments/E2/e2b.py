"""E2b: dependence-aware threshold estimation from a FINITE historical null stream (block / board bootstrap).

Usage: python -m revision.experiments.E2.e2b [--protocol-only] [--smoke]
Experiment id 12, distinct sub-stream namespace (condition ids >= 1000; see protocol()).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import beta
from revision import generators as G, calibration as C, rng as R
from revision.experiments.E2 import sim

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E2/E2b'
SMOKE = '--smoke' in sys.argv
EXP = 12
H, TAU, D0, TARGET = sim.H, sim.TAU, sim.D0, sim.TARGET
QS = sim.QS
VARS = ['base_iid', 'markov_phi0.3', 'markov_phi0.6', 'board_icc_r0.2']
MON = sim.MON_NAMES
K_HIST = 10 if SMOKE else 100        # independent historical streams per (variant, q)
B_BOOT = 500 if SMOKE else 2000      # bootstrap streams per history and block length
N_VAL = 1000 if SMOKE else 10000
N_EVAL = 1000 if SMOKE else 10000
H_HIST = 30000                        # history length in production items (~ one horizon)
BLOCKS = (50, 250, 1000)              # block lengths (items), starts on the 50-item board grid
GRID = 50
PATHS = {'null_const010': 0, 'null_path': 1, 'null_const005': 2, 'harm': 3}
FAMILY = len(VARS) * len(QS) * len(MON) * 2      # 112
NONE16 = np.uint16(65535)


def cond_eval(qi, path):        # fresh validation/evaluation streams, CRN across variants
    return 1000 + 10 * qi + PATHS[path]


def cond_hist(qi):              # historical null streams, CRN across variants
    return 2000 + 10 * qi


def protocol():
    return dict(
        experiment='E2b dependence-aware recalibration from finite history', experiment_id=EXP,
        written=time.strftime('%Y-%m-%d %H:%M:%S'),
        motivation='CODEX_REVIEW.md item 5: deployable alternative to oracle recalibration',
        conditions=dict(variants=VARS, q=QS, other='pi .1, d 500, H 29999, tau 10000, r .05->.20 harmful (as E2)'),
        monitors=MON,
        thresholds=dict(
            iid_nominal='E2 nominal (sim.calibrate base_iid, 20,000 calibration streams per null path, same seeds as E2)',
            oracle='E2 recalibrated (sim.calibrate on the true variant; same seeds as E2) - evidence of recoverability only',
            estimated=('from ONE historical null stream per replicate: H_hist=30,000 items generated from the true variant '
                       'at constant r=0.10 (boundary), same q and delay d=500, all labels with origin<=30,000 observed. '
                       'Moving-block bootstrap on the production-item axis: blocks of L items, starts uniform on the '
                       '50-item board grid {0,50,...,30000-L}; blocks concatenated to 29,999 items; labels keep their '
                       'within-block position; arrival = origin + 500; labels arriving after H dropped. '
                       f'B={B_BOOT} bootstrap streams per history and L; threshold = randomized calibrate_threshold at 0.02 '
                       'on bootstrap maxima (const-null only; justification: in E2 the const null bound h* in all 56 '
                       'oracle cells of these variants). PRIMARY estimated threshold = max over L in {50,250,1000} '
                       '(small conservative grid; L=50 = board/cluster bootstrap). Secondary: each L alone.'),
            same_procedure_all_variants='the auditor does not know the dependence type: the identical procedure is '
                                        'applied to iid, Markov and board histories'),
        history_randomness=f'K={K_HIST} independent histories per (variant,q) (role pilot, condition 2000+10*q_index, '
                           'replicate k; CRN across variants); bootstrap RNG SeedSequence(20261111, spawn_key=(8,12,'
                           '3000+100*variant_index+10*q_index, k, L_index))',
        validation=dict(n=N_VAL, role='validation', condition='1000+10*q_index+path_index (paths: const .10=0, '
                        '.05->.10 at tau=1, const .05=2)', replicates='0..N-1 (namespace disjoint from E2)'),
        evaluation=dict(n=N_EVAL, role='evaluation', condition='1000+10*q_index+3', deadlines=[5000, 10000, 20000],
                        endpoint='P(tau<=T<=tau+D-1)'),
        estimands=dict(
            unconditional_FA='FA of the procedure averaged over history randomness. PRIMARY test uses a diagonal '
                             'design: validation stream j is paired with history j mod K, so the N alarm indicators are '
                             'i.i.d. Bernoulli(unconditional FA) -> exact CP.',
            conditional_FA='each history threshold applied to all N validation streams: distribution over histories '
                           '(mean, p50, p90, max, P(FA_k>0.03)); the cross mean is the precise point estimate.',
            pD='mean over histories x streams (procedure p_D), distribution over histories, paired difference vs oracle '
               'and vs iid-nominal on identical streams (CI by normal approx over streams, conditional on histories).'),
        success_criterion=(f'estimated PRIMARY procedure: simultaneous one-sided 95% CP upper bound on the unconditional '
                           f'FA (diagonal design) <= 0.03 in every cell of the family = 4 variants x 2 q x 7 monitors x 2 '
                           f'null paths (const .10, path .05->.10) = {FAMILY} cells, Bonferroni alpha=0.05/{FAMILY}, '
                           'bound = beta.ppf(1-0.05/112, k+1, n-k). const .05 FA reported, not in family. Detection loss '
                           'is reported as the price, not a criterion. Secondary (descriptive): P_hist(conditional FA > '
                           '0.03).'),
        saved_per_stream='npz per (variant,q): uint16 alarm items (65535 = none) for nominal and oracle on all 4 paths; '
                         'for the primary estimated thresholds: harmful-path alarm items (K x N uint16) and null-path '
                         'alarm indicators (packbits, K x N); diagonal assignment j mod K.',
        resources='<=4 threads (NUMBA_NUM_THREADS=4), chunks of 5,000 streams, no GPU',
        not_done='history at r<0.10 (realistic in-control history) is NOT studied: the history is generated at the '
                 'boundary r0, as the oracle calibration is; design-effect inflation not used as a solution.')


# --------------------------------------------------------------------------- bootstrap
def history(variant, qi, k):
    cfg = sim.config(variant, 'null_const010', QS[qi])
    cfg = G.StreamConfig(**{**cfg.__dict__, 'H': H_HIST + D0, 'r': np.full(H_HIST + D0, 0.10)})
    b = G.generate(cfg, 'pilot', EXP, cond_hist(qi), np.array([k]))
    s = b.stream(0)
    keep = s['origin'] <= H_HIST
    return (s['origin'][keep] - 1).astype(np.int64), s['loss'][keep].astype(np.int8)


def boot_batch(idx, loss, L, B, rng, q):
    nb = -(-H // L)
    starts = rng.integers(0, (H_HIST - L) // GRID + 1, size=(B, nb)) * GRID
    lo = np.searchsorted(idx, starts); hi = np.searchsorted(idx, starts + L)
    cnt = (hi - lo).ravel()
    tot = int(cnt.sum())
    blk = np.repeat(np.arange(B * nb), cnt)
    first = np.repeat(np.cumsum(cnt) - cnt, cnt)
    src = np.repeat(lo.ravel(), cnt) + (np.arange(tot) - first)
    pos = (blk % nb) * L
    org = pos + (idx[src] - starts.ravel()[blk]) + 1
    arr = org + D0
    sid = blk // nb
    ok = arr <= H
    org, arr, ls, sid = org[ok], arr[ok], loss[src][ok], sid[ok]
    off = np.zeros(B + 1, np.int64); off[1:] = np.cumsum(np.bincount(sid, minlength=B))
    z = np.zeros(B, np.int64)
    return G.StreamBatch(H=H, offsets=off, origin=org, arrival=arr, loss=ls, true_loss=ls.copy(),
                         prob=np.full(len(ls), q), tiebreak=rng.random(B), n_audits=z, n_defects=z, n_misses=z,
                         n_informative=z, checkpoints=np.zeros(0, np.int64), counts_at=np.zeros((B, 0, 3), np.int64))


def estimate(variant, qi, mons):
    vi = VARS.index(variant)
    rows = []; prim = {m: [] for m in mons}; perL = {(m, L): [] for m in mons for L in BLOCKS}
    for k in range(K_HIST):
        idx, loss = history(variant, qi, k)
        per = {}
        for li, L in enumerate(BLOCKS):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                20261111, spawn_key=(8, EXP, 3000 + 100 * vi + 10 * qi, k, li))))
            b = boot_batch(idx, loss, L, B_BOOT, rng, QS[qi])
            for m, mo in mons.items():
                t = C.calibrate_threshold(mo.max_stat(b), TARGET, 'randomized')
                per[(m, L)] = t; perL[(m, L)].append(t)
                rows.append(dict(variant=variant, q=QS[qi], history=k, n_hist_labels=len(idx),
                                 hist_miss_rate=float(loss.mean()) if len(loss) else np.nan,
                                 monitor=m, L=L, h=t.h, gamma=t.gamma, h_hi=t.h_hi, h_lo=t.h_lo))
        for m in mons:
            ts = [per[(m, L)] for L in BLOCKS]
            best = ts[0]
            for t in ts[1:]:
                best = sim.conservative(best, t)
            prim[m].append(best)
    return rows, prim, perL


# --------------------------------------------------------------------------- evaluation
def eval_batches(variant, qi, path, n):
    cfg = sim.config(variant, path, QS[qi])
    return G.generate_chunks(cfg, 'validation' if path != 'harm' else 'evaluation', EXP, cond_eval(qi, path), n, 5000)


def thr_list(ts):
    """flatten Threshold list -> grid of (h_hi, h_lo) and gammas"""
    return np.array([t.h_hi for t in ts]), np.array([t.h_lo for t in ts]), np.array([t.gamma for t in ts])


def alarms_many(mo, b, ts):
    hi, lo, g = thr_list(ts)
    e = mo.evaluate(b, np.concatenate([hi, lo]), ())['alarm']
    n = len(ts)
    use_lo = b.tiebreak[:, None] < g[None, :]
    return np.where(use_lo, e[:, n:], e[:, :n])        # (N, n)


def to16(a):
    a = np.asarray(a, float)
    out = np.full(a.shape, NONE16, np.uint16)
    f = np.isfinite(a); out[f] = a[f].astype(np.uint16)
    return out


def cp_up(k, n, fam):
    return 1.0 if k >= n else float(beta.ppf(1 - 0.05 / fam, k + 1, n - k))


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = '_smoke' if SMOKE else ''
    if not SMOKE and not os.path.exists(f'{OUT}/protocol.json'):
        json.dump(protocol(), open(f'{OUT}/protocol.json', 'w'), indent=1, default=str)
    if '--protocol-only' in sys.argv:
        return
    allmons = sim.monitors(); mons = {m: allmons[m] for m in MON}
    thr_rows, fa_rows, hist_rows, det_rows = [], [], [], []
    only = os.environ.get('E2B_CELLS')
    for qi, q in enumerate(QS):
        t0 = time.time()
        nom = sim.calibrate('base_iid', q, mons)
        for variant in VARS:
            if only and f'{variant}@{q}' not in only.split(','):
                continue
            t1 = time.time()
            nomv = nom
            orc = nom if variant == 'base_iid' else sim.calibrate(variant, q, mons)
            rows, prim, perL = estimate(variant, qi, mons)
            thr_rows += rows
            print(f'q={q} {variant} thresholds {time.time()-t1:.0f}s', flush=True)
            K = K_HIST; save = {}
            for path in ('null_const010', 'null_path', 'null_const005', 'harm'):
                n = N_EVAL if path == 'harm' else N_VAL
                acc = {m: [] for m in mons}
                for bt in eval_batches(variant, qi, path, n):
                    for m, mo in mons.items():
                        ts = [nomv[m]['thr'], orc[m]['thr']] + prim[m] + sum((perL[(m, L)] for L in BLOCKS), [])
                        if path == 'harm':
                            acc[m].append(alarms_many(mo, bt, ts))
                        else:
                            mx = mo.max_stat(bt)
                            acc[m].append(np.stack([np.where(C.fa_from_maxima(mx, t, bt.tiebreak), 0.0, np.inf)
                                                    for t in ts], axis=1))
                for m in mons:
                    a = np.concatenate(acc[m])            # (n, 2+K+3K); null paths: 0 = alarm, inf = none
                    N = a.shape[0]
                    cols = {'nominal': a[:, [0]], 'oracle': a[:, [1]], 'estimated': a[:, 2:2 + K]}
                    for li, L in enumerate(BLOCKS):
                        cols[f'est_L{L}'] = a[:, 2 + K * (li + 1):2 + K * (li + 2)]
                    est = cols['estimated']
                    if path == 'harm':
                        save[f'{m}|{path}|nominal'] = to16(a[:, 0]); save[f'{m}|{path}|oracle'] = to16(a[:, 1])
                        save[f'{m}|{path}|est'] = to16(est.T)
                        for lab, col in cols.items():
                            for D in sim.DEADLINES:
                                s = (col >= TAU) & (col <= min(H, TAU + D - 1))
                                ph = s.mean(axis=0)
                                row = dict(q=q, variant=variant, monitor=m, threshold=lab, D=D, n=N,
                                           pD=float(s.mean()), early=float((col < TAU).mean()),
                                           pD_hist_p05=float(np.quantile(ph, .05)), pD_hist_p50=float(np.median(ph)),
                                           pD_hist_p95=float(np.quantile(ph, .95)))
                                if lab == 'estimated':
                                    so = ((a[:, 1] >= TAU) & (a[:, 1] <= min(H, TAU + D - 1))).astype(float)
                                    sn = ((a[:, 0] >= TAU) & (a[:, 0] <= min(H, TAU + D - 1))).astype(float)
                                    sj = s.mean(axis=1)
                                    for ref, v in (('oracle', so), ('nominal', sn)):
                                        dj = sj - v
                                        se = dj.std(ddof=1) / np.sqrt(N)
                                        row[f'diff_vs_{ref}'] = float(dj.mean())
                                        row[f'diff_vs_{ref}_lo'] = float(dj.mean() - 1.96 * se)
                                        row[f'diff_vs_{ref}_hi'] = float(dj.mean() + 1.96 * se)
                                det_rows.append(row)
                    else:
                        save[f'{m}|{path}|nominal_bits'] = np.packbits(a[:, 0] <= H)
                        save[f'{m}|{path}|oracle_bits'] = np.packbits(a[:, 1] <= H)
                        save[f'{m}|{path}|est_bits'] = np.packbits((est <= H).T, axis=1)
                        for lab, col in cols.items():
                            fa_mat = col <= H                     # (N, K) or (N, 1)
                            kc = fa_mat.shape[1]
                            x = fa_mat[np.arange(N), np.arange(N) % kc]   # diagonal design (identity for kc=1)
                            kk = int(x.sum())
                            lo, hi = C.clopper_pearson(kk, N, 0.95)
                            fk = fa_mat.mean(axis=0)
                            up = cp_up(kk, N, FAMILY)
                            fa_rows.append(dict(q=q, variant=variant, monitor=m, path=path, threshold=lab, n=N,
                                                alarms_diag=kk, fa_diag=kk / N, cp_lo=lo, cp_hi=hi, upper_sim112=up,
                                                pass_gate=(up <= 0.03) if path != 'null_const005' else None,
                                                K=kc, fa_cross_mean=float(fk.mean()),
                                                fa_hist_p10=float(np.quantile(fk, .1)), fa_hist_p50=float(np.median(fk)),
                                                fa_hist_p90=float(np.quantile(fk, .9)), fa_hist_max=float(fk.max()),
                                                P_hist_fa_gt_003=float((fk > 0.03).mean()),
                                                P_hist_fa_gt_002=float((fk > 0.02).mean())))
            np.savez_compressed(f'{OUT}/alarms_{variant}_q{q}{tag}.npz', **save)
            pd.DataFrame(thr_rows).to_csv(f'{OUT}/thresholds_est{tag}.csv', index=False)
            pd.DataFrame(fa_rows).to_csv(f'{OUT}/fa{tag}.csv', index=False)
            pd.DataFrame(det_rows).to_csv(f'{OUT}/detection{tag}.csv', index=False)
            print(f'q={q} {variant} done {time.time()-t1:.0f}s', flush=True)
    print('total done', flush=True)


if __name__ == '__main__':
    main()
