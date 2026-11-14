"""E3 follow-up (2026-11-13): no-leakage score model, AUC and tau sensitivity, diagnostics, P5 weight clip.
python -m revision.experiments.E3.followup [run|diag|summ] [--smoke]
Uses kernel2 (kernel.py v1 untouched).  Fresh replicate ids: START = 200000 (main run: 100000..120000, smoke 0..1999).
"""
import os, sys, json, time, pickle
import numpy as np, pandas as pd
from scipy.stats import norm
from revision import rng as R, calibration as C
from . import design as Dz
from .kernel2 import run_batch, labels_one, ST

OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E3/followup'
SMOKE = '--smoke' in sys.argv
NC, NV, NE = (2000, 1000, 1000) if SMOKE else (Dz.N_CAL, Dz.N_VAL, Dz.N_EVAL)
START = 900000 if SMOKE else 200000
BSL = 1.0
INF = np.inf
IDX = {s: i for i, s in enumerate(ST)}


class Scen:
    def __init__(self, name, auc=0.90, model=0, tau=Dz.TAU, cond=0, wclip=INF):
        self.name, self.auc, self.model, self.tau, self.cond, self.wclip = name, auc, model, tau, cond, wclip
        self.mu = np.sqrt(2) * norm.ppf(auc)
        self.c = self.mu + norm.ppf(Dz.R0)
        self.edges = self.c + np.array([-2.0, -1.0, -0.5, 0.0, 1.0, 2.0])
        self.a0 = np.sqrt(1 + BSL ** 2) * norm.ppf(norm.cdf(self.c))   # P(pass | Z=0) as in model 0
        self.pi = Dz.CONDITIONS[cond]['pi']; self.d = Dz.CONDITIONS[cond]['d']

    def a1(self, r):
        return self.mu + np.sqrt(1 + BSL ** 2) * norm.ppf(r)

    def risk(self, path):
        t = np.arange(1, Dz.H + 1)
        if path == 0: return np.full(Dz.H, Dz.R0)
        if path == 1: return np.where(t >= self.tau, Dz.R0, Dz.RPRE)
        if path == 2: return np.full(Dz.H, Dz.RPRE)
        return np.where(t >= self.tau, Dz.R1, Dz.RPRE)

    def hfun(self, s, r):
        if self.model == 0:
            h = (s < self.c).astype(float); return h, h
        return norm.cdf(self.a0 - BSL * s), norm.cdf(self.a1(r) - BSL * s)


def _solve(v, mass, target):
    lo, hi = 0.0, 1e7
    for _ in range(200):
        a = 0.5 * (lo + hi)
        m = np.sum(mass * np.clip(a * v, Dz.EPS, 1.0) * (v >= 0))
        lo, hi = (lo, a) if m > target else (a, hi)
    return 0.5 * (lo + hi)


def policies(sc, wf, names):
    pi = sc.pi
    s = np.linspace(-10, 14, 240001); ds = s[1] - s[0]
    f0 = norm.pdf(s) * ds * (1 - pi); f1 = norm.pdf(s - sc.mu) * ds * pi
    h0, h1 = sc.hfun(s, Dz.R0)
    k = np.searchsorted(sc.edges, s)
    NS = len(sc.edges) + 1
    agg = lambda x: np.bincount(k, weights=x, minlength=NS)
    if wf == 0:
        mass = agg(f0 + f1); pz = agg(f1) / mass
        EY = agg(f1 * (h1 * (1 - Dz.R0) - (1 - h1) * Dz.R0)) / mass
        EY2 = agg(f1 * (h1 * (1 - Dz.R0) ** 2 + (1 - h1) * Dz.R0 ** 2)) / mass
        v6 = np.sqrt(np.maximum(EY2 - EY ** 2, 0))
        fu = f0 + f1; u = f1 / (f0 + f1); elig = np.ones(NS, bool)
        neg = np.arange(NS) >= 4 if sc.model == 0 else np.ones(NS, bool)
    else:
        mp = agg(f0 * h0 + f1 * h1); elig = mp > 1e-12
        mass = mp; pz = np.where(elig, agg(f1 * h1) / np.maximum(mp, 1e-300), 0)
        v6 = np.sqrt(pz * (1 - pz))
        fu = f0 * h0 + f1 * h1; u = f1 * h1 / np.maximum(fu, 1e-300)
        neg = np.zeros(NS, bool)
    p_pass = float(np.sum(f0 * h0 + f1 * h1))
    a_unc = _solve(u * (1 - u), fu, Dz.QBAR)

    def strat(v):
        v = np.where(elig, v, -1.0); a = _solve(v, mass, Dz.QBAR)
        return np.where(elig, a * v, 0.0)
    base = dict(ptype=0, scope_block=False, base=0.0, boost=0.0, warn=False, warn_mode='linear', gtype=0,
                G=np.zeros(NS), a_unc=0.0, gmin_flag=1.0, ipw=0)
    out = {}
    def add(n, **kw):
        if n in names:
            d = dict(base); d.update(kw); out[n] = d
    if wf == 0:
        add('P1_horizon_uniform')
        for scb in (False, True):
            for b in (0.05, 0.20):
                tag = "block" if scb else "horizon"
                add(f'P3_{tag}_warn_q{int(b*100):02d}', ptype=1, scope_block=scb, base=b, boost=Dz.BOOST, warn=True)
                add(f'P3x_{tag}_logwarn_q{int(b*100):02d}', ptype=1, scope_block=scb, base=b, boost=Dz.BOOST, warn=True,
                    warn_mode='log')
        add('P4_uncertainty', ptype=2, gtype=2, a_unc=a_unc, gmin_flag=0.0, ipw=1)
        G5 = strat(pz); add('P5_score_stratified', ptype=2, gtype=1, G=G5, gmin_flag=float(G5[neg].min()), ipw=1)
        G6 = strat(v6); add('P6_neyman', ptype=2, gtype=1, G=G6, gmin_flag=float(G6[neg].min()), ipw=1)
    else:
        rho = Dz.QBAR / p_pass
        add('B1_passed_uniform', ptype=2, gtype=0, G=np.full(NS, rho), ipw=1)
        add('B3_passed_block_warn', ptype=1, scope_block=True, base=rho, boost=min(1.0, rho * Dz.BOOST / 0.2), warn=True, ipw=1)
        add('B4_passed_uncertainty', ptype=2, gtype=2, a_unc=a_unc, ipw=1)
        add('B5_passed_score_stratified', ptype=2, gtype=1, G=strat(pz), ipw=1)
        add('B6_passed_neyman', ptype=2, gtype=1, G=strat(v6), ipw=1)
    return out, dict(p_pass=p_pass, pz=pz.tolist(), mass=mass.tolist())


def kargs(sc, pol, path, wf, th=(INF,) * 4, h_warn=INF):
    r = sc.risk(path)
    mu1 = (sc.c - norm.ppf(r)) if sc.model == 0 else np.full(Dz.H, sc.mu)
    a1 = sc.a1(r) if sc.model == 1 else np.zeros(Dz.H)
    return (Dz.H, sc.tau, sc.pi, mu1, sc.c, sc.d, wf, pol['ptype'], pol['scope_block'], Dz.BLK, Dz.B, Dz.QBAR, Dz.EPS,
            pol['base'], pol['boost'], (h_warn if pol['warn'] else INF), pol['gtype'], sc.edges, pol['G'], pol['a_unc'],
            pol['gmin_flag'], pol['ipw'], sc.mu, Dz.R0, Dz.LAM, Dz.GAIN, Dz.BCAP, Dz.WIN, Dz.CAPF, np.asarray(th, float),
            Dz.D_PRIMARY, sc.model, sc.a0, a1, float(sc.a1(Dz.R0)), BSL, sc.wclip)


def states(sc, role, path, n, s0):
    return R.stream_states(role, Dz.EXP, 10 * sc.cond + path, np.arange(s0, s0 + n))


def sim(sc, pol, path, wf, role, n, th=(INF,) * 4, h_warn=INF, start=START, chunk=5000):
    args = kargs(sc, pol, path, wf, th, h_warn); O, S, T = [], [], []
    for s0 in range(0, n, chunk):
        st = states(sc, role, path, min(chunk, n - s0), start + s0)
        o, s = run_batch(st, *args); O.append(o); S.append(s); T.append(R.uniforms_for_tiebreak(st))
    return np.concatenate(O), np.concatenate(S), np.concatenate(T)


def alarm(out, tb, mon, thr):
    j = 2 if mon == 'msr' else 4
    return np.where(tb < thr.gamma, out[:, j + 1], out[:, j])


def cal_pair(sc, pol, wf, h_warn):
    mx = {p: sim(sc, pol, p, wf, 'calibration', NC, h_warn=h_warn)[0] for p in (0, 1)}
    thr = {}
    for mon, col in (('msr', 0), ('bet', 1)):
        t0 = C.calibrate_threshold(mx[0][:, col], Dz.TARGET_FA); t1 = C.calibrate_threshold(mx[1][:, col], Dz.TARGET_FA)
        thr[mon] = (t0 if t0.h >= t1.h else t1, t0.h, t1.h)
    return thr


def full(sc, wf, name, pol, raw, P, ref_h=None):
    fn = os.path.join(raw, f'{sc.name}_wf{wf}_{name}.pkl')
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    t0 = time.time(); iters = []
    if pol['warn']:
        h = ref_h
        for it in range(8):
            thr = cal_pair(sc, pol, wf, Dz.warn_level(h, pol['warn_mode'])); hn = thr['msr'][0].h
            iters.append((h, hn))
            if abs(hn - h) < 0.02: break
            h = hn
        h_warn = Dz.warn_level(h, pol['warn_mode'])
    else:
        thr = cal_pair(sc, pol, wf, INF); h_warn = INF
    tm, tb = thr['msr'][0], thr['bet'][0]; th = (tm.h_hi, tm.h_lo, tb.h_hi, tb.h_lo)
    val = {}
    for path in (0, 1, 2):
        o, s, t = sim(sc, pol, path, wf, 'validation', NV, th, h_warn)
        val[path] = dict(fa_msr=alarm(o, t, 'msr', tm) <= Dz.H, fa_bet=alarm(o, t, 'bet', tb) <= Dz.H,
                         native_bet=o[:, 1] > np.log(50), stats_mean=s.mean(0))
    o, s, t = sim(sc, pol, 3, wf, 'evaluation', NE, th, h_warn)
    res = dict(scen=sc.name, wf=wf, policy=name, thr=thr, h_warn=h_warn, iters=iters, val=val,
               ev=dict(alarm_msr=alarm(o, t, 'msr', tm), alarm_bet=alarm(o, t, 'bet', tb), stats=s))
    pickle.dump(res, open(fn, 'wb'))
    a = res['ev']['alarm_msr']; pdk = np.mean((a >= sc.tau) & (a <= min(Dz.H, sc.tau + 9999)))
    P(f'{sc.name} wf{wf} {name}: h={tm.h:.3f} FA={val[0]["fa_msr"].mean():.4f}/{val[1]["fa_msr"].mean():.4f} pD10k={pdk:.4f} {time.time()-t0:.0f}s')
    return res


WA4 = ['P1_horizon_uniform', 'P4_uncertainty', 'P5_score_stratified', 'P6_neyman']
WB4 = ['B1_passed_uniform', 'B4_passed_uncertainty', 'B5_passed_score_stratified', 'B6_passed_neyman']
WARN = ['P1_horizon_uniform'] + [f'P3{x}_{t}_{w}_q{q}' for x, w in (('', 'warn'), ('x', 'logwarn'))
                                  for t in ('horizon', 'block') for q in ('05', '20')]
SCENS = [  # (scenario, W-A policies, W-B policies)
    (Scen('noleak_pi0.10_d500', model=1, cond=0), WA4, WB4),
    (Scen('noleak_pi0.10_d0', model=1, cond=1), WA4, WB4),
    (Scen('auc0.80_pi0.10_d500', auc=0.80), WA4, WB4),
    (Scen('auc0.95_pi0.10_d500', auc=0.95), WA4, WB4),
    (Scen('tau5000_pi0.10_d500', tau=5000), WARN, ['B1_passed_uniform', 'B3_passed_block_warn']),
    (Scen('tau20000_pi0.10_d500', tau=20000), WARN, ['B1_passed_uniform', 'B3_passed_block_warn']),
    (Scen('P5clip2_pi0.10_d500', wclip=2.0), ['P5_score_stratified'], []),
    (Scen('P5clip2_noleak_pi0.10_d500', model=1, wclip=2.0), ['P5_score_stratified'], []),
]


def run():
    raw = os.path.join(OUT, 'raw_smoke' if SMOKE else 'raw'); os.makedirs(raw, exist_ok=True)
    log = open(os.path.join(OUT, 'run_smoke.log' if SMOKE else 'run.log'), 'a')
    def P(*a):
        s = ' '.join(map(str, a)); print(s, flush=True); log.write(s + '\n'); log.flush()
    for sc, wa, wb in SCENS:
        for wf, names in ((0, wa), (1, wb)):
            if not names: continue
            pols, _ = policies(sc, wf, names)
            ref_h = None
            for n in names:
                r = full(sc, wf, n, pols[n], raw, P, ref_h)
                if n in ('P1_horizon_uniform', 'B1_passed_uniform'):
                    ref_h = r['thr']['msr'][0].h
    P('done')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'run'
    if cmd == 'run':
        run()


# ------------------------------------------------------------------ diagnostics + summary
def boot3(x, y, seed):
    d = x.astype(int) - y.astype(int); n = len(d)
    p = np.array([(d == -1).mean(), (d == 0).mean(), (d == 1).mean()])
    cnt = np.random.default_rng(seed).multinomial(n, p, size=10000)
    bs = (cnt[:, 2] - cnt[:, 0]) / n
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def diag():
    """Main-run primary condition: same eval streams + thresholds through kernel2; extra diagnostics."""
    import glob
    MAIN = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E3/raw'
    sc = Scen('main_pi0.10_d500'); rows = []
    for fn in sorted(glob.glob(MAIN + '/c0_wf*.pkl')):
        r = pickle.load(open(fn, 'rb')); wf = r['wf']; name = r['policy']
        pol = Dz.policies(0.10, wf)[0][name]
        tm, tb = r['thr']['msr'][0], r['thr']['bet'][0]; th = (tm.h_hi, tm.h_lo, tb.h_hi, tb.h_lo)
        o, s, t = sim(sc, pol, 3, wf, 'evaluation', Dz.N_EVAL, th, r['h_warn'], start=100000)
        same = bool(np.array_equal(alarm(o, t, 'msr', tm), r['ev']['alarm_msr']) and np.array_equal(alarm(o, t, 'bet', tb), r['ev']['alarm_bet']))
        args = kargs(sc, pol, 3, wf, th, r['h_warn']); st = states(sc, 'evaluation', 3, 1000, 100000)
        W = []
        for i in range(1000):
            L, _, _ = labels_one(st[i], *args)
            W.append(L[2][L[1] == 1] if wf == 1 else L[2])   # W-B: paid-audit (miss) labels; W-A: all labels
        W = np.concatenate(W) if W else np.zeros(1)
        g = lambda k: s[:, IDX[k]]
        fe = g('first_exhaust'); ex = fe <= Dz.H
        a = alarm(o, t, 'msr', tm)
        rows.append(dict(workflow='W-A' if wf == 0 else 'W-B', policy=name, reproduces_main_alarms=same,
                         pD10k_msr=float(np.mean((a >= Dz.TAU) & (a <= Dz.TAU + 9999))),
                         audits_before_tau=g('aud_pre').mean(), audits_after_tau=g('aud_post').mean(),
                         paid_audits=g('n_audits').mean(), flagged_reviews=(g('n_flagged').mean() if wf == 1 else 0.0),
                         total_labels=g('n_labels').mean(), audit_labels=g('n_audit_inf').mean(),
                         window_miss_labels=g('miss_lab_win').mean(),
                         ess_window=float(np.mean(np.where(g('sw2_win') > 0, g('sw_win') ** 2 / np.maximum(g('sw2_win'), 1e-300), 0))),
                         w_median=float(np.median(W)), w_p99=float(np.quantile(W, .99)), w_max=float(W.max()),
                         frac_streams_exhausted=float(ex.mean()),
                         exhaust_t_q10=float(np.quantile(fe[ex], .1)) if ex.any() else np.nan,
                         exhaust_t_median=float(np.median(fe[ex])) if ex.any() else np.nan,
                         exhaust_t_q90=float(np.quantile(fe[ex], .9)) if ex.any() else np.nan,
                         floor_violations=g('n_floor_viol').mean(), forced=g('n_forced').mean()))
        print(name, same, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'diagnostics_main_primary.csv'), index=False); print(df.round(3).to_string())


def summ():
    import glob
    raw = os.path.join(OUT, 'raw'); R_ = {}
    for fn in sorted(glob.glob(raw + '/*.pkl')):
        r = pickle.load(open(fn, 'rb')); R_[(r['scen'], r['wf'], r['policy'])] = r
    fam = 2 * len(R_); sc_by = {s.name: s for s, _, _ in SCENS}
    V, E, K = [], [], []
    for (sn, wf, n), r in R_.items():
        sc = sc_by[sn]
        for path, pn in ((0, 'const_0.10'), (1, 'path_0.05to0.10'), (2, 'const_0.05')):
            v = r['val'][path]; row = dict(scenario=sn, workflow='W-A' if wf == 0 else 'W-B', policy=n, null=pn)
            for mon in ('msr', 'bet'):
                vv = C.validate(v['fa_' + mon], cap=0.03, family=fam)
                row[f'FA_{mon}'] = vv['rate']; row[f'FA_{mon}_upper_bonf'] = vv['upper_simultaneous']
            row['FA_bet_native_log50'] = float(v['native_bet'].mean())
            sm = v['stats_mean']; row['HT_rhat'] = sm[IDX['sum_wx']] / sm[IDX['sum_w']]
            V.append(row)
        st = r['ev']['stats']; row = dict(scenario=sn, workflow='W-A' if wf == 0 else 'W-B', policy=n,
                                          h_msr=r['thr']['msr'][0].h, h_warn=r['h_warn'])
        for mon in ('msr', 'bet'):
            a = r['ev']['alarm_' + mon]; row[f'early_{mon}'] = float(np.mean(a < sc.tau))
            for D in Dz.DEADLINES:
                k = (a >= sc.tau) & (a <= min(Dz.H, sc.tau + D - 1)); lo, hi = C.clopper_pearson(int(k.sum()), len(a))
                row[f'pD{D//1000}k_{mon}'] = k.mean(); row[f'pD{D//1000}k_{mon}_lo'] = lo; row[f'pD{D//1000}k_{mon}_hi'] = hi
        g = lambda k: st[:, IDX[k]]
        row.update(audits_before_tau=g('aud_pre').mean(), audits_after_tau=g('aud_post').mean(),
                   window_audits_arrived=g('aud_win_arrived').mean(), window_audit_labels=g('inf_win_arr').mean(),
                   window_miss_labels=g('miss_lab_win').mean(), flagged_reviews=g('n_flagged').mean() if wf == 1 else 0.0,
                   yield_overall=float(g('n_audit_inf').sum() / g('n_audits').sum()),
                   ess_window=float(np.mean(np.where(g('sw2_win') > 0, g('sw_win') ** 2 / np.maximum(g('sw2_win'), 1e-300), 0))),
                   max_w_mean=g('max_w').mean(), forced=g('n_forced').mean(), floor_violations=g('n_floor_viol').mean(),
                   frac_exhausted=float((g('first_exhaust') <= Dz.H).mean()),
                   exhaust_t_median=float(np.median(g('first_exhaust')[g('first_exhaust') <= Dz.H])) if (g('first_exhaust') <= Dz.H).any() else np.nan,
                   warn_occ_pre=g('warn_pre').mean() / (sc.tau - 1), warn_occ_window=g('warn_win').mean() / Dz.D_PRIMARY)
        E.append(row)
        ref = 'P1_horizon_uniform' if wf == 0 else 'B1_passed_uniform'
        if n != ref and (sn, wf, ref) in R_:
            rr = R_[(sn, wf, ref)]
            for mon in ('msr', 'bet'):
                for D in Dz.DEADLINES:
                    f = lambda a: (a >= sc.tau) & (a <= min(Dz.H, sc.tau + D - 1))
                    dm, lo, hi = boot3(f(r['ev']['alarm_' + mon]), f(rr['ev']['alarm_' + mon]), 77 + D)
                    K.append(dict(scenario=sn, workflow='W-A' if wf == 0 else 'W-B', policy=n, reference=ref, monitor=mon, D=D,
                                  diff_pD=dm, boot_lo=lo, boot_hi=hi))
    # clipped P5 vs P1 of the matching (unclipped) scenario
    for sn, refsn in (('P5clip2_pi0.10_d500', None), ('P5clip2_noleak_pi0.10_d500', 'noleak_pi0.10_d500')):
        if (sn, 0, 'P5_score_stratified') not in R_ or refsn is None or (refsn, 0, 'P1_horizon_uniform') not in R_:
            continue
        r, rr = R_[(sn, 0, 'P5_score_stratified')], R_[(refsn, 0, 'P1_horizon_uniform')]
        f = lambda a: (a >= Dz.TAU) & (a <= Dz.TAU + 9999)
        dm, lo, hi = boot3(f(r['ev']['alarm_msr']), f(rr['ev']['alarm_msr']), 99)
        K.append(dict(scenario=sn, workflow='W-A', policy='P5_score_stratified(clip2)', reference=refsn + ':P1', monitor='msr', D=10000,
                      diff_pD=dm, boot_lo=lo, boot_hi=hi))
    v = pd.DataFrame(V); e = pd.DataFrame(E); k = pd.DataFrame(K)
    v.to_csv(os.path.join(OUT, 'validation.csv'), index=False); e.to_csv(os.path.join(OUT, 'evaluation.csv'), index=False)
    k.to_csv(os.path.join(OUT, 'contrasts.csv'), index=False)
    vv = v[v.null != 'const_0.05']
    gate = dict(family=fam, msr_max_upper=float(vv.FA_msr_upper_bonf.max()), bet_max_upper=float(vv.FA_bet_upper_bonf.max()),
                msr_all_pass=bool((vv.FA_msr_upper_bonf <= 0.03).all()), bet_all_pass=bool((vv.FA_bet_upper_bonf <= 0.03).all()),
                native_bet_max=float(v.FA_bet_native_log50.max()))
    json.dump(gate, open(os.path.join(OUT, 'gate.json'), 'w'), indent=1); print(gate)


if __name__ == '__main__' and len(sys.argv) > 1:
    if sys.argv[1] == 'diag': diag()
    if sys.argv[1] == 'summ': summ()
