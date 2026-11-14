import json, datetime, numpy as np
from revision.experiments.E3 import design as Dz
OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E3/protocol.json'
pols = {}
for wf in (0, 1):
    for ci, c in Dz.CONDITIONS.items():
        p, info = Dz.policies(c['pi'], wf)
        pols[f'wf{wf}_cond{ci}'] = dict(strata_info=info, policies={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in p.items()})
proto = dict(
  experiment='E3 adaptive / targeted audit allocation', experiment_id=Dz.EXP, written=str(datetime.datetime.now()),
  status='prespecified before calibration, validation and evaluation (only a 4000-stream pilot with role=pilot for timing/kernel check was run)',
  scope='Outside Theorem 1: targeted policies make audit selection depend on the classifier score, which is informative about the miss indicator; Theorem 1 (oracle bound beta*) assumes noninformative audits. beta* at the uniform label rate is reported as the uniform-audit limit, not as a bound for targeted policies.',
  model=dict(H=Dz.H, tau=Dz.TAU, r0=Dz.R0, r_pre=Dz.RPRE, r1=Dz.R1,
    score='binormal: S = Z*mu1_t + N(0,1); inspector passes iff S < c; miss M = Z*1[S<c]; mu1_t = c - Phi^-1(r_t) so P(M=1|Z=1)=r_t exactly',
    AUC_at_r0=Dz.AUC_REF, mu_ref=float(Dz.MU_REF), c=float(Dz.C_THR), AUC_at_r=dict(r005=0.938, r010=0.90, r020=0.833),
    strata_edges=Dz.EDGES.tolist(), strata_note='7 strata by np.searchsorted(edges, S); strata 0-3 passed (S<c), 4-6 flagged',
    defect_posterior='u(S)=P(Z=1|S) under nominal (pi, r0) model; policies use the fixed nominal design (no adaptation of the score model)',
    delay='constant d (both W-B review labels and audit labels)'),
  conditions={v['label']: dict(cond_id=k, **v) for k, v in Dz.CONDITIONS.items()},
  seeds=dict(rng='revision.rng.stream_states(role, 13, 10*cond+path, replicate)', paths=Dz.PATHS,
             crn='all policies and both workflows share streams within (role, cond, path)'),
  budget=dict(B=Dz.B, qbar=Dz.QBAR, rule='pathwise hard cap B; W-A: forced p=1 when remaining scope items <= remaining scope budget (exact spending); W-B: pacing only (no forcing possible on passed items), audits used reported',
              block=dict(size=Dz.BLK, block_budget='round(B*end/H)-round(B*start/H)'),
              floor=Dz.EPS, floor_rule='paced targeted policies: p=clip(g(S)*kappa, eps, 1), kappa=(Brem/items_rem)/qbar; p=0 only when the (scope) budget is exhausted (counted as floor violations); P1/P2 exact-uniform p=Brem/items_rem; P3 original mechanics (base/boost, p=0 when exhausted, forced at end)'),
  workflows=dict(WA='any item auditable; audited defective -> label X=M', WB='flagged items (S>=c) always reviewed free (caught-defect labels X=0, weight 1, not charged); audits only on passed items; audited passed defective -> miss label X=1, weight 1/p_t'),
  estimand='r_t = P(M_t=1 | Z_t=1) = misses/(misses+caught defects) (both workflows); W-B HT ratio sum(w X)/sum(w) over defect labels',
  monitors=dict(primary='mixture SR (original stakes LAM=[.02,.05,.10,.20,.40]/.09, log mean of 5 SR components)', secondary='Betting (gain 4, cap 9, window 50, Hajek-weighted window mean)',
    increment='D = wt*(X - r0); W-A P1-P3 wt=1 (selection independent of current item given past: plain original monitors); W-A P4-P6 wt=qbar/p_t (IPW, normalized so uniform -> 1); W-B wt=1 caught, 1/p_t miss',
    clipping='no weight clipping beyond the floor eps (wt <= qbar/eps = 10 in W-A, 1/eps = 50 in W-B); stakes capped at capf/(r0*b_t), capf=0.99 (mSR), 0.9 (Betting: cap 9 * r0), b_t = F_{t-1}-measurable bound on the weight of any possible negative-increment label of item t (W-A IPW: qbar/clip(kappa_t*min_{s>=c} g(s), eps, 1); W-A P1-P3 and W-B: 1)',
    validity='item-level increments have E[D_t|F_{t-1}] = const*pi*(r_t - r0) <= 0 under every admissible path, stakes are predictable (constant delay => arrival order = origin order), every factor >= 0.01: Betting log-wealth is a nonnegative supermartingale (Ville: native threshold log(1/0.02) bounds FA over any horizon by 0.02); mixture SR is an e-SR (ARL guarantee). Thresholds actually used are empirically calibrated (two-point null). Native Betting FA reported as a theory check.'),
  policies_WA=dict(P1='horizon-uniform', P2='block-uniform (3000)', P3='warning-boost (original): desired q=base (0.05/0.20), 0.5 while mSR stat >= threshold/4; horizon/block budgets (4 variants)',
    P4='uncertainty: g = a*u(1-u)', P5='score-stratified: g_h proportional to P(Z=1|stratum h)', P6='Neyman: g_h proportional to sd of Z(M-r0) in stratum h, unit cost (cost-sensitive with equal costs = Neyman; no audit-cost data available). Deviation from brief: sd of the HT target Z(M-r0) used instead of sd of the label indicator Z, because Z(M-r0) is the quantity whose HT sum the monitor uses.'),
  policies_WB=dict(B1='uniform over passed items (paced)', B2='block-uniform over passed', B3='block warning-boost over passed (base qbar/P(pass), boost 2.5x base)', B4='uncertainty u(1-u) on passed', B5='score-stratified P(Z|h) on passed strata', B6='Neyman sqrt(p_h(1-p_h)) on passed strata'),
  design_parameters=pols,
  calibration=dict(n=Dz.N_CAL, role='calibration', target=Dz.TARGET_FA, method='randomized', rule='h* = max(h_const(r=0.10), h_path(0.05->0.10 at tau)); each (workflow, condition, policy, monitor) separately',
    feedback='warning policies calibrated as complete procedures: fixed-point iteration h_{k+1}=calibrated threshold with warning at h_k/4 (start: block-uniform threshold), stop when |h_{k+1}-h_k|<0.02 or 8 iterations; final procedure = warning at h_k/4, alarm threshold h_{k+1} (calibrated under that warning). Betting under warning policies: warning from mSR (fixed), Betting threshold calibrated on the same streams.'),
  validation=dict(n=Dz.N_VAL, role='validation', paths=['const 0.10', '0.05->0.10', 'const 0.05 (reported)'], gate='one-sided 95% CP upper bound <= 0.03, Bonferroni family = all (workflow x condition x policy) x 2 null paths for the given monitor = 90 (mSR family; Betting family separately 90)'),
  evaluation=dict(n=Dz.N_EVAL, role='evaluation', endpoint='p_D = P(tau <= T <= tau+D-1), D in (5000,10000,20000), primary D=10000'),
  primary_contrast=dict(what='W-A P5 score-stratified (IPW mSR) minus P1 horizon-uniform (mSR), pi=0.10, d=500, D=10000', ci='paired percentile bootstrap, 10000 resamples, 95%', decision='P5 better iff lower bound > 0; worse iff upper bound < 0; otherwise no evidence of difference'),
  secondary='all other contrasts vs P1 (W-A) or B1 (W-B), both monitors, all conditions: descriptive, unadjusted bootstrap CIs',
  diagnostics=['FA both null paths + const 0.05', 'informative labels per audit (yield)', 'Kish ESS of window labels (sum w)^2/sum w^2', 'audits within [tau, tau+D-1]', 'forced audits', 'floor violations', 'warning occupancy pre/post', 'beta*(alpha=0.02) at uniform label rate qbar*pi and full-information rate pi (theory/converse.np_power, L=D-d)', 'timing vs yield decomposition for warning-boost: window audit count, yield, and beta* at realized window label count (descriptive)'],
  resources='NUMBA_NUM_THREADS=4, chunks of 5000 streams, CPU only')
json.dump(proto, open(OUT, 'w'), indent=1, default=float)
print('written', OUT)
