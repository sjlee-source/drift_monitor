"""Stream generator: arrival-ordered informative audit labels for the drift-monitoring study.

Model (nominal defaults reproduce the original E2 model of simulation/run.py)
---------------------------------------------------------------------------
Production items t = 1..H.  For each item

* Z_t ~ Bernoulli(pi_t)                       defect indicator (prevalence path pi_t)
* M_t | Z_t = 1 ~ Bernoulli(r_t)              conditional miss (risk path r_t)
* A_t ~ Bernoulli(p_t)                        audit, p_t from the audit policy
* an audited defective item (A_t Z_t = 1) is *informative*: its observed loss
  Y_t = M_t XOR F_t (label noise, P(F=1|M=0)=e01, P(F=1|M=1)=e10) arrives at
  production item  a_t = t + D_t  (delay D_t).  Labels with a_t > H are not observed.

Labels are returned sorted by arrival item (ties: origin order); monitors must consume them
in that order.  The selection probability p_t of every label is recorded, so inverse
probability weights 1/p_t can be formed for informative (score/evidence-driven) policies.

Optional departures (each defaults to the nominal model):

* prevalence path: constant / step / sinusoid / user array  (``prevalence_path``)
* risk path: abrupt at tau / linear ramp / user array         (``risk_path``)
* Markov-persistent misses: M follows a two-state chain over successive *defective* items
  (``markov_over='defect'``, default) or over all items (``'item'``) with
  P(M_k=1 | M_{k-1}) = clip(r_k + rho (M_{k-1} - r_{k-1}), 0, 1).  For constant r this is
  the stationary two-state chain with marginal r and lag-1 autocorrelation rho; for
  time-varying r it keeps E[M_k] = r_k exactly whenever no clipping occurs (requires
  r_k + rho(1-r_{k-1}) <= 1 and r_k >= rho r_{k-1}).
* board clustering: items in consecutive boards of size m.  Board b draws
  B_r ~ Beta(mean r_{s_b}, precision phi_r = 1/icc_r - 1) (s_b = first item of board) and
  uses r_t * B_r / r_{s_b} for its items (exactly B_r unless r changes inside the board);
  this gives intra-board correlation icc_r of the misses of two defective items.  The same
  construction with icc_pi clusters defects.
* delay: constant d; geometric on {0,1,..} with mean; negative binomial = sum of k
  geometrics (mean, k); lognormal (mean, sigma of log; rounded to integer); batch release
  at the end of each shift of length s (after a base delay d0): a_t = ceil((t+d0)/s)*s.
* label noise e01, e10.
* audit policy: fixed probability path q_t (noninformative); score-stratified probabilities
  (numba fast path); or a Python ``AuditPolicy`` hook that sees only information available
  before item t0 (labels with arrival <= t0-1, past audit decisions, remaining budget and,
  optionally, scores of the items being decided).  Optional total audit budget (audits are
  refused once exhausted; optionally forced so that the budget is spent exactly).
* classifier scores: S_t ~ Normal(mu, sigma) with mu = mu0 if Z=0, mu_det if Z=1,M=0,
  mu_miss if Z=1,M=1 (binormal model).

Random numbers: see ``rng.py``.  The fast path (fixed/stratified policy) and the hook path
consume identical latent variables, so a hook returning constant q reproduces the fast
path exactly (tested).
"""
from dataclasses import dataclass, field
import numpy as np
from numba import njit, prange

from . import rng as R

NO_LABEL = -1

# ------------------------------------------------------------------ paths and specs


def risk_path(H, r_pre=0.05, r_post=None, tau=None, ramp=0, values=None):
    """Return r_t for t=1..H (array index t-1).

    abrupt: r_pre before tau, r_post from tau on (item tau is post-change, as in run.py);
    ramp>0: linear from r_pre at tau-1 to r_post at tau+ramp-1, constant afterwards;
    values: explicit array of length H (overrides).
    """
    if values is not None:
        v = np.asarray(values, float)
        assert v.shape == (H,)
        return v
    r = np.full(H, float(r_pre))
    if r_post is None or tau is None:
        return r
    t = np.arange(1, H + 1)
    if ramp <= 0:
        r[t >= tau] = r_post
    else:
        frac = np.clip((t - tau + 1) / ramp, 0, 1)
        r = np.where(t >= tau, r_pre + (r_post - r_pre) * frac, r_pre)
    return r


def prevalence_path(H, kind='constant', pi=0.1, pi_post=None, tau=None, amplitude=0.0,
                    period=10000, phase=0.0, drift_per_item=0.0, values=None):
    """pi_t for t=1..H.  kind: constant | step | sinusoid | drift | array."""
    t = np.arange(1, H + 1)
    if values is not None or kind == 'array':
        v = np.asarray(values, float)
        assert v.shape == (H,)
        return v
    if kind == 'constant':
        v = np.full(H, float(pi))
    elif kind == 'step':
        v = np.where(t >= tau, pi_post, pi).astype(float)
    elif kind == 'sinusoid':
        v = pi * (1 + amplitude * np.sin(2 * np.pi * t / period + phase))
    elif kind == 'drift':
        v = pi + drift_per_item * (t - 1)
    else:
        raise ValueError(kind)
    return np.clip(v, 0, 1)


@dataclass(frozen=True)
class Delay:
    """Label delay distribution.  Use the factory functions below."""
    kind: int = 0   # 0 constant, 1 geometric, 2 negbin, 3 lognormal, 4 batch
    a: float = 0.0
    b: float = 0.0

    @staticmethod
    def constant(d):
        return Delay(0, float(d), 0.0)

    @staticmethod
    def geometric(mean):
        return Delay(1, float(mean), 0.0)

    @staticmethod
    def negbin(mean, k):
        """Sum of k iid geometric(mean/k) on {0,1,..}: NB with size k, mean `mean`."""
        return Delay(2, float(mean), float(int(k)))

    @staticmethod
    def lognormal(mean, sigma):
        """round(exp(N(log(mean)-sigma^2/2, sigma^2))), so E[unrounded] = mean."""
        return Delay(3, float(mean), float(sigma))

    @staticmethod
    def batch(shift, base=0):
        """Released at the end of the shift containing t+base (shift length s)."""
        return Delay(4, float(int(shift)), float(int(base)))


@dataclass(frozen=True)
class ScoreModel:
    """Binormal classifier score: mean mu0 (Z=0), mu_det (Z=1,M=0), mu_miss (Z=1,M=1)."""
    mu0: float = 0.0
    mu_det: float = 3.0
    mu_miss: float = 1.5
    sigma: float = 1.0


@dataclass(frozen=True)
class StratifiedPolicy:
    """Numba fast path: p_t = probs[k] where k = #{edges < score_t} (np.searchsorted, side='left')."""
    edges: tuple
    probs: tuple


@dataclass
class StreamConfig:
    H: int
    pi: object = 0.1          # float or array (len H)
    r: object = 0.10          # float or array (len H)
    q: object = 0.2           # float or array (len H): fixed audit probability path
    policy: object = None     # None (fixed q) | StratifiedPolicy | AuditPolicy | callable
    budget: int = -1          # total audits allowed (-1 = unlimited)
    force_budget: bool = False  # force p=1 when remaining items == remaining budget
    delay: Delay = field(default_factory=lambda: Delay.constant(0))
    markov_rho: float = 0.0
    markov_over: str = 'defect'
    board_size: int = 1
    icc_r: float = 0.0
    icc_pi: float = 0.0
    e01: float = 0.0
    e10: float = 0.0
    score: object = None      # ScoreModel or None
    checkpoints: tuple = ()   # production items at which audit/defect/miss counts are recorded
    store_miss_items: bool = False

    def arrays(self):
        H = int(self.H)
        def arr(x):
            a = np.asarray(x, float)
            return np.full(H, float(a)) if a.ndim == 0 else a.astype(float)
        pi, r, q = arr(self.pi), arr(self.r), arr(self.q)
        assert pi.shape == r.shape == q.shape == (H,)
        return pi, r, q


def nominal_config(pi=0.1, q=0.2, d=500, H=29999, tau=10000, r_pre=0.05, r_post=0.20, **kw):
    """Original E2 model: constant pi, q, delay d; abrupt change r_pre -> r_post at tau."""
    return StreamConfig(H=H, pi=pi, r=risk_path(H, r_pre, r_post, tau), q=q, delay=Delay.constant(d), **kw)


def null_config(pi=0.1, q=0.2, d=500, H=29999, r=0.10, **kw):
    """Calibration/validation null: constant risk r (boundary r0=0.10 by default)."""
    return StreamConfig(H=H, pi=pi, r=float(r), q=q, delay=Delay.constant(d), **kw)


# ------------------------------------------------------------------ batch container


@dataclass
class StreamBatch:
    """Concatenated (CSR) label streams.  Stream i occupies [offsets[i], offsets[i+1])."""
    H: int
    offsets: np.ndarray        # int64 (N+1)
    origin: np.ndarray         # int64 production item of each label
    arrival: np.ndarray        # int64 arrival item (non-decreasing within a stream)
    loss: np.ndarray           # int8 observed loss (after label noise)
    true_loss: np.ndarray      # int8 true miss indicator
    prob: np.ndarray           # float64 selection probability of the label's audit
    tiebreak: np.ndarray       # float64 (N,) uniform for randomized threshold rules
    n_audits: np.ndarray       # int64 (N,) audits over 1..H
    n_defects: np.ndarray      # int64 (N,)
    n_misses: np.ndarray       # int64 (N,) production misses Z*M over 1..H
    n_informative: np.ndarray  # int64 (N,) audited defectives incl. labels arriving after H
    checkpoints: np.ndarray    # int64 (C,)
    counts_at: np.ndarray      # int64 (N, C, 3): audits, defects, misses with item <= checkpoint
    miss_offsets: object = None
    miss_items: object = None
    states: object = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.offsets) - 1

    def n_labels(self):
        return np.diff(self.offsets)

    def stream(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return dict(origin=self.origin[a:b], arrival=self.arrival[a:b], loss=self.loss[a:b],
                    true_loss=self.true_loss[a:b], prob=self.prob[a:b])

    def weights(self, normalize=None):
        """IPW weights 1/prob (optionally multiplied by `normalize`, e.g. the mean audit rate)."""
        w = 1.0 / self.prob
        return w * normalize if normalize is not None else w


def concat_batches(batches):
    b0 = batches[0]
    offs = [np.zeros(1, np.int64)]
    base = 0
    for b in batches:
        offs.append(b.offsets[1:] + base)
        base += b.offsets[-1]
    cat = lambda name: np.concatenate([getattr(b, name) for b in batches])
    mo = mi = None
    if b0.miss_offsets is not None:
        mo_l = [np.zeros(1, np.int64)]; base = 0
        for b in batches:
            mo_l.append(b.miss_offsets[1:] + base); base += b.miss_offsets[-1]
        mo = np.concatenate(mo_l); mi = cat('miss_items')
    return StreamBatch(H=b0.H, offsets=np.concatenate(offs), origin=cat('origin'), arrival=cat('arrival'),
                       loss=cat('loss'), true_loss=cat('true_loss'), prob=cat('prob'), tiebreak=cat('tiebreak'),
                       n_audits=cat('n_audits'), n_defects=cat('n_defects'), n_misses=cat('n_misses'),
                       n_informative=cat('n_informative'), checkpoints=b0.checkpoints,
                       counts_at=np.concatenate([b.counts_at for b in batches]), miss_offsets=mo, miss_items=mi,
                       states=None if b0.states is None else np.concatenate([b.states for b in batches]),
                       meta=b0.meta)


# ------------------------------------------------------------------ numba kernels


@njit
def _draw_delay(st, kind, a, b, t):
    if kind == 0:
        return int(a)
    if kind == 1:
        return R.geometric0(st, a)
    if kind == 2:
        k = int(b)
        tot = 0
        for _ in range(k):
            tot += R.geometric0(st, a / k)
        return tot
    if kind == 3:
        if a <= 0:
            return 0
        mu = np.log(a) - 0.5 * b * b
        return int(np.floor(np.exp(mu + b * R.normal(st)) + 0.5))
    if kind == 4:
        s = int(a)
        x = t + int(b)
        return ((x + s - 1) // s) * s - t
    return 0


@njit
def _latent(states, H, pi_arr, r_arr, rho, markov_item, m, phi_pi, phi_r, e01, e10,
            dkind, da, db, sc_on, mu0, mud, mum, sig, Z, Mt, Y, DL, UA, S):
    """Fill per-item latent arrays (index t-1 for item t) for one stream."""
    sd = states[R.C_DEFECT].copy(); sa = states[R.C_AUDIT].copy(); sm = states[R.C_MISS].copy()
    sn = states[R.C_NOISE].copy(); sdl = states[R.C_DELAY].copy(); ssc = states[R.C_SCORE].copy()
    sb = states[R.C_BOARD].copy()
    ratio_pi = 1.0
    ratio_r = 1.0
    prevM = 0.0
    prev_r = 0.0
    has_prev = False
    noise = (e01 > 0.0) or (e10 > 0.0)
    for t in range(H):
        if m > 1 and t % m == 0:
            if phi_pi > 0.0:
                mu = pi_arr[t]
                ratio_pi = R.beta_mean_phi(sb, mu, phi_pi) / mu if mu > 0.0 else 1.0
            if phi_r > 0.0:
                mu = r_arr[t]
                ratio_r = R.beta_mean_phi(sb, mu, phi_r) / mu if mu > 0.0 else 1.0
        p_def = min(1.0, pi_arr[t] * ratio_pi)
        rt = min(1.0, r_arr[t] * ratio_r)
        z = R.uniform(sd) < p_def
        um = R.uniform(sm)
        if markov_item or z:
            if has_prev and rho != 0.0:
                p = rt + rho * (prevM - prev_r)
                p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
            else:
                p = rt
            mm = um < p
            prevM = 1.0 if mm else 0.0
            prev_r = rt
            has_prev = True
        else:
            mm = False
        Z[t] = 1 if z else 0
        Mt[t] = 1 if (mm and z) else 0
        if noise:
            un = R.uniform(sn)
            flip = un < (e10 if mm else e01)
        else:
            flip = False
        Y[t] = (1 if mm else 0) ^ (1 if flip else 0)
        DL[t] = _draw_delay(sdl, dkind, da, db, t + 1)
        UA[t] = R.uniform(sa)
        if sc_on:
            mu = mu0 if not z else (mum if mm else mud)
            S[t] = mu + sig * R.normal(ssc)


@njit
def _select_fixed(H, Z, Mt, Y, DL, UA, S, q_arr, sp_on, sp_edges, sp_probs, budget, force, cps,
                  o_org, o_arr, o_y, o_m, o_p, o_miss, cnt):
    """Fixed/stratified audit selection.  Returns (n_labels, n_aud, n_def, n_miss, n_inf, n_missitems)."""
    naud = 0; ndef = 0; nmiss = 0; ninf = 0; n = 0; nmi = 0
    ci = 0; C = len(cps)
    monotone = True
    last_arr = -1
    for t in range(H):
        item = t + 1
        while ci < C and cps[ci] < item:
            cnt[ci, 0] = naud; cnt[ci, 1] = ndef; cnt[ci, 2] = nmiss; ci += 1
        if sp_on:
            k = np.searchsorted(sp_edges, S[t])
            p = sp_probs[k]
        else:
            p = q_arr[t]
        if budget >= 0:
            if naud >= budget:
                p = 0.0
            elif force and (H - t) <= budget - naud:
                p = 1.0
        a = UA[t] < p
        if a:
            naud += 1
        if Z[t] == 1:
            ndef += 1
            if Mt[t] == 1:
                nmiss += 1
                o_miss[nmi] = item; nmi += 1
            if a:
                ninf += 1
                ar = item + DL[t]
                if ar <= H:
                    o_org[n] = item; o_arr[n] = ar; o_y[n] = Y[t]; o_m[n] = Mt[t]; o_p[n] = p
                    if ar < last_arr:
                        monotone = False
                    last_arr = ar
                    n += 1
    while ci < C:
        cnt[ci, 0] = naud; cnt[ci, 1] = ndef; cnt[ci, 2] = nmiss; ci += 1
    if not monotone:
        idx = np.argsort(o_arr[:n], kind='mergesort')
        o_org[:n] = o_org[:n][idx].copy(); o_arr[:n] = o_arr[:n][idx].copy()
        o_y[:n] = o_y[:n][idx].copy(); o_m[:n] = o_m[:n][idx].copy(); o_p[:n] = o_p[:n][idx].copy()
    return n, naud, ndef, nmiss, ninf, nmi


@njit(parallel=True)
def _batch_fixed(states, fill, offsets, moffsets, H, pi_arr, r_arr, rho, markov_item, m, phi_pi, phi_r,
                 e01, e10, dkind, da, db, sc_on, mu0, mud, mum, sig, q_arr, sp_on, sp_edges, sp_probs,
                 budget, force, cps, store_miss,
                 out_org, out_arr, out_y, out_m, out_p, out_mi, n_lab, n_mi, n_aud, n_def, n_miss, n_inf, counts):
    N = states.shape[0]
    for i in prange(N):
        Z = np.empty(H, np.int8); Mt = np.empty(H, np.int8); Y = np.empty(H, np.int8)
        DL = np.empty(H, np.int64); UA = np.empty(H); S = np.empty(H if sc_on else 1)
        _latent(states[i], H, pi_arr, r_arr, rho, markov_item, m, phi_pi, phi_r, e01, e10, dkind, da, db,
                sc_on, mu0, mud, mum, sig, Z, Mt, Y, DL, UA, S)
        if fill:
            a = offsets[i]; b = offsets[i + 1]
            o_org = out_org[a:b]; o_arr = out_arr[a:b]; o_y = out_y[a:b]; o_m = out_m[a:b]; o_p = out_p[a:b]
            if store_miss:
                o_mi = out_mi[moffsets[i]:moffsets[i + 1]]
            else:
                o_mi = np.empty(H, np.int64)
            # local buffers: labels may need sorting, so select into H-sized scratch then copy
            lo = np.empty(H, np.int64); la = np.empty(H, np.int64); ly = np.empty(H, np.int8)
            lm = np.empty(H, np.int8); lp = np.empty(H); lmi = np.empty(H, np.int64)
            cnt = counts[i]
            n, na, nd, nm, ni, nmi = _select_fixed(H, Z, Mt, Y, DL, UA, S, q_arr, sp_on, sp_edges, sp_probs,
                                                   budget, force, cps, lo, la, ly, lm, lp, lmi, cnt)
            o_org[:] = lo[:n]; o_arr[:] = la[:n]; o_y[:] = ly[:n]; o_m[:] = lm[:n]; o_p[:] = lp[:n]
            if store_miss:
                o_mi[:] = lmi[:nmi]
        else:
            lo = np.empty(H, np.int64); la = np.empty(H, np.int64); ly = np.empty(H, np.int8)
            lm = np.empty(H, np.int8); lp = np.empty(H); lmi = np.empty(H, np.int64)
            cnt = np.empty((len(cps), 3), np.int64)
            n, na, nd, nm, ni, nmi = _select_fixed(H, Z, Mt, Y, DL, UA, S, q_arr, sp_on, sp_edges, sp_probs,
                                                   budget, force, cps, lo, la, ly, lm, lp, lmi, cnt)
            n_lab[i] = n; n_mi[i] = nmi; n_aud[i] = na; n_def[i] = nd; n_miss[i] = nm; n_inf[i] = ni


@njit
def latent_one(states_i, H, pi_arr, r_arr, rho, markov_item, m, phi_pi, phi_r, e01, e10, dkind, da, db,
               sc_on, mu0, mud, mum, sig):
    Z = np.empty(H, np.int8); Mt = np.empty(H, np.int8); Y = np.empty(H, np.int8)
    DL = np.empty(H, np.int64); UA = np.empty(H); S = np.zeros(H if sc_on else 1)
    _latent(states_i, H, pi_arr, r_arr, rho, markov_item, m, phi_pi, phi_r, e01, e10, dkind, da, db,
            sc_on, mu0, mud, mum, sig, Z, Mt, Y, DL, UA, S)
    return Z, Mt, Y, DL, UA, S


# ------------------------------------------------------------------ audit hook interface


class AuditView:
    """Information available when deciding audits for items t0..t1-1 (1-based, t1 exclusive).

    Attributes (numpy arrays are read-only views, built lazily on access):
      t0, t1, H
      arrived_origin, arrived_arrival, arrived_loss, arrived_prob : labels with arrival <= t0-1,
          in arrival order (observed, i.e. noisy, loss)
      n_arrived     : number of such labels
      past_prob, past_audit : selection probabilities / audit indicators for items 1..t0-1
      n_audits      : audits so far
      budget_remaining : remaining total budget (None if unlimited)
      scores        : classifier scores of items t0..t1-1 (None unless the policy sets
                      uses_scores=True and the config has a ScoreModel).  Scores are produced
                      with the item, so with epoch=1 this is strictly causal.
      state         : dict persisted across epochs of the current stream (policy scratch)
    """
    __slots__ = ('t0', 't1', 'H', 'n_arrived', 'n_audits', 'budget_remaining', 'state', '_c')

    @property
    def arrived_origin(self):
        return _ro(self._c['ao'][:self.n_arrived])

    @property
    def arrived_arrival(self):
        return _ro(self._c['aa'][:self.n_arrived])

    @property
    def arrived_loss(self):
        return _ro(self._c['ay'][:self.n_arrived])

    @property
    def arrived_prob(self):
        return _ro(self._c['ap'][:self.n_arrived])

    @property
    def past_prob(self):
        return _ro(self._c['probs'][:self.t0 - 1])

    @property
    def past_audit(self):
        return _ro(self._c['aud'][:self.t0 - 1])

    @property
    def scores(self):
        S = self._c['S']
        return None if S is None else _ro(S[self.t0 - 1:self.t1 - 1])


class AuditPolicy:
    """Base class for Python audit hooks.

    Subclasses implement ``decide(view) -> float or array of length view.t1-view.t0`` with
    selection probabilities in [0,1].  ``epoch`` (items per decision) trades speed for
    granularity.  ``reset(stream_index)`` is called at the start of every stream.
    The generator applies the budget rule after the hook (p := 0 once exhausted; if
    force_budget, p := 1 when remaining items equal remaining budget).
    """
    epoch = 1
    uses_scores = False

    def reset(self, stream_index):
        pass

    def decide(self, view):
        raise NotImplementedError


class CallablePolicy(AuditPolicy):
    def __init__(self, fn, epoch=1, uses_scores=False):
        self.fn, self.epoch, self.uses_scores = fn, epoch, uses_scores

    def decide(self, view):
        return self.fn(view)


def _ro(a):
    v = a.view()
    v.flags.writeable = False
    return v


def _generate_hook_one(cfg, st, policy, idx, arrays, extra):
    H = int(cfg.H)
    pi, r, q = arrays
    Z, Mt, Y, DL, UA, S = latent_one(st, H, pi, r, *extra)
    epoch = max(1, int(getattr(policy, 'epoch', 1)))
    uses_scores = bool(getattr(policy, 'uses_scores', False)) and cfg.score is not None
    probs = np.zeros(H); aud = np.zeros(H, np.int8)
    ao = np.empty(H, np.int64); aa = np.empty(H, np.int64); ay = np.empty(H, np.int8); am = np.empty(H, np.int8)
    ap = np.empty(H); nar = 0
    po = []; pa = []                      # pending labels (audited informative, not yet arrived)
    pmin = H + 1                          # earliest pending arrival
    naud = 0; ninf = 0
    budget = int(cfg.budget); force = bool(cfg.force_budget)
    policy.reset(idx)
    view = AuditView()
    view.H = H; view.state = {}
    view._c = dict(ao=ao, aa=aa, ay=ay, ap=ap, probs=probs, aud=aud, S=S if uses_scores else None)
    UAl = UA.tolist(); Zl = Z.tolist(); DLl = DL.tolist()
    for t0 in range(1, H + 1, epoch):
        t1 = min(t0 + epoch, H + 1)
        if pmin <= t0 - 1:
            ready = [(a_, o_) for o_, a_ in zip(po, pa) if a_ <= t0 - 1]
            ready.sort()
            for a_, o_ in ready:
                ao[nar] = o_; aa[nar] = a_; ay[nar] = Y[o_ - 1]; am[nar] = Mt[o_ - 1]; ap[nar] = probs[o_ - 1]
                nar += 1
            keep = [(o_, a_) for o_, a_ in zip(po, pa) if a_ > t0 - 1]
            po = [k[0] for k in keep]; pa = [k[1] for k in keep]
            pmin = min(pa) if pa else H + 1
        view.t0 = t0; view.t1 = t1; view.n_arrived = nar; view.n_audits = naud
        view.budget_remaining = None if budget < 0 else budget - naud
        p = policy.decide(view)
        if np.ndim(p) == 0:
            p = float(p)
            if not (0.0 <= p <= 1.0):
                raise ValueError('audit probabilities must lie in [0,1]')
            plist = [p] * (t1 - t0)
        else:
            plist = np.broadcast_to(np.asarray(p, float), (t1 - t0,)).tolist()
            if any(not (0.0 <= v <= 1.0) for v in plist):
                raise ValueError('audit probabilities must lie in [0,1]')
        if budget < 0 and t1 - t0 > 16:          # vectorized epoch (no budget rule)
            pv = np.asarray(plist); sl = slice(t0 - 1, t1 - 1)
            probs[sl] = pv
            a = UA[sl] < pv
            aud[sl] = a
            naud += int(a.sum())
            for t in (np.flatnonzero(a & (Z[sl] == 1)) + (t0 - 1)).tolist():
                ninf += 1
                arr = t + 1 + DLl[t]
                if arr <= H:
                    po.append(t + 1); pa.append(arr)
                    if arr < pmin:
                        pmin = arr
            continue
        for j in range(t1 - t0):
            t = t0 - 1 + j
            pj = plist[j]
            if budget >= 0:
                if naud >= budget:
                    pj = 0.0
                elif force and (H - t) <= budget - naud:
                    pj = 1.0
            probs[t] = pj
            if UAl[t] < pj:
                aud[t] = 1; naud += 1
                if Zl[t] == 1:
                    ninf += 1
                    arr = t + 1 + DLl[t]
                    if arr <= H:
                        po.append(t + 1); pa.append(arr)
                        if arr < pmin:
                            pmin = arr
    if pa:
        rest = sorted(zip(pa, po))
        for a_, o_ in rest:
            ao[nar] = o_; aa[nar] = a_; ay[nar] = Y[o_ - 1]; am[nar] = Mt[o_ - 1]; ap[nar] = probs[o_ - 1]; nar += 1
    zz = Z == 1
    mm = zz & (Mt == 1)
    cps = np.asarray(cfg.checkpoints, np.int64)
    ca = np.cumsum(aud); cd = np.cumsum(zz); cm = np.cumsum(mm)
    cnt = np.zeros((len(cps), 3), np.int64)
    for c, cp in enumerate(cps):
        j = min(int(cp), H) - 1
        if j >= 0:
            cnt[c] = (ca[j], cd[j], cm[j])
    return dict(origin=ao[:nar].copy(), arrival=aa[:nar].copy(), loss=ay[:nar].copy(), true_loss=am[:nar].copy(),
                prob=ap[:nar].copy(), n_aud=naud, n_def=int(zz.sum()), n_miss=int(mm.sum()), n_inf=ninf,
                miss_items=np.flatnonzero(mm).astype(np.int64) + 1, counts=cnt)


# ------------------------------------------------------------------ public entry point


def _extra(cfg):
    assert cfg.markov_over in ('defect', 'item')
    assert 0 <= cfg.icc_r < 1 and 0 <= cfg.icc_pi < 1
    phi_r = 1.0 / cfg.icc_r - 1.0 if cfg.icc_r > 0 else 0.0
    phi_pi = 1.0 / cfg.icc_pi - 1.0 if cfg.icc_pi > 0 else 0.0
    sc = cfg.score
    sc_on = sc is not None
    if not sc_on:
        sc = ScoreModel()
    d = cfg.delay
    return (float(cfg.markov_rho), cfg.markov_over == 'item', int(cfg.board_size), phi_pi, phi_r,
            float(cfg.e01), float(cfg.e10), int(d.kind), float(d.a), float(d.b), sc_on,
            float(sc.mu0), float(sc.mu_det), float(sc.mu_miss), float(sc.sigma))


def generate(cfg, role='evaluation', experiment=0, condition=0, replicates=1000, states=None, keep_states=False):
    """Generate a StreamBatch.

    replicates: int N (replicates 0..N-1) or an explicit array of replicate indices.
    states: optional precomputed (N, NCOMP, 4) uint64 states (overrides the seed key).
    Fast numba path for policy None / StratifiedPolicy; Python hook path otherwise.
    """
    if states is None:
        reps = np.arange(replicates) if np.ndim(replicates) == 0 else np.asarray(replicates)
        states = R.stream_states(role, experiment, condition, reps)
    N = states.shape[0]
    H = int(cfg.H)
    pi, r, q = cfg.arrays()
    extra = _extra(cfg)
    cps = np.asarray(sorted(cfg.checkpoints), np.int64)
    tie = R.uniforms_for_tiebreak(states)
    meta = dict(role=role, experiment=experiment, condition=condition)
    pol = cfg.policy
    if pol is None or isinstance(pol, StratifiedPolicy):
        sp_on = isinstance(pol, StratifiedPolicy)
        if sp_on:
            assert cfg.score is not None, 'StratifiedPolicy requires a ScoreModel'
            sp_edges = np.asarray(pol.edges, float); sp_probs = np.asarray(pol.probs, float)
            assert len(sp_probs) == len(sp_edges) + 1
        else:
            sp_edges = np.zeros(0); sp_probs = np.zeros(1)
        z1 = np.zeros(1, np.int64)
        n_lab = np.zeros(N, np.int64); n_mi = np.zeros(N, np.int64); n_aud = np.zeros(N, np.int64)
        n_def = np.zeros(N, np.int64); n_miss = np.zeros(N, np.int64); n_inf = np.zeros(N, np.int64)
        counts = np.zeros((N, len(cps), 3), np.int64)
        args = (H, pi, r) + extra + (q, sp_on, sp_edges, sp_probs, int(cfg.budget), bool(cfg.force_budget), cps,
                bool(cfg.store_miss_items))
        _batch_fixed(states, False, z1, z1, *args, np.zeros(1, np.int64), np.zeros(1, np.int64),
                     np.zeros(1, np.int8), np.zeros(1, np.int8), np.zeros(1), np.zeros(1, np.int64),
                     n_lab, n_mi, n_aud, n_def, n_miss, n_inf, counts)
        offsets = np.zeros(N + 1, np.int64); offsets[1:] = np.cumsum(n_lab)
        moff = np.zeros(N + 1, np.int64); moff[1:] = np.cumsum(n_mi)
        L = int(offsets[-1])
        org = np.empty(L, np.int64); arr = np.empty(L, np.int64); y = np.empty(L, np.int8); mt = np.empty(L, np.int8)
        p = np.empty(L); mi = np.empty(int(moff[-1]) if cfg.store_miss_items else 1, np.int64)
        _batch_fixed(states, True, offsets, moff, *args, org, arr, y, mt, p, mi,
                     n_lab, n_mi, n_aud, n_def, n_miss, n_inf, counts)
        return StreamBatch(H=H, offsets=offsets, origin=org, arrival=arr, loss=y, true_loss=mt, prob=p, tiebreak=tie,
                           n_audits=n_aud, n_defects=n_def, n_misses=n_miss, n_informative=n_inf, checkpoints=cps,
                           counts_at=counts, miss_offsets=moff if cfg.store_miss_items else None,
                           miss_items=mi if cfg.store_miss_items else None,
                           states=states if keep_states else None, meta=meta)
    # ---- Python hook path
    policy = pol if isinstance(pol, AuditPolicy) else CallablePolicy(pol)
    cfg_c = StreamConfig(**{**cfg.__dict__, 'checkpoints': tuple(cps)})
    res = [_generate_hook_one(cfg_c, states[i], policy, i, (pi, r, q), extra) for i in range(N)]
    offsets = np.zeros(N + 1, np.int64); offsets[1:] = np.cumsum([len(x['origin']) for x in res])
    moff = np.zeros(N + 1, np.int64); moff[1:] = np.cumsum([len(x['miss_items']) for x in res])
    cat = lambda k, dt: np.concatenate([x[k] for x in res]).astype(dt) if N else np.zeros(0, dt)
    return StreamBatch(H=H, offsets=offsets, origin=cat('origin', np.int64), arrival=cat('arrival', np.int64),
                       loss=cat('loss', np.int8), true_loss=cat('true_loss', np.int8), prob=cat('prob', float),
                       tiebreak=tie, n_audits=np.array([x['n_aud'] for x in res], np.int64),
                       n_defects=np.array([x['n_def'] for x in res], np.int64),
                       n_misses=np.array([x['n_miss'] for x in res], np.int64),
                       n_informative=np.array([x['n_inf'] for x in res], np.int64), checkpoints=cps,
                       counts_at=np.stack([x['counts'] for x in res]) if N else np.zeros((0, len(cps), 3), np.int64),
                       miss_offsets=moff if cfg.store_miss_items else None,
                       miss_items=cat('miss_items', np.int64) if cfg.store_miss_items else None,
                       states=states if keep_states else None, meta=meta)


def generate_chunks(cfg, role, experiment, condition, n, chunk=2000):
    """Yield StreamBatches of at most `chunk` replicates (replicate ids are global 0..n-1)."""
    for s in range(0, n, chunk):
        yield generate(cfg, role, experiment, condition, np.arange(s, min(n, s + chunk)))


def _gen_worker(args):
    cfg, role, experiment, condition, reps = args
    return generate(cfg, role, experiment, condition, reps)


def generate_parallel(cfg, role='evaluation', experiment=0, condition=0, replicates=1000, n_jobs=8, chunk=None):
    """Process-parallel `generate` (for Python audit hooks, which run serially in-process).

    The config (including the policy object) must be picklable.  Uses the 'spawn' start
    method; results are identical to `generate` (same seed keys), concatenated in replicate
    order.  Keep n_jobs <= 8 on the shared machine.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    reps = np.arange(replicates) if np.ndim(replicates) == 0 else np.asarray(replicates)
    chunk = chunk or max(1, int(np.ceil(len(reps) / (4 * n_jobs))))
    parts = [reps[i:i + chunk] for i in range(0, len(reps), chunk)]
    with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context('spawn')) as ex:
        out = list(ex.map(_gen_worker, [(cfg, role, experiment, condition, p) for p in parts]))
    return concat_batches(out)


class WarningBoostPolicy(AuditPolicy):
    """EXAMPLE evidence-adaptive hook (not the original allocation policy).

    Audits with probability `base`; switches to `boost` while the mixture-SR log statistic of
    the labels arrived so far (updated incrementally in view.state) exceeds `warn`.
    Demonstrates the hook API; selection probabilities are recorded for IPW.
    """
    def __init__(self, base=0.1, boost=0.5, warn=2.0, r0=0.10, epoch=10):
        self.base, self.boost, self.warn, self.r0, self.epoch = base, boost, warn, r0, epoch
        self.lam = np.array([.02, .05, .10, .20, .40]) / .09

    def reset(self, i):
        pass

    def decide(self, view):
        st = view.state
        if 'logR' not in st:
            st['logR'] = np.full(5, -np.inf); st['seen'] = 0
        n = view.n_arrived
        if n > st['seen']:
            for x in view.arrived_loss[st['seen']:n]:
                st['logR'] = np.logaddexp(st['logR'], 0.0) + np.log1p(self.lam * (x - self.r0))
            st['seen'] = n
        s = np.logaddexp.reduce(st['logR']) - np.log(5) if n else -np.inf
        return self.boost if s > self.warn else self.base
