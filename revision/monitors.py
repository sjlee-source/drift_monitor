"""Sequential monitors for the conditional false-negative risk r_t = P(miss | defective).

Every monitor consumes the informative labels of one stream in *arrival order*,
``(arrival_item, loss[, weight])`` with loss in {0,1}, and produces a statistic path
``(t_k, S_k)``: for label-driven monitors one value after each label (t_k = its arrival
production item); for the periodic test one value per scheduled check.  The alarm rule is
always

    T(h) = first t_k with S_k > h      (strict; T = inf if none)

so first-crossing times for ANY threshold follow from the running maximum of S (sweep
mode).  Randomized tie-breaking (alarm on S >= h with probability gamma) is supported by
``first_alarm(..., gamma=...)`` using the per-stream ``batch.tiebreak`` uniform.

Implemented monitors (null boundary r0 = 0.10 unless stated):

  Betting      : original predictable-stake betting monitor (log wealth).
  CUSUM        : Bernoulli CUSUM (Page 1954), continuous or integer-score chart.
  MixtureSR    : original mixture Shiryaev-Roberts (five stakes /0.09).
  EDetector    : Shin, Ramdas & Rinaldo (2024) mixture e-SR / e-CUSUM (bounded-mean baselines,
                 grid and weights by their Algorithm 3).
  Periodic     : original periodic exact binomial check (disjoint intervals).
  EWMA         : Bernoulli EWMA chart, standardized with exact time-varying variance.
  GLRCUSUM     : windowed Bernoulli GLR-CUSUM, unknown r1 > r0.
  WindowBinomial: exact binomial tail test on the last W labels at every arrival.
  ADWIN        : one-sided ADWIN (Bifet & Gavalda 2007, ADWIN2 bound).
  DDM          : Drift Detection Method (Gama et al. 2004).
  CSDetector   : backward confidence-sequence detector: alarm when an anytime-valid
                 betting lower confidence bound (Waudby-Smith & Ramdas 2024, PrPl stakes)
                 started at a recent label exceeds r0 (cf. Shekhar & Ramdas 2023).

Weights (IPW) are supported by Betting, CUSUM, MixtureSR and EDetector (betting form);
others raise if weights are passed.  See each class docstring for formulas.
"""
import math
import numpy as np
from numba import njit, prange
from scipy.special import logsumexp
from scipy.stats import binom
from scipy.optimize import brentq

NO_ALARM = np.int64(2 ** 62)
NEG_INF = -np.inf


# ======================================================================= numba helpers

@njit(inline='always')
def _logaddexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit
def log_binom_sf(k, n, p):
    """log P(Bin(n,p) >= k), computed by direct summation of the pmf in the short tail."""
    if k <= 0:
        return 0.0
    if k > n:
        return -np.inf
    lp = math.log(p)
    lq = math.log1p(-p)
    lg_n = math.lgamma(n + 1.0)
    if k > n * p:
        lt = lg_n - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0) + k * lp + (n - k) * lq
        s = 1.0
        term = 1.0
        ratio = p / (1.0 - p)
        for j in range(k, n):
            term *= (n - j) / (j + 1.0) * ratio
            s += term
            if term < 1e-17 * s:
                break
        return lt + math.log(s)
    # complement of the lower tail P(X <= k-1)
    j0 = k - 1
    lt = lg_n - math.lgamma(j0 + 1.0) - math.lgamma(n - j0 + 1.0) + j0 * lp + (n - j0) * lq
    s = 1.0
    term = 1.0
    ratio = (1.0 - p) / p
    for j in range(j0, 0, -1):
        term *= j / (n - j + 1.0) * ratio
        s += term
        if term < 1e-17 * s:
            break
    lower = math.exp(lt) * s
    if lower >= 1.0:
        return -745.0
    return math.log1p(-lower)


# ======================================================================= path kernels
# signature: f(items, loss, w, H, t0, params, table, h_stop, out_t, out_s) -> n_out
# items/loss: labels with arrival > t0 in arrival order; w: weights or empty array.

@njit
def _path_betting(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    r0 = params[0]; gain = params[1]; cap = params[2]; win = int(params[3])
    n = len(items)
    weighted = len(w) > 0
    bx = np.zeros(win); bw = np.zeros(win)
    sx = 0.0; sw = 0.0; cnt = 0
    logw = 0.0
    for i in range(n):
        if cnt > 0:
            prev = sx / sw
        else:
            prev = 0.0
        lam = gain * (prev - r0)
        if lam < 0.0:
            lam = 0.0
        if lam > cap:
            lam = cap
        x = float(loss[i])
        wi = w[i] if weighted else 1.0
        if weighted and lam * wi > cap:
            lam = cap / wi
        logw += math.log1p(lam * wi * (x - r0))
        out_t[i] = items[i]; out_s[i] = logw
        # update window (current observation enters only after its own bet)
        j = i % win
        if cnt == win:
            sx -= bx[j] * bw[j]; sw -= bw[j]
        else:
            cnt += 1
        bx[j] = x; bw[j] = wi
        sx += x * wi; sw += wi
        if cnt == win and i % 4096 == 4095:   # refresh sums against round-off drift
            sx = 0.0; sw = 0.0
            for k in range(win):
                sx += bx[k] * bw[k]; sw += bw[k]
        if logw > h_stop:
            return i + 1
    return n


@njit
def _path_cusum(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    r0 = params[0]; p1 = params[1]; A = params[2]; B = params[3]
    integer = A > 0
    if integer:
        up = A; dn = -B
    else:
        up = math.log(p1 / r0); dn = math.log((1.0 - p1) / (1.0 - r0))
    weighted = len(w) > 0
    s = 0.0
    for i in range(len(items)):
        inc = up if loss[i] == 1 else dn
        if weighted:
            inc *= w[i]
        s = s + inc
        if s < 0.0:
            s = 0.0
        out_t[i] = items[i]; out_s[i] = s
        if s > h_stop:
            return i + 1
    return len(items)


@njit
def _path_mix(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    """Mixture of SR / CUSUM baseline e-detectors.  params:
    [r0, K, mode(0 SR,1 CUSUM), inc(0 betting,1 exponential), cap_frac, stakes(K), logw(K)].

    Computed in a rescaled linear domain: R_k = exp(c) * Rs_k with a common log-scale c, so the
    SR update (R+1)L becomes Rs <- (Rs + e^{-c}) L and the CUSUM update max(R,1)L becomes
    Rs <- max(Rs, e^{-c}) L; c is increased whenever max Rs > 1e100 (and decreased toward 0
    when all Rs < 1e-100).  S_n = c + log sum_k w_k Rs_k.
    """
    r0 = params[0]; K = int(params[1]); mode = int(params[2]); inc = int(params[3]); capf = params[4]
    stakes = params[5:5 + K]; wts = np.exp(params[5 + K:5 + 2 * K])
    weighted = len(w) > 0
    L0 = np.empty(K); L1 = np.empty(K)
    for k in range(K):
        if inc == 0:
            L0[k] = 1.0 - stakes[k] * r0
            L1[k] = 1.0 + stakes[k] * (1.0 - r0)
        else:
            lam = stakes[k] * r0           # Eq. (72) parametrization lambda in (0,1)
            psi = -math.log1p(-lam) - lam
            s0 = -1.0; s1 = 1.0 / r0 - 1.0
            L0[k] = math.exp(lam * s0 - psi * s0 * s0)
            L1[k] = math.exp(lam * s1 - psi * s1 * s1)
    Rs = np.zeros(K)
    c = 0.0
    one = 1.0          # e^{-c}
    for i in range(len(items)):
        x = loss[i]
        tot = 0.0
        mx = 0.0
        for k in range(K):
            if weighted:
                st = stakes[k]
                if st * w[i] * r0 > capf:
                    st = capf / (w[i] * r0)
                li = 1.0 + st * w[i] * (x - r0)
            else:
                li = L1[k] if x == 1 else L0[k]
            if mode == 0:
                v = (Rs[k] + one) * li
            else:
                v = (Rs[k] if Rs[k] > one else one) * li
            Rs[k] = v
            tot += wts[k] * v
            if v > mx:
                mx = v
        s = c + math.log(tot) if tot > 0.0 else -np.inf
        out_t[i] = items[i]; out_s[i] = s
        if s > h_stop:
            return i + 1
        if mx > 1e100 or (mx < 1e-100 and c > 0.0):
            sh = math.log(mx) if mx > 0.0 else -c
            if c + sh < 0.0:
                sh = -c
            for k in range(K):
                Rs[k] = Rs[k] * math.exp(-sh)
            c += sh
            one = math.exp(-c)
    return len(items)


@njit
def _path_periodic(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    r0 = params[0]; interval = int(params[1])
    n = len(items)
    j = 0
    k = 0
    first = (t0 // interval + 1) * interval
    c = first
    while True:
        if c > H:
            if (H % interval != 0) and H > t0 and (k == 0 or out_t[k - 1] < H):
                c = H
            else:
                break
        m = 0; s = 0
        while j < n and items[j] <= c:
            m += 1; s += loss[j]; j += 1
        st = 0.0 if m == 0 else -log_binom_sf(s, m, r0)
        out_t[k] = c; out_s[k] = st; k += 1
        if st > h_stop or c == H:
            break
        c += interval
    return k


@njit
def _path_ewma(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    r0 = params[0]; lam = params[1]; z = params[2]; asym = params[3] > 0
    v_inf = r0 * (1.0 - r0) * lam / (2.0 - lam)
    decay = 1.0
    for i in range(len(items)):
        z = (1.0 - lam) * z + lam * loss[i]
        decay *= (1.0 - lam) * (1.0 - lam)
        v = v_inf if asym else v_inf * (1.0 - decay)
        s = (z - r0) / math.sqrt(v)
        out_t[i] = items[i]; out_s[i] = s
        if s > h_stop:
            return i + 1
    return len(items)


@njit
def _path_glr(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    r0 = params[0]; W = int(params[1])
    n = len(items)
    cs = np.zeros(n + 1)
    mpos = np.empty(n, np.int64); nm = 0; lo = 0
    lr0 = math.log(r0); lq0 = math.log1p(-r0)
    for i in range(n):
        cs[i + 1] = cs[i] + loss[i]
        if loss[i] == 1:
            mpos[nm] = i; nm += 1
        while lo < nm and mpos[lo] <= i - W:
            lo += 1
        best = 0.0
        # optimal segment start is a miss (dropping a leading non-miss raises the sup-LLR)
        for q in range(nm - 1, lo - 1, -1):
            k = mpos[q]
            mlen = i - k + 1
            s = cs[i + 1] - cs[k]
            ph = s / mlen
            if ph > r0:
                if s < mlen:
                    llr = s * (math.log(ph) - lr0) + (mlen - s) * (math.log1p(-ph) - lq0)
                else:
                    llr = s * (-lr0)
                if llr > best:
                    best = llr
        out_t[i] = items[i]; out_s[i] = best
        if best > h_stop:
            return i + 1
    return n


@njit
def _path_window_binom(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    W = int(params[1])
    n = len(items)
    s = 0
    for i in range(n):
        s += loss[i]
        if i >= W:
            s -= loss[i - W]
        m = i + 1 if i + 1 < W else W
        st = table[m, s]
        out_t[i] = items[i]; out_s[i] = st
        if st > h_stop:
            return i + 1
    return n


@njit
def _path_adwin(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    min_len = int(params[0]); M = params[1]; maxw = int(params[2]); clock = int(params[3])
    n = len(items)
    cs = np.zeros(n + 1)
    for i in range(n):
        cs[i + 1] = cs[i] + loss[i]
        nw = i + 1
        ws = 0
        if maxw > 0 and nw > maxw:
            ws = nw - maxw
            nw = maxw
        best = -np.inf
        if nw >= 2 * min_len and nw >= 3 and (i + 1) % clock == 0:
            tot = cs[i + 1] - cs[ws]
            p = tot / nw
            var = p * (1.0 - p)
            n1 = min_len
            while nw - n1 >= min_len:
                n0 = nw - n1
                mu1 = (cs[i + 1] - cs[i + 1 - n1]) / n1
                mu0 = (cs[i + 1 - n1] - cs[ws]) / n0
                d = mu1 - mu0
                if d > 0.0:
                    m = 1.0 / (1.0 / n0 + 1.0 / n1)
                    a = 2.0 / (3.0 * m)
                    b = math.sqrt(2.0 * var / m)
                    x = (-b + math.sqrt(b * b + 4.0 * a * d)) / (2.0 * a)
                    if x * x > best:
                        best = x * x
                nn = int(n1 * (1.0 + 1.0 / M))
                n1 = nn if nn > n1 else n1 + 1
            if best > -np.inf:
                best = best - math.log(2.0 * math.log(nw))
        out_t[i] = items[i]; out_s[i] = best
        if best > h_stop:
            return i + 1
    return n


@njit
def _path_ddm(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    n_min = int(params[0])
    err = 0.0
    pmin = np.inf; smin = np.inf
    for i in range(len(items)):
        err += loss[i]
        k = i + 1
        p = err / k
        s = math.sqrt(p * (1.0 - p) / k)
        st = -np.inf
        if k >= n_min:
            if p + s <= pmin + smin:
                pmin = p; smin = s
            if smin > 0.0:
                st = (p + s - pmin) / smin
            else:
                st = np.inf if p + s > pmin else 0.0
        out_t[i] = items[i]; out_s[i] = st
        if st > h_stop:
            return i + 1
    return len(items)


@njit
def _path_cs(items, loss, w, H, t0, params, table, h_stop, out_t, out_s):
    r0 = params[0]; la = math.log(1.0 / params[1]); G = int(params[2]); K = int(params[3]); c = params[4]
    cnt = np.zeros(K, np.int64); sx = np.zeros(K); ssq = np.zeros(K); sig2 = np.full(K, 0.25)
    logk = np.zeros(K); act = np.zeros(K, np.bool_)
    nstart = 0
    cap = c / r0
    for i in range(len(items)):
        if i % G == 0:
            sl = nstart % K
            cnt[sl] = 0; sx[sl] = 0.0; ssq[sl] = 0.0; sig2[sl] = 0.25; logk[sl] = 0.0; act[sl] = True
            nstart += 1
        x = float(loss[i])
        best = -np.inf
        for k in range(K):
            if not act[k]:
                continue
            t = cnt[k] + 1
            lam = math.sqrt(2.0 * la / (sig2[k] * t * math.log(1.0 + t)))
            if lam > cap:
                lam = cap
            logk[k] += math.log1p(lam * (x - r0))
            sx[k] += x
            mu = (0.5 + sx[k]) / (t + 1.0)
            ssq[k] += (x - mu) * (x - mu)
            sig2[k] = (0.25 + ssq[k]) / (t + 1.0)
            cnt[k] = t
            if logk[k] > best:
                best = logk[k]
        out_t[i] = items[i]; out_s[i] = best
        if best > h_stop:
            return i + 1
    return len(items)


# ======================================================================= batch drivers

def _make_drivers(path_fn):
    @njit(parallel=True)
    def drv_eval(offsets, items, loss, w, wflag, H, params, table, extra, ths, cps, stop, alarm, mx):
        N = len(offsets) - 1
        G = len(ths); C = len(cps)
        for i in prange(N):
            a = offsets[i]; b = offsets[i + 1]
            it = items[a:b]; ls = loss[a:b]
            ww = w[a:b] if wflag else w[0:0]
            ot = np.empty(b - a + extra, np.int64); os_ = np.empty(b - a + extra)
            n = path_fn(it, ls, ww, H, 0, params, table, stop, ot, os_)
            g = 0; c = 0; run = -np.inf
            for k in range(n):
                t = ot[k]
                while c < C and cps[c] < t:
                    mx[i, c] = run; c += 1
                if os_[k] > run:
                    run = os_[k]
                    while g < G and run > ths[g]:
                        alarm[i, g] = t; g += 1
            while c < C:
                mx[i, c] = run; c += 1
            while g < G:
                alarm[i, g] = NO_ALARM; g += 1

    @njit(parallel=True)
    def drv_records(offsets, items, loss, w, wflag, H, params, table, extra, fill, roff, rt, rv, rn):
        N = len(offsets) - 1
        for i in prange(N):
            a = offsets[i]; b = offsets[i + 1]
            it = items[a:b]; ls = loss[a:b]
            ww = w[a:b] if wflag else w[0:0]
            ot = np.empty(b - a + extra, np.int64); os_ = np.empty(b - a + extra)
            n = path_fn(it, ls, ww, H, 0, params, table, np.inf, ot, os_)
            run = -np.inf; r = 0
            base = roff[i] if fill else 0
            for k in range(n):
                if os_[k] > run:
                    run = os_[k]
                    if fill:
                        rt[base + r] = ot[k]; rv[base + r] = run
                    r += 1
            if not fill:
                rn[i] = r

    @njit(parallel=True)
    def drv_restart(offsets, items, loss, w, wflag, H, params, table, extra, h, n_alarm, last_t, first_t):
        N = len(offsets) - 1
        for i in prange(N):
            a = offsets[i]; b = offsets[i + 1]
            ot = np.empty(b - a + extra, np.int64); os_ = np.empty(b - a + extra)
            s0 = a; tl = 0; na = 0; ft = NO_ALARM
            while True:
                it = items[s0:b]; ls = loss[s0:b]
                ww = w[s0:b] if wflag else w[0:0]
                n = path_fn(it, ls, ww, H, tl, params, table, h, ot, os_)
                hit = -1
                for k in range(n):
                    if os_[k] > h:
                        hit = k; break
                if hit < 0:
                    break
                tl = ot[hit]; na += 1
                if ft == NO_ALARM:
                    ft = tl
                while s0 < b and items[s0] <= tl:
                    s0 += 1
                if tl >= H:
                    break
            n_alarm[i] = na; last_t[i] = tl; first_t[i] = ft

    return drv_eval, drv_records, drv_restart


# ======================================================================= Python API

class Monitor:
    """Base class.  Subclasses set: name, _fn (path kernel), params, table, extra_outputs(H)."""
    name = 'monitor'
    supports_weights = False
    _drivers_cache = {}

    def __init__(self):
        self.table = np.zeros((1, 1))

    # -- kernels
    @classmethod
    def _drivers(cls):
        key = cls._fn
        if key not in Monitor._drivers_cache:
            Monitor._drivers_cache[key] = _make_drivers(cls._fn)
        return Monitor._drivers_cache[key]

    def extra_outputs(self, H):
        return 1

    def native_threshold(self):
        """Threshold with a design-based guarantee (see class docstring), or None."""
        return None

    def _w(self, weights, n):
        if weights is None:
            return np.zeros(1), False
        if not self.supports_weights:
            raise NotImplementedError(f'{self.name} does not support weights')
        w = np.asarray(weights, float)
        assert w.shape == (n,)
        return w, True

    # -- single stream
    def path(self, arrival, loss, weight=None, H=None, t0=0):
        """Statistic path for one stream: (times, S).  Labels must be in arrival order."""
        arrival = np.ascontiguousarray(arrival, np.int64); loss = np.ascontiguousarray(loss, np.int8)
        assert np.all(np.diff(arrival) >= 0), 'labels must be in arrival order'
        H = int(arrival[-1]) if H is None and len(arrival) else int(H or 0)
        w, wf = self._w(weight, len(arrival))
        ot = np.empty(len(arrival) + self.extra_outputs(H), np.int64); os_ = np.empty(len(ot))
        sel = arrival > t0
        n = type(self)._fn(arrival[sel], loss[sel], w[sel] if wf else np.zeros(0), H, t0, self.params, self.table,
                           np.inf, ot, os_)
        return ot[:n].copy(), os_[:n].copy()

    # -- batch
    def evaluate(self, batch, thresholds=(), checkpoints=(), weights=None):
        """Return dict(alarm=(N,G) float first-crossing items for each threshold (inf if none),
        max=(N,C) running max of S over outputs with time <= checkpoint (-inf if none))."""
        ths = np.asarray(thresholds, float).ravel(); order = np.argsort(ths, kind='stable')
        cps = np.asarray(checkpoints, np.int64).ravel(); corder = np.argsort(cps, kind='stable')
        w, wf = self._w(weights, len(batch.loss))
        N = batch.n
        alarm = np.empty((N, len(ths)), np.int64); mx = np.empty((N, len(cps)))
        stop = np.inf if len(cps) or not len(ths) else float(ths.max())
        self._drivers()[0](batch.offsets, batch.arrival, batch.loss, w, wf, int(batch.H), self.params, self.table,
                           self.extra_outputs(batch.H), ths[order], cps[corder], stop, alarm, mx)
        a = np.empty_like(alarm); a[:, order] = alarm
        m = np.empty_like(mx); m[:, corder] = mx
        a = a.astype(float); a[a >= NO_ALARM] = np.inf
        return dict(alarm=a, max=m)

    def max_stat(self, batch, horizon=None, weights=None):
        """Max of S over outputs with time <= horizon (default H); -inf if no output."""
        return self.evaluate(batch, (), (batch.H if horizon is None else horizon,), weights)['max'][:, 0]

    def first_alarm(self, batch, h, gamma=0.0, weights=None):
        """First alarm item (float, inf if none) for threshold h.  If gamma>0, a stream whose
        tiebreak uniform V < gamma alarms at S >= h instead of S > h (randomized rule)."""
        if gamma <= 0:
            return self.evaluate(batch, (h,), (), weights)['alarm'][:, 0]
        e = self.evaluate(batch, (h, np.nextafter(h, -np.inf)), (), weights)['alarm']
        return np.where(batch.tiebreak < gamma, e[:, 1], e[:, 0])

    def alarm_grid(self, batch, grid, weights=None):
        """(N,G) first-crossing items for a threshold grid (sweep mode)."""
        return self.evaluate(batch, grid, (), weights)['alarm']

    def records(self, batch, weights=None):
        """Strict running-max records per stream (CSR): offsets, times, values.  First crossing of
        h = time of the first record with value > h (see records_first_crossing)."""
        w, wf = self._w(weights, len(batch.loss))
        N = batch.n; d = self._drivers()[1]; ex = self.extra_outputs(batch.H)
        rn = np.zeros(N, np.int64)
        d(batch.offsets, batch.arrival, batch.loss, w, wf, int(batch.H), self.params, self.table, ex, False,
          np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1), rn)
        roff = np.zeros(N + 1, np.int64); roff[1:] = np.cumsum(rn)
        rt = np.empty(roff[-1], np.int64); rv = np.empty(roff[-1])
        d(batch.offsets, batch.arrival, batch.loss, w, wf, int(batch.H), self.params, self.table, ex, True,
          roff, rt, rv, rn)
        return roff, rt, rv

    def restart_run(self, batch, h, weights=None):
        """Run with restart after every alarm (statistic reset; labels arriving after the alarm
        item are used).  Returns dict(n_alarms, last_alarm, first_alarm) per stream."""
        w, wf = self._w(weights, len(batch.loss))
        N = batch.n
        na = np.zeros(N, np.int64); lt = np.zeros(N, np.int64); ft = np.zeros(N, np.int64)
        self._drivers()[2](batch.offsets, batch.arrival, batch.loss, w, wf, int(batch.H), self.params, self.table,
                           self.extra_outputs(batch.H), float(h), na, lt, ft)
        f = ft.astype(float); f[ft >= NO_ALARM] = np.inf
        return dict(n_alarms=na, last_alarm=lt, first_alarm=f)

    def __repr__(self):
        return f'{type(self).__name__}({self.name})'


def records_first_crossing(roff, rt, rv, h):
    """First crossing time (float, inf if none) of threshold h from records()."""
    N = len(roff) - 1
    out = np.full(N, np.inf)
    for i in range(N):
        v = rv[roff[i]:roff[i + 1]]
        k = np.searchsorted(v, h, side='right')   # records strictly increasing
        if k < len(v):
            out[i] = rt[roff[i] + k]
    return out


# ----------------------------------------------------------------------- concrete monitors

class Betting(Monitor):
    r"""Original betting monitor (manuscript Sec. III; Waudby-Smith & Ramdas 2024; Ville 1939).

    log W_i = sum_{j<=i} log{1 + lambda_j (X_j - r0)},
    lambda_j = min{cap, gain * max(mean(X_{j-50..j-1}) - r0, 0)},  lambda_1 = 0
    (mean of the preceding <= `window` losses; current observation excluded).
    Defaults gain=4, cap=9 < 1/r0, window=50.  Under E[X_j | F_{j-1}] <= r0, W is a
    nonnegative supermartingale, so by Ville P(sup W >= 1/delta) <= delta: native threshold
    log(1/delta) (delta = 0.02) controls the probability of ANY false alarm over an infinite
    horizon.  Weighted (IPW) version: factor 1 + lambda_j w_j (X_j - r0) with lambda_j capped
    at cap/w_j, window mean Hajek-weighted (valid if E[w_j (X_j-r0) | past] <= 0).
    """
    name = 'betting'
    supports_weights = True
    _fn = _path_betting

    def __init__(self, r0=0.10, gain=4.0, cap=9.0, window=50, delta=0.02):
        super().__init__()
        assert cap < 1 / r0
        self.delta = delta
        self.params = np.array([r0, gain, cap, window], float)

    def native_threshold(self, delta=None):
        return float(np.log(1 / (self.delta if delta is None else delta)))


class CUSUM(Monitor):
    r"""Bernoulli CUSUM (Page 1954; Reynolds & Stoumbos 1999 for integer scores).

    S_0 = 0, S_i = max(0, S_{i-1} + w_i * [X_i log(p1/r0) + (1-X_i) log((1-p1)/(1-r0))]).
    Alarm when S_i > h (strict).  integer_scale=B>0 gives the integer chart of
    theory/exact_cusum.py: miss -> +A, non-miss -> -B with A=round(log(p1/r0)*B/(-log((1-p1)/(1-r0)))),
    statistic in integer units (alarm when S > c).  No native threshold (calibrate).
    """
    supports_weights = True
    _fn = _path_cusum

    def __init__(self, p1=0.20, r0=0.10, integer_scale=0):
        super().__init__()
        self.name = f'cusum_{p1:g}' + (f'_int{integer_scale}' if integer_scale else '')
        A = B = 0.0
        if integer_scale:
            a = np.log(p1 / r0); b = np.log((1 - p1) / (1 - r0))
            s = integer_scale / (-b)
            A, B = float(int(round(a * s))), float(integer_scale)
            self.scale = s
        self.params = np.array([r0, p1, A, B], float)


class MixtureSR(Monitor):
    r"""Original mixture Shiryaev-Roberts (manuscript Eq. (1)).

    R_i^(l) = (R_{i-1}^(l) + 1) {1 + l (X_i - r0)}, R_0 = 0,
    l in {0.02, 0.05, 0.10, 0.20, 0.40} / 0.09  (0.09 = r0(1-r0)),
    S_i = log( mean_l R_i^(l) ).  This is exactly a mixture e-SR detector of Shin, Ramdas &
    Rinaldo (2024, Eqs. (26)-(27)) with bounded-mean baseline increments (Eq. (70) with
    lambda = l*r0) and uniform weights, so the native threshold log(1/alpha) guarantees
    ARL >= 1/alpha (their Thm 2.4); it does NOT bound P(false alarm within H).
    """
    name = 'msr'
    supports_weights = True
    _fn = _path_mix
    LAM = np.array([.02, .05, .10, .20, .40]) / .09

    def __init__(self, r0=0.10, stakes=None, weights=None, alpha_arl=None, cap_frac=0.99):
        super().__init__()
        st = self.LAM if stakes is None else np.asarray(stakes, float)
        assert np.all(st * r0 < 1)
        K = len(st)
        lw = np.full(K, -np.log(K)) if weights is None else np.log(np.asarray(weights, float) / np.sum(weights))
        self.alpha_arl = alpha_arl
        self.params = np.concatenate([[r0, K, 0, 0, cap_frac], st, lw]).astype(float)

    def native_threshold(self, alpha=None):
        a = self.alpha_arl if alpha is None else alpha
        return None if a is None else float(np.log(1 / a))


def compute_baseline(alpha, delta_L, delta_U, k_max=1000, v_min=0.0):
    """Shin, Ramdas & Rinaldo (2024), Algorithm 3 (computeBaseline) for the sub-exponential
    psi_E(l) = -log(1-l) - l (Eq. (72)); psi_E*(u) = u - log(1+u), grad psi_E*(u) = u/(1+u).

    Returns (lambdas, weights, info) with lambdas in the Eq. (70) parametrization (0,1),
    lambda_0 = lambda_U > ... > lambda_K = lambda_L, weights from Eq. (96).
    """
    psis = lambda u: u - np.log1p(u)
    glam = lambda u: u / (1 + u)
    DL, DU = psis(delta_L), psis(delta_U)
    lL, lU = glam(delta_L), glam(delta_U)
    if np.log(1 / alpha) <= v_min * DL:
        return np.array([lL]), np.array([1.0]), dict(g=np.log(1 / alpha), K=1, eta=np.nan)
    ks = np.arange(1, k_max + 1)
    ratio = DU / DL

    def f(g):  # Eq. (94) left-hand side
        return np.exp(-g) * (g > v_min * DU) + np.min(ks * np.exp(-g * ratio ** (-1.0 / ks)))

    lo = np.log(1 / alpha)
    hi = lo + 1.0
    while f(hi) > alpha:
        hi = lo + 2 * (hi - lo)
    # f is decreasing except for the jump of the indicator at v_min*DU; bisection on the infimum
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) <= alpha:
            hi = mid
        else:
            lo = mid
    g = hi
    K = int(ks[np.argmin(ks * np.exp(-g * ratio ** (-1.0 / ks)))])   # Eq. (95)
    eta = ratio ** (1.0 / K)
    lams = [lU]
    for k in range(1, K):
        target = DU * eta ** (-k)
        dk = brentq(lambda z: psis(z) - target, 1e-15, delta_U * 10 + 10)
        lams.append(glam(dk))
    lams.append(lL)
    ind = float(g > v_min * DU)
    Wn = np.exp(-g) * ind + K * np.exp(-g / eta)
    w = np.array([np.exp(-g) * ind / Wn] + [np.exp(-g / eta) / Wn] * K)
    keep = w > 0
    return np.array(lams)[keep], w[keep], dict(g=g, K=K, eta=eta, W=Wn)


class EDetector(Monitor):
    r"""Mixture e-detector of Shin, Ramdas & Rinaldo (2024), arXiv:2203.03532v4.

    Baseline increment for the bounded-mean pre-change class {E[X_n | F_{n-1}] <= m}, m=r0
    (Sec. 5.2, Eq. (70)):  L_n^l = 1 + l (X_n/m - 1),  l in (0,1)   [increment='betting'];
    or the exponential lower bound Eq. (72): exp{l s - psi_E(l) s^2}, s = X_n/m - 1
    [increment='exponential'].
    e-SR (Eq. (27)):    M_n(k) = L_n(k) (M_{n-1}(k) + 1);
    e-CUSUM (Eq. (28)): M_n(k) = L_n(k) max(M_{n-1}(k), 1);   M_0(k) = 0,
    mixture (Eq. (26)): M_n = sum_k w_k M_n(k);  statistic S_n = log M_n.
    Grid/weights: Algorithm 3 (computeBaseline) with Delta_L = m d/(1-m)^2,
    Delta_U = m(1-m)/d^2 (Eq. (75); d = minimum mean increase, default 0.02), ARL design
    parameter alpha, k_max=1000, v_min = min_x (x/m-1)^2 over x in {0,1} (=1 for binary losses).
    Native threshold log(1/alpha): ARL >= 1/alpha (Thm 2.4, Eqs. (15)-(16)); for e-CUSUM this
    is conservative (the paper's c_alpha may be smaller).  Can also be given explicit
    `lambdas` (Eq. (70) parametrization) and `weights`.
    """
    supports_weights = True
    _fn = _path_mix

    def __init__(self, r0=0.10, kind='SR', alpha=0.001, min_gap=0.02, increment='betting', lambdas=None,
                 weights=None, k_max=1000, v_min=None, cap_frac=0.99):
        super().__init__()
        assert kind in ('SR', 'CUSUM') and increment in ('betting', 'exponential')
        m = r0
        if lambdas is None:
            vm = min((0 / m - 1) ** 2, (1 / m - 1) ** 2) if v_min is None else v_min
            dL = m * min_gap / (1 - m) ** 2
            dU = m * (1 - m) / min_gap ** 2
            lambdas, weights, self.info = compute_baseline(alpha, dL, dU, k_max, vm)
        else:
            lambdas = np.asarray(lambdas, float)
            weights = np.full(len(lambdas), 1 / len(lambdas)) if weights is None else np.asarray(weights, float)
            self.info = {}
        self.lambdas, self.weights, self.alpha = np.asarray(lambdas), np.asarray(weights) / np.sum(weights), alpha
        stakes = self.lambdas / m            # 1 + l(X/m - 1) = 1 + (l/m)(X - m)
        self.increment = increment
        if increment == 'exponential':
            self.supports_weights = False
        self.name = f'e{kind}_{increment}'
        self.params = np.concatenate([[r0, len(stakes), 0 if kind == 'SR' else 1, 0 if increment == 'betting' else 1,
                                       cap_frac], stakes, np.log(self.weights)]).astype(float)

    def native_threshold(self, alpha=None):
        return float(np.log(1 / (self.alpha if alpha is None else alpha)))


class Periodic(Monitor):
    r"""Original periodic exact binomial reference (manuscript Sec. III).

    Checks at c = interval, 2*interval, ..., plus H if H is not a multiple.  At check c the test
    uses labels arrived in (previous check, c]: m labels, s misses,
    p = P(Bin(m, r0) >= s) (p=1 if m=0); statistic S = -log p.  Native rule p <= alpha/K with
    K = number of checks in 1..H (Bonferroni), i.e. S >= -log(alpha/K); implemented as
    S > -log(alpha/K) - 1e-9 (tolerance for round-off).  A calibrated threshold on S is the
    'matched' version (exact FA matching in expectation with randomized tie-breaking).
    With restart (ARL runs) checks continue on the same production schedule.
    """
    name = 'periodic'
    _fn = _path_periodic

    def __init__(self, r0=0.10, interval=3000, alpha=0.02):
        super().__init__()
        self.alpha, self.interval = alpha, interval
        self.params = np.array([r0, interval], float)

    def n_checks(self, H):
        return len(np.unique(np.r_[np.arange(self.interval, H + 1, self.interval), H]))

    def extra_outputs(self, H):
        return self.n_checks(H) + 2

    def native_threshold(self, H=None, alpha=None):
        assert H is not None, 'Periodic.native_threshold needs H'
        a = self.alpha if alpha is None else alpha
        return float(-np.log(a / self.n_checks(H)) - 1e-9)


class EWMA(Monitor):
    r"""Bernoulli EWMA chart (Roberts 1959; one-sided upper limit).

    Z_0 = z0 (default r0), Z_i = (1-lam) Z_{i-1} + lam X_i (lam in label units, default 0.05),
    S_i = (Z_i - r0) / sigma_i, sigma_i^2 = r0(1-r0) lam/(2-lam) [1 - (1-lam)^{2i}]
    (exact variance under constant r0; asymptotic=True drops the bracket).  Alarm S_i > L.
    Textbook L=3 is returned as native_threshold (no finite-horizon FA guarantee; calibrate).
    """
    _fn = _path_ewma

    def __init__(self, r0=0.10, lam=0.05, z0=None, asymptotic=False, L=3.0):
        super().__init__()
        self.name = f'ewma_{lam:g}'
        self.L = L
        self.params = np.array([r0, lam, r0 if z0 is None else z0, 1.0 if asymptotic else 0.0], float)

    def native_threshold(self):
        return float(self.L)


class GLRCUSUM(Monitor):
    r"""Windowed Bernoulli GLR-CUSUM for unknown r1 > r0 (Lorden 1971; Lai 1995 window-limited GLR).

    S_n = max_{n-W < k <= n} sup_{p > r0} sum_{i=k}^n log[p^X_i (1-p)^{1-X_i} / (r0^X_i (1-r0)^{1-X_i})]
        = max_k { s log(p^/r0) + (m-s) log((1-p^)/(1-r0)) } with m = n-k+1, s = misses, p^ = s/m,
    term set to 0 when p^ <= r0 (constrained MLE).  Only segments starting with a miss are
    evaluated (exact: dropping a leading non-miss increases the sup-LLR).  Default W = 500
    labels (covers >= 25 expected misses at r1=0.05..0.2; documented choice).  Calibrate.
    """
    _fn = _path_glr

    def __init__(self, r0=0.10, window=500):
        super().__init__()
        self.name = f'glr_W{window}'
        self.params = np.array([r0, window], float)


class WindowBinomial(Monitor):
    r"""Sliding-window exact binomial test: at every arrival n, with the last m = min(n, W)
    labels containing s misses, S_n = -log P(Bin(m, r0) >= s) (scipy table).  Calibrate."""
    _fn = _path_window_binom

    def __init__(self, r0=0.10, window=200):
        super().__init__()
        self.name = f'winbin_W{window}'
        m = np.arange(window + 1)[:, None]; s = np.arange(window + 2)[None, :]
        with np.errstate(divide='ignore'):
            t = -binom.logsf(s - 1, m, r0)
        t[:, 0] = 0.0
        t = np.where(s > m, np.inf, t)
        self.table = np.ascontiguousarray(np.maximum(t, 0.0))
        self.params = np.array([r0, window], float)


class ADWIN(Monitor):
    r"""One-sided ADWIN (Bifet & Gavalda 2007, ADWIN2 cut bound).

    Window = all labels since start (optionally the last `max_window`); no cuts before the first
    alarm (one-sided adaptation: we stop at the first increase).  For each split into older W0
    (n0) and newer W1 (n1) sub-windows with n0, n1 >= min_len, m = 1/(1/n0+1/n1) and
    sigma^2 = window variance, ADWIN2 cuts when
        mu1 - mu0 > eps = sqrt(2 sigma^2 ln(2/delta') / m) + 2 ln(2/delta') / (3m),  delta' = delta/ln n.
    Solving for ln(2/delta') gives the threshold-free statistic
        S_n = max_splits L*(split) - ln(2 ln n),   alarm  S_n > ln(1/delta)  <=>  ADWIN2 increase cut.
    Splits are taken on a geometric grid n1 <- max(n1+1, floor(n1(1+1/M))) (resolution of the
    ADWIN2 exponential histogram with M buckets per level; M=5 default).  Native delta=0.002.
    """
    _fn = _path_adwin

    def __init__(self, delta=0.002, min_len=5, M=5, max_window=0, clock=1):
        super().__init__()
        self.name = 'adwin'
        self.delta = delta
        self.params = np.array([min_len, M, max_window, clock], float)

    def native_threshold(self, delta=None):
        return float(np.log(1 / (self.delta if delta is None else delta)))


class DDM(Monitor):
    r"""Drift Detection Method (Gama, Medas, Castillo & Rodrigues 2004).

    p_i = (#misses in first i labels)/i, s_i = sqrt(p_i(1-p_i)/i).  After i >= n_min (30), the
    minimum (p_min, s_min) of p_i + s_i is updated (<=) and the drift level is
    p_i + s_i > p_min + 3 s_min.  Statistic S_i = (p_i + s_i - p_min)/s_min (inf if s_min = 0 and
    p_i + s_i > p_min); native threshold 3 (warning level 2).  One-sided by construction.
    Known faithful flaw: if the first n_min losses are all 0 then s_min = 0 and the first miss
    alarms (P = (1-r)^30, ~21% at r=0.05).
    """
    name = 'ddm'
    _fn = _path_ddm

    def __init__(self, n_min=30, level=3.0):
        super().__init__()
        self.level = level
        self.params = np.array([n_min], float)

    def native_threshold(self):
        return float(self.level)


class CSDetector(Monitor):
    r"""Backward confidence-sequence (BCS) detector with one-sided betting confidence sequences.

    Choice (documented): following the reduction of change detection to sequential estimation
    (Shekhar & Ramdas 2023, arXiv:2309.09111), a new one-sided confidence sequence for the mean
    is started every G labels; the K most recent remain active.  Start j uses the WSR (Waudby-
    Smith & Ramdas 2024, JRSSB) predictable plug-in betting capital for H0: mean <= r0,
        K_n^(j) = prod_{i=j}^n {1 + l_i (X_i - r0)},
        l_i = min{ sqrt(2 log(1/a) / (sig2_{t-1} t log(1+t))), c/r0 },  t = i-j+1,
        mu_t = (1/2 + sum X)/(t+1),  sig2_t = (1/4 + sum (X_i - mu_i)^2)/(t+1),  sig2_0 = 1/4.
    The lower confidence bound of start j exceeds r0 iff K^(j) >= 1/a, so the alarm 'some recent
    LCB > r0' is S_n = max_j log K_n^(j) > log(1/a).  Stakes are designed with alpha_design;
    native threshold log(1/alpha_design): each start is a level-a test (Ville), and since
    sum_j K^(j) is an e-detector with starts every G labels, ARL (in labels) >= G(1/a - 1).
    """
    name = 'cs_detector'
    _fn = _path_cs

    def __init__(self, r0=0.10, alpha_design=0.02, start_every=25, n_active=40, c=0.5):
        super().__init__()
        self.alpha_design = alpha_design
        self.params = np.array([r0, alpha_design, start_every, n_active, c], float)

    def native_threshold(self):
        return float(np.log(1 / self.alpha_design))


def original_monitors(r0=0.10):
    """The five monitors of the original study, in archived threshold order."""
    return [Betting(r0), CUSUM(0.12, r0), CUSUM(0.20, r0), MixtureSR(r0), Periodic(r0)]


def default_suite(r0=0.10):
    """All monitors with default settings (name -> Monitor)."""
    ms = original_monitors(r0) + [EDetector(r0, 'SR'), EDetector(r0, 'CUSUM'), EWMA(r0), GLRCUSUM(r0),
                                  WindowBinomial(r0), ADWIN(), DDM(), CSDetector(r0)]
    names = ['betting', 'cusum_0.12', 'cusum_0.2', 'msr', 'periodic', 'eSR', 'eCUSUM', 'ewma', 'glr',
             'window_binomial', 'adwin', 'ddm', 'cs_detector']
    return dict(zip(names, ms))
