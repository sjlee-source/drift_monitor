"""Window likelihood-ratio scan (E1, new monitor).

At every label arrival t (arrival item of the label), the statistic is the fixed-alternative
log-likelihood ratio of r1_design vs r0 over the labels that have ARRIVED by t and whose
ORIGIN lies in the origin window [t - D + 1, t]:

    S_t = s * log(r1/r0) + (m - s) * log((1 - r1)/(1 - r0)),
    m = #such labels, s = #misses among them.

With constant delay d, labels arrived by t have origin <= t - d, so the window is the last
(D - d) origins [t - D + 1, t - d] -- exactly the data set of the oracle Neyman-Pearson test in
theory/converse.py when t = tau + D - 1.  Alarm: first arrival t with S_t > h (strict),
alarm time = arrival item of the triggering label.  Calibrate h on the per-stream max of S
(no native threshold).  The window is defined on origins, so the kernel handles general
(out-of-order) delays: a label is added at its arrival only if its origin is still in the
window, and removed (in origin order) when its origin leaves the window.

The class duck-types revision.monitors.Monitor (evaluate / max_stat / alarm_grid /
restart_run / path), so revision.calibration helpers (alarm_times, fa_from_maxima,
tradeoff_curve, arl0) work unchanged.
"""
import math
import numpy as np
from numba import njit, prange

NO_ALARM = np.int64(2 ** 62)


@njit
def _path(origin, items, loss, H, t0, D, la, lb, h_stop, out_t, out_s):
    """One stream; labels in arrival order.  Only labels with arrival > t0 are used."""
    n = len(items)
    order = np.argsort(origin, kind='mergesort')   # removal order (by origin)
    added = np.zeros(n, np.bool_)
    rp = 0                 # pointer into `order`
    m = 0; s = 0; k_out = 0
    for i in range(n):
        t = items[i]
        if t <= t0:
            continue
        lo = t - D + 1
        # remove labels whose origin left the window
        while rp < n and origin[order[rp]] < lo:
            j = order[rp]
            if added[j]:
                added[j] = False
                m -= 1
                s -= loss[j]
            rp += 1
        if origin[i] >= lo:
            added[i] = True
            m += 1
            s += loss[i]
        val = s * la + (m - s) * lb
        out_t[k_out] = t; out_s[k_out] = val; k_out += 1
        if val > h_stop:
            return k_out
    return k_out


@njit(parallel=True)
def _drv_eval(offsets, origin, items, loss, H, D, la, lb, ths, cps, stop, alarm, mx):
    N = len(offsets) - 1
    G = len(ths); C = len(cps)
    for i in prange(N):
        a = offsets[i]; b = offsets[i + 1]
        ot = np.empty(b - a + 1, np.int64); os_ = np.empty(b - a + 1)
        n = _path(origin[a:b], items[a:b], loss[a:b], H, 0, D, la, lb, stop, ot, os_)
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
def _drv_restart(offsets, origin, items, loss, H, D, la, lb, h, n_alarm, last_t, first_t):
    N = len(offsets) - 1
    for i in prange(N):
        a = offsets[i]; b = offsets[i + 1]
        ot = np.empty(b - a + 1, np.int64); os_ = np.empty(b - a + 1)
        tl = 0; na = 0; ft = NO_ALARM
        while True:
            n = _path(origin[a:b], items[a:b], loss[a:b], H, tl, D, la, lb, h, ot, os_)
            if n == 0 or not (os_[n - 1] > h):
                break
            tl = ot[n - 1]; na += 1
            if ft == NO_ALARM:
                ft = tl
            if tl >= H:
                break
        n_alarm[i] = na; last_t[i] = tl; first_t[i] = ft


class WindowLR:
    supports_weights = False

    def __init__(self, D=10000, r1=0.20, r0=0.10):
        self.D, self.r1, self.r0 = int(D), float(r1), float(r0)
        self.la = math.log(r1 / r0); self.lb = math.log((1 - r1) / (1 - r0))
        self.name = f'winlr_D{D}'

    def native_threshold(self):
        return None

    def path(self, origin, arrival, loss, H=None, t0=0):
        arrival = np.ascontiguousarray(arrival, np.int64); origin = np.ascontiguousarray(origin, np.int64)
        loss = np.ascontiguousarray(loss, np.int8)
        H = int(arrival[-1]) if (H is None and len(arrival)) else int(H or 0)
        ot = np.empty(len(arrival) + 1, np.int64); os_ = np.empty(len(arrival) + 1)
        n = _path(origin, arrival, loss, H, t0, self.D, self.la, self.lb, np.inf, ot, os_)
        return ot[:n].copy(), os_[:n].copy()

    def evaluate(self, batch, thresholds=(), checkpoints=(), weights=None):
        assert weights is None
        ths = np.asarray(thresholds, float).ravel(); order = np.argsort(ths, kind='stable')
        cps = np.asarray(checkpoints, np.int64).ravel(); corder = np.argsort(cps, kind='stable')
        N = batch.n
        alarm = np.empty((N, len(ths)), np.int64); mx = np.empty((N, len(cps)))
        stop = np.inf if len(cps) or not len(ths) else float(ths.max())
        _drv_eval(batch.offsets, batch.origin, batch.arrival, batch.loss, int(batch.H), self.D, self.la, self.lb,
                  ths[order], cps[corder], stop, alarm, mx)
        a = np.empty_like(alarm); a[:, order] = alarm
        m = np.empty_like(mx); m[:, corder] = mx
        a = a.astype(float); a[a >= NO_ALARM] = np.inf
        return dict(alarm=a, max=m)

    def max_stat(self, batch, horizon=None, weights=None):
        return self.evaluate(batch, (), (batch.H if horizon is None else horizon,))['max'][:, 0]

    def alarm_grid(self, batch, grid, weights=None):
        return self.evaluate(batch, grid, ())['alarm']

    def restart_run(self, batch, h, weights=None):
        N = batch.n
        na = np.zeros(N, np.int64); lt = np.zeros(N, np.int64); ft = np.zeros(N, np.int64)
        _drv_restart(batch.offsets, batch.origin, batch.arrival, batch.loss, int(batch.H), self.D, self.la, self.lb,
                     float(h), na, lt, ft)
        f = ft.astype(float); f[ft >= NO_ALARM] = np.inf
        return dict(n_alarms=na, last_alarm=lt, first_alarm=f)

    def __repr__(self):
        return f'WindowLR({self.name})'


def naive_path(origin, arrival, loss, D, r1=0.20, r0=0.10, t0=0):
    """Literal O(n^2) reference: at each arrival t > t0, recount labels with arrival in (t0, t] and
    origin in [t-D+1, t]  (labels sharing the arrival item are added one at a time, in order)."""
    la = math.log(r1 / r0); lb = math.log((1 - r1) / (1 - r0))
    ts, vs = [], []
    for i in range(len(arrival)):
        t = arrival[i]
        if t <= t0:
            continue
        m = s = 0
        for j in range(i + 1):
            if arrival[j] > t0 and origin[j] >= t - D + 1:
                m += 1; s += int(loss[j])
        ts.append(t); vs.append(s * la + (m - s) * lb)
    return np.array(ts, np.int64), np.array(vs)
