"""E3 follow-up kernel (copy of kernel.py v1 + no-leakage model, weight clip, extra stats).

E3 fused numba kernel: production stream + audit policy + online monitors (experiment id 13).

Item model (per item t = 1..H; random numbers from revision.rng components, fixed draws/item):
  Z_t ~ Bern(pi)                                  (C_DEFECT, 1 uniform)
  S_t = Z_t * mu1_t + N(0,1)                      (C_SCORE, 1 Box-Muller normal = 2 uniforms)
  inspector passes the item iff S_t < c; miss M_t = Z_t * 1[S_t < c]
  => P(M_t = 1 | Z_t = 1) = Phi(c - mu1_t) = r_t exactly (mu1_t = c - Phi^-1(r_t)).
  audit A_t ~ Bern(p_t)                           (C_AUDIT, 1 uniform)
Conditions differing only in r share Z, the normal draws and the audit uniforms (CRN); misses are
monotonically coupled (lower mu1 -> lower S).

Workflows
  W-A (wf=0): any item may be audited; an audited defective item yields label X = M_t.
  W-B (wf=1): flagged items (S >= c) are always reviewed for free (a flagged defective gives a
       caught-defect label X=0, weight 1); audits target passed items only (S < c); an audited
       passed defective gives a miss label X=1 with weight 1/p_t.  Same delay d for both labels.

Monitor increments (per label): D = wt * (X - r0).  Every label carries (wt, b) where b is an
F_{t-1}-measurable bound on the weight of any label that could have a negative increment
(X=0) from item t; stakes are capped at capf/(r0*b) so every factor is >= 1-capf > 0.
  noninformative W-A policies (P1-P3): wt = 1, b = 1 (plain original monitors).
  informative W-A policies (P4-P6): wt = qbar/p_t (qbar = B/H), b = qbar/min_{s>=c} p_t(s).
  W-B: caught wt = 1, b = 1; miss wt = 1/p_t (only positive increments carry large weights).
Under any null path with r_t <= r0, E[D_t | F_{t-1}] = const * pi (r_t - r0) <= 0 at the item level,
so with predictable stakes both monitors are nonnegative supermartingale / e-SR constructions.

Policies (ptype):
  0 exact-uniform   p = Brem_scope / items_remaining_scope                  (W-A P1, P2)
  1 warning-boost   q = boost if mSR stat >= h_warn else base; p=0 if scope budget exhausted;
                    p=1 if items_remaining_scope <= Brem_scope (forced)       (P3, original)
  2 paced targeted  p = clip(g(S) * kappa, eps, 1), kappa = (Brem/items_rem)/qbar;
                    p=0 if exhausted; p=1 if items_rem <= Brem (W-A only)    (P4-P6, W-B P1/P2)
  g: gtype 0 const G[0]; 1 strata G[searchsorted(E, S)]; 2 a * u(1-u), u = P(Z=1|S) nominal.
  W-B: items with S >= c are not eligible (p = 0, not charged).
"""
import math
import numpy as np
from numba import njit, prange
from revision import rng as R

# stats columns
ST = ['n_audits', 'n_forced', 'n_floor_viol', 'n_labels', 'n_audit_inf', 'aud_win', 'inf_win_arr',
      'miss_lab_win', 'sw_win', 'sw2_win', 'aud_pre', 'true_miss_win', 'def_win', 'warn_pre', 'warn_win',
      'sum_wx', 'sum_w', 'aud_inf_win', 'free_lab_win', 'aud_win_arrived', 'n_flagged', 'first_exhaust', 'max_w', 'aud_post']
NST = len(ST)

LAM = np.array([.02, .05, .10, .20, .40]) / .09


@njit
def _Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit
def _gval(s, gtype, E, G, a_unc, pi, mu_ref, ok, cpass, a0, a1r, bsl):
    if not ok:
        return 0.0
    if gtype == 0:
        return G[0]
    if gtype == 1:
        return G[np.searchsorted(E, s)]
    # posterior defect probability under nominal model
    lr = mu_ref * s - 0.5 * mu_ref * mu_ref
    if cpass:   # no-leakage model, W-B: P(Z=1 | S, passed)
        f = math.exp(lr) * _Phi(a1r - bsl * s) / _Phi(a0 - bsl * s) if lr < 700 else 1e300
        u = pi * f / (pi * f + 1.0 - pi)
    else:
        u = 1.0 / (1.0 + (1.0 - pi) / pi * math.exp(-lr)) if lr > -700 else 0.0
    return a_unc * u * (1.0 - u)


@njit
def _logaddexp0(x):
    # log(exp(x) + 1)
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


@njit
def run_one(st, H, tau, pi, mu1, c, d, wf, ptype, scope_block, blk, B, qbar, eps, base, boost,
            h_warn, gtype, E, G, a_unc, gmin_flag, ipw, mu_ref, r0, lam, gain, bcap, win, capf,
            th, Dwin, stats, ws, model, a0, a1, a1r, bsl, wclip):
    """One stream.  th = [msr_hi, msr_lo, bet_hi, bet_lo]; returns (max_msr, max_bet, a_msr_hi,
    a_msr_lo, a_bet_hi, a_bet_lo).  ws: scratch arrays (arrival, x, wt, b) of length H."""
    sd = st[R.C_DEFECT].copy(); sa = st[R.C_AUDIT].copy(); ss = st[R.C_SCORE].copy(); smi = st[R.C_MISS].copy()
    l_arr = ws[0]; l_x = ws[1]; l_w = ws[2]; l_b = ws[3]
    nl = 0; head = 0
    K = len(lam)
    logR = np.full(K, -np.inf)
    # betting
    bx = np.zeros(win); bw = np.zeros(win); sx = 0.0; sw = 0.0; cnt = 0; nb = 0
    logW = 0.0
    smsr = -np.inf
    mx_m = -np.inf; mx_b = 0.0
    a_m_hi = np.inf; a_m_lo = np.inf; a_b_hi = np.inf; a_b_lo = np.inf
    nblk = (H + blk - 1) // blk
    naud = 0; brem_h = B
    # block budgets: round(qbar*end) - round(qbar*start) with qbar*H*... use B/H scaling
    cur_blk = -1; brem_b = 0; blk_end = 0
    win_lo = tau; win_hi = tau + Dwin - 1
    for k in range(stats.shape[0]):
        stats[k] = 0.0
    stats[21] = H + 1
    for t in range(1, H + 2):
        # ---- process labels arrived by t-1
        while head < nl and l_arr[head] <= t - 1:
            x = l_x[head]; w = l_w[head]; b = l_b[head]; ta = l_arr[head]
            inc = w * (x - r0)
            # mixture SR (log domain)
            for j in range(K):
                sj = lam[j]
                cj = capf / (r0 * b)
                if sj > cj:
                    sj = cj
                lr = math.log1p(sj * inc)
                if logR[j] == -np.inf:
                    logR[j] = lr
                else:
                    logR[j] = _logaddexp0(logR[j]) + lr
            m = logR[0]
            for j in range(1, K):
                if logR[j] > m:
                    m = logR[j]
            tot = 0.0
            for j in range(K):
                tot += math.exp(logR[j] - m)
            smsr = m + math.log(tot / K)
            if smsr > mx_m:
                mx_m = smsr
            if smsr > th[0] and a_m_hi == np.inf:
                a_m_hi = ta
            if smsr > th[1] and a_m_lo == np.inf:
                a_m_lo = ta
            # betting (Hajek window mean of past labels)
            prev = sx / sw if cnt > 0 else 0.0
            lb = gain * (prev - r0)
            if lb < 0.0:
                lb = 0.0
            if lb > bcap:
                lb = bcap
            cb = bcap / b
            if lb > cb:
                lb = cb
            logW += math.log1p(lb * inc)
            if logW > mx_b:
                mx_b = logW
            if logW > th[2] and a_b_hi == np.inf:
                a_b_hi = ta
            if logW > th[3] and a_b_lo == np.inf:
                a_b_lo = ta
            jj = nb % win
            if cnt == win:
                sx -= bx[jj] * bw[jj]; sw -= bw[jj]
            else:
                cnt += 1
            bx[jj] = x; bw[jj] = w; sx += x * w; sw += w
            nb += 1
            if cnt == win and nb % 4096 == 0:
                sx = 0.0; sw = 0.0
                for q in range(win):
                    sx += bx[q] * bw[q]; sw += bw[q]
            head += 1
        if t > H:
            break
        # ---- latent item
        z = R.uniform(sd) < pi
        s = R.normal(ss)
        if z:
            s += mu1[t - 1]
        ua = R.uniform(sa)
        if model == 0:
            passed = s < c
        else:
            upm = R.uniform(smi)
            passed = upm < _Phi((a1[t - 1] if z else a0) - bsl * s)
        if not passed:
            stats[20] += 1
        mm = z and passed
        inwin = (t >= win_lo) and (t <= win_hi)
        if inwin:
            if z:
                stats[12] += 1
            if mm:
                stats[11] += 1
        # ---- budget scope
        if scope_block:
            bi = (t - 1) // blk
            if bi != cur_blk:
                cur_blk = bi
                st_i = bi * blk
                blk_end = min(H, st_i + blk)
                brem_b = int(round(B * blk_end / H)) - int(round(B * st_i / H))
            brem = brem_b
            irem = blk_end - t + 1
        else:
            brem = brem_h
            irem = H - t + 1
        eligible = True if wf == 0 else passed
        warn = smsr >= h_warn
        if warn:
            if t < tau:
                stats[13] += 1
            elif inwin:
                stats[14] += 1
        forced = False
        bnd = 1.0
        if not eligible:
            p = 0.0
        elif brem <= 0:
            p = 0.0
            if stats[21] > H:
                stats[21] = t
        elif ptype == 0:
            p = brem / irem
            if p >= 1.0:
                p = 1.0
                forced = True
        elif ptype == 1:
            if wf == 0 and irem <= brem:
                p = 1.0; forced = True
            else:
                p = boost if warn else base
        else:
            if wf == 0 and irem <= brem:
                p = 1.0; forced = True
                bnd = qbar
            else:
                kap = (brem / irem) / qbar
                g = _gval(s, gtype, E, G, a_unc, pi, mu_ref, True, model == 1 and wf == 1, a0, a1r, bsl)
                p = g * kap
                if p < eps:
                    p = eps
                if p > 1.0:
                    p = 1.0
                pm = gmin_flag * kap
                if pm < eps:
                    pm = eps
                if pm > 1.0:
                    pm = 1.0
                bnd = qbar / pm
        if eligible and p < eps:
            stats[2] += 1
        a = ua < p
        if a:
            naud += 1
            stats[0] += 1
            if forced:
                stats[1] += 1
            if scope_block:
                brem_b -= 1
            brem_h -= 1
            if t < tau:
                stats[10] += 1
            else:
                stats[23] += 1
            if inwin:
                stats[5] += 1
                if t + d <= win_hi:
                    stats[19] += 1
        # ---- labels
        ar = t + d
        lab = False; x = 0.0; wt = 1.0; bb = 1.0; free = False
        if wf == 0:
            if a and z:
                lab = True
                x = 1.0 if mm else 0.0
                if ipw:
                    wt = qbar / p; bb = bnd
                    if wt > wclip:
                        wt = wclip
                    if bb > wclip:
                        bb = wclip
        else:
            if z and not passed:
                lab = True; free = True; x = 0.0; wt = 1.0; bb = 1.0
            elif z and a:
                lab = True; x = 1.0; wt = 1.0 / p; bb = 1.0
                if wt > wclip:
                    wt = wclip
        if lab:
            if not free:
                stats[4] += 1
                if inwin:
                    stats[17] += 1
            if ar <= H:
                l_arr[nl] = ar; l_x[nl] = x; l_w[nl] = wt; l_b[nl] = bb; nl += 1
                stats[3] += 1
                stats[15] += wt * x; stats[16] += wt
                if wt > stats[22]:
                    stats[22] = wt
                if inwin and ar <= win_hi:
                    stats[6] += 0.0 if free else 1.0
                    stats[18] += 1.0 if free else 0.0
                    if x == 1.0:
                        stats[7] += 1
                    stats[8] += wt; stats[9] += wt * wt
    return mx_m, mx_b, a_m_hi, a_m_lo, a_b_hi, a_b_lo


@njit(parallel=True)
def run_batch(states, H, tau, pi, mu1, c, d, wf, ptype, scope_block, blk, B, qbar, eps, base, boost,
              h_warn, gtype, E, G, a_unc, gmin_flag, ipw, mu_ref, r0, lam, gain, bcap, win, capf, th, Dwin,
              model, a0, a1, a1r, bsl, wclip):
    N = states.shape[0]
    out = np.empty((N, 6))
    stats = np.zeros((N, NST))
    for i in prange(N):
        ws = np.empty((4, H + 1))
        r = run_one(states[i], H, tau, pi, mu1, c, d, wf, ptype, scope_block, blk, B, qbar, eps, base, boost,
                    h_warn, gtype, E, G, a_unc, gmin_flag, ipw, mu_ref, r0, lam, gain, bcap, win, capf,
                    th, Dwin, stats[i], ws, model, a0, a1, a1r, bsl, wclip)
        for k in range(6):
            out[i, k] = r[k]
    return out, stats


@njit
def labels_one(st, H, tau, pi, mu1, c, d, wf, ptype, scope_block, blk, B, qbar, eps, base, boost,
               h_warn, gtype, E, G, a_unc, gmin_flag, ipw, mu_ref, r0, lam, gain, bcap, win, capf, th, Dwin,
               model, a0, a1, a1r, bsl, wclip):
    """Debug: return the label arrays of one stream (for checks against library monitors)."""
    ws = np.empty((4, H + 1))
    stats = np.zeros(NST)
    r = run_one(st, H, tau, pi, mu1, c, d, wf, ptype, scope_block, blk, B, qbar, eps, base, boost,
                h_warn, gtype, E, G, a_unc, gmin_flag, ipw, mu_ref, r0, lam, gain, bcap, win, capf, th, Dwin,
                stats, ws, model, a0, a1, a1r, bsl, wclip)
    n = int(stats[3])
    return ws[:, :n].copy(), r, stats
