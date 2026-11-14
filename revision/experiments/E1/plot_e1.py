"""E1 figures (run with /usr/bin/python3: matplotlib + numpy, csv module only). PNG + PDF in exp_E1/figures/."""
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.environ.get('E1_ROOT', '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1')
FIG = os.environ.get('E1_FIG', os.path.join(ROOT, 'figures')); os.makedirs(FIG, exist_ok=True)
CORE = ['cusum_0.12', 'cusum_0.20', 'betting', 'msr', 'eSR', 'ewma_0.05', 'glr_W500', 'winbin_W200', 'winlr_D10000',
        'periodic', 'adwin', 'ddm']
SUPP = ['eCUSUM', 'cs_detector']
cmap = plt.get_cmap('tab20')
COL = {m: cmap(i) for i, m in enumerate(CORE + SUPP)}
TRADEOFF = [(0.10, 0.20, 0.20, 500), (0.10, 0.12, 1.00, 500), (0.01, 0.30, 1.00, 0), (0.10, 0.30, 0.10, 500)]


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=150)
    plt.close(fig)


oc = rd(os.path.join(ROOT, 'oracle_curves.csv'))
key = lambda c: tuple(round(float(x), 4) for x in c)

# 1. trade-off curves
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, (pi, r1, q, d) in zip(axs.flat, TRADEOFF):
    t = rd(os.path.join(ROOT, 'raw', f'tradeoff_pi{pi:g}_r{r1:g}_q{q:g}_d{d}.csv'))
    by = defaultdict(list)
    for r in t:
        by[r['monitor']].append((float(r['fa_comp']), float(r['pD_10000'])))
    for m in CORE + SUPP:
        if m not in by:
            continue
        xy = sorted(by[m]); x, y = zip(*xy)
        ax.plot(x, y, color=COL[m], lw=1.2 if m in CORE else 0.8, ls='-' if m in CORE else ':', label=m)
    o = sorted((float(r['alpha']), float(r['beta'])) for r in oc if r['kind'] == 'fa' and
               key((r['pi'], r['r1'], r['q'], r['d'])) == key((pi, r1, q, d)))
    ax.plot(*zip(*o), 'k--', lw=2, label='oracle NP bound β*')
    ax.axvline(0.02, color='grey', lw=0.6)
    ax.set_xscale('log'); ax.set_xlim(1e-3, 0.5); ax.set_ylim(0, 1.01)
    ax.set_xlabel('composite FA = max(FA const 0.10, FA 0.05→0.10)'); ax.set_ylabel('p_D, D = 10,000')
    ax.set_title(f'π={pi}, r1={r1}, q={q}, d={d}', fontsize=10)
axs.flat[0].legend(fontsize=6.5, ncol=2, loc='lower right')
save(fig, 'fig_tradeoff_4conditions')

# 2. p_D vs q with oracle envelope (D = 10k), all scenarios; fine grid for pi=.1, r1=.2
ml = [r for r in rd(os.path.join(ROOT, 'metrics_long.csv')) if r['D'] == '10000' and r['family'] == 'core']
fq = rd(os.path.join(ROOT, 'metrics_fineq.csv')) if os.path.exists(os.path.join(ROOT, 'metrics_fineq.csv')) else []
fq = [r for r in fq if r['D'] == '10000']
fig, axs = plt.subplots(2, 6, figsize=(24, 8.5), sharey=True)
for i, (pi, r1, d) in enumerate([(pi, r1, d) for pi in (0.01, 0.1) for r1 in (0.12, 0.2, 0.3) for d in (0, 500)]):
    ax = axs.flat[i]
    for m in CORE:
        pts = sorted((float(r['q']), float(r['pD'])) for r in ml + fq if r['variant'] == m and
                     key((r['pi'], r['r1'], r['d'])) == key((pi, r1, d)))
        if pts:
            ax.plot(*zip(*pts), marker='.', ms=3, lw=1, color=COL[m], label=m)
    o = sorted((float(r['q']), float(r['beta'])) for r in oc if r['kind'] == 'q' and key((r['pi'], r['r1'], r['d'])) == key((pi, r1, d)))
    ax.plot(*zip(*o), 'k--', lw=2, label='oracle β*(0.02)')
    ax.axhline(0.8, color='grey', lw=0.6); ax.set_xscale('log'); ax.set_xlim(0.09, 1.05); ax.set_ylim(0, 1.01)
    ax.set_title(f'π={pi}, r1={r1}, d={d}' + (' (fine q grid)' if (pi, r1) == (0.1, 0.2) else ''), fontsize=9)
    ax.set_xlabel('audit rate q')
axs.flat[0].set_ylabel('p_D, D = 10,000'); axs.flat[6].set_ylabel('p_D, D = 10,000')
axs.flat[0].legend(fontsize=6)
save(fig, 'fig_pD_vs_q_oracle')

# 2b. zoom on fine grid with CIs
if fq:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, d in zip(axs, (0, 500)):
        for m in CORE:
            pts = sorted((float(r['q']), float(r['pD']), float(r['pD_lo']), float(r['pD_hi'])) for r in ml + fq
                         if r['variant'] == m and key((r['pi'], r['r1'], r['d'])) == key((0.1, 0.2, d)) and float(r['q']) <= 0.5)
            if pts:
                q_, p_, lo, hi = map(np.array, zip(*pts))
                ax.errorbar(q_, p_, yerr=[p_ - lo, hi - p_], marker='.', ms=3, lw=1, capsize=1.5, color=COL[m], label=m)
        o = sorted((float(r['q']), float(r['beta'])) for r in oc if r['kind'] == 'q' and key((r['pi'], r['r1'], r['d'])) == key((0.1, 0.2, d)))
        ax.plot(*zip(*o), 'k--', lw=2, label='oracle β*(0.02)')
        ax.axhline(0.8, color='grey', lw=0.6); ax.set_xlim(0.09, 0.52); ax.set_xlabel('audit rate q')
        ax.set_title(f'π=0.1, r1=0.2, d={d}, D=10k (95% CP bars)', fontsize=10)
    axs[0].set_ylabel('p_D'); axs[0].legend(fontsize=6, ncol=2)
    save(fig, 'fig_pD_vs_q_finegrid')

# 3. timely-delay distributions: P(tau <= T <= tau + t)
de = rd(os.path.join(ROOT, 'delay_ecdf.csv'))
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, (pi, r1, q, d) in zip(axs.flat, TRADEOFF):
    for m in CORE + SUPP:
        pts = sorted((float(r['t']), float(r['cdf'])) for r in de if r['monitor'] == m and key((r['pi'], r['r1'], r['q'], r['d'])) == key((pi, r1, q, d)))
        if pts:
            ax.plot(*zip(*pts), color=COL[m], lw=1.2 if m in CORE else 0.8, ls='-' if m in CORE else ':', label=m)
    for D in (5000, 10000, 20000):
        ax.axvline(D, color='grey', lw=0.5)
    ax.set_xlabel('t = items after change'); ax.set_ylabel('P(τ ≤ T ≤ τ + t)')
    ax.set_title(f'π={pi}, r1={r1}, q={q}, d={d}', fontsize=10); ax.set_ylim(0, 1.01)
axs.flat[0].legend(fontsize=6.5, ncol=2)
save(fig, 'fig_delay_ecdf')
print('figures written to', FIG)
