# `revision/`: shared simulation library for the IEEE Access revision

This is the library for all revision experiments: baselines, robustness, adaptive auditing and real-data replay. It monitors the conditional false-negative risk r_t = P(miss | defective) from sparse, delayed audit labels.

Hypothesis: the composite admissible null is r_t ≤ r0 = 0.10. Three null paths are used:
- calibration/validation at the boundary (constant r = 0.10 over 1..H);
- FA reported under 0.05 → 0.10 at τ;
- FA reported under constant 0.05.

Harmful scenarios use r = 0.05 before τ and r1 after.

Nothing outside `revision/` was modified. The original `simulation/run.py` is only *imported* (by `legacy.py`) for bit-exact checks.

## Setup

```bash
/usr/bin/python3 -m venv ~/.venvs/drift-rev
~/.venvs/drift-rev/bin/pip install numpy scipy pandas numba pytest   # numpy 2.5.4, scipy 1.18.1, numba 0.68.1, pandas 3.1.1
cd /home/sjlee/Research/drift_monitor
export PYTHONPATH=$PWD NUMBA_NUM_THREADS=8          # <= 8 cores on the shared machine
~/.venvs/drift-rev/bin/python -m pytest -q revision/tests     # 63 tests, ~3.5 min, peak RSS ~2.1 GB
~/.venvs/drift-rev/bin/python -m revision.bench --n 20000 --q 0.5
```

The library is CPU only and does not use the GPU. The first call of each kernel spends about 1 s in numba compilation. Kernels are not cached on disk, so a script's first call is slow.

## Modules and API

### `rng.py`: seeding and common random numbers (CRN)

```python
stream_states(role, experiment, condition, replicates, master=20261111) -> uint64 (N, 8, 4)
ROLES = {'calibration':1, 'validation':2, 'evaluation':3, 'pilot':4, 'arl':5, 'replay':6, 'test':7, 'sweep':8}
```

Stream key: `SeedSequence(entropy=20261111, spawn_key=(role, experiment, condition, replicate))`. It yields 8 independent xoshiro256** states, one per *component*:

| Component | Use |
|---|---|
| defect | defect indicator |
| audit | audit selection |
| miss | miss indicator |
| noise | label noise |
| delay | label delay |
| score | classifier score |
| board | board random effects |
| tiebreak | randomized threshold rule |

Each component draws a fixed number of variates per item, independent of the other features. For a given key, conditions that differ only in r_t share defects, audits and delays item-by-item; the misses are monotonically coupled (tested). Turning label noise, scores or boards on or off does not move the other components.

- **Coupling rules.** Replicate ids are global, so replicates 10..19 of one call equal the slice of a 0..49 call. Use different `condition` ids for conditions that must be independent; the original study used independent streams per audit rate.
- **CRN across monitors.** All monitors evaluated on one `StreamBatch` see identical data.
- **Namespaces.** Use the role code for the calibration / validation / evaluation separation. Give each experiment its own `experiment` id: baselines 1, robustness 2, adaptive 3, replay 4, … (suggested; record your choice).

### `generators.py`: streams

```python
risk_path(H, r_pre=0.05, r_post=None, tau=None, ramp=0, values=None) -> r[t-1]   # item tau is post-change
prevalence_path(H, kind='constant'|'step'|'sinusoid'|'drift'|'array', pi=..., pi_post=, tau=, amplitude=, period=, phase=, drift_per_item=, values=)
Delay.constant(d) | Delay.geometric(mean) | Delay.negbin(mean, k) | Delay.lognormal(mean, sigma) | Delay.batch(shift, base=0)
ScoreModel(mu0=0, mu_det=3, mu_miss=1.5, sigma=1)      # binormal score given (Z, M)
StratifiedPolicy(edges, probs)                          # numba fast path, p = probs[searchsorted(edges, score)]
StreamConfig(H, pi=0.1, r=0.10, q=0.2, policy=None, budget=-1, force_budget=False, delay=Delay.constant(0),
             markov_rho=0, markov_over='defect'|'item', board_size=1, icc_r=0, icc_pi=0, e01=0, e10=0,
             score=None, checkpoints=(), store_miss_items=False)
nominal_config(pi=0.1, q=0.2, d=500, H=29999, tau=10000, r_pre=0.05, r_post=0.20, **kw)  # original E2 model
null_config(pi=0.1, q=0.2, d=500, H=29999, r=0.10, **kw)
generate(cfg, role='evaluation', experiment=0, condition=0, replicates=1000|array, states=None) -> StreamBatch
generate_chunks(cfg, role, experiment, condition, n, chunk=2000)     # iterator, bounded memory
generate_parallel(cfg, role, experiment, condition, replicates, n_jobs=8)   # for Python hooks (spawn)
latent_one(states_i, H, pi, r, *_extra(cfg)) -> Z, M, Y, delay, U_audit, score   # per-item latent arrays
concat_batches(list_of_batches)
```

`StreamBatch` is CSR-like: stream i occupies `[offsets[i], offsets[i+1])`. Its fields:

| Field | Content |
|---|---|
| `origin`, `arrival` | int64 items; arrival is non-decreasing, ties in origin order |
| `loss` | int8 observed loss (after noise) |
| `true_loss` | int8 true miss |
| `prob` | selection probability of the label's audit (IPW: `batch.weights()` = 1/prob) |
| `tiebreak` | (N,) per-stream uniform |
| `n_audits`, `n_defects`, `n_misses`, `n_informative` | per-stream totals |
| `counts_at` | (N, C, 3) audits / defects / misses at `checkpoints` |
| `miss_offsets`, `miss_items` | production miss items (CSR), if `store_miss_items` |

`batch.stream(i)` returns a dict of one stream's arrays. Labels with arrival > H are dropped, as in the original. The alarm time is the arrival item of the triggering label.

Model details are in the module docstring:
- **Markov misses:** P(M_k = 1 | M_{k−1}) = clip(r_k + ρ(M_{k−1} − r_{k−1})). This gives an exact marginal r_k and lag-1 correlation ρ for constant r. The chain runs over successive defective items by default.
- **Board effect:** Beta with precision 1/icc − 1 on r and/or π. The ICC is that of two defective items' misses in the same board.
- **Delays:** geometric is on {0, 1, …}; NB is the sum of k geometrics; lognormal is rounded with E[unrounded] = mean. Batch release gives a_t = ceil((t + base)/s)·s.

**Audit hook.** Subclass `AuditPolicy`, or pass a callable, which is wrapped by `CallablePolicy`:

```python
class MyPolicy(G.AuditPolicy):
    epoch = 1            # items decided per call (speed vs granularity)
    uses_scores = False  # True -> view.scores = scores of items t0..t1-1 (needs cfg.score)
    def reset(self, stream_index): ...
    def decide(self, view):   # -> float or array of length view.t1 - view.t0, values in [0,1]
        view.t0, view.t1, view.H, view.n_arrived, view.n_audits, view.budget_remaining, view.state (dict)
        view.arrived_origin / arrived_arrival / arrived_loss / arrived_prob   # arrival <= t0-1 only, read-only
        view.past_prob / view.past_audit                                     # items 1..t0-1
```

The generator applies the budget after the hook: p := 0 once the budget is exhausted, and with `force_budget`, p := 1 when the remaining items equal the remaining budget. A hook returning constant q reproduces the fast path bit-for-bit; this is tested for epoch 1 and 37 and for constant and geometric delays. `WarningBoostPolicy` is an *example* hook, not the original allocation policy.

### `monitors.py`: detectors

All monitors share one interface. The statistic path is S_k at output times t_k: after every label, or at every check for `Periodic`. The alarm is the first t_k with **S_k > h (strict)**.

```python
m.path(arrival, loss, weight=None, H=None, t0=0) -> (times, S)            # one stream
m.evaluate(batch, thresholds=(...), checkpoints=(...), weights=None) -> {'alarm': (N,G) float (inf=none), 'max': (N,C)}
m.max_stat(batch, horizon=None)          # running max up to horizon (default H); for calibration
m.first_alarm(batch, h, gamma=0.0)       # gamma>0: streams with tiebreak<gamma alarm at S>=h
m.alarm_grid(batch, grid)                # sweep mode: first crossing for every grid threshold, O(n+G) per stream
m.records(batch) -> (roff, times, values) # strict running-max records; records_first_crossing(roff, rt, rv, h)
m.restart_run(batch, h)                  # restart after each alarm (ARL0 runs)
m.native_threshold(...)                  # design-based threshold or None
original_monitors()   # [Betting, CUSUM(.12), CUSUM(.20), MixtureSR, Periodic] = archived threshold order
default_suite()       # dict of all 13 configured monitors
```

| class | statistic (see docstrings for full formula and reference) | native threshold | weights |
|---|---|---|---|
| `Betting(r0, gain=4, cap=9, window=50, delta=.02)` | log wealth, predictable stake (original) | log(1/δ): Ville, P(any FA, ∞ horizon) ≤ δ | yes |
| `CUSUM(p1, r0, integer_scale=0)` | Bernoulli CUSUM; integer chart = theory/exact_cusum.py | none | yes |
| `MixtureSR(r0, stakes=LAM, alpha_arl=None)` | log mean of 5 SR components (original) = e-SR | log(1/α): ARL ≥ 1/α (labels) | yes |
| `EDetector(r0, kind='SR'/'CUSUM', alpha=1e-3, min_gap=.02, increment='betting'/'exponential')` | SRR 2024 mixture e-SR/e-CUSUM, Eqs. (26)-(28), (70)/(72); grid + weights by Algorithm 3 (Eqs. (94)-(96), (75)) | log(1/α): ARL ≥ 1/α | betting form |
| `Periodic(r0, interval=3000, alpha=.02)` | −log exact binomial p on newly arrived labels at each check (original) | −log(α/K) − 1e-9 | no |
| `EWMA(r0, lam=.05)` | standardized Bernoulli EWMA, exact variance | L = 3 (textbook, no guarantee) | no |
| `GLRCUSUM(r0, window=500)` | windowed GLR, MLE constrained > r0 | none | no |
| `WindowBinomial(r0, window=200)` | −log P(Bin(m, r0) ≥ s) on last W labels | none | no |
| `ADWIN(delta=.002, min_len=5, M=5)` | one-sided ADWIN2 cut, inverted to a threshold-free statistic | ln(1/δ) | no |
| `DDM(n_min=30)` | (p+s − p_min)/s_min | 3 | no |
| `CSDetector(r0, alpha_design=.02, start_every=25, n_active=40, c=.5)` | max over recent starts of WSR-PrPl betting capital at r0 (backward CS, Shekhar & Ramdas 2023) | log(1/α_d) per start; ARL ≥ G(1/α − 1) labels | no |

Weighted versions use the factor 1 + λ w (X − r0), with λ capped so that λ w r0 ≤ cap. They are valid when E[w(X − r0) | past] ≤ 0; this is *not* separately validated by a test beyond the w ≡ 1 identity.

### `calibration.py`

```python
calibrate_threshold(maxima, target=0.02, method='randomized'|'order_statistic', tol=1e-9) -> Threshold(h, gamma, h_hi, h_lo, attained_cal, n_tie, ...)
alarm_times(monitor, batch, thr) -> (N,) float           # applies the randomized rule with batch.tiebreak
any_alarm(monitor, batch, thr, horizon=None) ; fa_from_maxima(maxima, thr, tiebreak)   # same decision, cheap
strict(h)                                                 # plain S > h (native or archived thresholds)
validate(alarmed, cap=0.05, family=1, conf=0.95)          # CP two-sided + one-sided Bonferroni upper, pass_gate
clopper_pearson(k, n, conf, sided='two'|'upper', family=1) ; poisson_ci(k, exposure)
detection_metrics(alarm, tau, H, deadlines=(5000,10000,20000), quantiles=(.5,.9)) -> pD_D (+k, CI), rmd_D, early_rate, no_alarm_rate, cond_mean_delay, delay_q (non-early streams, inf = no alarm)
arl0(monitor, batch, thr) -> alarms_per_10k (+Poisson CI), arl_items (+CI), any_alarm_rate   # restart after alarm
threshold_grid(null_maxima, fa_levels=geomspace(1e-3,.5,120)) ; tradeoff_curve(monitor, null_batch, alt_batch, tau, grid) -> DataFrame(h, fa, early, pD_D) ; pd_at_fa(curve, fa, D)
```

**Randomized calibration.** Values within tol·max(1, |h|) of h form one tie class. The rule is:
- alarm at the first S > h_hi;
- streams with tiebreak < γ instead alarm at the first S > h_lo.

The attained calibration FA then equals the target exactly in expectation. `order_statistic` is the original rule. Use `max_stat` (fast) for calibration and `alarm_times` for evaluation. The paper's metric conventions are listed at the top of `calibration.py`.

### `legacy.py`

```python
conditions(); archived_thresholds(); cal_key(c); legacy_batch(c, role='final', n=2000)   # original seeding
original_trajectories(x); original_periodic(x, arrival, H)
```

## Validation results (pytest, 63 tests)

### (a) Generator + integer CUSUM vs exact DP (`theory/exact_cusum.py`)

20,000 streams per quantity. Thresholds come from archived E2 `thresholds.json` (c = floor(h·s), B = 80). Every row has |z| < 2:

| condition (π, q, d, p1, D) | FA r=.10 exact / MC | FA path .05→.10 | FA r=.05 | p_D exact / MC |
|---|---|---|---|---|
| .1, .1, 500, .20, 10k | .0194 / .0182 | .0113 / .0101 | 0 / 0 | .4802 / .4814 |
| .1, .2, 500, .20, 10k | .0203 / .0205 | .0125 / .0129 | 0 / .0001 | .8283 / .8249 |
| .1, .5, 500, .20, 10k | .0197 / .0193 | .0126 / .0115 | 0 / 0 | .9953 / .9950 |
| .1, .2, 0, .12, 20k | .0199 / .0193 | .0070 / .0067 | 0 / 0 | .1130 / .1149 |
| .01, 1, 500, .20, 20k (τ=20k) | .0194 / .0175 | .0076 / .0083 | 0 / 0 | .8686 / .8675 |

The continuous chart at the archived threshold is within 0.005 of the integer-chart exact p_D in all five conditions.

### (b) Archived operating map

**Bit-level.** `legacy_batch` regenerates the archived final streams for 7 conditions (E2_088, 089, 001, 121, 057, 108, 132). That is 105 cells (5 methods × 3 deadlines), plus prealarm and no-alarm counts.
- 101/105 reproduce *exactly*.
- The 4 others are CUSUM(0.20) cells off by 1–2 streams: E2_001 at all three deadlines and E2_089 at D = 5000.
- In each, the archived threshold lies on a lattice value that many streams reach exactly. For example, 40 of 2,000 streams in E2_001 flip under a ±1e-9 change of the threshold, so round-off decided them (old: cumsum − running min; new: recursion).
- The new trajectories equal `run.trajectory` to ≤ 1.5e-13.

**Statistical.** The new generator (10,000 independent streams) gives max |z| = 2.06 over 30 cells versus the archived rates.

### (c) Native validity, H = 29,999, N = 20,000

- **Betting:** native FA at log(1/0.02) is 0.0094 at r = 0.10, 0.0043 on the .05→.10 path and 0 at r = 0.05, all ≤ δ = 0.02. Periodic native FA ≤ 0.02.
- **Restart ARL** (labels; 40 streams × 4e5 items, 0.5 labels/item, α = 0.002, bound 500): e-SR 698, mSR 656, e-CUSUM 5935. All 95% lower bounds are ≥ 500.

### (d) Generator variants

All checks pass:
- marginal π, q, r and label supply;
- Markov lag-1 ρ ∈ {0.3, 0.6} (±0.02), and the exact time-varying marginal;
- board ICC 0.2 on r and 0.1 on π (±0.03 / ±0.02);
- all four delay laws: means, sorted arrival, origin-ordered ties, genuinely out-of-order origins, batch release on shift ends;
- label-noise rates;
- prevalence step and sinusoid;
- binormal score means;
- stratified-policy IPW: the Hajek estimate is unbiased while the naive mean is biased;
- budget cap and forced spending;
- determinism, namespaces and CRN coupling;
- process-parallel equals serial.

### (e) No look-ahead

An evidence-adaptive hook sees only labels with arrival ≤ t0 − 1 and read-only arrays. Two runs that differ only in r after T* produce identical decisions for items ≤ T* + d (d = 0, 50), and they differ afterwards.

### Monitor kernels vs naive pure-Python references

Every monitor is checked against `tests/reference.py`: literal e-SR/e-CUSUM sums, brute-force GLR over all starts, direct ADWIN ε test at δ ∈ {.5, .05, .002}, direct DDM rule, and so on. The numba binomial tail is checked against scipy on 3,000 cases. Batch drivers are checked against single-stream paths for all 13 monitors (alarm grid, checkpoint maxima, records), restart counts against a manual loop, and long-stream rescaling of the mixture kernel against `run.py`'s logsumexp.

Per-test numbers are written to `tests/_results/*.json`.

## Performance

8 numba threads on the shared GB10 (aarch64), with a GPU job running, so timings vary about ±2×. Setting: 20,000 streams × H = 29,999, π = 0.1, q = 0.5, d = 500, which gives 1,475 labels per stream (`benchmarks/bench_q0.5.json`).

| step | 20k streams | streams/s |
|---|---|---|
| generate (fixed q, numba, two-pass) | 3–4.5 s | 4.5–6.3k |
| Betting / CUSUM / EWMA / DDM / WindowBinomial / Periodic | 0.02–0.08 s | 0.25–1.4M |
| MixtureSR | 0.1 s | 200k |
| eSR / eCUSUM (K = 223 components) | 1.8 s | 11k |
| ADWIN / CSDetector / GLR (W = 500) | 1–4.5 s | 4.5–18k |
| Python hook, epoch = 1 (serial; use `generate_parallel`) | — | ~16 per process |
| Python hook, epoch = 100 | — | ~160 per process |

Peak RSS is about 0.8 GB with chunks of 5,000. Memory is roughly 26 bytes per label, so chunk with `generate_chunks` for large q·π·H.

## Design choices and known limitations

- **Ties and discrete statistics.**
  - CUSUM, GLR, the binomial tests and DDM are lattice-valued. Archived thresholds placed on a lattice value make the strict rule sensitive to round-off (see (b)).
  - Use `calibrate_threshold(..., 'randomized')` for exact FA matching. Use `strict(h)` only for native or archived thresholds.
- **ADWIN, DDM and other relative-change detectors** detect change *relative to the past*, not exceedance of r0.
  - Calibrated at the r = 0.10 boundary, they exploit the 0.05 pre-change baseline of the harmful scenarios.
  - Measured (π = .1, q = .2, d = 500, 5,000 streams, `benchmarks/fa_composite_null_pi0.1_q0.2_d500.txt`): at thresholds calibrated to 0.02 at r = 0.10, FA on the admissible 0.05 → 0.10 path is **0.18 for ADWIN** and **0.096 for DDM**, against 0.008–0.015 for every r0-anchored monitor. Always report that path.
  - ADWIN never shrinks its window before the first alarm (one-sided adaptation). This is a documented deviation from the original adaptive window.
- **DDM is faithful to Gama et al. (2004),** including its flaw: if the first 30 losses are all 0, then s_min = 0 and the next miss alarms (about 4% of r = 0.10 streams). This makes the calibrated threshold +∞ with randomization at typical targets. `n_min` is a parameter.
- **EDetector:**
  - Default α = 1e-3 (ARL design), Δ from a minimum gap of 0.02 (r1 ≥ 0.12), and v_min = 1 for binary losses. The ω0 component is then 0 by Eq. (96) and is dropped.
  - The statistic depends on the design α, so a calibrated threshold should be reported together with the design α.
  - Native thresholds guarantee ARL, *not* P(FA within H). With α = 1e-3 the native FA within H is about 0.2 at 590 labels/stream.
  - Algorithm 3's g_α is solved by bisection assuming f(g) is decreasing; the indicator jump at v_min·D_U (≈ 220 here) is far from the root.
- **CSDetector** is our own documented instantiation (starts on a grid, K active, one-sided WSR PrPl stakes with c = 0.5). It is *not* the exact algorithm of Shekhar & Ramdas (2023); see the docstring.
- **Periodic:** the "matched" version is the same statistic with a calibrated threshold (randomized ties). A per-check randomized UMP binomial test is not implemented. Under restart, checks continue on the global schedule.
- **Model scope:**
  - Label noise flips only the losses of informative labels. Reference errors on defect status are not modelled.
  - Board effects use the r (or π) of the board's first item, scaled if the path changes inside the board.
  - Combinations of Markov and board effects run but were not separately validated.
- **Hook policies** run serially in Python. Scores of the items in an epoch are visible to the decision for that epoch, so use epoch = 1 for strict causality. Budgets are applied item-by-item after the hook.
- **IPW-weighted monitors** are implemented, but their validity and power under informative policies are not yet studied. The w ≡ 1 identity is tested.
- **No numba on-disk cache** is used, so every new process recompiles (about 1 s per kernel).
- **Not done here:** the big experiments, real-data replay loaders, and the original block-adaptive allocation policy (its code is not in this repo; only its numerical evidence is). Agents need to build these on the hook API.
