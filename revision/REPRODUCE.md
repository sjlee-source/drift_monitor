# Reproducing the IEEE Access revision results

This file explains how to regenerate every number in the revised manuscript and its supplement:
- the exact computations (Theorem 1 limit, exact CUSUM, q* search),
- simulation experiments E1–E4,
- the supplement tables.

The library API is documented in `README_REVISION.md`; the supplement (`supplementary.tex`, Sections S1–S8) gives the model, monitor formulas, policy logic and protocol deviations.

## 0. Layout and paths

| What | Location |
|---|---|
| Library (`rng`, `generators`, `monitors`, `calibration`, `legacy`) | `drift_monitor/revision/` |
| Experiment code | `drift_monitor/revision/experiments/E1 … E4/` |
| Library tests | `drift_monitor/revision/tests/` |
| Exact-computation scripts | `access_revision_2026-11-11/theory/` |
| Independent re-implementation | `access_revision_2026-11-11/verify_independent/` |
| Archived outputs (CSV/JSON, `protocol.json`, `REPORT.md`) | `access_revision_2026-11-11/exp_E1 … exp_E4/` |
| Supplement table generator | `access_revision_2026-11-11/manuscript_v2/supp_src/make_supp_tables.py` |

Below, `$DM=/home/sjlee/Research/drift_monitor` and `$REV=/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11`.

**Output paths are absolute.** Experiment scripts write to `$REV/exp_E*/` through an `OUT` constant at the top of each script (E1 shell scripts: `OUT=`/`FU=`; E1 aggregation also accepts `E1_ROOT`). Rerunning therefore **overwrites the archived outputs**. To keep the archive for comparison:
1. Copy `$REV/exp_E*` to a backup directory first, or
2. Change `OUT` to a scratch directory.

The comparison recipe in Section 7 assumes a backup in `$REV/archive/exp_E*`.

## 1. Environment

The simulation code is CPU only (no GPU). The original runs used:
- aarch64, NVIDIA GB10, Linux 6.17;
- 4 numba threads;
- under 16 GB RAM (typical peak RSS 1–3 GB).

```bash
/usr/bin/python3 -m venv ~/.venvs/drift-rev
~/.venvs/drift-rev/bin/pip install numpy==2.5.4 scipy==1.18.1 pandas==3.1.1 numba==0.68.1 pytest
cd $DM
export PYTHONPATH=$PWD NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1
```

The public-data steps of E4 need **scikit-learn 1.9.1** for the gradient-boosting model.

numba compiles every kernel on first use, about 1 s per kernel; there is no on-disk cache.

## 2. Tests (library validation)

```bash
cd $DM && ~/.venvs/drift-rev/bin/python -m pytest -q revision/tests        # ~3.5 min, peak RSS ~2.1 GB
~/.venvs/drift-rev/bin/python -m pytest -q revision/experiments/E1/test_winlr.py   # window LR scan vs naive reference
```

The tests cover:
- the generator against exact DP;
- bit-level reproduction of the archived operating map (101/105 cells exact; the 4 others are explained in `README_REVISION.md`);
- every monitor against naive pure-Python references;
- native-threshold validity;
- generator variants and no-look-ahead of audit hooks.

Per-test numbers are written to `revision/tests/_results/*.json`.

## 3. Seeds

Every stream is defined by `SeedSequence(entropy=20261111, spawn_key=(role, experiment, condition, replicate))` (`revision/rng.py`):
- Role codes: calibration 1, validation 2, evaluation 3, pilot 4, arl 5, replay 6, test 7, sweep 8.
- Experiment ids: E1 = 11, E2 = 12, E3 = 13, E4 = 14.
- Condition-id formulas are listed in each `protocol.json` and in Supplement Table S2.

With the same library version, regenerated streams, thresholds and alarm times are **bit-identical**. Any single stream can be regenerated directly:

```python
from revision import generators as G
cfg = G.nominal_config(pi=0.1, q=0.2, d=500)             # harmful path 0.05 -> 0.20 at tau=10,000
b = G.generate(cfg, role='evaluation', experiment=11, condition=500 + 3*7 + 1, replicates=range(10))
```

In this example, setting 7 is (π=0.10, q=0.10, d=500) and r1 index 1 is r1 = 0.20 (E1 `e1_common.py`).

## 4. Exact computations (theory)

The scripts import each other as local modules, so run them from inside their own directory with the drift-rev interpreter.

```bash
cd $REV/theory
P=~/.venvs/drift-rev/bin/python
$P qstar.py      > qstar_v2.log     # q*_any and q*_CUSUM on the 176-cell grid -> qstar_grid.csv   (~20-40 min, 10 processes)
$P check_408.py                     # exact vs archived MC for the 408 CUSUM cells -> exact_vs_mc_408.csv (~5 min, 8 processes)
$P scale_sens.py > scale_sens.log   # integer scale B = 40/80/160 convergence (log only)
$P validate.py                      # unit checks of exact_cusum.py / converse.py
cd $REV/verify_independent          # independent lattice implementation (no integer rounding)
$P taskA.py exact > taskA_exact.log; $P taskA.py mc > taskA_mc.log   # -> taskA_exact.json, taskA_mc.json (MC: 200,000 reps/cell)
$P taskB.py                         # Poisson label counts lambda* (stdout only: 29.86 / 100.11 / 354.69 / 2025.15)
$P taskC.py > taskC.log             # crossing of 0.80 near q*_CUSUM -> taskC.json
```

Notes:
- `qstar.py` is version 2: a scan-and-refine search that records pass-to-fail transitions. Version 1 output is kept as `qstar_grid_v1.csv`.
- `check_408.py` reads the archived operating map and thresholds from `$DM/supplementary/evidence/simulation/`.
- The Proposition-1 certification table additionally reads `$DM/supplementary/analysis/information_planning.csv`.

## 5. Experiments

All commands start from `cd $DM` with the environment variables of Section 1. Every experiment works the same way:
1. `protocol.json` is written before the evaluation run.
2. Raw outputs go to `exp_E*/raw*/`.
3. Aggregation scripts produce the CSV tables.

Runtimes are wall-clock times from the original logs at 4 threads on a shared machine; they can vary by about ±2×.

### E1 — matched comparison (experiment id 11)

**Confirmatory version (100,000 calibration streams per path)** — this is what the manuscript reports:

```bash
bash revision/experiments/E1/run_cal100k.sh            # 24 main settings + 18 fine-q settings, resumable; ~1.5-2 h
~/.venvs/drift-rev/bin/python -m revision.experiments.E1.exact_cusum_check   # exact DP for CUSUM(0.20) -> cusum_exact_dp.csv (~3 min)
~/.venvs/drift-rev/bin/python -m revision.experiments.E1.aggregate_cal100k   # -> exp_E1/*_cal100k.csv, compare_20k_vs_100k_*.csv
```

This produces:
- per-setting JSON and alarm files in `exp_E1/raw_cal100k/` and `exp_E1/followup_cal100k/`;
- CSV files `exp_E1/*_cal100k.csv`, including `metrics_long`, `fa_validation`, `main_pD_D10000`, `ranking_confirmatory`, `crn_paired_diff`, `qmin_interp_fineq`, `tradeoff_summary`, `restart_fa_incidence` and `cusum_mc_vs_exact`.

`aggregate_cal100k.py` reruns `crn_paired.py` and the figure script with `E1_ROOT` set to the 100k tree.

**20,000-stream history** (reported as history; not confirmatory):

```bash
bash revision/experiments/E1/run_all.sh                 # 24 settings -> exp_E1/raw/ (~40 min at 3 threads)
bash revision/experiments/E1/run_followup.sh            # A1 gate repair (4 cells) + A3 fine-q grid -> exp_E1/followup/ (~25 min)
~/.venvs/drift-rev/bin/python -m revision.experiments.E1.aggregate
~/.venvs/drift-rev/bin/python -m revision.experiments.E1.aggregate_followup
~/.venvs/drift-rev/bin/python -m revision.experiments.E1.crn_paired
/usr/bin/python3 revision/experiments/E1/plot_e1.py     # figures (needs matplotlib)
```

A single setting can be run with `python -m revision.experiments.E1.run_setting100k IDX --out DIR`, where IDX is 0–23 and indexes (π, q, d) in product order.

### E2 — assumption violations (experiment id 12)

```bash
P=~/.venvs/drift-rev/bin/python
$P -m revision.experiments.E2.sim                 # 13 stressors x 2 q x 7 monitors -> fa.csv, detection.csv, thresholds.csv (~15-30 min)
$P -m revision.experiments.E2.plan_dp             # planning misspecification + pilot uncertainty, exact DP (~10 min)
$P -m revision.experiments.E2.summarize           # summary_fa_nominal/recalibrated, summary_detection, labelnoise_analytic
$P -m revision.experiments.E2.posthoc_reval       # post hoc re-validation (replicates 10,000-19,999)
# E2b: thresholds from one finite null history (block bootstrap)
$P -m revision.experiments.E2.e2b                 # prespecified procedure (~30 min)
$P -m revision.experiments.E2.e2b_ph PH1; $P -m revision.experiments.E2.e2b_ph PH2   # post hoc recentring (~30 min)
$P -m revision.experiments.E2.e2b_cluster         # history-cluster (two-way bootstrap) inference
$P -m revision.experiments.E2.e2b_summary
# E2c: confirmatory recentred bootstrap from long histories
for a in primary curve1H curve5H incontrol; do $P -m revision.experiments.E2.e2c --arm $a; done   # ~1-2.5 h per arm
$P -m revision.experiments.E2.e2c_summary
```

Each of `sim`, `e2b` and `e2c` accepts:
- `--protocol-only`: write `protocol.json` without running;
- `--smoke`: small development run.

E2c stores per-history summaries only; per-stream alarms can be regenerated from the seeds in `E2c/protocol.json`.

### E3 — adaptive/targeted audit allocation (experiment id 13)

This is the complete pipeline for the allocation experiment:
- **Policy definitions:** `experiments/E3/design.py`.
- **Fused stream + policy + monitor kernel:** `kernel.py` (main run) and `kernel2.py` (follow-up). It contains the score model, budget, floor, forcing, IPW weights and capped stakes.
- **Drivers:** `core.py` and `run.py`.

```bash
P=~/.venvs/drift-rev/bin/python
$P -m revision.experiments.E3.write_protocol      # protocol.json incl. design parameters (strata, g_h, scale constants)
$P -m revision.experiments.E3.run                 # main run: calibrate (20,000/path; fixed-point for warning policies),
                                                  # validate (10,000/path), evaluate (10,000) for 3 conditions x 2 workflows x 19 policies
                                                  # -> raw/*.pkl, run.log (~25-40 min)
$P -m revision.experiments.E3.summarize           # thresholds.csv, validation.csv, evaluation.csv, contrasts.csv, beta_star.csv, summary.json
$P -m revision.experiments.E3.followup run        # no-leakage, AUC 0.80/0.95, tau 5k/20k, P5 weight clip -> followup/raw (~30-45 min)
$P -m revision.experiments.E3.followup diag       # diagnostics_main_primary.csv (re-runs the main primary condition, checks alarm reproduction)
$P -m revision.experiments.E3.followup summ       # followup/evaluation.csv, validation.csv, contrasts.csv, gate.json
$P -m revision.experiments.E3.recal               # 100,000-stream recalibration of all W-B AUC-0.95 procedures -> followup/recal/
```

Replicate-id ranges:

| Run | Replicate ids |
|---|---|
| Smoke run (`--smoke`, development only) | 0–1,999 |
| Main run | 100,000 onward |
| Follow-up | 200,000 onward |
| Recalibration | 300,000 onward |

The protocol records SHA-256 hashes of the code used in the main run (`code_hashes_main_run_sha256`).

Checks:
- Paired-bootstrap contrasts use 10,000 resamples with fixed seeds, so they reproduce exactly.
- The primary contrast is the row with `primary=True` in `contrasts.csv`: P5 − P1, mSR, π=0.10, d=500, D=10,000.

### E4 — retrospective AOI replays (experiment id 14)

The PCB replay of the manuscript uses proprietary data; its code and results are not part of this repository.

Public AOI data (Pfab et al.):

```bash
SK=~/.venvs/smt-pub/bin/python                                 # any Python env with scikit-learn 1.9.1
P=~/.venvs/drift-rev/bin/python
$SK -I revision/experiments/E4/pfab_01_hgb.py        # refits the pilot HGB; asserts the pilot miss counts 16/140/217
for t in 0 1 2; do $P revision/experiments/E4/pfab_02_replay.py $t > $REV/exp_E4/pfab/02_replay_t$t.log; done   # ~25-30 min each
```

Post hoc follow-up and summaries:

```bash
for t in 0 1 2; do for c in boardcal influence recentred; do
  $P revision/experiments/E4/fu_pfab.py $c $t > $REV/exp_E4/followup/${c}_t$t.log; done; done   # see run_fu.sh (~1-2 h)
```

`run_chain.sh` and `run_fu.sh` contain the exact sequences that were run.

Data locations used by the scripts:

| Data | Path |
|---|---|
| Public data | `/home/sjlee/Dataset/pfab_aoi_2024/dataset.csv` |

Model and split parameters are recorded in `pfab_01_hgb.py` and `exp_E4/pfab/hgb_summary.json`.

## 6. Supplement tables

```bash
cd $REV/manuscript_v2
PYTHONPATH=$DM ~/.venvs/drift-rev/bin/python supp_src/make_supp_tables.py   # -> supp_src/tables/*.tex, numbers.tex, SOURCES.txt
docker run --rm --network none --user $(id -u):$(id -g) -v "$PWD":/w -w /w texlive/texlive:latest pdflatex -interaction=nonstopmode supplementary.tex   # run twice
```

`supp_src/tables/SOURCES.txt` lists every file read and its row count.

## 7. Verifying against the archived numbers

Because seeds and code are fixed, a rerun reproduces the archived CSVs exactly. The only expected differences are floating-point summation order in multithreaded numba reductions. They do not affect alarm times, which come from per-stream serial loops.

To compare a regenerated table with the archive, back up the archive first (Section 0):

```python
import pandas as pd, numpy as np
a = pd.read_csv('$REV/archive/exp_E1/metrics_long_cal100k.csv')   # archived copy
b = pd.read_csv('$REV/exp_E1/metrics_long_cal100k.csv')           # regenerated
key = ['variant', 'pi', 'r1', 'q', 'd', 'D']
m = a.merge(b, on=key, suffixes=('_a', '_b'))
print(len(a), len(m), np.abs(m.pD_a - m.pD_b).max())             # expect equal lengths and 0
```

Spot checks against the manuscript:
- `manuscript_v2/facts.json` maps each number in the paper to its source file and row.
- Examples: E1 gate 573/576 (`fa_validation_cal100k.csv`, gated rows); E3 primary contrast −0.569 (`exp_E3/contrasts.csv`); exact q*_CUSUM 0.188 at (π, r1, d, D) = (0.10, 0.20, 500, 10,000) (`theory/qstar_grid.csv`).
- `make_supp_tables.py` regenerates the supplement from the same files, so recompiling after a rerun is a full consistency check.

The library-level checks are:
- `pytest`: bit-level reproduction of the original study and exact-DP agreement;
- `exp_E3/followup/diagnostics_main_primary.csv`, column `reproduces_main_alarms`: the follow-up kernel reproduces every main-run alarm time.

## 8. Data availability

- **Simulated data:** fully regenerable from the seeds; no data files are needed.
- **Public AOI data:** Pfab, K. (2024), *Data of automated optical inspection of surface-mounted technology electronic production*, Mendeley Data, V2, https://doi.org/10.17632/99jzmh9658.2. Place `dataset.csv` at the path used by `pfab_01_hgb.py`.
- **PCB reinspection data:** proprietary data from an operating production line; the data, the code that reads them and the corresponding results are not distributed. Aggregate results appear in the manuscript and supplement.
