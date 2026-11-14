# Audit-limited manufacturing inspection monitoring

## Revision (v2.0.0)

The `revision/` directory contains the material for the revised manuscript: a tested simulation library
(`revision/README_REVISION.md`), the exact CUSUM and detection-limit computations, the experiment pipelines E1–E4,
prespecified protocols and the numerical results (`revision/results/`). Step-by-step rerun instructions are in
`revision/REPRODUCE.md`. Every stream is regenerated from `SeedSequence(20261111, spawn_key=(role, experiment,
condition, replicate))`. Only the public AOI data are used in the published replay code; the PCB replay of the manuscript relies on
proprietary data, and its code and results are not included. The public AOI data are available from Mendeley Data
(doi:10.17632/99jzmh9658.2). The original archive described below is retained unchanged.


Reproducibility code and numerical evidence for **Manufacturing Inspection Monitoring under Sparse and Delayed Audits: Detection Limits and Audit Planning**.

The contribution is operational characterization and audit planning. Five established monitors are compared: predictable betting, CUSUM with alternatives 0.12 and 0.20, mixture Shiryaev–Roberts, and periodic exact-binomial checks. No algorithmic superiority is claimed for the betting implementation.

## Quick start

Python 3 with the packages in `requirements.txt` is required. These are the versions used for the archived analytical checks.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python analysis/planning_extensions.py
```

This lightweight, CPU-only command reproduces the periodic deadline bounds, 72 exhaustive small-sequence checks, and multiplicity sensitivity from archived inputs. Results are written to `supplementary/analysis/`.

```bash
python analysis/information_planning.py
python analysis/build_result_figures.py
python analysis/plot_representative_stream.py
```

The plotting commands require an installed **Times New Roman** font. They additionally check `~/.local/share/fonts/mfg-times-new-roman/*.TTF`. Fonts are not bundled. Generated figures go to `figures/`. The analytical extension does not require fonts or a GPU.

## Reproduction scope

- `analysis/`: deterministic CUSUM and periodic-binomial planning, multiplicity sensitivity, and numerical plotting routines.
- `supplementary/evidence/`: conditions, calibrated thresholds, null-validation records, operating maps, audit-policy summaries and image-replay numerical outcomes.
- `supplementary/analysis/`: archived finite-sum and sensitivity outputs.
- `supplementary/representative_stream/`: fixed synthetic replicate and metadata used for the trajectory illustration.
- `simulation/`: full main Bernoulli simulation and operating-map analysis. To rerun it in a fresh output directory, run `python simulation/run.py` and then `python simulation/analyze.py`. The full run is computationally substantial (5,000 calibration, 3,000 validation and 2,000 final streams per relevant condition); it is not part of the quick start. A failed null-validation gate blocks final evaluation. `python simulation/run.py --pilot-only` runs implementation checks and the pilot.

The full simulation rerun writes under `simulation/`; the lightweight analyses intentionally use archived evidence under `supplementary/evidence/`. To analyze a fresh full run, explicitly copy its verified outputs into a separate evidence package rather than silently mixing them with the archived results.

This repository does not constitute a full raw-data or classifier-training release. NEU images, third-party classifier weights, proprietary fonts and manuscript publishing-template assets are not bundled. Image-replay results are conditional on a fixed image pool; the evaluated checkpoint's training/evaluation independence is unverified. Documented retraining is separate ongoing work and is not represented as completed here. Audit-policy results are supplied as numerical evidence, not as a full rerun pipeline.

## Statistical interpretation

The main grid contains 136 conditions, including 68 harmful conditions. The 1,020 operating-map cells use all 2,000 final streams in each denominator, with early alarms and nondetections counted as failures. Detection is measured in production items. A common 5% validation cap does not imply equal attained false-alarm rates.

CUSUM planning uses actual calibrated log thresholds, random informative-label counts, and a conservative early-alarm subtraction. The periodic extension integrates random label counts and uses complete post-change-origin checking intervals; mixed intervals are omitted conservatively. These calculations require independent Bernoulli items, noninformative auditing and a fixed delay. They do not establish factory performance or bounds for arbitrary drift.

At the 10,000-item deadline, supported/insufficient/inconclusive counts across 340 method-condition cells are 41/291/8 with global Bonferroni, 41/293/6 with directional Holm, and 41/292/7 with deadline-specific Bonferroni families. Holm outputs are test decisions, not simultaneous confidence intervals. Deadline-specific families change the error-control scope.

`WH_finiteH` and `WH_finite` are retained machine identifiers for the neutrally labelled **Betting monitor**. Method names in recorded data are not novelty claims.

## Integrity and citation

`MANIFEST.json` records SHA-256 hashes of numerical evidence and source scripts. Source provenance is recorded in `PROVENANCE.json`. Release v2.0.0 is archived on Zenodo: [doi:10.5281/zenodo.24210607](https://doi.org/10.5281/zenodo.24210607). The code is released under the MIT License (see LICENSE).

## Alarm convention and betting-gain sensitivity

All four continuous monitors use strict `statistic > calibrated_threshold`; the periodic test uses `p_value <= 0.02/K`. Calibration, null validation and evaluation use the same comparisons. For the betting monitor the statistic is log wealth; for mixture SR it is the log of the component mean.

The implementation uses gain 4 and cap 9. Gains 2, 4 and 8 were checked with separate calibration and common streams in a six-condition representative slice. Their maximum stakes are 1.8, 3.6 and 7.2, so the cap is inactive.

```bash
python supplementary/stake_sensitivity/run_sensitivity.py
```

All 18 null-validation comparisons pass (maximum simultaneous upper bound 3.55%). Gain 4 reproduces the six archived first-alarm arrays exactly. For the strong change, q=0.50 is the lowest evaluated audit rate supporting 80% detection for all gains. This is a sensitivity result, not a claim that gain 4 is optimal. The analysis uses an 18-cell final family, distinct from the primary 1,020-cell family. CSV outputs retain full precision.
