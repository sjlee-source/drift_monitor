"""Shared simulation library for the IEEE Access revision (audit-limited drift monitoring).

Modules: rng (seeding), generators (streams), monitors (detectors), calibration (thresholds,
validation, metrics), legacy (bit-exact reproduction of the original simulation/run.py).
"""
import os as _os
_os.environ.setdefault('NUMBA_NUM_THREADS', '8')
