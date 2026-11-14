"""E1 shared definitions: grid, seeding ids, monitors, threshold variants."""
import itertools, math
import numpy as np
from revision import monitors as MM
from revision.experiments.E1.winlr import WindowLR

EXP = 11
H, TAU, R0, RPRE = 29999, 10000, 0.10, 0.05
DEADLINES = (5000, 10000, 20000)
PIS = (0.01, 0.10)
QS = (0.10, 0.15, 0.20, 0.30, 0.50, 1.00)
DS = (0, 500)
R1S = (0.12, 0.20, 0.30)
SETTINGS = list(itertools.product(PIS, QS, DS))          # null settings (pi, q, d), idx 0..23
TRADEOFF = [(0.10, 0.20, 0.20, 500), (0.10, 0.12, 1.00, 500), (0.01, 0.30, 1.00, 0), (0.10, 0.30, 0.10, 500)]
PATHS = ('const10', 'path', 'const05')
FA_TARGET = 0.02

N_CAL, N_VAL, N_EVAL, N_ARL = 20000, 10000, 10000, 2500
CHUNK = 2500


def null_cond(idx, path):
    return 100 + 4 * idx + PATHS.index(path)


def harm_cond(idx, r1):
    return 500 + 3 * idx + R1S.index(r1)


def make_monitors():
    return {
        'cusum_0.12': MM.CUSUM(0.12, R0), 'cusum_0.20': MM.CUSUM(0.20, R0), 'betting': MM.Betting(R0),
        'msr': MM.MixtureSR(R0), 'eSR': MM.EDetector(R0, 'SR'), 'ewma_0.05': MM.EWMA(R0, 0.05),
        'glr_W500': MM.GLRCUSUM(R0, 500), 'winbin_W200': MM.WindowBinomial(R0, 200),
        'winlr_D10000': WindowLR(10000, 0.20, R0), 'periodic': MM.Periodic(R0), 'adwin': MM.ADWIN(),
        'ddm': MM.DDM(), 'eCUSUM': MM.EDetector(R0, 'CUSUM'), 'cs_detector': MM.CSDetector(R0),
    }


CORE = ['cusum_0.12', 'cusum_0.20', 'betting', 'msr', 'eSR', 'ewma_0.05', 'glr_W500', 'winbin_W200',
        'winlr_D10000', 'periodic', 'adwin', 'ddm']
SUPP = ['eCUSUM', 'cs_detector']
# native / reference variants: name -> (monitor key, threshold function)
NATIVE = {
    'betting_native': ('betting', lambda m: m.native_threshold()),            # log(1/0.02), Ville
    'eSR_native': ('eSR', lambda m: m.native_threshold()),                    # log(1/1e-3), ARL design
    'eCUSUM_native': ('eCUSUM', lambda m: m.native_threshold()),
    'periodic_bonferroni': ('periodic', lambda m: m.native_threshold(H=H)),   # original rule
    'adwin_native': ('adwin', lambda m: m.native_threshold()),
    'ddm_native': ('ddm', lambda m: m.native_threshold()),
    'cs_native': ('cs_detector', lambda m: m.native_threshold()),
}
CONSTCAL = {'adwin_constcal': 'adwin', 'ddm_constcal': 'ddm'}   # calibrated on constant 0.10 only


def stricter(a, b):
    """Composite threshold: the stricter of two randomized Thresholds (larger h; tie -> smaller gamma)."""
    ka = (a.h, -a.gamma); kb = (b.h, -b.gamma)
    return a if ka >= kb else b


def oracle_beta(pi, q, d, r1, D, alpha=FA_TARGET):
    import sys
    sys.path.insert(0, '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory')
    from converse import np_power
    return np_power(max(0, D - d), pi * q, R0, r1, alpha)
