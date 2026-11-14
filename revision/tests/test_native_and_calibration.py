"""(c) Native thresholds keep their guarantees; calibration/validation/metrics mechanics."""
import json
from pathlib import Path
import numpy as np
import pytest

from revision import generators as G, monitors as M, calibration as C

OUT = Path(__file__).resolve().parent / '_results'
OUT.mkdir(exist_ok=True)


@pytest.mark.parametrize('rpath', ['const0.10', 'const0.05', 'path0.05to0.10'])
def test_betting_native_fa_below_delta(rpath):
    H, N, delta = 29999, 20000, 0.02
    r = {'const0.10': 0.10, 'const0.05': 0.05, 'path0.05to0.10': G.risk_path(H, 0.05, 0.10, 10000)}[rpath]
    cfg = G.StreamConfig(H=H, pi=0.1, q=1.0, r=r, delay=G.Delay.constant(0))
    b = G.generate(cfg, 'test', 300, ['const0.10', 'const0.05', 'path0.05to0.10'].index(rpath), N)
    m = M.Betting(delta=delta)
    fa = np.mean(m.max_stat(b) > m.native_threshold())
    lo, hi = C.clopper_pearson(int(fa * N), N, 0.99)
    (OUT / f'betting_native_{rpath}.json').write_text(json.dumps(dict(fa=fa, cp99=(lo, hi), N=N, delta=delta)))
    assert lo <= delta, (fa, lo)


@pytest.mark.parametrize('mon', ['eSR', 'eCUSUM', 'msr'])
def test_edetector_native_arl(mon):
    """ARL (in labels) >= 1/alpha under the boundary null r=0.10 (SRR Thm 2.4)."""
    alpha = 0.002
    m = {'eSR': M.EDetector(kind='SR', alpha=alpha), 'eCUSUM': M.EDetector(kind='CUSUM', alpha=alpha),
         'msr': M.MixtureSR(alpha_arl=alpha)}[mon]
    H = 400000
    cfg = G.StreamConfig(H=H, pi=0.5, q=1.0, r=0.10)        # 0.5 labels per item
    b = G.generate(cfg, 'test', 301, 0, 40)
    res = C.arl0(m, b, m.native_threshold())
    arl_labels = res['arl_items'] * 0.5
    arl_labels_lo = res['arl_items_lo'] * 0.5
    (OUT / f'native_arl_{mon}.json').write_text(json.dumps(dict(res, arl_labels=arl_labels, arl_labels_lo=arl_labels_lo,
                                                               bound=1 / alpha), default=float))
    assert arl_labels_lo >= 1 / alpha * 0.9 or res['alarms'] == 0, res


def test_periodic_native_fa_below_alpha():
    H, N = 29999, 20000
    b = G.generate(G.null_config(pi=0.1, q=0.5, d=500, H=H), 'test', 302, 0, N)
    m = M.Periodic()
    fa = np.mean(m.max_stat(b) > m.native_threshold(H))
    lo, _ = C.clopper_pearson(int(fa * N), N, 0.99)
    assert lo <= 0.02


def test_randomized_calibration_hits_target_and_validates():
    H = 29999
    cal = G.generate(G.null_config(pi=0.01, q=0.2, d=0, H=H), 'calibration', 303, 0, 5000)
    val = G.generate(G.null_config(pi=0.01, q=0.2, d=0, H=H), 'validation', 303, 0, 20000)
    out = {}
    for name, m in [('cusum_0.2', M.CUSUM(0.2)), ('periodic', M.Periodic()), ('winbin', M.WindowBinomial()),
                    ('glr', M.GLRCUSUM())]:
        mx = m.max_stat(cal)
        thr = C.calibrate_threshold(mx, 0.02)
        att = np.mean(C.fa_from_maxima(mx, thr, cal.tiebreak))
        assert abs(thr.attained_cal - 0.02) < 1e-12
        assert abs(att - 0.02) < 4 * np.sqrt(0.02 * 0.98 / 5000) + 0.002   # realized tiebreaks
        a = C.alarm_times(m, val, thr)
        v = C.validate(np.isfinite(a), cap=0.05)
        assert v['cp_lo'] < 0.02 + 0.01 and v['cp_hi'] > 0.02 - 0.01, (name, v)
        # alarm_times and the maxima shortcut agree
        assert np.array_equal(np.isfinite(a), C.fa_from_maxima(m.max_stat(val), thr, val.tiebreak))
        so = C.calibrate_threshold(mx, 0.02, 'order_statistic')
        out[name] = dict(h=thr.h, gamma=thr.gamma, n_tie=thr.n_tie, val_rate=v['rate'], strict_os_rate=so.attained_cal)
    (OUT / 'calibration_mechanics.json').write_text(json.dumps(out, indent=1, default=float))


def test_order_statistic_reproduces_original_rule():
    x = np.random.default_rng(0).normal(size=5000)
    thr = C.calibrate_threshold(x, 0.02, 'order_statistic')
    assert thr.h == np.sort(x)[int(np.ceil(5001 * 0.98)) - 1]


def test_detection_metrics_and_tradeoff():
    a = np.array([5, 10, 12, 30, np.inf, 9])
    m = C.detection_metrics(a, tau=10, H=100, deadlines=(5,), quantiles=(0.5,))
    assert m['early'] == 2 and m['pD_5_k'] == 2 and m['rmd_5'] == pytest.approx((5 + 0 + 2 + 5 + 5 + 5) / 6)
    nb = G.generate(G.null_config(pi=0.1, q=0.2, d=500), 'validation', 304, 0, 2000)
    ab = G.generate(G.nominal_config(pi=0.1, q=0.2, d=500), 'evaluation', 304, 0, 2000)
    cur = C.tradeoff_curve(M.CUSUM(0.2), nb, ab, tau=10000)
    assert np.all(np.diff(cur.fa.values) <= 0) and cur.fa.max() > 0.3 and cur.fa.min() < 0.01
    assert 0.5 < C.pd_at_fa(cur, 0.02, 10000) < 1


def test_arl_restart_horizon_streams():
    b = G.generate(G.null_config(pi=0.1, q=0.5, d=0), 'validation', 305, 0, 500)
    m = M.CUSUM(0.2)
    r = C.arl0(m, b, 3.0)
    assert r['alarms'] > 0 and r['alarms_per_10k'] == pytest.approx(1e4 * r['alarms'] / (500 * b.H))
