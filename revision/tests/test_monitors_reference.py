"""Numba monitor kernels vs naive reference implementations, plus API mechanics."""
import numpy as np
import pytest
from scipy.stats import binom

import reference as ref
from revision import monitors as M, generators as G, calibration as C


def seqs(n_seq=8, n=400):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n_seq):
        p = np.where(np.arange(n) < n // 2, 0.05 + 0.02 * i, 0.10 + 0.04 * i)
        x = (rng.random(n) < p).astype(np.int8)
        arr = np.cumsum(rng.integers(0, 30, n)) + 1
        out.append((arr.astype(np.int64), x))
    return out


def test_log_binom_sf_vs_scipy():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(1, 3000)); p = float(rng.choice([0.05, 0.1, 0.2, 0.5]))
        k = int(rng.integers(0, n + 2))
        a = M.log_binom_sf(k, n, p)
        b = binom.logsf(k - 1, n, p) if k > 0 else 0.0
        if np.isfinite(b) and b > -700:
            assert abs(a - b) <= 1e-9 * max(1, abs(b)) or abs(np.exp(a) - np.exp(b)) < 1e-15, (k, n, p, a, b)


@pytest.mark.parametrize('name', ['betting', 'cusum12', 'cusum20', 'ewma', 'glr', 'winbin', 'cs'])
def test_label_monitors_vs_reference(name):
    mon, fn = {
        'betting': (M.Betting(), ref.betting),
        'cusum12': (M.CUSUM(0.12), lambda x: ref.cusum(x, 0.12)),
        'cusum20': (M.CUSUM(0.20), lambda x: ref.cusum(x, 0.20)),
        'ewma': (M.EWMA(lam=0.05), ref.ewma),
        'glr': (M.GLRCUSUM(window=60), lambda x: ref.glr(x, W=60)),
        'winbin': (M.WindowBinomial(window=50), lambda x: ref.window_binom(x, W=50)),
        'cs': (M.CSDetector(start_every=10, n_active=6), lambda x: ref.cs_detector(x, G=10, K=6)),
    }[name]
    for arr, x in seqs(4 if name in ('glr', 'cs') else 8, 250 if name in ('glr', 'cs') else 400):
        t, s = mon.path(arr, x)
        r = fn(x.astype(float))
        assert np.array_equal(t, arr)
        assert np.allclose(s, r, rtol=1e-10, atol=1e-10), (name, np.max(np.abs(s - r)))


@pytest.mark.parametrize('kind', ['SR', 'CUSUM'])
def test_edetector_vs_literal_sums(kind):
    lam = np.array([0.2, 0.5, 0.9]); w = np.array([0.5, 0.3, 0.2])
    mon = M.EDetector(kind=kind, lambdas=lam, weights=w)
    for arr, x in seqs(4, 120):
        _, s = mon.path(arr, x)
        r = ref.edetector_direct(x, lam / 0.1, w, kind=kind)
        assert np.allclose(s, r, rtol=1e-9, atol=1e-9)


def test_msr_equals_esr_with_original_grid():
    m1 = M.MixtureSR(); m2 = M.EDetector(kind='SR', lambdas=M.MixtureSR.LAM * 0.1)
    for arr, x in seqs():
        assert np.allclose(m1.path(arr, x)[1], m2.path(arr, x)[1], rtol=1e-12, atol=1e-12)


def test_compute_baseline_weights():
    lam, w, info = M.compute_baseline(1e-3, 0.02 * 0.1 / 0.81, 0.09 / 0.02 ** 2, 1000, v_min=1.0)
    assert np.isclose(w.sum(), 1) and np.all(np.diff(lam) < 0) and np.all((lam > 0) & (lam < 1))
    assert info['g'] > np.log(1e3)


def test_periodic_vs_reference_and_restart_schedule():
    m = M.Periodic()
    for arr, x in seqs():
        H = int(arr[-1]) + 1234
        t, s = m.path(arr, x, H=H)
        ct, cs = ref.periodic(x, arr, H)
        assert np.array_equal(t, ct) and np.allclose(s, cs, rtol=1e-9, atol=1e-9)
    assert m.native_threshold(29999) == pytest.approx(-np.log(0.02 / 10), abs=1e-8)


def test_adwin_and_ddm_first_alarm_equivalence():
    rng = np.random.default_rng(3)
    ad, dd = M.ADWIN(), M.DDM()
    nchk = 0
    for rep in range(30):
        n = 600
        p = np.where(np.arange(n) < 300, 0.08, 0.25)
        x = (rng.random(n) < p).astype(np.int8); arr = np.arange(1, n + 1, dtype=np.int64)
        t, s = ad.path(arr, x)
        for delta in [0.5, 0.05, 0.002]:
            i_ref = ref.adwin_first_alarm(x, delta)
            hit = np.flatnonzero(s > np.log(1 / delta))
            assert (hit[0] if len(hit) else -1) == i_ref
            nchk += 1
        t, s = dd.path(arr, x)
        hit = np.flatnonzero(s > 3.0)
        assert (hit[0] if len(hit) else -1) == ref.ddm_first_alarm(x)
    assert nchk == 90


def test_batch_drivers_consistent_with_path():
    b = G.generate(G.nominal_config(pi=0.1, q=0.2, d=500), 'test', 9, 0, 40)
    for name, m in M.default_suite().items():
        grid = np.array([0.5, 2.0, 5.0, 8.0])
        ev = m.evaluate(b, grid, (10000, b.H))
        roff, rt, rv = m.records(b)
        for i in range(b.n):
            s = b.stream(i)
            t, v = m.path(s['arrival'], s['loss'], H=b.H)
            for g, h in enumerate(grid):
                hit = np.flatnonzero(v > h)
                exp = t[hit[0]] if len(hit) else np.inf
                assert ev['alarm'][i, g] == exp, name
            for c, cp in enumerate((10000, b.H)):
                sel = t <= cp
                assert ev['max'][i, c] == (v[sel].max() if sel.any() else -np.inf)
        for g, h in enumerate(grid):
            assert np.array_equal(M.records_first_crossing(roff, rt, rv, h), ev['alarm'][:, g]), name


def test_restart_counts_match_manual():
    b = G.generate(G.null_config(pi=0.1, q=1.0, d=0, H=20000, r=0.12), 'test', 10, 0, 20)
    for m, h in [(M.CUSUM(0.2), 2.0), (M.Betting(), 1.0), (M.Periodic(), 2.0)]:
        r = m.restart_run(b, h)
        for i in range(b.n):
            s = b.stream(i); tl, n = 0, 0
            while True:
                sel = s['arrival'] > tl
                t, v = m.path(s['arrival'][sel], s['loss'][sel], H=b.H, t0=tl)
                hit = np.flatnonzero(v > h)
                if not len(hit):
                    break
                tl = t[hit[0]]; n += 1
                if tl >= b.H:
                    break
            assert r['n_alarms'][i] == n


def test_weights_unit_weight_identity_and_rejection():
    b = G.generate(G.nominal_config(pi=0.1, q=0.3, d=0), 'test', 11, 0, 30)
    w = np.ones(len(b.loss))
    for m in [M.Betting(), M.CUSUM(0.2), M.MixtureSR(), M.EDetector(kind='SR')]:
        assert np.allclose(m.max_stat(b), m.max_stat(b, weights=w), atol=1e-12)
    with pytest.raises(NotImplementedError):
        M.EWMA().max_stat(b, weights=w)


def test_mixture_rescaling_long_streams_vs_original_logdomain():
    """Statistic reaching hundreds of nats (rescaling active) still equals run.py's logsumexp mSR."""
    from revision import legacy as L
    rng = np.random.default_rng(7)
    x = (rng.random(6000) < np.where(np.arange(6000) < 1000, 0.05, 0.35)).astype(np.int8)
    arr = np.arange(1, 6001, dtype=np.int64)
    s = M.MixtureSR().path(arr, x)[1]
    ref_ = L.original_trajectories(x)[3]
    assert s.max() > 500
    assert np.allclose(s, ref_, rtol=1e-11, atol=1e-9)
    # e-CUSUM: rescaled linear domain vs literal running max on a short stream with large values
    lam = np.array([0.3, 0.9]); w = np.array([0.5, 0.5])
    xs = x[900:1150]
    sc = M.EDetector(kind='CUSUM', lambdas=lam, weights=w).path(arr[:250], xs)[1]
    assert np.allclose(sc, ref.edetector_direct(xs, lam / 0.1, w, kind='CUSUM'), rtol=1e-9, atol=1e-9)
