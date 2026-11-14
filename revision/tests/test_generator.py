"""Generator sanity checks (marginals, dependence, delays, label noise, policies) and no look-ahead."""
import numpy as np
import pytest

from revision import generators as G, rng as R


def se(p, n):
    return np.sqrt(p * (1 - p) / n)


def test_nominal_marginals_and_label_supply():
    cfg = G.nominal_config(pi=0.1, q=0.2, d=500, H=29999, tau=10000, r_pre=0.05, r_post=0.2)
    b = G.generate(cfg, 'test', 1, 0, 2000)
    H = b.H
    assert abs(b.n_defects.mean() / H - 0.1) < 4 * se(0.1, H * 2000)
    assert abs(b.n_audits.mean() / H - 0.2) < 4 * se(0.2, H * 2000)
    exp_labels = 0.02 * (H - 500)
    assert abs(b.n_labels().mean() - exp_labels) < 4 * np.sqrt(exp_labels / 2000)
    pre = b.origin < 10000
    assert abs(b.loss[pre].mean() - 0.05) < 4 * se(0.05, pre.sum())
    assert abs(b.loss[~pre].mean() - 0.2) < 4 * se(0.2, (~pre).sum())
    assert np.all(b.arrival == b.origin + 500) and np.all(b.arrival <= H)
    assert np.allclose(b.prob, 0.2)


def test_determinism_and_namespaces():
    cfg = G.nominal_config()
    a = G.generate(cfg, 'calibration', 3, 7, 50); b = G.generate(cfg, 'calibration', 3, 7, 50)
    c = G.generate(cfg, 'validation', 3, 7, 50)
    assert np.array_equal(a.origin, b.origin) and np.array_equal(a.loss, b.loss)
    assert not np.array_equal(a.offsets, c.offsets)
    # replicate ids are global: generating 10..19 equals the slice of 0..49
    d = G.generate(cfg, 'calibration', 3, 7, np.arange(10, 20))
    assert np.array_equal(d.origin, a.origin[a.offsets[10]:a.offsets[20]])
    # CRN: changing r1 keeps defects/audits/origins identical, misses coupled monotonically
    e = G.generate(G.nominal_config(r_post=0.12), 'calibration', 3, 7, 50)
    assert np.array_equal(a.origin, e.origin)
    post = a.origin >= 10000
    assert np.all(e.loss[post] <= a.loss[post]) and np.array_equal(e.loss[~post], a.loss[~post])


def test_states_shape_and_tiebreak_uniform():
    st = R.stream_states('evaluation', 1, 2, np.arange(5000))
    assert st.shape == (5000, R.NCOMP, 4)
    u = R.uniforms_for_tiebreak(st)
    assert 0 < u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.02


def latent(cfg, n=1, role='test', exp=50):
    pi, r, q = cfg.arrays()
    st = R.stream_states(role, exp, 0, np.arange(n))
    return [G.latent_one(st[i], cfg.H, pi, r, *G._extra(cfg)) for i in range(n)]


@pytest.mark.parametrize('rho', [0.3, 0.6])
def test_markov_persistence(rho):
    cfg = G.StreamConfig(H=400000, pi=0.2, r=0.1, markov_rho=rho)
    Z, Mt, *_ = latent(cfg)[0]
    m = Mt[Z == 1].astype(float)
    assert abs(m.mean() - 0.1) < 0.005
    c = np.corrcoef(m[:-1], m[1:])[0, 1]
    assert abs(c - rho) < 0.02, c


def test_markov_time_varying_marginal():
    H = 400000
    cfg = G.StreamConfig(H=H, pi=0.3, r=G.risk_path(H, 0.05, 0.2, H // 2), markov_rho=0.5)
    Z, Mt, *_ = latent(cfg)[0]
    t = np.arange(1, H + 1)
    pre, post = (Z == 1) & (t < H // 2), (Z == 1) & (t >= H // 2 + 1000)
    assert abs(Mt[pre].mean() - 0.05) < 0.004 and abs(Mt[post].mean() - 0.2) < 0.006


def test_board_icc_on_risk_and_prevalence():
    m = 20
    cfg = G.StreamConfig(H=400000, pi=0.5, r=0.1, board_size=m, icc_r=0.2)
    Z, Mt, *_ = latent(cfg)[0]
    b = np.arange(cfg.H) // m
    # ANOVA-free pair estimator: correlation of misses between two defective items of the same board
    xs, ys = [], []
    for k in range(cfg.H // m):
        idx = np.flatnonzero(Z[k * m:(k + 1) * m] == 1)
        if len(idx) >= 2:
            v = Mt[k * m + idx]
            xs.append(v[0]); ys.append(v[1])
    xs, ys = np.array(xs, float), np.array(ys, float)
    assert abs(Mt[Z == 1].mean() - 0.1) < 0.005
    assert abs(np.corrcoef(xs, ys)[0, 1] - 0.2) < 0.03
    cfg2 = G.StreamConfig(H=400000, pi=0.1, r=0.1, board_size=m, icc_pi=0.1)
    Z2, *_ = latent(cfg2)[0]
    zb = Z2.reshape(-1, m).astype(float)
    assert abs(zb.mean() - 0.1) < 0.004
    assert abs(np.corrcoef(zb[:, 0], zb[:, 1])[0, 1] - 0.1) < 0.02


@pytest.mark.parametrize('delay,mean', [(G.Delay.geometric(300), 300), (G.Delay.negbin(300, 4), 300),
                                        (G.Delay.lognormal(300, 0.8), 300), (G.Delay.batch(1000, 100), None)])
def test_delay_distributions_and_arrival_order(delay, mean):
    cfg = G.StreamConfig(H=29999, pi=0.1, r=0.1, q=0.5, delay=delay)
    b = G.generate(cfg, 'test', 4, delay.kind, 300)
    d = b.arrival - b.origin
    assert np.all(d >= 0) and np.all(b.arrival <= b.H)
    for i in range(b.n):
        s = b.stream(i)
        assert np.all(np.diff(s['arrival']) >= 0)
        same = np.diff(s['arrival']) == 0
        assert np.all(np.diff(s['origin'])[same] > 0)   # ties in origin order
    if mean is not None:
        # unconditional delay mean from the latent draws (observed d is truncated by arrival <= H)
        Z, Mt, Y, DL, UA, S = latent(cfg, role='test', exp=60)[0]
        assert abs(DL.mean() - mean) < 0.05 * mean
        assert np.mean(np.diff(b.stream(0)['origin']) < 0) > 0.01   # genuinely out of origin order
    else:
        assert np.all(b.arrival % 1000 == 0) or np.all((b.arrival % 1000 == 0) | (b.arrival == b.H))
        assert np.all(b.arrival >= b.origin + 100)


def test_label_noise_rates():
    cfg = G.StreamConfig(H=29999, pi=0.2, r=0.2, q=1.0, e01=0.03, e10=0.15)
    b = G.generate(cfg, 'test', 5, 0, 300)
    m0, m1 = b.true_loss == 0, b.true_loss == 1
    assert abs(b.loss[m0].mean() - 0.03) < 4 * se(0.03, m0.sum())
    assert abs((1 - b.loss[m1]).mean() - 0.15) < 4 * se(0.15, m1.sum())


def test_prevalence_step_and_sinusoid():
    H = 200000
    cfg = G.StreamConfig(H=H, pi=G.prevalence_path(H, 'step', 0.05, 0.15, tau=H // 2), r=0.1, q=0.0)
    Z, *_ = latent(cfg)[0]
    assert abs(Z[:H // 2 - 1].mean() - 0.05) < 0.003 and abs(Z[H // 2:].mean() - 0.15) < 0.004
    p = G.prevalence_path(H, 'sinusoid', 0.1, amplitude=0.5, period=1000)
    assert abs(p.mean() - 0.1) < 1e-3 and p.max() == pytest.approx(0.15, abs=1e-4)


def test_scores_and_stratified_policy_ipw():
    sm = G.ScoreModel(mu0=0.0, mu_det=3.0, mu_miss=1.5, sigma=1.0)
    pol = G.StratifiedPolicy(edges=(1.0, 2.0), probs=(0.05, 0.3, 0.9))
    cfg = G.StreamConfig(H=29999, pi=0.1, r=0.1, score=sm, policy=pol)
    b = G.generate(cfg, 'test', 6, 0, 400)
    assert set(np.unique(b.prob)) <= {0.05, 0.3, 0.9}
    naive = b.loss.mean()
    w = 1 / b.prob
    hajek = np.sum(w * b.loss) / np.sum(w)
    assert abs(hajek - 0.1) < 0.006, hajek
    assert abs(naive - 0.1) > 0.02          # informative selection biases the naive mean
    Z, Mt, Y, DL, UA, S = latent(cfg)[0]
    assert abs(S[Z == 0].mean() - 0) < 0.03 and abs(S[(Z == 1) & (Mt == 0)].mean() - 3) < 0.05
    assert abs(S[(Z == 1) & (Mt == 1)].mean() - 1.5) < 0.15


def test_budget_cap_and_force():
    cfg = G.StreamConfig(H=29999, pi=0.1, r=0.1, q=0.5, budget=5999)
    b = G.generate(cfg, 'test', 7, 0, 50)
    assert np.all(b.n_audits == 5999)       # q=0.5 exhausts the budget early
    cfg = G.StreamConfig(H=29999, pi=0.1, r=0.1, q=0.1, budget=5999, force_budget=True)
    b = G.generate(cfg, 'test', 7, 1, 50)
    assert np.all(b.n_audits == 5999)


class ConstQ(G.AuditPolicy):
    def __init__(self, q, epoch):
        self.q, self.epoch = q, epoch

    def decide(self, view):
        return self.q


@pytest.mark.parametrize('epoch', [1, 37])
def test_hook_constant_q_reproduces_fast_path(epoch):
    for delay in [G.Delay.constant(500), G.Delay.geometric(200)]:
        base = G.StreamConfig(H=29999, pi=0.1, r=G.risk_path(29999, 0.05, 0.2, 10000), q=0.2, delay=delay,
                              checkpoints=(10000, 29999))
        fast = G.generate(base, 'test', 8, 0, 8)
        hook = G.generate(G.StreamConfig(**{**base.__dict__, 'policy': ConstQ(0.2, epoch)}), 'test', 8, 0, 8)
        for k in ['offsets', 'origin', 'arrival', 'loss', 'true_loss', 'prob', 'n_audits', 'n_defects', 'n_misses',
                  'n_informative', 'counts_at']:
            assert np.array_equal(getattr(fast, k), getattr(hook, k)), k


class Recorder(G.AuditPolicy):
    """Evidence-adaptive policy that records everything it sees."""
    epoch = 1
    uses_scores = True

    def __init__(self):
        self.log = []

    def reset(self, i):
        self.log = []

    def decide(self, view):
        assert len(view.past_prob) == view.t0 - 1
        if len(view.arrived_arrival):
            assert view.arrived_arrival.max() <= view.t0 - 1
            assert view.arrived_origin.max() <= view.t0 - 1
        assert not view.arrived_loss.flags.writeable
        recent = view.arrived_loss[-30:]
        p = 0.6 if (len(recent) and recent.mean() > 0.12) else 0.1
        self.log.append(p)
        return p


@pytest.mark.parametrize('d', [0, 50])
def test_no_lookahead_in_audit_hook(d):
    H, Tstar = 6000, 3000
    out = []
    for r_post in [0.05, 0.4]:
        pol = Recorder()
        cfg = G.StreamConfig(H=H, pi=0.3, r=G.risk_path(H, 0.05, r_post, Tstar), q=0.2, delay=G.Delay.constant(d),
                             policy=pol, score=G.ScoreModel())
        b = G.generate(cfg, 'test', 12, 0, 1)
        out.append(np.array(pol.log))
    # decisions for items t <= Tstar + d use only labels from items < Tstar  -> identical
    assert np.array_equal(out[0][:Tstar + d], out[1][:Tstar + d])
    assert not np.array_equal(out[0], out[1])           # the policy does react afterwards


def test_generate_parallel_identical_to_serial():
    cfg = G.StreamConfig(**{**G.nominal_config(q=0.1, H=8000, tau=4000).__dict__,
                            'policy': G.WarningBoostPolicy(epoch=25)})
    a = G.generate(cfg, 'test', 21, 0, 6)
    b = G.generate_parallel(cfg, 'test', 21, 0, 6, n_jobs=2, chunk=3)
    for k in ['offsets', 'origin', 'arrival', 'loss', 'prob', 'n_audits']:
        assert np.array_equal(getattr(a, k), getattr(b, k))
