"""Deterministic, numba-friendly random numbers with documented seed namespaces.

Seeding scheme (common random numbers, CRN)
-------------------------------------------
Every replicate stream is keyed by the tuple

    SeedSequence(entropy=MASTER_ENTROPY,
                 spawn_key=(role, experiment, condition, replicate))

where ``role`` is an integer namespace (see ``ROLES``: calibration=1, validation=2,
evaluation=3, ...), ``experiment`` and ``condition`` are user integers and ``replicate`` is
0..N-1.  ``generate_state(4*NCOMP, uint64)`` gives NCOMP independent xoshiro256** states,
one per *component* (defect, audit, miss, noise, delay, score, board, tiebreak).  Each
component draws a fixed number of variates per production item (or per board), so
switching a feature on/off or changing one parameter does not shift the random numbers of
the other components: e.g. the same (role, experiment, condition, replicate) with r1=0.12
and r1=0.20 shares defects, audits and miss uniforms item-by-item.  All monitors evaluated
on one generated batch see identical data.

Different roles never share streams.  Use different ``condition`` ids for conditions that
must be independent (e.g. different audit rates, as in the original study), and the same
id where coupling is wanted.
"""
import numpy as np
from numba import njit

MASTER_ENTROPY = 20261111
ROLES = {'calibration': 1, 'validation': 2, 'evaluation': 3, 'pilot': 4, 'arl': 5,
         'replay': 6, 'test': 7, 'sweep': 8}

# component indices
C_DEFECT, C_AUDIT, C_MISS, C_NOISE, C_DELAY, C_SCORE, C_BOARD, C_TIEBREAK = range(8)
NCOMP = 8


def role_code(role):
    if isinstance(role, (int, np.integer)):
        return int(role)
    return ROLES[role]


def stream_states(role, experiment, condition, replicates, master=MASTER_ENTROPY):
    """Return uint64 array (N, NCOMP, 4) of xoshiro256** states for the given replicates."""
    reps = np.atleast_1d(np.asarray(replicates, dtype=np.int64))
    out = np.empty((len(reps), NCOMP, 4), dtype=np.uint64)
    rc = role_code(role)
    for i, rep in enumerate(reps):
        ss = np.random.SeedSequence(entropy=master, spawn_key=(rc, int(experiment), int(condition), int(rep)))
        out[i] = ss.generate_state(4 * NCOMP, np.uint64).reshape(NCOMP, 4)
    return out


# ---------------------------------------------------------------- xoshiro256** core
@njit(inline='always')
def _rotl(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@njit(inline='always')
def next_u64(s):
    """xoshiro256** (Blackman & Vigna 2018); s is a length-4 uint64 array updated in place."""
    result = _rotl(s[1] * np.uint64(5), 7) * np.uint64(9)
    t = s[1] << np.uint64(17)
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl(s[3], 45)
    return result


@njit(inline='always')
def uniform(s):
    """Uniform on the open interval (0,1) with 53-bit resolution."""
    return ((next_u64(s) >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)


@njit
def normal(s):
    """Standard normal via Box-Muller (uses exactly two uniforms)."""
    u1 = uniform(s)
    u2 = uniform(s)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


@njit
def gamma(s, a):
    """Gamma(a,1), Marsaglia & Tsang (2000); a<1 via the u^(1/a) boost."""
    if a <= 0.0:
        return 0.0
    boost = 1.0
    if a < 1.0:
        boost = uniform(s) ** (1.0 / a)
        a = a + 1.0
    d = a - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    while True:
        x = normal(s)
        v = 1.0 + c * x
        if v <= 0.0:
            continue
        v = v * v * v
        u = uniform(s)
        if np.log(u) < 0.5 * x * x + d - d * v + d * np.log(v):
            return d * v * boost


@njit
def beta_mean_phi(s, mu, phi):
    """Beta with mean mu and precision phi=a+b (ICC = 1/(1+phi) for Bernoulli outcomes)."""
    if mu <= 0.0:
        return 0.0
    if mu >= 1.0:
        return 1.0
    x = gamma(s, mu * phi)
    y = gamma(s, (1.0 - mu) * phi)
    if x + y <= 0.0:
        return mu
    return x / (x + y)


@njit
def geometric0(s, mean):
    """Geometric on {0,1,2,...} with the given mean (inversion; one uniform)."""
    if mean <= 0.0:
        return 0
    p = 1.0 / (1.0 + mean)
    return int(np.floor(np.log(uniform(s)) / np.log1p(-p)))


def uniforms_for_tiebreak(states):
    """One uniform per stream from the TIEBREAK component (for randomized threshold rules)."""
    return _tiebreak(states)


@njit
def _tiebreak(states):
    n = states.shape[0]
    out = np.empty(n)
    for i in range(n):
        st = states[i, C_TIEBREAK].copy()
        out[i] = uniform(st)
    return out
