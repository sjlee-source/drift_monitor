"""E3 driver helpers."""
import numpy as np
from revision import rng as R
from . import design as Dz
from .kernel import run_batch, labels_one, ST

INF = np.inf


def kernel_args(pol, cond, path, wf, th=(INF, INF, INF, INF), h_warn=INF, Dwin=Dz.D_PRIMARY):
    c = Dz.CONDITIONS[cond]
    mu1 = Dz.mu1_of_r(Dz.risk_arr(path))
    return (Dz.H, Dz.TAU, c['pi'], mu1, Dz.C_THR, c['d'], wf, pol['ptype'], pol['scope_block'], Dz.BLK, Dz.B,
            Dz.QBAR, Dz.EPS, pol['base'], pol['boost'], (h_warn if pol['warn'] else INF), pol['gtype'],
            Dz.EDGES, pol['G'], pol['a_unc'], pol['gmin_flag'], pol['ipw'], Dz.MU_REF, Dz.R0, Dz.LAM,
            Dz.GAIN, Dz.BCAP, Dz.WIN, Dz.CAPF, np.asarray(th, float), Dwin)


def states_for(role, cond, path, n, start=0):
    return R.stream_states(role, Dz.EXP, 10 * cond + path, np.arange(start, start + n))


def simulate(pol, cond, path, wf, role, n, th=(INF,) * 4, h_warn=INF, chunk=5000, start=0):
    outs, sts, tbs = [], [], []
    args = kernel_args(pol, cond, path, wf, th, h_warn)
    for s0 in range(0, n, chunk):
        st = states_for(role, cond, path, min(chunk, n - s0), start + s0)
        o, s = run_batch(st, *args)
        outs.append(o); sts.append(s); tbs.append(R.uniforms_for_tiebreak(st))
    return np.concatenate(outs), np.concatenate(sts), np.concatenate(tbs)


def alarm(out, tb, mon, thr):
    """First alarm (randomized rule) for mon in {'msr','bet'} given Threshold."""
    j = 2 if mon == 'msr' else 4
    return np.where(tb < thr.gamma, out[:, j + 1], out[:, j])


def stats_df(st):
    import pandas as pd
    return pd.DataFrame(st, columns=ST)
