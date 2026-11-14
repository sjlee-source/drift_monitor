"""(b) Original five monitors reproduce the archived operating_map.csv.

Part 1 (bit-level): archived 'final' streams are regenerated with the original seeding via
legacy.py and the new monitors are run with the archived thresholds; success / prealarm /
no-alarm counts must equal the archive except for streams whose decision flips under a
1e-9 perturbation of the threshold (float ties of lattice-valued statistics).
Part 2 (statistical): independent streams from the NEW generator reproduce archived rates
within Monte Carlo error (two-sample z, |z| < 4).
"""
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from revision import generators as G, monitors as M, legacy as L

OUT = Path(__file__).resolve().parent / '_results'
OUT.mkdir(exist_ok=True)
NAMES = ['WH_finiteH', 'CUSUM_p1_012', 'CUSUM_p1_020', 'mSR_finiteH', 'periodic_binomial']
COND_IDS = ['E2_089', 'E2_088', 'E2_001', 'E2_121', 'E2_057', 'E2_108', 'E2_132']


def omap():
    rows = list(csv.DictReader((L.EVIDENCE / 'operating_map.csv').open()))
    return {(r['condition'], r['method'], int(r['deadline_D'])): r for r in rows}


def thresholds(c):
    th = L.archived_thresholds()[L.cal_key(c)]
    mons = M.original_monitors()
    return mons, [th[0], th[1], th[2], th[3], mons[4].native_threshold(int(c['horizon']))]


@pytest.mark.parametrize('cid', COND_IDS)
def test_bit_level_reproduction(cid):
    c = L.conditions()[cid]
    if c['regime'] == 'null':
        pytest.skip('operating map holds harmful conditions only')
    om = omap()
    b = L.legacy_batch(c, 'final', 2000)
    tau, H = int(c['first_change']), int(c['horizon'])
    mons, ths = thresholds(c)
    out = []
    for j, (m, h) in enumerate(zip(mons, ths)):
        a = m.first_alarm(b, h)
        ap, am = m.first_alarm(b, h + 1e-9 * max(1, abs(h))), m.first_alarm(b, h - 1e-9 * max(1, abs(h)))
        amb = int(np.sum(ap != am))
        for D in (5000, 10000, 20000):
            r = om[(cid, NAMES[j], D)]
            k = int(((a >= tau) & (a <= min(H, tau + D - 1))).sum())
            pre = int((a < tau).sum()); na = int(np.isinf(a).sum())
            out.append(dict(cond=cid, method=NAMES[j], D=D, new=k, archived=int(r['successes']), prealarm_new=pre,
                            prealarm_arch=int(r['prealarms']), noalarm_new=na, noalarm_arch=int(r['no_alarm']),
                            ambiguous_streams=amb))
            assert abs(k - int(r['successes'])) <= amb
            assert abs(pre - int(r['prealarms'])) <= amb
            assert abs(na - int(r['no_alarm'])) <= amb
    (OUT / f'legacy_{cid}.json').write_text(json.dumps(out, indent=1))


@pytest.mark.parametrize('cid', ['E2_089', 'E2_057'])
def test_new_generator_statistically_matches_archive(cid):
    c = L.conditions()[cid]
    om = omap()
    cfg = G.nominal_config(c['prevalence'], c['audit_rate'], int(c['label_delay']), int(c['horizon']),
                           int(c['first_change']), c['pre_risk'], c['post_risk'])
    n = 10000
    b = G.generate(cfg, 'test', 200, int(cid.split('_')[1]), n)
    tau, H = int(c['first_change']), int(c['horizon'])
    mons, ths = thresholds(c)
    out = []
    for j, (m, h) in enumerate(zip(mons, ths)):
        a = m.first_alarm(b, h)
        for D in (5000, 10000, 20000):
            k1 = int(((a >= tau) & (a <= min(H, tau + D - 1))).sum()); k0 = int(om[(cid, NAMES[j], D)]['successes'])
            p1, p0 = k1 / n, k0 / 2000
            pp = (k1 + k0) / (n + 2000)
            se = np.sqrt(max(pp * (1 - pp), 1e-6) * (1 / n + 1 / 2000))
            z = (p1 - p0) / se
            out.append(dict(cond=cid, method=NAMES[j], D=D, new_rate=p1, archived_rate=p0, z=z))
            assert abs(z) < 4, (NAMES[j], D, p1, p0, z)
    (OUT / f'newgen_vs_archive_{cid}.json').write_text(json.dumps(out, indent=1))
