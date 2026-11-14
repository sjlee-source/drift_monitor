"""Throughput benchmark: python -m revision.bench [--n 20000] [--q 0.5]  (8 numba threads)."""
import argparse
import json
import os
import time

os.environ.setdefault('NUMBA_NUM_THREADS', '8')
import numpy as np

from revision import generators as G, monitors as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=5000)
    ap.add_argument('--pi', type=float, default=0.1)
    ap.add_argument('--q', type=float, default=0.5)
    ap.add_argument('--H', type=int, default=29999)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    cfg = G.nominal_config(pi=a.pi, q=a.q, d=500, H=a.H)
    suite = M.default_suite()
    G.generate(cfg, 'pilot', 0, 0, 4)                 # compile
    for m in suite.values():
        m.max_stat(G.generate(cfg, 'pilot', 0, 0, 4))
    tg = 0.0; tm = {k: 0.0 for k in suite}; labels = 0
    for s in range(0, a.n, a.chunk):
        t = time.perf_counter(); b = G.generate(cfg, 'pilot', 1, 0, np.arange(s, min(a.n, s + a.chunk)))
        tg += time.perf_counter() - t; labels += int(b.offsets[-1])
        for k, m in suite.items():
            t = time.perf_counter(); m.evaluate(b, (5.0,), (b.H,)); tm[k] += time.perf_counter() - t
    res = dict(n=a.n, H=a.H, pi=a.pi, q=a.q, mean_labels=labels / a.n, threads=os.environ['NUMBA_NUM_THREADS'],
               generate_seconds=tg, generate_streams_per_s=a.n / tg,
               monitor_seconds=tm, monitor_streams_per_s={k: a.n / v for k, v in tm.items()})
    # Python hook path (per-stream cost)
    class Q(G.AuditPolicy):
        epoch = 1
        def decide(self, v):
            return a.q
    cfgh = G.StreamConfig(**{**cfg.__dict__, 'policy': Q()})
    t = time.perf_counter(); G.generate(cfgh, 'pilot', 2, 0, 20); res['hook_epoch1_streams_per_s'] = 20 / (time.perf_counter() - t)
    Q.epoch = 100
    t = time.perf_counter(); G.generate(cfgh, 'pilot', 2, 0, 50); res['hook_epoch100_streams_per_s'] = 50 / (time.perf_counter() - t)
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, 'w').write(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
