"""Minimum audit rate q* as a continuous function of (pi, r1, d, D):
 - converse: no monitor with FA<=alpha under the admissible path (0.05 -> 0.10) can reach target below q*_oracle
 - achievable: matched integer CUSUM calibrated EXACTLY to FA<=alpha under constant r=0.10 over H reaches target at q*_cusum
"""
import csv,itertools,math,sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from exact_cusum import scores,detection_prob,calibrate
from converse import np_power
R0,RPRE,ALPHA,TARGET,TAU,H,SB=0.10,0.05,0.02,0.80,10000,29999,40
def oracle(pi,r1,d,D,q): return np_power(max(0,D-d),pi*q,R0,r1,ALPHA)
def cusum(pi,r1,d,D,q):
    A,B,s=scores(r1,R0,SB); c,fa=calibrate(H,d,pi*q,A,B,ALPHA,R0)
    return detection_prob(H,d,pi*q,TAU,RPRE,r1,D,A,B,c)[0],c/s,fa
def solve(f,lo=1e-4,hi=1.0,rel=0.005):
    if f(hi)<TARGET: return math.inf
    while hi/lo>1+rel:
        mid=math.sqrt(lo*hi)
        if f(mid)>=TARGET: hi=mid
        else: lo=mid
    return hi
def job(args):
    pi,r1,d,D=args
    qo=solve(lambda q:oracle(pi,r1,d,D,q))
    qc=solve(lambda q:cusum(pi,r1,d,D,q)[0])
    extra=cusum(pi,r1,d,D,qc) if math.isfinite(qc) else (None,None,None)
    nlab=lambda q: pi*q*max(0,D-d) if math.isfinite(q) else math.inf
    return dict(prevalence=pi,post_risk=r1,delay=d,deadline=D,q_oracle=qo,q_cusum=qc,ratio=qc/qo if math.isfinite(qc) and math.isfinite(qo) else None,
                labels_oracle=nlab(qo),labels_cusum=nlab(qc),cusum_pD_at_q=extra[0],cusum_logh=extra[1],cusum_FA=extra[2])
if __name__=='__main__':
    grid=list(itertools.product([0.003,0.01,0.03,0.10],[0.12,0.15,0.20,0.30],[0,500,2000],[2000,5000,10000,20000]))
    grid=[g for g in grid if g[3]>g[2]]
    with ProcessPoolExecutor(8) as ex:
        out=[]
        for i,o in enumerate(ex.map(job,grid)):
            out.append(o); print(i,o,flush=True)
    w=csv.DictWriter(open('qstar_grid.csv','w'),fieldnames=out[0]);w.writeheader();w.writerows(out)
