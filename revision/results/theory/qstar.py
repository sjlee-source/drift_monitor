"""v2: minimum audit-rate curve with a safe search (Codex review 2026-11-11).
CUSUM power under per-q exact recalibration need not be monotone in q, so we
(1) scan q on a log grid, (2) take the first grid point with p_D >= target,
(3) refine by a dense local scan between the last failing and first passing grid point,
(4) record any non-monotone (pass -> fail) transitions on the scan.
Oracle power is monotone in q (Blackwell: higher q can simulate lower q by discarding audits); bisection used.
"""
import csv,itertools,math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from exact_cusum import scores,detection_prob,calibrate
import converse
from converse import np_power
R0,RPRE,ALPHA,TARGET,TAU,H=0.10,0.05,0.02,0.80,10000,29999
SB=40
QGRID=np.geomspace(1e-3,1,49)
def oracle(pi,r1,d,D,q): return np_power(max(0,D-d),pi*q,R0,r1,ALPHA)
def cusum(pi,r1,d,D,q,sb=SB):
    A,B,s=scores(r1,R0,sb); c,fa=calibrate(H,d,pi*q,A,B,ALPHA,R0)
    return detection_prob(H,d,pi*q,TAU,RPRE,r1,D,A,B,c)[0],c,c/s,fa
def bisect(f,lo=1e-4,hi=1.0,rel=0.002):
    if f(hi)<TARGET: return math.inf
    while hi/lo>1+rel:
        mid=math.sqrt(lo*hi)
        if f(mid)>=TARGET: hi=mid
        else: lo=mid
    return hi
def job(args):
    pi,r1,d,D=args
    qo=bisect(lambda q:oracle(pi,r1,d,D,q))
    lab=lambda q: pi*q*max(0,D-d) if math.isfinite(q) else math.inf
    res=dict(prevalence=pi,post_risk=r1,delay=d,deadline=D,q_oracle=qo,labels_oracle=lab(qo))
    # CUSUM: only scan q >= q_oracle (below it no monitor can reach the target)
    if not math.isfinite(qo):
        res.update(q_cusum=math.inf,labels_cusum=math.inf,ratio=None,nonmonotone=0,cusum_pD_at_q=None,cusum_logh=None,cusum_FA=None); return res
    grid=[q for q in QGRID if q>=qo*0.999]
    if grid[0]>qo: grid=[qo]+grid
    vals=[cusum(pi,r1,d,D,q)[0] for q in grid]
    passed=[v>=TARGET for v in vals]
    nonmono=sum(1 for a,b in zip(passed,passed[1:]) if a and not b)
    if not any(passed):
        res.update(q_cusum=math.inf,labels_cusum=math.inf,ratio=None,nonmonotone=nonmono,cusum_pD_at_q=None,cusum_logh=None,cusum_FA=None); return res
    i=passed.index(True)
    if i==0: qc=grid[0]
    else:
        fine=np.geomspace(grid[i-1],grid[i],41)[1:]
        fv=[cusum(pi,r1,d,D,q)[0] for q in fine]
        fp=[v>=TARGET for v in fv]
        nonmono+=sum(1 for a,b in zip(fp,fp[1:]) if a and not b)
        qc=fine[fp.index(True)]
    pD,c,logh,fa=cusum(pi,r1,d,D,qc)
    res.update(q_cusum=qc,labels_cusum=lab(qc),ratio=qc/qo,nonmonotone=nonmono,cusum_pD_at_q=pD,cusum_logh=logh,cusum_FA=fa)
    return res
if __name__=='__main__':
    grid=list(itertools.product([0.003,0.01,0.03,0.10],[0.12,0.15,0.20,0.30],[0,500,2000],[2000,5000,10000,20000]))
    grid=[g for g in grid if g[3]>g[2]]
    out=[]
    with ProcessPoolExecutor(10) as ex:
        for i,o in enumerate(ex.map(job,grid)):
            out.append(o); print(i,o,flush=True)
    w=csv.DictWriter(open('qstar_grid.csv','w'),fieldnames=list(out[0].keys())+[k for k in out[-1] if k not in out[0]]);w.writeheader();w.writerows(out)
    print('max omitted N mass',converse.OMITTED[0])
