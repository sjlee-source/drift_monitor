import json,time,sys
import numpy as np
from exact_cusum import *
from converse import np_power
T=json.load(open('/home/sjlee/Research/drift_monitor/supplementary/evidence/simulation/E2/thresholds.json'))
pi,d,tau,H,D,r1=0.1,500,10000,29999,10000,0.2
for q,obs in [(0.1,.478),(0.2,.831),(0.5,.9955)]:
  h=T[f'pi{pi}_q{q}_d{d}_H{H}'][2]
  for sb in [10,20,40,80]:
    A,B,s=scores(0.2,0.1,sb); c=int(np.floor(h*s))
    t=time.time(); pd_,early=detection_prob(H,d,pi*q,tau,0.05,r1,D,A,B,c)
    fa=false_alarm(H,d,pi*q,lambda s_:0.10,A,B,c)
    print(f'q={q} h={h:.3f} B={sb} A={A} c={c}  pD={pd_:.4f} early={early:.4f} FA(0.10)={fa:.4f}  MC={obs}  {time.time()-t:.1f}s',flush=True)
  L=D-d
  for al in [0.02,0.05]:
    print(f'   oracle NP power (0.10 vs 0.20, alpha={al}) = {np_power(L,pi*q,0.1,0.2,al):.4f}')
