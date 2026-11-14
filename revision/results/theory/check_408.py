"""Exact integer-CUSUM probabilities for all 408 CUSUM operating-map cells vs Monte Carlo."""
import json,csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from exact_cusum import scores, detection_prob
S='/home/sjlee/Research/drift_monitor/supplementary/evidence/simulation/'
T=json.load(open(S+'E2/thresholds.json'))
cond={r['condition_id']:r for r in csv.DictReader(open(S+'E2_production_conditions.csv'))}
rows=[r for r in csv.DictReader(open(S+'operating_map.csv')) if r['method'] in('CUSUM_p1_012','CUSUM_p1_020')]
def job(r):
    c=cond[r['condition']];p=.12 if r['method'].endswith('012') else .2
    pi,q,d,tau,D,H=float(r['prevalence']),float(r['audit_rate']),int(r['delay']),int(r['tau']),int(r['deadline']),int(c['horizon'])
    h=T[f'pi{pi}_q{q}_d{d}_H{H}'][1 if p==.12 else 2]
    A,B,s=scores(p,0.1,80); cc=int(np.floor(h*s))
    pD,early=detection_prob(H,d,pi*q,tau,float(r['pre_risk']),float(r['post_risk']),D,A,B,cc)
    return dict(condition=r['condition'],method=r['method'],prevalence=pi,audit_rate=q,delay=d,tau=tau,deadline=D,post_risk=float(r['post_risk']),
                exact=pD,exact_early=early,mc=float(r['rate']),mc_lo=float(r['simultaneous_lo']),mc_hi=float(r['simultaneous_hi']),decision=r['decision'])
if __name__=='__main__':
    with ProcessPoolExecutor(8) as ex: out=list(ex.map(job,rows))
    w=csv.DictWriter(open('exact_vs_mc_408.csv','w'),fieldnames=out[0]);w.writeheader();w.writerows(out)
    e=np.array([o['exact'] for o in out]);m=np.array([o['mc'] for o in out])
    se=np.sqrt(np.clip(e*(1-e),1e-12,None)/2000)
    z=(m-e)/np.where(se>0,se,np.nan)
    print('cells',len(out),'max|diff|',np.abs(e-m).max(),'mean|diff|',np.abs(e-m).mean())
    print('|z|>1.96:',np.nansum(np.abs(z)>1.96),'|z|>3:',np.nansum(np.abs(z)>3),'of',np.isfinite(z).sum())
    inside=sum(o['mc_lo']<=o['exact']<=o['mc_hi'] for o in out); print('exact within simultaneous MC CI:',inside)
    for lab in ['supported','insufficient','inconclusive']:
        sub=[o for o in out if o['decision']==lab]
        print(lab,len(sub),'exact>=0.8:',sum(o['exact']>=.8 for o in sub))
    inc=[o for o in out if o['decision']=='inconclusive']
    for o in inc: print('  inconclusive:',o['method'],o['prevalence'],o['audit_rate'],o['delay'],o['tau'],o['deadline'],o['post_risk'],'MC',o['mc'],'exact',round(o['exact'],4))
