import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,hashlib,time
import numpy as np
from scipy.stats import beta
B=Path(__file__).resolve().parent;S=B/'inputs';G=np.array([2.,4.,8.]);ROLE={'calibration':20000000,'validation':30000000,'final':40000000}
rows=list(csv.DictReader((S/'E2_production_conditions.csv').open()))
for c in rows:
 for k,v in c.copy().items():
  try:c[k]=float(v) if '.' in v else int(v)
  except (ValueError,TypeError):pass

def key(c):return f"pi{c['prevalence']}_q{c['audit_rate']}_d{c['label_delay']}_H{c['horizon']}"
def gen(c,role,i):
 rng=np.random.default_rng(np.random.SeedSequence([ROLE[role],20260922,2,int(c['condition_id'].split('_')[1]),i]));H=c['horizon'];z=rng.random(H)<c['prevalence'];a=rng.random(H)<c['audit_rate'];u=rng.random(H);idx=np.flatnonzero(z&a);idx=idx[idx+c['label_delay']<H];r=np.where(idx+1>=c['first_change'],c['post_risk'],c['pre_risk']);return (u[idx]<r).astype(float),idx+1+c['label_delay']
def wealth(x):
 n=len(x);cs=np.r_[0.,np.cumsum(x)];i=np.arange(n);p=(cs[i]-cs[np.maximum(i-50,0)])/np.maximum(np.minimum(i,50),1)
 return np.cumsum(np.log1p(np.clip(G[:,None]*np.maximum(p-.1,0),0,9)*(x[None,:]-.1)),axis=1)
def alarm(v,arr,h):
 return np.array([arr[z[0]] if len(z:=np.flatnonzero(y>hh)) else 0 for y,hh in zip(v,h)])
def save(name,obj):(B/name).write_text(json.dumps(obj,indent=2)+'\n')
def write(name,rr):
 with (B/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
# Scalar recurrence and exact-boundary checks.
mx=0.
for n in [0,1,49,50,51,300]:
 x=np.random.default_rng(n).binomial(1,.2,n);v=wealth(x)
 for j,g in enumerate(G):
  total=0.;seq=[]
  for i,xx in enumerate(x):
   past=x[max(0,i-50):i].mean() if i else 0.;total+=np.log1p(min(9,g*max(past-.1,0))*(xx-.1));seq.append(total)
  if n:mx=max(mx,float(np.max(abs(v[j]-seq))))
assert mx<1e-12
assert np.array_equal(alarm(np.array([[0.,1.],[0.,1.],[0.,1.]]),np.array([1,2]),[1.,1.,1.]),[0,0,0])
ccs={}
for c in rows:
 if key(c) not in ccs:ccs[key(c)]=dict(c,pre_risk=.1,post_risk=.1,regime='null',condition_id=f'E2_{900+len(ccs):03d}')
selected=[c for c in rows if c['prevalence']==.1 and c['audit_rate'] in [.1,.2,.5] and c['label_delay']==500 and c['first_change']==10000]
thresholds={};original=json.loads((S/'E2/thresholds.json').read_text())
for k in dict.fromkeys(key(c) for c in selected):
 c=ccs[k];vals=[]
 for i in range(5000):
  x,arr=gen(c,'calibration',i);v=wealth(x);vals.append(v.max(axis=1) if len(x) else [-np.inf]*3)
 thresholds[k]=np.sort(vals,axis=0)[int(np.ceil(5001*.98))-1].tolist();assert abs(thresholds[k][1]-original[k][0])<1e-10
 save('status.json',dict(stage='calibration',completed=len(thresholds),updated=time.time()));print('calibrated',k,flush=True)
save('thresholds.json',thresholds)
val=[]
for c in selected:
 if c['regime']!='null':continue
 a=np.array([alarm(wealth(x),arr,thresholds[key(c)]) for i in range(3000) for x,arr in [gen(c,'validation',i)]])
 np.savez_compressed(B/(c['condition_id']+'_validation.npz'),alarms=a)
 for j,g in enumerate(G):
  k=int((a[:,j]>0).sum());upper=float(beta.ppf(1-.05/18,k+1,3000-k)) if k<3000 else 1.
  val.append(dict(condition=c['condition_id'],gain=int(g),alarms=k,n=3000,rate=k/3000,simultaneous_upper=upper,passed=upper<=.05))
 print('validated',c['condition_id'],flush=True)
write('validation.csv',val);assert all(r['passed'] for r in val),'Null gate blocks final'
final=[];matches=[]
for c in selected:
 if c['regime']=='null':continue
 a=np.array([alarm(wealth(x),arr,thresholds[key(c)]) for i in range(2000) for x,arr in [gen(c,'final',i)]])
 ref=np.load(S/'E2/final'/f"{c['condition_id']}.npz")['alarms'][:,0];assert np.array_equal(a[:,1],ref);matches.append(c['condition_id'])
 np.savez_compressed(B/(c['condition_id']+'_final.npz'),alarms=a)
 for j,g in enumerate(G):
  good=(a[:,j]>=10000)&(a[:,j]<=19999);k=int(good.sum());lo=float(beta.ppf(.025/18,k,2001-k)) if k else 0.;hi=float(beta.ppf(1-.025/18,k+1,2000-k)) if k<2000 else 1.
  final.append(dict(condition=c['condition_id'],audit_rate=c['audit_rate'],post_risk=c['post_risk'],gain=int(g),n=2000,successes=k,rate=k/2000,simultaneous_lower=lo,simultaneous_upper=hi,decision='supported' if lo>=.8 else 'insufficient' if hi<.8 else 'inconclusive',prealarms=int(((a[:,j]>0)&(a[:,j]<10000)).sum()),restricted_delay=float(np.where(good,a[:,j]-10000,10000).mean())))
 print('evaluated',c['condition_id'],flush=True)
write('results.csv',final)
save('SUMMARY.json',dict(status='complete',gains=G.tolist(),ncal=5000,nval=3000,nfinal=2000,null_family=18,final_family=18,max_null_upper=max(r['simultaneous_upper'] for r in val),coefficient4_exact_alarm_matches=matches,scalar_vector_max_abs_error=mx,strict_equality_check='passed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=final));save('status.json',dict(stage='complete',updated=time.time()))
