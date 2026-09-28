"""Frozen E1/E2 CPU simulation; independent seed roles and no test before null gate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import csv,json,time,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, binom
OUT=Path(__file__).resolve().parent
DES=OUT
METHODS=['WH_finiteH','CUSUM_p1_012','CUSUM_p1_020','mSR_finiteH','periodic_binomial']
LAM=np.array([.02,.05,.10,.20,.40])/.09
ROLES={'pilot':10000000,'calibration':20000000,'validation':30000000,'final':40000000}
N={'pilot':100,'calibration':5000,'validation':3000,'final':2000}
VERSION=20260922
START=time.time()
def js(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def status(**kw):js(OUT/'status.json',dict(updated=time.strftime('%Y-%m-%d %H:%M:%S %z'),pid=os.getpid(),elapsed_seconds=time.time()-START,**kw))
def read(e):
    name='E1_detector_conditions.csv' if e==1 else 'E2_production_conditions.csv'
    rows=list(csv.DictReader((DES/name).open()))
    for c in rows:
      for k,v in list(c.items()):
        try:c[k]=float(v) if '.' in v else int(v)
        except (ValueError,TypeError):pass
    return rows

def trajectory(x):
    x=np.asarray(x,float);n=len(x)
    if not n:return [np.empty(0)]*4
    cs=np.r_[0.,np.cumsum(x)];t=np.arange(n)
    prev=(cs[t]-cs[np.maximum(t-50,0)])/np.maximum(np.minimum(t,50),1)
    lam=np.clip(4*np.maximum(prev-.1,0),0,9.)
    wh=np.cumsum(np.log1p(lam*(x-.1)))
    ll=x*np.log(.12/.10)+(1-x)*np.log(.88/.90)
    s=np.cumsum(ll);cu=np.maximum(s-np.minimum.accumulate(np.r_[0.,s])[:-1],0)
    logs=np.cumsum(np.log1p(LAM[:,None]*(x[None,:]-.1)),axis=1)
    before=np.concatenate([np.zeros((5,1)),logs[:,:-1]],axis=1)
    sr=logsumexp(logs+np.logaddexp.accumulate(-before,axis=1),axis=0)-np.log(5)
    ll2=x*np.log(.20/.10)+(1-x)*np.log(.80/.90)
    s2=np.cumsum(ll2);cu2=np.maximum(s2-np.minimum.accumulate(np.r_[0.,s2])[:-1],0)
    return [wh,cu,cu2,sr]

def generate(e,c,role,i):
    cid=int(str(c['condition_id']).split('_')[-1]);rng=np.random.default_rng(np.random.SeedSequence([ROLES[role],VERSION,e,cid,i]))
    H=c['horizon'];tau=c['first_change'];pre=c['pre_risk'];post=c['post_risk'];harmful=post>.1
    if e==1:
      origin=np.arange(1,H+1);request=origin;arrival=origin;p=np.full(H,pre)
      if harmful:
        ramp=c['ramp'];p[tau-1:]=post if ramp==0 else pre+(post-pre)*np.minimum((origin[tau-1:]-tau+1)/ramp,1.)
      x=(rng.random(H)<p).astype(np.int8);harm=int(origin[p>.1][0]) if harmful else 0
      all_defects=H;totalmiss=int(x.sum());production_misses=x
    else:
      prod=np.arange(1,H+1);truth=rng.random(H)<c['prevalence'];audit=rng.random(H)<c['audit_rate'];missu=rng.random(H)
      p=np.where(prod>=tau,post,pre);origin=prod[truth&audit];request=prod[audit]
      keep=origin+c['label_delay']<=H;origin=origin[keep];arrival=origin+c['label_delay']
      x=(missu[origin-1]<p[origin-1]).astype(np.int8);harm=tau if harmful else 0
      all_defects=int(truth.sum());production_misses=(truth&(missu<p)).astype(np.int8);totalmiss=int(production_misses.sum())
    return x,origin,arrival,request,harm,all_defects,totalmiss,production_misses

def run_cell(e,c,role,thresholds=None):
    path=OUT/f'E{e}'/role/(c['condition_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return dict(np.load(path))
    n=N[role];maxima=np.full((n,5),-np.inf);alarms=np.zeros((n,5),int);alarmidx=np.zeros_like(alarms);origins=np.zeros_like(alarms)
    reqalarm=np.zeros_like(alarms);arralarm=np.zeros_like(alarms)
    requests=np.zeros(n,int);arrived=np.zeros(n,int);defects=np.zeros(n,int);misses=np.zeros(n,int);harms=np.zeros(n,int)
    burden=np.zeros((n,5),int);requests_deadline=np.zeros((n,3),int);positive_arrivals_deadline=np.zeros((n,3),int)
    tick=time.time()
    for i in range(n):
      x,origin,arrival,request,harm,nd,nmiss,production_misses=generate(e,c,role,i);tr=trajectory(x)
      maxima[i,:4]=[v.max() if len(v) else -np.inf for v in tr]
      checks,pvalues=periodic(x,arrival,c['horizon'])
      maxima[i,4]=float((-np.log(np.maximum(pvalues,1e-300))).max())
      burden[i,:]=nmiss
      for di,D in enumerate([5000,10000,20000]):
        deadline=min(c['horizon'],c['first_change']+D-1)
        requests_deadline[i,di]=np.searchsorted(request,deadline,side='right')
        positive_arrivals_deadline[i,di]=np.searchsorted(arrival,deadline,side='right')
      miss_cumsum=np.cumsum(production_misses)
      requests[i]=len(request);arrived[i]=len(x);defects[i]=nd;misses[i]=nmiss;harms[i]=harm
      if thresholds is not None:
        for j,v in enumerate(tr):
          hit=np.flatnonzero(v>thresholds[j])
          if len(hit):
            k=hit[0];burden[i,j]=miss_cumsum[arrival[k]-1];alarms[i,j]=arrival[k];alarmidx[i,j]=k+1;origins[i,j]=origin[k]
            reqalarm[i,j]=np.searchsorted(request,arrival[k],side='right')
            arralarm[i,j]=np.searchsorted(request,arrival[k]-(c.get('label_delay',0)),side='right')
      if thresholds is not None:
        hit=np.flatnonzero(pvalues<=.02/len(checks))
        if len(hit):
          t=int(checks[hit[0]]);j=4;k=int(np.searchsorted(arrival,t,side='right'))
          alarms[i,j]=t;alarmidx[i,j]=k;origins[i,j]=int(origin[k-1]) if k else 0
          burden[i,j]=miss_cumsum[t-1];reqalarm[i,j]=np.searchsorted(request,t,side='right');arralarm[i,j]=np.searchsorted(request,t-c.get('label_delay',0),side='right')
      if i%250==0:status(state='running',experiment=f'E{e}',phase=role,condition=c['condition_id'],replicate=i,n=n)
    result=dict(misses_through_first_alarm_or_horizon=burden,requests_by_deadline=requests_deadline,positive_arrivals_by_deadline=positive_arrivals_deadline,maxima=maxima,alarms=alarms,alarm_positive_index=alarmidx,alarm_origin=origins,audit_requests_at_alarm=reqalarm,audit_arrivals_at_alarm=arralarm,requests=requests,arrived_defects=arrived,total_defects=defects,total_misses=misses,harms=harms)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**result);temp.replace(path)
    print(e,role,c['condition_id'],round(time.time()-tick,2),flush=True)
    return result

def periodic(x,arrival,H):
    checks=np.unique(np.r_[np.arange(3000,H+1,3000),H])
    edges=np.r_[0,np.searchsorted(arrival,checks,side='right')]
    cs=np.r_[0,np.cumsum(x)];n=np.diff(edges);k=np.diff(cs[edges])
    pv=np.where(n>0,binom.sf(k-1,n,.1),1.)
    return checks,pv

def verify():
    errors=[]
    for seed in range(12):
      x=np.random.default_rng(seed).binomial(1,.1 if seed<6 else .3,500);tr=trajectory(x)
      wh=0.;cu=0.;cu2=0.;r=np.zeros(5);history=[];out=[[],[],[],[]]
      for v in x:
        prev=np.mean(history[-50:]) if history else 0.;lam=min(9.,4*max(prev-.1,0))
        wh+=np.log1p(lam*(v-.1));cu=max(0.,cu+v*np.log(1.2)+(1-v)*np.log(.88/.9));cu2=max(0.,cu2+v*np.log(2.)+(1-v)*np.log(.8/.9));r=(r+1)*(1+LAM*(v-.1))
        history.append(v)
        for z,y in zip(out,[wh,cu,cu2,np.log(r.mean())]):z.append(y)
      for j in range(4):
        err=float(np.max(np.abs(tr[j]-out[j])));assert err<1e-8,(seed,j,err);errors.append(err)
    checks,pv=periodic(np.array([1,0,1]),np.array([3000,3001,6000]),6001)
    assert checks.tolist()==[3000,6000,6001]
    assert np.allclose(pv,[.1,.19,1.])
    assert periodic(np.array([],int),np.array([],int),6000)[1].tolist()==[1.,1.]
    for c in read(2):
      x,origin,arrival,request,harm,nd,nmiss,pm=generate(2,c,'pilot',0)
      assert len(x)==len(origin)==len(arrival) and np.all(arrival<=c['horizon'])
      assert np.all(arrival==origin+c['label_delay'])
      if c['regime']=='null':assert harm==0
      else:assert harm>=c['first_change']
    js(OUT/'numerical_validation.json',dict(status='passed',max_abs_error=max(errors),checks=['48 scalar-vector recurrences','all 136 E2 clocks','periodic boundary includes arrival at check','empty interval never alarms','final H check','CUSUM p1=.12 and .20']))

def pilot():
    if (OUT/'pilot_runtime.json').exists():return
    verify();timings={}
    for e in [2]:
      # Longest path for resource estimate, development streams only.
      c=max(read(e),key=lambda c:c['horizon']*(1 if e==1 else c['prevalence']*c['audit_rate']))
      t=time.time();run_cell(e,c,'pilot',[np.log(20)]*5);timings[f'E{e}_100_seconds']=time.time()-t
    js(OUT/'pilot_runtime.json',dict(timings=timings,planned_counts=N,counts_unchanged=True,performance_not_used_for_design=True))

def cal_key(e,c):
    if e==1:return 'common25000'
    return f"pi{c['prevalence']}_q{c['audit_rate']}_d{c['label_delay']}_H{c['horizon']}"
def calibration(e,conditions):
    file=OUT/f'E{e}'/'thresholds.json'
    if file.exists():return json.loads(file.read_text())
    cells={}
    for c in conditions:
      key=cal_key(e,c)
      if key in cells:continue
      cc=dict(c,pre_risk=.1,post_risk=.1,regime='null')
      if e==1:cc.update(horizon=25000,pre_length=20000,first_change=20001,condition_id='E1_900')
      else:cc['condition_id']=f"E2_{900+len(cells):03d}"
      cells[key]=cc
    ans={};rank=int(np.ceil((N['calibration']+1)*.98))-1
    for key,c in cells.items():
      r=run_cell(e,c,'calibration');h=np.sort(r['maxima'],axis=0)[rank].tolist();h[4]=float(-np.log(.02/len(np.unique(np.r_[np.arange(3000,c['horizon']+1,3000),c['horizon']]))))
      ans[key]=h
    js(file,ans);return ans

def phase(e):
    conditions=read(e);th=calibration(e,conditions)
    nulls=[c for c in conditions if c['regime']=='null'];family=len(nulls)*5;gate=[]
    for c in nulls:
      r=run_cell(e,c,'validation',th[cal_key(e,c)])
      for j,m in enumerate(METHODS):
        k=int((r['alarms'][:,j]>0).sum());n=N['validation'];upper=1. if k==n else float(beta.ppf(1-.05/family,k+1,n-k))
        gate.append(dict(condition=c['condition_id'],method=m,n=n,alarms=k,rate=k/n,simultaneous_upper=upper,pass_gate=upper<=.05))
    operational=all(g['pass_gate'] for g in gate)
    js(OUT/f'E{e}'/'validation_gate.json',dict(family_size=family,confidence=.95,gate_rule='all operational methods x all planned null cells upper<=.05',passed=operational,native_is_reference_not_gate=False,results=gate))
    if not operational:
      print('FINAL BLOCKED null gate',e,flush=True);return False
    for c in conditions:
      run_cell(e,c,'final',th[cal_key(e,c)])
    summarize(e,conditions)
    return True

def summarize(e,conditions):
    rows=[]
    for c in conditions:
      r=dict(np.load(OUT/f'E{e}'/'final'/(c['condition_id']+'.npz')));n=len(r['harms']);harm=r['harms'];H=c['horizon'];isharm=c['regime']!='null'
      for j,m in enumerate(METHODS):
        a=r['alarms'][:,j];pre=(a>0)&(a<harm) if isharm else np.zeros(n,bool);det=(a>=harm)&(a>0) if isharm else np.zeros(n,bool)
        row=dict(condition=c['condition_id'],method=m,regime=c['regime'],n=n,any_alarm=int((a>0).sum()),prealarm=int(pre.sum()),detected=int(det.sum()),no_alarm=int((a==0).sum()),mean_requests=float(r['requests'].mean()),mean_arrived_defects=float(r['arrived_defects'].mean()),mean_total_misses=float(r['total_misses'].mean()))
        for D in ([500,1000,5000] if e==1 else [5000,10000,20000]):
          k=int((det&(a<=np.minimum(H,harm+D-1))).sum());row[f'detect_D{D}']=k;row[f'detect_D{D}_rate']=k/n
          row[f'detect_D{D}_lo']=0. if k==0 else float(beta.ppf(.025,k,n-k+1));row[f'detect_D{D}_hi']=1. if k==n else float(beta.ppf(.975,k+1,n-k))
        row['conditional_detected_delay']=float(np.mean(a[det]-harm[det])) if det.any() else ''
        row['restricted_delay_no_prealarm']=float(np.mean(np.where(det,a-harm,H+1-harm)[~pre])) if isharm and (~pre).any() else ''
        rows.append(row)
    with (OUT/f'E{e}'/'summary.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    config=dict(version=VERSION,methods=METHODS,counts=N,seed_roles=ROLES,seed_key=['role','version','experiment','condition','replicate'],risk=.1,cusum_p1=[.12,.20],periodic_interval=3000,periodic_total_alpha=.02,msr_lambda=LAM.tolist(),operational_alarm='strict > calibrated log threshold',calibration_target=.02,validation_cap=.05,E2_parameter_holdout='intermediate settings included with own null calibration; not transfer holdout',E1_secondary_CUSUM='not part of E2v2',source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),condition_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DES.glob('*.csv')})
    existing=OUT/'execution_config.json'
    if existing.exists():assert json.loads(existing.read_text())==config,'config changed; use new output directory'
    else:js(existing,config)
    pilot()
    if '--pilot-only' in sys.argv:status(state='pilot_complete');return
    results={f'E{e}':phase(e) for e in [2]}
    status(state='complete' if all(results.values()) else 'finished_with_gate_blocks',results=results)
if __name__=='__main__':
    import fcntl
    _lock=(OUT/'run.lock').open('w');fcntl.flock(_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception as exc:
      status(state='failed',error=repr(exc));raise
