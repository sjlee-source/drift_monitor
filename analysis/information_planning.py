"""Deterministic analytical comparison using recorded CUSUM thresholds; no fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in (Path.home()/'.local/share/fonts/mfg-times-new-roman').glob('*.TTF'): font_manager.fontManager.addfont(str(fp))
font_manager.findfont('Times New Roman', fallback_to_default=False)
P=Path(__file__).resolve().parents[1];S=P/'supplementary/evidence/simulation';O=P/'supplementary/analysis';O.mkdir(parents=True,exist_ok=True);F=P/'figures';F.mkdir(exist_ok=True)
rows=list(csv.DictReader((S/'operating_map.csv').open()));thresholds=json.loads((S/'E2/thresholds.json').read_text());gate=json.loads((S/'E2/validation_gate.json').read_text());conditions={r['condition_id']:r for r in csv.DictReader((S/'E2_production_conditions.csv').open())}
# Probability that the post-change segment log-likelihood exceeds h at the deadline.
# Exact finite binomial mixture, evaluated numerically without tail truncation.
def endpoint(L,rho,r,p,h):
 n=np.arange(L+1);a=np.log(p/.1);b=np.log((1-p)/.9)
 k=np.floor((h-n*b)/(a-b)).astype(int)+1
 return float(np.sum(binom.pmf(n,L,rho)*binom.sf(k-1,n,r)))
# Tiny exhaustive categorical-sequence check, including strict threshold crossing.
import itertools
for L in range(1,6):
 rho=.2;r=.2;p=.2;h=.3;z=[0,np.log(2),np.log(.8/.9)];probs=[1-rho,rho*r,rho*(1-r)]
 brute=sum(np.prod([probs[i] for i in v]) for v in itertools.product(range(3),repeat=L) if sum(z[i] for i in v)>h)
 assert abs(brute-endpoint(L,rho,r,p,h))<1e-12
out=[]
for row in rows:
 method=row['method']
 if method not in ['CUSUM_p1_012','CUSUM_p1_020']:continue
 c=conditions[row['condition']];r=float(row['post_risk']);p=.12 if method.endswith('012') else .2;idx=1 if p==.12 else 2
 pi=float(row['prevalence']);q=float(row['audit_rate']);d=int(row['delay']);D=int(row['deadline']);tau=int(row['tau']);H=int(c['horizon'])
 key=f'pi{pi}_q{q}_d{d}_H{H}';h=thresholds[key][idx];rho=pi*q
 mu=r*np.log(p/.1)+(1-r)*np.log((1-p)/.9);kl=r*np.log(r/.1)+(1-r)*np.log((1-r)/.9)
 # Horizon null probability upper-bounds early alarms under an identical pre-change prefix.
 null=[g['simultaneous_upper'] for g in gate['results'] if g['method']==method and all(conditions[g['condition']][k]==c[k] for k in ['prevalence','audit_rate','label_delay','horizon']) and float(conditions[g['condition']]['pre_risk'])==float(c['pre_risk'])]
 assert len(null)==1
 a_pre=null[0];prob=endpoint(max(0,D-d),rho,r,p,h);lower=max(0,prob-a_pre)
 out.append(dict(condition=row['condition'],method=method,prevalence=pi,audit_rate=q,delay=d,tau=tau,deadline=D,post_risk=r,alternative=p,log_threshold=h,mean_increment=mu,kl=kl,planning_items=d+h/(rho*mu) if mu>0 else None,endpoint_probability=prob,early_alarm_upper=a_pre,analytical_lower=lower,observed_rate=float(row['rate']),observed_lower=float(row['simultaneous_lo']),observed_upper=float(row['simultaneous_hi']),decision=row['decision']))
with (O/'information_planning.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
matched=[r for r in out if abs(r['post_risk']-r['alternative'])<1e-10]
summary={'rows':len(out),'matched_rows':len(matched),'analytical_supported':sum(r['analytical_lower']>=.8 for r in out),'analytical_supported_empirical_insufficient':sum(r['analytical_lower']>=.8 and r['decision']=='insufficient' for r in out),'lower_exceeds_empirical_upper':sum(r['analytical_lower']>r['observed_upper'] for r in out),'tiny_exhaustive_test':'passed','fit_parameters':0,'source_sha256':{str(x.relative_to(S)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [S/'operating_map.csv',S/'E2/thresholds.json',S/'E2/validation_gate.json']},'examples':[r for r in out if r['method']=='CUSUM_p1_020' and r['post_risk']==.2 and r['prevalence']==.1 and r['delay']==500 and r['tau']==10000 and r['deadline']==10000]}
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7.1,3.3),layout='constrained')
for p,col,mark in [(.12,'#0072B2','o'),(.2,'#D55E00','^')]:
 rr=[r for r in matched if r['alternative']==p]
 axes[0].scatter([(r['deadline']-r['delay'])*r['prevalence']*r['audit_rate']*r['kl']/r['log_threshold'] for r in rr],[r['observed_rate'] for r in rr],s=14,alpha=.65,label=f'Matched CUSUM {p:.2f}',color=col,marker=mark)
 axes[1].scatter([r['analytical_lower'] for r in rr],[r['observed_rate'] for r in rr],s=14,alpha=.65,color=col,marker=mark)
axes[0].axvline(1,color='#666666',ls=':',lw=1);axes[0].set_xscale('log');axes[0].set_xlabel('Expected log evidence / calibrated threshold');axes[0].set_ylabel('Observed timely detection');axes[0].set_title('(a) Information-scale approximation');axes[0].legend(fontsize=7)
axes[1].plot([0,1],[0,1],color='#666666',ls=':',lw=1);axes[1].set_xlabel('Analytical probability lower bound');axes[1].set_ylabel('Observed timely detection');axes[1].set_title('(b) Finite-deadline bound')
for ax in axes:ax.set_ylim(-.03,1.03);ax.axhline(.8,color='#999999',ls='--',lw=.7)
for ext in ['pdf','png']:fig.savefig(F/f'figure5_information_planning.{ext}',dpi=300)
print(json.dumps(summary,indent=2))
