"""Periodic deadline lower bounds and multiplicity sensitivity; no refitting.
Run from any working directory. Requires numpy and scipy.
"""
from pathlib import Path
from functools import lru_cache
import csv,json,itertools
import numpy as np
from scipy.stats import binom,beta
P=Path(__file__).resolve().parents[1];S=P/'supplementary/evidence/simulation';O=P/'supplementary/analysis'
rows=list(csv.DictReader((S/'operating_map.csv').open()))
conditions={r['condition_id']:r for r in csv.DictReader((S/'E2_production_conditions.csv').open())}
gate=json.loads((S/'E2/validation_gate.json').read_text())
@lru_cache(None)
def batch_probability(length,rho,risk,alpha):
 n=np.arange(length+1)
 # Smallest integer count whose null upper tail is <= alpha.
 k=binom.isf(alpha,n,.1).astype(int)+1
 return float(np.dot(binom.pmf(n,length,rho),binom.sf(k-1,n,risk)))
# Independent implementation: enumerate absent/nonmiss/miss item outcomes,
# directly calculate a tail with explicit binomial coefficients.
from math import comb
errors=[]
for L in range(1,7):
 for rho in [.2,.7]:
  for r in [.12,.4]:
   for alpha in [.001,.02,.2]:
    brute=0.
    for seq in itertools.product(range(3),repeat=L):
     n=sum(x!=0 for x in seq);k=sum(x==2 for x in seq)
     tail=sum(comb(n,j)*.1**j*.9**(n-j) for j in range(k,n+1))
     if n and tail<=alpha:
      brute+=np.prod([[1-rho,rho*(1-r),rho*r][x] for x in seq])
    errors.append(abs(brute-batch_probability(L,rho,r,alpha)))
assert max(errors)<1e-12
out=[]
for row in rows:
 if row['method']!='periodic_binomial':continue
 c=conditions[row['condition']];H=int(c['horizon']);tau=int(row['tau']);d=int(row['delay']);D=int(row['deadline']);rho=float(row['prevalence'])*float(row['audit_rate']);r=float(row['post_risk'])
 looks=list(range(3000,H+1,3000))
 if looks[-1]!=H:looks.append(H)
 prev=0;probs=[];eligible=[]
 for t in looks:
  lo=max(1,prev+1-d);hi=min(H,t-d)
  if lo>=tau and hi>=lo and t<=tau+D-1:
   probs.append(batch_probability(hi-lo+1,rho,r,.02/len(looks)));eligible.append(t)
  prev=t
 union=1-float(np.prod(1-np.array(probs)))
 null=[g['simultaneous_upper'] for g in gate['results'] if g['method']==row['method'] and all(conditions[g['condition']][k]==c[k] for k in ['prevalence','audit_rate','label_delay','horizon']) and float(conditions[g['condition']]['pre_risk'])==float(c['pre_risk'])]
 assert len(null)==1
 out.append(dict(condition=row['condition'],deadline=D,eligible_looks=';'.join(map(str,eligible)),batch_union_probability=union,early_alarm_upper=null[0],analytical_lower=max(0,union-null[0]),observed_rate=float(row['rate']),observed_upper=float(row['simultaneous_hi']),decision=row['decision']))
with (O/'periodic_planning.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
def holm(p,alpha):
 p=np.array(p);order=np.argsort(p);reject=np.zeros(len(p),bool)
 for i,j in enumerate(order):
  if p[j]>alpha/(len(p)-i):break
  reject[j]=True
 return reject
# Two directional families at alpha .025 each maintain joint .05 FWER.
pplus=[binom.sf(int(r['success'])-1,int(r['n']),.8) for r in rows]
pminus=[binom.cdf(int(r['success']),int(r['n']),.8) for r in rows]
sup=holm(pplus,.025);ins=holm(pminus,.025)
sens=[]
for i,r in enumerate(rows):
 k=int(r['success']);n=int(r['n']);m=sum(x['deadline']==r['deadline'] for x in rows)
 lo=beta.ppf(.025/m,k,n-k+1) if k else 0.;hi=beta.ppf(1-.025/m,k+1,n-k) if k<n else 1.
 sens.append(dict(condition=r['condition'],method=r['method'],deadline=r['deadline'],rate=r['rate'],global_bonferroni=r['decision'],global_holm='supported' if sup[i] else 'insufficient' if ins[i] else 'inconclusive',deadline_family_bonferroni='supported' if lo>=.8 else 'insufficient' if hi<.8 else 'inconclusive',deadline_family_lower=lo,deadline_family_upper=hi))
with (O/'multiplicity_sensitivity.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=sens[0]);w.writeheader();w.writerows(sens)
summary={'periodic_cells':len(out),'enumeration_cases':len(errors),'enumeration_max_error':max(errors),'analytic_supported':sum(r['analytical_lower']>=.8 for r in out),'bound_above_empirical_upper':sum(r['analytical_lower']>r['observed_upper'] for r in out),'primary_counts':{name:{decision:sum(r['deadline']=='10000' and r[name]==decision for r in sens) for decision in ['supported','insufficient','inconclusive']} for name in ['global_bonferroni','global_holm','deadline_family_bonferroni']},'example':[r for r in sens if r['condition']=='E2_097' and r['deadline']=='10000'],'scope':'Holm is a directional exact-test classification, not simultaneous confidence intervals. Deadline-specific families do not control the full cross-deadline family.'}
(O/'EXTENSIONS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
