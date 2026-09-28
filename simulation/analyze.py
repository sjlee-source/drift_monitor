import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import beta,binomtest
import run
P=Path(__file__).resolve().parent

def write(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 gate=json.loads((P/'E2/validation_gate.json').read_text());assert gate['passed']
 conditions=run.read(2);harmful=[c for c in conditions if c['regime']!='null'];family=len(harmful)*len(run.METHODS)*3
 rows=[];pairs=[];cost=[]
 for c in conditions:
  z=np.load(P/'E2/final'/f"{c['condition_id']}.npz");n=len(z['harms']);assert n==2000
  a=z['alarms'];tau=c['first_change'];H=c['horizon']
  for j,m in enumerate(run.METHODS):
   b=z['misses_through_first_alarm_or_horizon'][:,j]
   cost.append(dict(condition=c['condition_id'],regime=c['regime'],method=m,mean_requests=float(z['requests'].mean()),mean_arrived_positive_labels=float(z['arrived_defects'].mean()),mean_total_misses=float(z['total_misses'].mean()),mean_misses_through_alarm_or_H=float(b.mean()),median_misses_through_alarm_or_H=float(np.median(b)),prealarm_count=int(((a[:,j]>0)&(a[:,j]<tau)).sum())))
  if c['regime']=='null':continue
  for D in [5000,10000,20000]:
   success=(a>=tau)&(a<=min(H,tau+D-1))
   for j,m in enumerate(run.METHODS):
    k=int(success[:,j].sum());tail=.05/(2*family)
    lo=0. if k==0 else float(beta.ppf(tail,k,n-k+1));hi=1. if k==n else float(beta.ppf(1-tail,k+1,n-k))
    rows.append(dict(condition=c['condition_id'],method=m,prevalence=c['prevalence'],audit_rate=c['audit_rate'],delay=c['label_delay'],tau=tau,pre_risk=c['pre_risk'],post_risk=c['post_risk'],deadline_D=D,n=n,successes=k,rate=k/n,simultaneous_lower=lo,simultaneous_upper=hi,meets_provisional_80pct=lo>=.8,prealarms=int(((a[:,j]>0)&(a[:,j]<tau)).sum()),no_alarm=int((a[:,j]==0).sum())))
   for j in [0,1,3]:
    x=success[:,2].astype(int)-success[:,j].astype(int);delta=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n));wins=int((x==1).sum());losses=int((x==-1).sum())
    pairs.append(dict(condition=c['condition_id'],deadline_D=D,method_A='CUSUM_p1_020',method_B=run.METHODS[j],n=n,delta_A_minus_B=delta,descriptive_normal95_lo=max(-1,delta-1.96*se),descriptive_normal95_hi=min(1,delta+1.96*se),A_only=wins,B_only=losses,unadjusted_exact_mcnemar_p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.,multiplicity_adjusted=False))
 for r in rows:
  r.update(label_delay=r['delay'],first_change=r['tau'],deadline=r['deadline_D'],success=r['successes'],simultaneous_lo=r['simultaneous_lower'],simultaneous_hi=r['simultaneous_upper'],decision='supported' if r['meets_provisional_80pct'] else ('insufficient' if r['simultaneous_upper']<.8 else 'inconclusive'))
 write('operating_map.csv',rows);write('paired_comparisons_descriptive.csv',pairs);write('cost_summary.csv',cost)
 counts={}
 for m in run.METHODS:
  rr=[r for r in rows if r['method']==m and r['deadline_D']==10000]
  counts[m]=dict(conditions=len(rr),meets80=sum(r['meets_provisional_80pct'] for r in rr),mean_condition_detection=float(np.mean([r['rate'] for r in rr])))
 contrasts={}
 for m in [run.METHODS[j] for j in [0,1,3]]:
  rr=[r for r in pairs if r['method_B']==m and r['deadline_D']==10000]
  contrasts[m]=dict(CUSUM20_point_higher=sum(r['delta_A_minus_B']>0 for r in rr),equal=sum(r['delta_A_minus_B']==0 for r in rr),lower=sum(r['delta_A_minus_B']<0 for r in rr))
 out=dict(status='complete',validation_max_upper=max(r['simultaneous_upper'] for r in gate['results']),validation_family=gate['family_size'],operating_ci_family=family,conditions=len(conditions),final_replicates_per_condition=2000,final_streams=len(conditions)*2000,detector_evaluations=len(conditions)*2000*5,method_summary_D10000=counts,descriptive_CUSUM20_comparisons_D10000=contrasts)
 (P/'ANALYSIS.json').write_text(json.dumps(out,indent=2))
 report=['# 추가 대조군 실험 결과','',f"완료: 136조건 × 2000 paired streams × 5방법. calibration5000, validation3000, final2000. 검증 family={gate['family_size']}, 동시 오경보 상한 최대={out['validation_max_upper']:.6%}. 전체 gate 통과.",'','| 방법 | D10000 평균 조건별 검출률 | 80% 동시하한 충족 조건/68 |','|---|---:|---:|']
 for m,v in counts.items():report.append(f"| {m} | {v['mean_condition_detection']:.2%} | {v['meets80']}/68 |")
 report+=['','조건별 평균은 지정된 격자의 기술요약이며 생산조건 분포의 평균효과가 아니다. operating_map.csv는 68변화조건×5방법×3deadline 전체 1020셀에 95% 동시 양측 Clopper–Pearson 구간을 적용했다. 80%는 잠정적인 운영 목표이며 산업 허용 기준이 아니다.','', 'paired_comparisons_descriptive.csv의 CUSUM.20 비교는 같은 스트림의 차이이며 기술적 95% 정규근사 구간과 미보정 exact McNemar p를 제공한다. 다중검정 보정된 유의성 또는 사후 최적 CUSUM 주장을 하지 않는다.','', '주기적 정확 이항검정은 매3000개 생산 및 마지막 H의 disjoint arrived actual-positive labels에서 검정한다. empty interval은 경보없음. 총 alpha=.02 분할 기준이므로 실제 오경보율이 연속 감시기와 정확히 같다는 주장은 하지 않는다.','', '비용표는 첫경보 또는 H까지 누적 오류이며 조기경보의 낮은 비용은 성공으로 해석하면 안 된다. 운영표의 유효검출률은 조기경보를 실패로 처리한다.','', '자료는 독립 Monte Carlo 생산 스트림이며 실제 생산 cycle을 추가 확보한 결과가 아니다. 네 detector의 한정기간 보정 임계값에는 무한기간 보장을 전이하지 않는다.','']
 (P/'REPORT_KR.md').write_text('\n'.join(report))
 files=[p for p in P.rglob('*') if p.is_file() and p.suffix in ['.py','.csv','.json','.npz','.md'] and p.name not in ['ARTIFACT_SHA256.json','pid.json','status.json'] and '__pycache__' not in str(p)]
 (P/'ARTIFACT_SHA256.json').write_text(json.dumps({str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
