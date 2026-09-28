"""Fixed displays of all five controls; no post-result method selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,sys
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in (Path.home()/'.local/share/fonts/mfg-times-new-roman').glob('*.TTF'): font_manager.fontManager.addfont(str(fp))
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'supplementary/evidence';F=ROOT/'figures';F.mkdir(exist_ok=True)
OLD=P
M=['WH_finiteH','CUSUM_p1_012','CUSUM_p1_020','mSR_finiteH','periodic_binomial']
VM=['WH_finite','CUSUM_p12','CUSUM_p20','mSR','periodic_binomial']
LABEL=['Betting monitor','CUSUM .12','CUSUM .20','Mixture SR','Periodic binomial']
COLORS=['#0072B2','#E69F00','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'font.family':'Times New Roman','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def cp(k,n):
 return (0 if k==0 else beta.ppf(.025,k,n-k+1),1 if k==n else beta.ppf(.975,k+1,n-k))
def policy():
 rows=json.loads((OLD/'policy/SUMMARY.json').read_text());rr=[r for r in rows if r['condition']['id']==5]
 labels=['Horizon\nuniform','Block\nuniform','Horizon\nadapt .05','Horizon\nadapt .20','Block\nadapt .05','Block\nadapt .20']
 x=np.arange(6);rates=np.array([r['timely'] for r in rr]);ci=np.array([cp(round(v*2000),2000) for v in rates]);colors=['#777777','#0072B2','#D55E00','#E69F00','#CC79A7','#009E73']
 fig,ax=plt.subplots(2,1,figsize=(7.1,5.4),layout='constrained',sharex=True)
 ax[0].bar(x,100*rates,color=colors,yerr=100*np.array([rates-ci[:,0],ci[:,1]-rates]),capsize=3)
 ax[0].set_ylabel('Timely detection (%)');ax[0].set_title('(a) Primary scenario: prevalence 1%, label delay 500')
 ax[1].bar(x,[r['mean_requests_deadline'] for r in rr],color=colors)
 ax[1].set_ylabel('Audits in response window');ax[1].set_title('(b) Same total budget: 5,999 requests per stream')
 ax[1].set_xticks(x,labels);save(fig,'figure4_audit_policy')
def simulations():
 rows=list(csv.DictReader((P/'simulation/operating_map.csv').open()))
 def match(pi,post,m,q,D):
  z=[r for r in rows if float(r['prevalence'])==pi and float(r['post_risk'])==post and int(r['first_change'])==10000 and int(r['label_delay'])==500 and r['method']==m and float(r['audit_rate'])==q and int(r['deadline'])==D];assert len(z)==1;return z[0]
 fig,axes=plt.subplots(2,2,figsize=(7.1,5.6),layout='constrained')
 cmap=ListedColormap(['#E4B3A8','#DDDDDD','#7AB9AC']);code={'insufficient':0,'inconclusive':1,'supported':2};symbol={'insufficient':'I','inconclusive':'?','supported':'S'}
 for ax,(pi,post) in zip(axes.flat,[(.01,.12),(.01,.2),(.1,.12),(.1,.2)]):
  zz=[[match(pi,post,m,q,10000) for q in [.1,.2,.5,1.]] for m in M]
  ax.imshow([[code[r['decision']] for r in rr] for rr in zz],cmap=cmap,vmin=-.5,vmax=2.5,aspect='auto')
  for i,rr in enumerate(zz):
   for j,r in enumerate(rr):ax.text(j,i,f"{100*float(r['rate']):.1f}\n{symbol[r['decision']]}",ha='center',va='center',fontsize=8)
  ax.set_xticks(range(4),['10','20','50','100']);ax.set_yticks(range(5),LABEL);ax.set_xlabel('Audit rate (%)');ax.set_title(f'Prevalence {pi:.0%}, post-change risk {post:.0%}')
 fig.suptitle('10,000-item deadline; change at 10,000; label delay 500',fontsize=10)
 fig.legend(handles=[Patch(facecolor=cmap(i),label=t) for i,t in enumerate(['I: upper bound < 80%','?: inconclusive','S: lower bound >= 80%'])],loc='outside lower center',ncol=3,fontsize=8)
 save(fig,'figure1_operating_map')
 fig,axes=plt.subplots(2,2,figsize=(7.1,5.5),sharex=True,sharey=True,layout='constrained')
 for ax,(pi,post) in zip(axes.flat,[(.01,.12),(.01,.2),(.1,.12),(.1,.2)]):
  for m,label,col,marker in zip(M,LABEL,COLORS,['o','s','^','D','v']):
   zz=[match(pi,post,m,.2,D) for D in [5000,10000,20000]];y=np.array([float(r['rate']) for r in zz]);lo=np.array([float(r['simultaneous_lo']) for r in zz]);hi=np.array([float(r['simultaneous_hi']) for r in zz])
   ax.errorbar([5,10,20],100*y,yerr=100*np.array([y-lo,hi-y]),label=label,color=col,marker=marker,capsize=2,lw=1)
  ax.axhline(80,color='#777777',ls=':',lw=1);ax.set_title(f'Prevalence {pi:.0%}, post-change risk {post:.0%}');ax.set_ylim(-3,103);ax.set_xticks([5,10,20]);ax.grid(alpha=.15)
 for ax in axes[-1]:ax.set_xlabel('Response window (1,000 production items)')
 for ax in axes[:,0]:ax.set_ylabel('Timely detection (%)')
 fig.suptitle('Audit rate 20%; change at 10,000; label delay 500',fontsize=10);h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3);save(fig,'figure2_detection_curves')
def video():
 rows=list(csv.DictReader((P/'video/runs.csv').open()));fig,axes=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained',sharey=True)
 for ax,(q,d) in zip(axes.flat,[(.1,0),(.1,500),(.2,0),(.2,500)]):
  means=[];err=[]
  for m in VM:
   a=np.array([float(r['restricted_delay']) for r in rows if r['condition']=='blurblend_0.8' and float(r['q'])==q and int(r['delay'])==d and r['method']==m]);assert len(a)==2000
   means.append(a.mean());err.append(1.96*a.std(ddof=1)/np.sqrt(len(a)))
  ax.bar(range(5),means,color=COLORS,yerr=err,capsize=3);ax.set_xticks(range(5),['Betting','C .12','C .20','mSR','Periodic'],rotation=25);ax.set_title(f'Audit rate {q:.0%}; label delay {d}');ax.set_ylabel('Restricted delay (production items)')
 fig.suptitle('Fixed NEU strong-shift pool; conditional Monte Carlo intervals',fontsize=10);save(fig,'figure3_video_controls')
if __name__=='__main__':
 policy()
 if '--policy-only' not in sys.argv:simulations();video()
