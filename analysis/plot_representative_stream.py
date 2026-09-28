"""Plot the fixed representative trajectory; no simulation or path selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
P=Path(__file__).resolve().parents[1];O=P/'supplementary/representative_stream'
for p in (Path.home()/'.local/share/fonts/mfg-times-new-roman').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
a=np.load(O/'trajectory.npz');m=json.loads((O/'metadata.json').read_text());c=m['condition_parameters'];tau=c['first_change'];end=m['display_end'];deadline=m['deadline_item'];grid=np.arange(1,end+1)
col=['#0072B2','#E69F00','#D55E00','#009E73','#CC79A7'];labels=['Betting monitor','CUSUM .12','CUSUM .20','Mixture SR','Periodic binomial']
fig,axes=plt.subplots(4,1,figsize=(7.1,7.7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[.8,1.25,1.65,1.2]})
axes[0].step([1,tau-1,tau,end],[.05,.05,.2,.2],where='post',color='#173c60',lw=1.7,label='True conditional miss risk')
axes[0].axhline(.1,color='#777777',ls=':',label='Admissible risk 0.10');axes[0].set_ylim(0,.25);axes[0].set_ylabel('Miss probability');axes[0].set_title('(a) Controlled risk change',loc='left');axes[0].legend(loc='upper left',fontsize=8)
for data,label,color,style in [('origin','Audited defective items','#173c60','-'),('arrival','Arrived informative labels','#D55E00','--')]:
 axes[1].plot(grid,np.searchsorted(a[data],grid,side='right'),label=label,color=color,ls=style,lw=1.3)
axr=axes[1].twinx();axr.plot(grid,np.searchsorted(a['requests'],grid,side='right'),color='#888888',alpha=.6,lw=.9,ls=':',label='All audit requests');axr.set_ylabel('Total audit requests',color='#666666');axr.spines['top'].set_visible(False);axr.tick_params(axis='y',labelsize=8)
axes[1].set_ylabel('Informative count');axes[1].set_title('(b) Audit acquisition and label availability (delay = 500 items)',loc='left');axes[1].legend(fontsize=8,loc='upper left');axr.legend(fontsize=8,loc='lower right')
# Stop each trace at its first alarm; offset log statistics by their calibrated thresholds.
for j in range(4):
 stop=m['alarms'][j] or end;keep=(a['arrival']<=min(stop,end));axes[2].step(a['arrival'][keep],a['statistics'][j][keep]-a['thresholds'][j],where='post',color=col[j],lw=1.15,ls=['-','--','-.',':'][j],label=labels[j])
 if stop<=end:
  idx=np.flatnonzero(a['arrival']==stop)[0];axes[2].plot(stop,a['statistics'][j][idx]-a['thresholds'][j],'o',color=col[j],ms=4)
axes[2].axhline(0,color='black',lw=.9);axes[2].set_ylabel('Log statistic − log threshold');axes[2].set_title('(c) Continuous-monitor evidence, shown until first alarm',loc='left');axes[2].legend(fontsize=8,ncol=4,loc='lower left')
for j,t in enumerate(m['alarms']):
 axes[3].hlines(j,tau,deadline,color='#dddddd',lw=2)
 if t and t<=end:axes[3].scatter(t,j,color=col[j],s=24,zorder=5);axes[3].annotate(f'{t:,}',(t,j),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
axes[3].set_yticks(range(5),labels);axes[3].invert_yaxis();axes[3].set_ylim(4.6,-.6);axes[3].set_title('(d) First alarms on the same production stream',loc='left');axes[3].set_xlabel('Production item')
for ax in axes:
 ax.axvline(tau,color='#555555',ls='--',lw=.8);ax.axvline(deadline,color='#555555',ls=':',lw=.8);ax.axvspan(tau,tau+500,color='#E69F00',alpha=.09);ax.set_xlim(0,end);ax.grid(axis='y',alpha=.12)
axes[0].text(tau+150,.235,'Change',fontsize=8);axes[0].text(deadline-200,.235,'Deadline',ha='right',fontsize=8)
axes[-1].set_xticks([0,5000,10000,15000,19999],['0','5,000','10,000','15,000','19,999'])
for ext in ['pdf','png']:fig.savefig(P/'figures'/f'figure6_representative_stream.{ext}',dpi=300)
