# -*- coding: utf-8 -*-
"""NNDR S4 — Commutator gap and finite-data recovery (V19 T11).

Standalone, plot-only script: open in Spyder, press F5. All source data
are copied into ./data. Neither this script nor the figure alters the
frozen synthetic results or reruns any simulations.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, FuncFormatter
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

INK = '#25272A'
ORANGE = '#D55E00'       # manuscript M2
GREEN = '#009E73'        # manuscript M3
MUTED = '#777F86'
PALE = '#D7DDE2'
PANEL = '#FFFFFF'
FRACS = [0.0, 0.5, 0.9, 0.98]
FRAC_TEXT = [r'$0$', r'$0.50$', r'$0.90$', r'$0.98$']
REGIMES = ['M1_T40', 'M25_T3', 'M25_T4', 'M1_T240', 'M4_T160', 'M16_T240']


def lighten(color, f=.57):
    c = np.array(to_rgb(color))
    return tuple(c*(1.-f)+np.ones(3)*f)


def configure_style():
    mpl.rcParams.update({
        'font.family':'serif',
        'font.serif':['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':'cm',
        'font.size':9.0,
        'axes.labelsize':9.7,
        'xtick.labelsize':8.2,
        'ytick.labelsize':8.2,
        'legend.fontsize':8.0,
        'axes.linewidth':.85,
        'lines.linewidth':1.30,
        'xtick.direction':'in','ytick.direction':'in',
        'xtick.top':True,'ytick.right':True,
        'xtick.major.size':4.2,'ytick.major.size':4.2,
        'xtick.minor.size':2.0,'ytick.minor.size':2.0,
        'xtick.major.width':.8,'ytick.major.width':.8,
        'xtick.minor.width':.62,'ytick.minor.width':.62,
        'figure.facecolor':'white','axes.facecolor':'white',
        'pdf.fonttype':42,'ps.fonttype':42,
        'figure.dpi':170,'savefig.dpi':550,
    })


def fmt_axis(ax, minor_x=True, minor_y=True):
    for sp in ax.spines.values():
        sp.set_linewidth(.85)
    ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.grid(False)


def tag(ax, label):
    ax.text(-.105,1.047,'('+label+')',transform=ax.transAxes,ha='left',
            va='bottom',fontweight='bold',fontsize=11.4,color=INK,clip_on=False)


def competitor_axis(ax):
    ax.set_xlim(-.40,3.40)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(FRAC_TEXT)
    ax.set_xlabel(r'Competing fraction $c$')
    fmt_axis(ax,minor_x=False)


def panel_a(ax, rows):
    g = rows.groupby('competitor_fraction')['commutator_gap'].median().reindex(FRACS)
    val=g.to_numpy(float)
    assert np.isfinite(val).all()
    assert np.allclose(val,val[0]*(1.-np.asarray(FRACS)),rtol=1e-10,atol=1e-10)
    ax.plot(np.arange(4),val,color=INK,lw=1.7,marker='o',ms=5.4,
            markerfacecolor='white',markeredgewidth=1.35,zorder=4)
    ax.set_ylabel(r'Commutator gap $g_C$')
    ax.set_yscale('log')
    ax.set_ylim(.009,1.0)
    ax.set_yticks([.01,.03,.1,.3,1.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: {0.01:'0.01',0.03:'0.03',0.1:'0.1',0.3:'0.3',1.0:'1'}.get(round(float(y),2),'')))
    competitor_axis(ax)
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=20))
    tag(ax,'a')


def panel_b(ax, smry):
    series=[(0.,GREEN,'$c=0$', 'o','-'),(.98,lighten(GREEN,.53),r'$c=0.98$', '^','--')]
    for c,color,label,mark,ls in series:
        d=smry[np.isclose(smry['competitor_fraction'],c)].set_index('regime').reindex(REGIMES)
        assert len(d)==6 and d['informative_n'].notna().all()
        x=d['informative_n'].to_numpy(float)
        y=d['median_M3_angle_deg'].to_numpy(float)
        lo=d['angle_CI95_low'].to_numpy(float)
        hi=d['angle_CI95_high'].to_numpy(float)
        ax.errorbar(x,y,yerr=[y-lo,hi-y],color=color,marker=mark,
                    linestyle=ls,linewidth=1.45,markersize=5.1,
                    markerfacecolor='white',markeredgewidth=1.25,
                    capsize=2.1,elinewidth=.85,alpha=1.,label=label,zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(29,5600)
    ax.set_xticks([40,75,240,640,3800])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'))
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9],numticks=100))
    ax.set_ylim(0,97)
    ax.set_yticks([0,20,40,60,80])
    ax.set_ylabel(r'$\operatorname{med}\,\theta_{\rm M3}$ ($^\circ$)')
    ax.set_xlabel(r'Training pairs $n=M(T-1)$')
    fmt_axis(ax,minor_x=False)
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=60))
    ax.legend(loc='upper right',ncol=1,frameon=False,fontsize=8.0,
              borderaxespad=.5,handlelength=1.4,labelspacing=.34)
    tag(ax,'b')


def stratified_boot_ci(rows, c, method, rng_seed):
    # Fixed N-stratified bootstrap of six independent seeds in each of N=16,32,64.
    # Unlike points from the same subject at differing c, these are not treated
    # as new independent participants.  Returns 95% percentile CI for median.
    d=rows[(rows['regime']=='M16_T240') & np.isclose(rows['competitor_fraction'],c)]
    assert len(d)==18
    column='m2_angle_deg' if method=='M2' else 'M3_angle_deg'
    arr_by_n=[]
    for N in (16,32,64):
        a=d.loc[d.N==N,column].to_numpy(float)
        assert len(a)==6 and np.isfinite(a).all()
        arr_by_n.append(a)
    rng=np.random.default_rng(rng_seed)
    draws=np.empty(5000)
    for b in range(5000):
        samples=np.concatenate([a[rng.integers(0,6,size=6)] for a in arr_by_n])
        draws[b]=float(np.median(samples))
    return float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def panel_c(ax, rows, summary):
    x=np.arange(4,dtype=float)
    for method,color,marker,dx in [('M2',ORANGE,'o',-.10),('M3',GREEN,'^',+.10)]:
        yy, low,high=[],[],[]
        for i,c in enumerate(FRACS):
            d=rows[(rows.regime=='M16_T240')&np.isclose(rows.competitor_fraction,c)]
            col='m2_angle_deg' if method=='M2' else 'M3_angle_deg'
            median=float(np.median(d[col]))
            yy.append(median)
            # Same fixed-dimension-stratified bootstrap for both methods.
            lo,hi=stratified_boot_ci(rows,c,method,
                                    20261120+21*i+(0 if method=='M2' else 1000))
            low.append(lo);high.append(hi)
        ax.errorbar(x+dx,yy,yerr=[np.asarray(yy)-low,np.asarray(high)-yy],
                    color=color,marker=marker,markersize=5.25,
                    markerfacecolor='white',markeredgewidth=1.3,
                    linewidth=1.35,elinewidth=.86,capsize=2.4,
                    linestyle='-',label=method,zorder=4)
    ax.set_ylim(0,45)
    ax.set_yticks([0,10,20,30,40])
    ax.set_ylabel(r'$\operatorname{med}\,\theta_\star$ ($^\circ$)')
    competitor_axis(ax)
    ax.legend(loc='upper left',ncol=2,frameon=False,fontsize=8.0,
              borderaxespad=.3,handlelength=1.1,columnspacing=.7)
    tag(ax,'c')


def panel_d(ax, summary):
    x=np.arange(4,dtype=float)
    styles=[('M16_T240',INK,'o','-',r'$M=16,\ T=240$'),
            ('M4_T160',MUTED,'s','--',r'$M=4,\ T=160$')]
    for reg,col,mark,ls,label in styles:
        d=summary[summary['regime']==reg].set_index('competitor_fraction').reindex(FRACS)
        vals=d['nonvacuous_DK'].to_numpy(float)
        n=d['systems'].to_numpy(int)
        assert (n==18).all()
        ax.plot(x,vals,marker=mark,markersize=5.1,linestyle=ls,color=col,
                linewidth=1.45,markerfacecolor='white',markeredgewidth=1.25,
                label=label,zorder=3)
    ax.set_ylim(-.9,20)
    ax.set_yticks([0,6,12,18])
    ax.set_ylabel(r'Informative bound (of $18$)')
    competitor_axis(ax)
    ax.legend(loc='upper right',frameon=False,fontsize=7.8,
              handlelength=1.45,handletextpad=.55,borderaxespad=.36)
    tag(ax,'d')


def build():
    configure_style()
    raw=pd.read_csv(DATA/'T11_full_gap_sample_rows.csv')
    summary=pd.read_csv(DATA/'T11_cluster_bootstrap_summary.csv')
    assert len(raw)==432 and len(summary)==24
    assert set(np.round(raw.competitor_fraction,2))==set(FRACS)
    assert raw.groupby(['regime','competitor_fraction']).size().eq(18).all()
    assert int(raw.DK_nonvacuous.sum())==36
    assert int(raw.DK_holds.sum())==432
    assert int((raw['n_pairs']<raw['N']).sum())==48
    assert int((raw.m3_status!='ok').sum())==135
    assert int((raw.m2_status!='ok').sum())==175
    assert summary['systems'].eq(18).all()
    fig,ax=plt.subplots(2,2,figsize=(7.65,5.50))
    panel_a(ax[0,0],raw)
    panel_b(ax[0,1],summary)
    panel_c(ax[1,0],raw,summary)
    panel_d(ax[1,1],summary)
    fig.subplots_adjust(left=.13,right=.983,top=.944,bottom=.13,wspace=.36,hspace=.40)
    stem=FIG/'SuppFig04_Gap_Sample_Recovery_REVISED'
    fig.savefig(str(stem)+'.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(str(stem)+'.png',dpi=550,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return stem


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    stem=build()
    docs={'figure':'S4 candidate: commutator gap and finite-data validation',
          'data_source':'V19 T11, 18 independent base subjects; 432 finite-data fits',
          'data_files':['T11_full_gap_sample_rows.csv','T11_cluster_bootstrap_summary.csv'],
          'checks':{'n_fits':432,'DK_holds':432,'DK_informative':36,'n_lt_N':48,
                    'M3_RA_inadmissible':135,'M2_RA_inadmissible':175},
          'output_files':{str(p.relative_to(HERE)):sha256(p)
                          for p in [Path(str(stem)+'.pdf'),Path(str(stem)+'.png')]}}
    (HERE/'manifest.json').write_text(json.dumps(docs,indent=2))
    print('Created:', stem)
