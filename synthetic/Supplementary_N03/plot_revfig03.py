# -*- coding: utf-8 -*-
"""NNDR Reviewer Figure 3: model adequacy, detection and uncertainty.

Spyder-ready, plot only. Reads CSVs in ./data (frozen from V19), creates
RA-only and RA plus faint Rsigma companion versions. No simulations rerun.
No titles/subtitles; manuscript palette, symbols and inward minor ticks.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator, MaxNLocator, PercentFormatter

BASE = Path(__file__).resolve().parent
DATA = BASE/'data'
FIG = BASE/'figures'
FIG.mkdir(exist_ok=True)

ORANGE='#D55E00'  # M2, original manuscript role
GREEN='#009E73'   # M3, reserved for method curves
BLACK='#232323'
GREY='#727272'
PALE='#A5A5A5'


def configure_style():
    mpl.rcParams.update({
        'font.family':'serif',
        'font.serif':['Computer Modern Roman','Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'cm',
        'font.size':9.2,
        'axes.labelsize':9.2,
        'xtick.labelsize':8.,
        'ytick.labelsize':8.,
        'legend.fontsize':7.,
        'axes.linewidth':.80,
        'lines.linewidth':1.35,
        'xtick.direction':'in',
        'ytick.direction':'in',
        'xtick.top':True,
        'ytick.right':True,
        'xtick.major.size':4.0,
        'ytick.major.size':4.0,
        'xtick.minor.size':2.0,
        'ytick.minor.size':2.0,
        'savefig.dpi':550,
        'figure.dpi':170,
        'pdf.fonttype':42,
        'ps.fonttype':42,
        'legend.frameon':False,
    })


def clean_axis(ax):
    ax.tick_params(which='major',direction='in',top=True,right=True,length=4)
    ax.tick_params(which='minor',direction='in',top=True,right=True,length=2)
    for sp in ax.spines.values():sp.set_linewidth(.80)
    ax.grid(False)


def label_panel(ax, txt):
    ax.text(-.075,1.045,f'({txt})',transform=ax.transAxes,fontsize=10.7,
            fontweight='bold',ha='left',va='bottom',clip_on=False)


def wilson(k,n,z=1.959963984540054):
    assert 0 <= k <= n and n>0
    p=k/n
    den=1+z*z/n
    cen=(p+z*z/(2*n))/den
    rad=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0.,cen-rad),min(1.,cen+rad)


def panel_a(ax,df):
    # Held-out family results; each classification rate uses n=48 recordings
    selected=[('gaussian_VAR1','VAR(1)'),
              ('cubic_saturation','Cubic'),
              ('VAR2','VAR(2)'),
              ('coloured_spatial','Coloured'),
              ('common_mode_bursts','Bursts'),
              ('observation_noise','Obs. noise'),
              ('switching','Switch')]
    yy=np.arange(len(selected))
    for i,(code,_label) in enumerate(selected):
        sub=df[(df.dataset=='T07')&(df.perturbation==code)]
        normal=sub[sub.label=='normal'].iloc[0]
        positive=sub[sub.label=='positive'].iloc[0]
        # Slight row offset: two independently reported phase labels
        x0=float(normal.gate_fail_rate)
        x1=float(positive.gate_fail_rate)
        assert normal.n==48 and positive.n==48
        ax.plot([x0,x1],[i,i],lw=.95,color='#B4B4B4',zorder=1)
        ax.scatter([x0],[i-.075],marker='o',s=34,facecolors='white',
                   edgecolors=GREY,lw=1.20,zorder=3)
        ax.scatter([x1],[i+.075],marker='o',s=36,color=BLACK,
                   edgecolors='white',lw=.5,zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels([y for _x,y in selected])
    ax.invert_yaxis()
    ax.set_xlim(-.045,1.07)
    ax.set_xticks([0,.25,.5,.75,1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    ax.xaxis.set_minor_locator(MultipleLocator(.125))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Adequacy rejection fraction')
    l=ax.legend(handles=[
        Line2D([0],[0],marker='o',ls='none',mfc='white',mec=GREY,ms=5,label='Normal'),
        Line2D([0],[0],marker='o',ls='none',mfc=BLACK,mec=BLACK,ms=5,label='Positive')],
        loc='lower left',bbox_to_anchor=(.015,.02),ncol=2,handletextpad=.4,
        columnspacing=.95,borderaxespad=0,fontsize=7.3)
    clean_axis(ax)


def rates_from_t08(df,metric):
    d=df[(df.dataset=='T07')&(df.method=='M2')]
    out={}
    for label in ['positive','normal']:
        q=d[d.label==label]
        n=int(q.denominator_total.sum())
        if metric=='R_A':
            nr=int(q.raw_RA_detect_count.sum())
            ns=int(q.accepted_RA_detect_count.sum())
        else:
            nr=int(q.raw_Rsigma_detect_count.sum())
            ns=int(q.accepted_Rsigma_detect_count.sum())
        out[label]=(nr,ns,n)
    return out


def panel_b(ax,table,cluster,include_companion):
    # T07 M2, denominator of ALL held-out records, before/after five-part screen.
    pri=rates_from_t08(table,'R_A')
    sec=rates_from_t08(table,'R_sigma')
    xbase={'positive':0.,'normal':1.}
    for label in ['positive','normal']:
        raw,passed,n=pri[label]
        for status,k,dx in [('Raw',raw,-.115),('Screened',passed,.115)]:
            x=xbase[label]+dx
            y=k/n
            est_name=('T07_'+label+'_'+ ('raw_power' if label=='positive' else 'raw_FPR')+'_M2'
                      if status=='Raw' else
                      'T07_'+label+'_screened_per_all_M2' if label=='positive' else
                      'T07_'+label+'_screened_FPR_M2')
            # T08 subject-cluster bootstrap intervals for R_A; for zero counts
            # Wilson upper bound is used descriptively, since percentile CI degenerates.
            row=cluster[cluster.estimand==est_name]
            assert len(row)==1,(est_name,len(row))
            row=row.iloc[0]
            lo=float(row.cluster_bootstrap95_low)
            hi=(float(row.wilson95_high_descriptive) if k==0 else float(row.cluster_bootstrap95_high))
            assert abs(float(row.fraction)-y)<1e-11
            color=GREY if status=='Raw' else ORANGE
            ax.vlines(x,lo,hi,color=color,lw=1.05,alpha=.75,zorder=3)
            ax.hlines([lo,hi],x-.02,x+.02,color=color,lw=.80,zorder=3)
            if status=='Raw':
                ax.scatter(x,y,marker='o',s=40,facecolors='white',edgecolors=color,lw=1.25,zorder=5)
            else:
                ax.scatter(x,y,marker='o',s=40,c=color,edgecolors='white',lw=.55,zorder=5)
        # The secondary metric is descriptive only; the main uncertainty bars above
        # apply to R_A. Secondary overlays are muted diamonds to avoid clutter.
        if include_companion:
            sr,ss,sn=sec[label]
            assert sn==n
            for k,dx in [(sr,-.185),(ss,.185)]:
                ax.scatter(xbase[label]+dx,k/n,marker='D',s=24,facecolors='white',
                           edgecolors=PALE,lw=1.,zorder=4)
    ax.set_xlim(-.39,1.39)
    ax.set_ylim(-.025,1.065)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Positive','Normal'])
    ax.set_yticks([0,.25,.5,.75,1.])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    ax.yaxis.set_minor_locator(MultipleLocator(.125))
    ax.set_ylabel(r'$p_{\mathrm{detect}}$ (all records)')
    handles=[Line2D([0],[0],ls='none',marker='o',ms=5,mfc='white',mec=GREY,label='Raw'),
             Line2D([0],[0],ls='none',marker='o',ms=5,mfc=ORANGE,mec=ORANGE,label='Screened')]
    if include_companion:
        handles.append(Line2D([0],[0],ls='none',marker='D',ms=4,mfc='white',mec=PALE,label=r'$R_\sigma$'))
    ax.legend(handles=handles,loc='upper right',bbox_to_anchor=(1.0,.98),ncol=1 if include_companion else 2,
              handletextpad=.3,columnspacing=.65,fontsize=7.2,borderaxespad=.2)
    clean_axis(ax)


def coverage(df,metric):
    s=df[(df.metric==metric)&(df.level==.95)&(df.method=='percentile')]
    regimes=['M1_T90','M1_T240','M8_T240']
    out={}
    for regime in regimes:
        for mode in ['oracle','selected_M2']:
            q=s[(s.regime==regime)&(s['mode']==mode)]
            n=len(q)
            assert n==96,(metric,regime,mode,n)
            covered=int(q.covered.sum())
            estimable=int(q.estimable.sum())
            out[regime,mode]={'k':covered,'n':n,'estimable':estimable,'p':covered/n,'ci':wilson(covered,n)}
    return out


def panel_c(ax,intervals,include_companion):
    pri=coverage(intervals,'R_A')
    sec=coverage(intervals,'R_sigma')
    regimes=['M1_T90','M1_T240','M8_T240']
    lab=['$M=1$\n$T=90$','$M=1$\n$T=240$','$M=8$\n$T=240$']
    for j,r in enumerate(regimes):
        for mode,dx,c in [('oracle',-.14,BLACK),('selected_M2',.14,ORANGE)]:
            entry=pri[r,mode]
            x=j+dx
            y=entry['p']
            lo,hi=entry['ci']
            ax.vlines(x,lo,hi,color=c,lw=1.35,alpha=.72,zorder=3)
            ax.hlines([lo,hi],x-.025,x+.025,lw=.9,color=c,zorder=3)
            ax.scatter(x,y,s=44,color=c,edgecolors='white',lw=.5,zorder=5)
            if include_companion:
                e=sec[r,mode]
                sdx=dx + (-.058 if mode=='oracle' else .058)
                sx=j+sdx
                ax.scatter(sx,e['p'],s=28,marker='D',facecolors='white',
                           edgecolors=PALE,lw=1.1,zorder=4)
                l,h=e['ci']
                ax.vlines(sx,l,h,color=PALE,lw=.7,zorder=2,alpha=.75)
    ax.axhline(.95,ls=':',color='#909090',lw=1.,zorder=0)
    ax.set_ylim(0,1.055)
    ax.set_xlim(-.45,2.45)
    ax.set_xticks(range(3))
    ax.set_xticklabels(lab,fontsize=7.8)
    ax.set_ylabel('Unconditional 95% CI coverage')
    ax.set_yticks([0,.25,.5,.75,1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    ax.yaxis.set_minor_locator(MultipleLocator(.125))
    ax.xaxis.set_minor_locator(NullLocator())
    handles=[Line2D([0],[0],marker='o',ls='none',color=BLACK,ms=5,label='Oracle plane'),
             Line2D([0],[0],marker='o',ls='none',color=ORANGE,ms=5,label='Selected M2')]
    ax.legend(handles=handles,loc='lower left',ncol=2,bbox_to_anchor=(.07,1.025),
              handletextpad=.4,borderaxespad=.15,fontsize=7.1)
    clean_axis(ax)


def panel_d(ax,solver):
    regimes=['M1_T90','M1_T240','M8_T240']
    labels=['$M=1$\n$T=90$','$M=1$\n$T=240$','$M=8$\n$T=240$']
    rng=np.random.default_rng(270)
    for j,r in enumerate(regimes):
        q=solver[solver.regime==r]['single_start_fraction_within_1pct'].to_numpy(float)
        assert len(q)==48,(r,len(q))
        jitter=rng.uniform(-.09,.09,len(q))
        ax.scatter(np.full(len(q),j)+jitter,q,s=14,facecolors='none',
                   edgecolors=ORANGE,lw=.65,alpha=.34,zorder=2)
        q25,med,q75=np.quantile(q,[.25,.5,.75])
        ax.vlines(j,q25,q75,color=BLACK,lw=3.0,zorder=4)
        ax.hlines(med,j-.155,j+.155,color=BLACK,lw=2.0,zorder=5)
    ax.set_xlim(-.45,2.45)
    ax.set_ylim(-.03,1.045)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels,fontsize=7.8)
    ax.set_yticks([0,.25,.5,.75,1])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    ax.yaxis.set_minor_locator(MultipleLocator(.125))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(r'$p_{\mathrm{hit}}$')
    clean_axis(ax)


def make_figure(include_companion=False):
    configure_style()
    diag=pd.read_csv(DATA/'T08_holdout_diagnostic_rejection_by_family.csv')
    power=pd.read_csv(DATA/'T08_holdout_power_FPR_and_rejection.csv')
    intervals=pd.read_csv(DATA/'T12_interval_results.csv')
    clustered=pd.read_csv(DATA/'T08_subject_cluster_bootstrap_intervals.csv')
    solver=pd.read_csv(DATA/'T12_solver_audit.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.65))
    fig.subplots_adjust(left=.145,right=.985,top=.92,bottom=.12,wspace=.38,hspace=.50)
    panel_a(axs[0,0],diag)
    panel_b(axs[0,1],power,clustered,include_companion)
    panel_c(axs[1,0],intervals,include_companion)
    panel_d(axs[1,1],solver)
    for label,ax in zip('abcd',axs.flat):label_panel(ax,label)
    suffix='RA_Rsigma' if include_companion else 'RA_only'
    pdf=FIG/f'RevFig03_{suffix}.pdf'
    png=FIG/f'RevFig03_{suffix}.png'
    fig.savefig(pdf,bbox_inches='tight',facecolor='white')
    fig.savefig(png,dpi=550,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return pdf,png


def main():
    outputs=[]
    for companion in (False,True):
        outputs.extend(make_figure(companion))
    manifest={'source':'V19 T08 and T12 frozen data','files':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    (BASE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Created',*[str(p) for p in outputs],sep='\n  ')


if __name__=='__main__':main()
