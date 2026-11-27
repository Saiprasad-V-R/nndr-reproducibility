# -*- coding: utf-8 -*-
"""NNDR broad-singular-spectrum experiment, original plotting layout.

This is Figure 8 in the submitted manuscript, though the supplied original
script is named `fig7_broad_singular_value_nndr.py`.

Run in Spyder with F5. No simulation: this code reads the frozen V19
per-method numerical results and the exact analytically computed singular
spectra for the rates-recalibrated operators. Requires numpy,pandas,matplotlib.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
OUT=HERE/'outputs_synthetic_figures'
OUT.mkdir(exist_ok=True)

# Original source plotting constants (unchanged).
def set_publication_style():
    mpl.rcParams.update({
        "font.family":"serif",
        "font.serif":["Computer Modern Roman","Times New Roman","DejaVu Serif"],
        "mathtext.fontset":"cm",
        "font.size":9.0,"axes.labelsize":9.0,"axes.titlesize":9.6,
        "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.2,
        "axes.linewidth":0.8,"lines.linewidth":1.25,
        "xtick.major.width":0.8,"ytick.major.width":0.8,
        "xtick.minor.width":0.6,"ytick.minor.width":0.6,
        "xtick.major.size":4.0,"ytick.major.size":4.0,
        "xtick.minor.size":2.0,"ytick.minor.size":2.0,
        "xtick.direction":"in","ytick.direction":"in",
        "xtick.top":True,"ytick.right":True,
        "savefig.dpi":500,"figure.dpi":150,"pdf.fonttype":42,"ps.fonttype":42,
        "legend.frameon":True,"legend.facecolor":"white",
        "legend.framealpha":0.86,"legend.edgecolor":"none",
    })


def format_axis(ax,nbx=5,nby=5,logx=False,logy=False):
    ax.minorticks_on()
    ax.tick_params(which="minor",direction="in",top=True,right=True,length=2)
    ax.tick_params(which="major",direction="in",top=True,right=True,length=4)
    if not logx:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbx))
    if not logy:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nby))


def add_panel_label(ax,label,x=-0.16,y=1.05):
    ax.text(x,y,f"({label})",transform=ax.transAxes,
            fontsize=10.5,fontweight="bold",ha="left",va="bottom")


def mean_sem(df,value):
    # Reject invalid R only for plots that actually require a valid R.
    g=df.groupby(["method","beta"])[value]
    out=g.agg(["mean","std","count"]).reset_index()
    out["sem"]=out["std"]/np.sqrt(out["count"].clip(lower=1))
    return out


def plot_figure(include_rsigma=False):
    set_publication_style()
    df=pd.read_csv(DATA/'T01_fig08_full_results.csv')
    sp=pd.read_csv(DATA/'Fig08_spectra_rates_recalibrated.csv')
    assert len(df)==320 and set(df.method)=={'M2','M3'}
    assert ((df['R_status']=='ok') == df['R_hat'].notna()).all(), 'R_A admissibility must be preserved'
    assert int(df['R_hat'].isna().sum())==32
    assert int(df['R_sigma_hat'].notna().sum())==289
    assert np.allclose(df['abs_log10_error'].dropna(),
                       np.abs(np.log10(df.loc[df['abs_log10_error'].notna(),'R_hat'] /
                                       df.loc[df['abs_log10_error'].notna(),'R_star'])),atol=1e-12)

    # Correct companion error using matching R_sigma reference, not R_A.
    df['E_Rsigma']=np.abs(np.log10(df['R_sigma_hat']/df['R_sigma_star']))
    fig,axes=plt.subplots(2,2,figsize=(7.15,5.6))
    axa,axb,axc,axd=axes.ravel()

    # (a) Singular spectra from the actual rates-recalibrated F.
    # Orthogonal embedding does not change the singular values.
    for beta in sorted(sp['beta'].unique()):
        ds=sp[sp.beta==beta]
        axa.loglog(ds['index'],ds['sigma_normalized'],marker='o',markersize=2.3,linewidth=1.0,
                   label=fr'$\beta={beta:.2g}$')
    axa.set_xlabel('Singular-value index $j$')
    axa.set_ylabel(r'$\sigma_j(F)/\sigma_1(F)$')
    axa.legend(loc='lower left',ncol=1)
    format_axis(axa,logx=True,logy=True);add_panel_label(axa,'a')

    # (b) Plane-angle error, using ALL fits, including invalid reduced ratios.
    styles={'M2':{'marker':'o','linestyle':'-','label':'Optimization (M2)','color':'#1f77b4'},
            'M3':{'marker':'^','linestyle':'--','label':'Commutator (M3)','color':'#ff7f0e'}}
    stat=mean_sem(df,'angle_deg')
    for method in ['M2','M3']:
        d=stat[stat.method==method]
        axb.errorbar(d.beta,d['mean'],yerr=d['sem'],capsize=2.5,
                     marker=styles[method]['marker'],linestyle=styles[method]['linestyle'],
                     color=styles[method]['color'],label=styles[method]['label'])
    axb.set_xlabel(r'Power-law exponent $\beta$')
    axb.set_ylabel(r'Plane error $\theta$ (deg)')
    format_axis(axb);add_panel_label(axb,'b')

    # (c) Mean absolute log10 ratio error among valid maps ONLY.
    stat=mean_sem(df,'abs_log10_error')
    for method in ['M2','M3']:
        d=stat[stat.method==method]
        axc.errorbar(d.beta,d['mean'],yerr=d['sem'],capsize=2.5,
                     marker=styles[method]['marker'],linestyle=styles[method]['linestyle'],
                     color=styles[method]['color'])
    if include_rsigma:
        # Companion presented as light/transparent dotted curves, not a
        # second definition of the primary R_A statistic.
        stat_s=mean_sem(df,'E_Rsigma')
        for method in ['M2','M3']:
            ds=stat_s[stat_s.method==method]
            axc.errorbar(ds.beta,ds['mean'],yerr=ds['sem'],capsize=1.8,
                         marker=styles[method]['marker'],markersize=3.5,linestyle=':',
                         color=styles[method]['color'],alpha=0.30,zorder=1,
                         elinewidth=.8)
        axc.text(.03,.95,r'Faint dotted: $R_\sigma$',transform=axc.transAxes,
                 color='#9CA3AF',fontsize=7,va='top',ha='left')
    axc.set_xlabel(r'Power-law exponent $\beta$')
    axc.set_ylabel(r'$R_A$-recovery error $\mathcal{E}_{R_A}$')
    format_axis(axc);add_panel_label(axc,'c')

    # (d) Reference vs inferred; inadmissible R_A estimates stay absent.
    for method in ['M2','M3']:
        dd=df[(df.method==method) & df.R_hat.notna()]
        axd.scatter(dd.R_star,dd.R_hat,s=15,alpha=.72,
                    marker=styles[method]['marker'],color=styles[method]['color'],
                    label=styles[method]['label'])
    lo=min(df.R_star.min(),df.R_hat.min())*.75
    hi=max(df.R_star.max(),df.R_hat.max())*1.35
    if include_rsigma:
        for method in ['M2','M3']:
            dd=df[(df.method==method)&df.R_sigma_hat.notna()]
            axd.scatter(dd.R_sigma_star,dd.R_sigma_hat,s=15,alpha=.28,
                        marker=styles[method]['marker'],color=styles[method]['color'],
                        facecolors='none',linewidths=.7,zorder=1)
        lo=min(lo,df.R_sigma_star.min()*.75,df.R_sigma_hat.min()*.75)
        hi=max(hi,df.R_sigma_star.max()*1.35,df.R_sigma_hat.max()*1.35)
        axd.text(.03,.96,r'Faint open: $R_\sigma$',transform=axd.transAxes,
                 ha='left',va='top',color='#9CA3AF',fontsize=7)
    axd.plot([lo,hi],[lo,hi],'k--',linewidth=.9,label='identity')
    axd.set_xscale('log');axd.set_yscale('log')
    axd.set_xlim(lo,hi);axd.set_ylim(lo,hi)
    axd.set_xlabel(r'Reference $R_{A,\star}$')
    axd.set_ylabel(r'Inferred $\widehat R_A$')
    format_axis(axd,logx=True,logy=True);add_panel_label(axd,'d')

    # Reuse the precise source legend placement.
    handles,labels=axb.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.01),
               ncol=2,frameon=False,handlelength=1.8,columnspacing=1.2)
    fig.tight_layout(rect=(0.,.09,1.,1.),pad=1.1,w_pad=1.2,h_pad=1.2)
    tag='RA_Rsigma' if include_rsigma else 'RA_only'
    fig.savefig(OUT/f'Fig08_broad_{tag}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'Fig08_broad_{tag}.png',bbox_inches='tight',dpi=600)
    plt.close(fig)


if __name__=='__main__':
    plot_figure(False)
    plot_figure(True)
    print('Figure 8 broad-spectrum package regenerated from frozen V19 data.')
