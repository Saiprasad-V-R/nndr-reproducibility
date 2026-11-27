# -*- coding: utf-8 -*-
"""NNDR reviewer-facing method comparison, manuscript style, no titles.

Plot-only, reproducible from the bundled V19 T02 CSV tables. Open this script
in Spyder and press F5. No global paths, external modules, or simulations.
"""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FixedLocator, ScalarFormatter
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
OUT = BASE / 'figures'
OUT.mkdir(exist_ok=True)

# The user's Oct 14 manuscript figure guide, including its explicit style.
def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.2,
        "axes.labelsize": 9.2,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.80,
        "lines.linewidth": 1.35,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "savefig.dpi": 550,
        "figure.dpi": 170,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.86,
        "legend.edgecolor": "none",
    })

def format_axis(ax, nbx=5, nby=5) -> None:
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbx))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nby))

def panel_label(ax, label: str, x=-0.18, y=1.09) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10.7, fontweight='bold', clip_on=False)

def ell(ax, cx, cy, w, h, ang, fc, ec, al=0.80, lw=1.0, z=4):
    ax.add_patch(Ellipse((cx,cy),w,h,angle=ang,fc=fc,ec=ec,alpha=al,lw=lw,zorder=z))

def dcirc(ax,cx,cy,r,z=3):
    ax.add_patch(Circle((cx,cy),r,fill=False,ls='--',ec='#888888',lw=0.9,zorder=z))

def arr(ax,x0,y0,x1,y1,col='#222222',lw=1.2,rad=0.0,ms=10,z=6):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),connectionstyle=f'arc3,rad={rad}',
                                arrowstyle='-|>',mutation_scale=ms,color=col,lw=lw,zorder=z))
BLUE='#0B3C6D'
RED='#D62728'
GREEN='#2CA02C'
C_RED='#B03A2E'; C_BLU='#1A5276'; C_PUR='#6C3483'
C_ORG='#A04000'; C_GRN='#1E8449'
F_RED='#F1948A'; F_BLU='#AED6F1'; F_PUR='#D2B4DE'

METHS = ['M1_eigenbasis','M2','M3','SVD_leading_pair','snapshot_Tb','PCA2','random2']
MLAB = {'M1_eigenbasis':'M1', 'M2':'M2','M3':'M3','SVD_leading_pair':'SVD',
        'snapshot_Tb':'Snapshot','PCA2':'PCA','random2':'Random'}
MCOL = {'M1_eigenbasis':'#0072B2','M2':'#D55E00','M3':'#009E73','SVD_leading_pair':RED,
        'snapshot_Tb':C_PUR,'PCA2':'#777777','random2':'#B7B7B7'}
FAM = {'dominant_embedded':('Dominant',BLUE),
       'competing_nonnormal_background':('Competing',RED),
       'two_near_tied_blocks':('Near-tied','#687A82')}


def pale(color, weight=.47):
    rgb=np.array(mpl.colors.to_rgb(color))
    return tuple((1-weight)*rgb+weight*np.ones(3))


def get_inputs():
    summary=pd.read_csv(DATA/'T02_group_summary.csv')
    ci=pd.read_csv(DATA/'T02_angle_bootstrap_intervals.csv')
    horizon=pd.read_csv(DATA/'T02_snapshot_horizons.csv')
    return summary,ci,horizon


def short_labels(ax):
    ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
    for s in ax.spines.values():s.set_linewidth(.8)
    ax.grid(False)


def panel_methods(ax, fam, summary, cis, legend=False):
    ys=np.arange(len(METHS))
    for i,m in enumerate(METHS):
        q=summary[(summary.family==fam) & (summary.method==m)]
        old=float(q.loc[q.M==1,'median_planted_angle'].iloc[0]); new=float(q.loc[q.M==18,'median_planted_angle'].iloc[0])
        c=MCOL[m]
        # Thin neutral connector makes the change in observation count visible.
        ax.plot([old,new],[i,i],color='#BABABA',lw=.85,zorder=1)
        # Bootstrap 95% CI only for M=18; report seed-count limitation in caption.
        r=cis[(cis.family==fam)&(cis.M==18)&(cis.method==m)].iloc[0]
        low=float(r.CI95_low_deg); high=float(r.CI95_high_deg)
        ax.plot([low,high],[i,i],color=c,lw=1.7,alpha=.53,zorder=2)
        ax.plot([low,low],[i-.095,i+.095],color=c,lw=.9,alpha=.65,zorder=2)
        ax.plot([high,high],[i-.095,i+.095],color=c,lw=.9,alpha=.65,zorder=2)
        ax.scatter([old],[i],facecolors='white',edgecolors=pale(c,.38),
                   linewidths=1.35,s=34,zorder=4)
        ax.scatter([new],[i],facecolors=c,edgecolors='white',
                   linewidths=.50,s=40,zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([MLAB[x] for x in METHS],fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(-1.5,94)
    ax.set_xticks([0,30,60,90])
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel(r'$\theta_\star\ ({}^\circ)$')
    short_labels(ax)
    if legend:
        handles=[Line2D([0],[0],marker='o',linestyle='None',markersize=5,
                        markerfacecolor='white',markeredgecolor='#8B8B8B',label=r'$M=1$'),
                 Line2D([0],[0],marker='o',linestyle='None',markersize=5,
                        markerfacecolor='#525252',markeredgecolor='#525252',label=r'$M=18$')]
        leg=ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.04,.98),
                      frameon=False,handlelength=.8,labelspacing=.25,fontsize=7.5)


def panel_horizon(ax,df):
    d=df[df.M==18]
    for fam,(label,col) in FAM.items():
        sub=d[d.family==fam]
        vals=sub.groupby('T',sort=True)['planted_angle_deg'].median()
        ax.plot(vals.index.to_numpy(dtype=float),vals.to_numpy(dtype=float),
                lw=1.45,marker='o',markersize=3.8,markeredgecolor='white',
                markeredgewidth=.5,color=col,label=label)
    ax.set_xscale('log')
    ax.set_xlim(.9,60)
    ax.set_ylim(0,95)
    ax.set_yticks([0,30,60,90])
    ax.set_xticks([1,2,5,10,20,50])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10,subs=np.arange(2,10)*.1))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel(r'Horizon $T$')
    ax.set_ylabel(r'$\theta_\star(T)\ ({}^\circ)$')
    short_labels(ax)
    leg=ax.legend(loc='lower right',ncol=1,frameon=False,fontsize=7.0,
                  borderpad=.2,handlelength=1.4,labelspacing=.35)


def panel_gain(ax,df):
    d=df[(df.family=='dominant_embedded') & (df.M==18)]
    ys=np.arange(len(METHS))
    for i,m in enumerate(METHS):
        v=float(d.loc[d.method==m,'median_relative_growth_capture'].iloc[0])
        c=MCOL[m]
        ax.hlines(y=i,xmin=0,xmax=v,color='#C2C2C2',lw=1.0,zorder=1)
        ax.scatter([v],[i],s=43,marker='o',color=c,edgecolors='white',lw=.5,zorder=3)
    ax.axvline(1.0,ls=':',lw=1,color='#A0A0A0',zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([MLAB[x] for x in METHS],fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(-.02,1.06)
    ax.set_xticks([0,.25,.5,.75,1.0])
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2g'))
    ax.xaxis.set_minor_locator(MultipleLocator(.125))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel(r'$\eta_1(Q)$')
    short_labels(ax)


def make_figure():
    configure_style()
    summary,ci,horizon=get_inputs()
    assert (summary[(summary.family=='dominant_embedded')&(summary.M==18)&(summary.method=='M2')]['n'].iloc[0]==8)
    assert len(horizon[(horizon.family=='dominant_embedded')&(horizon.M==18)])==48
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.60),layout=None)
    plt.subplots_adjust(left=.105,right=.985,top=.92,bottom=.115,wspace=.37,hspace=.44)
    panel_methods(axs[0,0],'dominant_embedded',summary,ci)
    panel_methods(axs[0,1],'competing_nonnormal_background',summary,ci,legend=True)
    panel_horizon(axs[1,0],horizon)
    panel_gain(axs[1,1],summary)
    for lab,ax in zip('abcd',axs.flat):
        panel_label(ax,lab,x=-.07,y=1.045)
    # Panels deliberately have no heading/title/subtitle. Caption identifies the families.
    pdf=OUT/'RevFig01_methods.pdf'
    png=OUT/'RevFig01_methods.png'
    fig.savefig(pdf,bbox_inches='tight')
    fig.savefig(png,dpi=550,bbox_inches='tight')
    plt.close(fig)
    return pdf,png


def main():
    pdf,png=make_figure()
    print('Generated',pdf,'and',png)

if __name__=='__main__':
    main()
