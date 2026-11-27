# -*- coding: utf-8 -*-
"""Publication-style Supplementary Figure S1: exact sampling, plane reselection,
reduced-ratio admissibility and complex-frequency aliasing.

Plot-only, reproducible in Spyder with F5. Uses accompanying frozen V19 CSVs.
No simulation is rerun; no invalid ratios are imputed.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Prescribed manuscript style (from Manuscript_Figure_Prompt_Didier_Virgile).
def configure_style() -> None:
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9.2,
        'axes.labelsize': 9.2,
        'axes.titlesize': 10.2,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'legend.fontsize': 7.0,
        'axes.linewidth': 0.80,
        'lines.linewidth': 1.35,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'savefig.dpi': 550,
        'figure.dpi': 170,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.framealpha': 0.86,
        'legend.edgecolor': 'none',
    })

def format_axis(ax, nbx=5, nby=5) -> None:
    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', top=True, right=True, length=2)
    ax.tick_params(which='major', direction='in', top=True, right=True, length=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbx))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nby))

def panel_label(ax, label: str, x=-0.15, y=1.055) -> None:
    ax.text(x, y, f'({label})', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10.7, fontweight='bold', clip_on=False)

# Prescribed manuscript colors; where colors distinguish *operator families*
# they are not reused to suggest M2 or M3.
BLUE = '#0B3C6D'
RED = '#D62728'
GREEN = '#2CA02C'
GREY = '#75838F'
DARK = '#202020'
ORANGE = '#D55E00'


def set_fixed_lag_axis(ax):
    ax.set_xlim(0.85, 8.65)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel(r'Lag $\tau$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='major', direction='in', top=True, right=True, length=4)
    ax.tick_params(which='minor', direction='in', top=True, right=True, length=2)
    ax.grid(False)


def legend(ax, **kwargs):
    leg=ax.legend(frameon=True, borderpad=.28, labelspacing=.30, handlelength=1.7,
                  handletextpad=.42, **kwargs)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(.87)
    leg.get_frame().set_edgecolor('none')
    return leg


def draw():
    configure_style()
    d=pd.read_csv(DATA/'T03_all_lags.csv')
    alias=pd.read_csv(DATA/'T03_complex_alias.csv').sort_values('lag')
    assert set([1,2,4,8]).issubset(set(alias['lag']))
    # The exact two-dimensional audit keeps the spectral rates fixed for the
    # following four geometric strengths. Do not pool families with the same case IDs.
    fam2 = d[(d.N==2)&(d.family=='vary_geometry_fixed_spectrum')]
    fam40=d[d.N==40]
    assert len(fam2)>0 and len(fam40)>0

    fig,axs=plt.subplots(2,2,figsize=(7.08,5.02),constrained_layout=False)
    a,b,c,e=axs.ravel()
    # (a) True two-dimensional R_A is lag-invariant on the real log branch.
    # Directly label each tested geometric parameter, not an opaque case number.
    palette=['#A7B5C3', '#65829B', '#35658B', BLUE]
    for case,col in zip(sorted(fam2['case_index'].unique()),palette):
        rows=fam2[fam2.case_index==case]
        series=rows.groupby('lag')['R_pop'].median().sort_index()
        kappa=float(rows['kappa_eigenvectors'].iloc[0])
        a.plot(series.index,series.values,marker='o',markersize=3.7,
               lw=1.35,color=col,label=rf'$\kappa={kappa:g}$')
    a.axhline(1.0,color='0.63',ls=':',lw=.95,zorder=0)
    a.set_ylabel(r'$R_A$ (known $2\times2$)')
    a.set_ylim(0,3.9)
    set_fixed_lag_axis(a)
    legend(a,loc='upper center',bbox_to_anchor=(0.53,0.81),ncol=2,fontsize=7.3)

    # (b) Known operator, M2-selected plane: lag can reselect a distinct plane.
    families=[('vary_spectrum_fixed_geometry',GREY,'Spectrum varied','o'),
              ('vary_geometry_fixed_spectrum',BLUE,'Geometry varied','^')]
    for fam,color,label,mark in families:
        ss=fam40[fam40.family==fam].groupby('lag')['pop_plane_to_lag1_deg'].median().sort_index()
        b.plot(ss.index,ss.values,color=color,marker=mark,ms=4.0,lw=1.5,label=label)
    b.set_ylim(-3,76)
    b.set_ylabel(r'$\theta_{\max}(Q_\tau,Q_1)$ ($^\circ$)')
    set_fixed_lag_axis(b)
    legend(b,loc='lower right',fontsize=7.2)

    # (c) Proportion of admissible fitted rate ratios; denominators include
    # inadmissible fits, with 32 fitted instances per lag and M.
    for M,color,line,mark,fill in [(1,'#9DAAB6','--','o','white'),
                                   (25,BLUE,'-','s',BLUE)]:
        sub=fam40[fam40.M==M]
        agg=sub.groupby('lag')['R_hat'].agg(valid='count',total='size').sort_index()
        p=100*agg.valid/agg.total
        c.plot(p.index,p.values,color=color,lw=1.55,ls=line,marker=mark,ms=4.7,
               mfc=fill,mec=color,mew=1.0,label=rf'$M={M}$')
    c.set_ylabel(r'Admissible $\widehat R_A$ (\%)')
    c.set_ylim(-3,104)
    c.set_yticks([0,25,50,75,100])
    set_fixed_lag_axis(c)
    legend(c,loc='lower right',fontsize=7.5)

    # (d) Frequency alias: known physical frequency and the apparent principal
    # logarithm frequency, with the theoretical branch boundary.
    w0=float(alias['true_omega'].iloc[0])
    assert np.allclose(alias.true_omega,w0)
    e.axhline(w0,color=DARK,ls='--',lw=1.2,label=r'Known $\omega_0$')
    e.plot(alias.lag,alias.apparent_omega,color=BLUE,lw=1.55,marker='o',
           ms=4.5,label=r'Principal-log $\widehat\omega$')
    tau_branch=np.pi/w0
    e.axvline(tau_branch,color='0.68',ls=':',lw=.9)
    e.text(tau_branch+.15,.18,r'$\pi/\omega_0$',fontsize=7.4,color='0.45',ha='left',va='bottom')
    e.set_ylim(0,.88)
    e.set_ylabel(r'$\omega$ (per unit time)')
    set_fixed_lag_axis(e)
    legend(e,loc='lower left',fontsize=7.4)

    for ax,label in zip(axs.ravel(),'abcd'):
        panel_label(ax,label)

    fig.subplots_adjust(left=.115,right=.984,top=.948,bottom=.108,
                        wspace=.31,hspace=.355)
    fig.savefig(OUT/'SuppFig01_Lag_Sampling_REVISED.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/'SuppFig01_Lag_Sampling_REVISED.png',dpi=550,bbox_inches='tight',facecolor='white')
    plt.close(fig)

    # Automated numerical consistency checks, no synthetic reruns.
    exact=fam2.groupby(['family','case_index'])['R_pop'].agg(lambda q:q.max()-q.min())
    max_span=float(exact.max())
    assert max_span < 1e-10, max_span
    valid=[]
    for M in (1,25):
        q=fam40[fam40.M==M].groupby('lag')['R_hat'].agg(valid='count',total='size')
        assert (q.total==32).all(),q
        valid.append(dict(M=M,nperlag=32,valid_counts=q.valid.to_dict()))
    return {'max_population_RA_span':max_span,
            'n_N40_perlag_each_M':32,
            'admissibility':valid,
            'complex_alias_lags':alias[alias.alias.astype(bool)].lag.tolist(),
            'Nyquist_branch_boundary':tau_branch}

if __name__=='__main__':
    import json
    report=draw()
    (ROOT/'VALIDATION.json').write_text(json.dumps(report,indent=2))
    print('Rendered Supplementary Figure S1 and checked archived data.\n',json.dumps(report,indent=2))
