# -*- coding: utf-8 -*-
"""Reviewer Fig. 2, carefully redesigned for a two-column research manuscript.

Uses frozen V19 tables. Run this file directly in Spyder (F5).
Requires numpy, pandas, matplotlib. Outputs editable vector PDFs and 600-dpi PNGs.
The primary and companion plots have identical data and geometry except for the
additional light-gray discrete-time R_sigma distribution in panel (c).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DAT = ROOT / 'data'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

ORANGE = '#D55E00'  # M2, from the original manuscript time-varying figure
GREEN = '#009E73'   # M3
GRAY = '#858585'    # baseline-fixed M2
BLACK = '#262626'   # population and R_A
LIGHT = '#a6aab1'   # companion R_sigma
RNG = np.random.default_rng(20261120)  # visual jitter only

INTERVENTIONS = ('geometry_only', 'spectrum_only', 'plane_rotation', 'combined')
XTICKS = ('Geometry', 'Spectrum', 'Rotation', 'Combined')
METHODS = (
    ('M2', ORANGE, 'o', 'M2'),
    ('M3', GREEN, '^', 'M3'),
    ('baseline_fixed_M2', GRAY, 's', 'Fixed-plane M2'),
)
OFFSETS = (-0.19, 0, 0.19)


def publication_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 8.4, 'axes.labelsize': 9.0,
        'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0,
        'legend.fontsize': 8.0,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
        'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def axis_style(ax, label):
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.grid(False)
    ax.text(-0.105, 1.075, f'({label})', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=11, fontweight='bold', clip_on=False)


def plot_summary(ax, summ, metric, low, high):
    # Categorical interventions: no lines through unrelated categories.
    for j,(method,color,marker,name) in enumerate(METHODS):
        for i, interv in enumerate(INTERVENTIONS):
            line = summ[(summ.intervention == interv) & (summ.method == method)]
            if len(line) != 1:
                raise ValueError(f'Expected one row for {interv}/{method}, got {len(line)}')
            row = line.iloc[0]
            y = float(row[metric]); lo = float(row[low]); hi = float(row[high]);
            if not (np.isfinite(y) and np.isfinite(lo) and np.isfinite(hi)):
                continue
            if not lo <= y <= hi:
                raise ValueError(f'Invalid interval: {metric}, {interv}, {method}')
            x = i + OFFSETS[j]
            ax.plot([x,x], [lo,hi], color=color, lw=1.15, alpha=0.78, solid_capstyle='round', zorder=2)
            ax.plot([x-0.045,x+0.045],[lo,lo], color=color, lw=0.8, alpha=0.78)
            ax.plot([x-0.045,x+0.045],[hi,hi], color=color, lw=0.8, alpha=0.78)
            ax.scatter([x],[y], s=24, facecolors='white' if method != 'baseline_fixed_M2' else GRAY,
                       edgecolors=color, marker=marker, linewidths=1.2, zorder=4)
    ax.set_xlim(-0.55, 3.55)
    ax.set_xticks(np.arange(4), labels=XTICKS)


def normal_controls(ax, normal, include_rsigma):
    known = normal[normal['mode']=='population'].sort_values('anchor_id')
    fitted = normal[normal['mode']=='inferred'].sort_values('anchor_id')
    assert known.anchor_id.tolist() == fitted.anchor_id.tolist()
    assert len(known) == len(fitted) == 30
    specs = [('R', BLACK, 'o', -0.105, r'$R_A$')]
    if include_rsigma:
        specs.append(('R_sigma', LIGHT, '^', +0.105, r'$R_\sigma$'))
    for key,color,marker,offset,label in specs:
        for j, group in enumerate((known,fitted)):
            yy = group[key].to_numpy(dtype=float)
            xx = np.full(len(yy), float(j) + offset) + RNG.uniform(-0.043,0.043,size=len(yy))
            # Known-normal values are numerically ~0, so draw a population marker
            # instead of placing 30 coincident dots on top of each other.
            if j == 0:
                ax.scatter([j + offset],[np.median(yy)], s=35,
                           facecolors='white',edgecolors=color,linewidths=1.3,
                           marker=marker, zorder=5)
            else:
                ax.scatter(xx, yy, s=15, facecolors='white', edgecolors=color,
                           linewidths=0.7, alpha=0.70, marker=marker,zorder=3)
                median=float(np.median(yy))
                # Median is a heavy horizontal bar, not a spurious curve.
                ax.plot([j + offset-0.067,j+offset+0.067],[median,median],
                        color=color,lw=2.2,solid_capstyle='round',zorder=5)
    ax.set_xticks([0,1], labels=('Known $F$', 'Fitted $\\widehat F$'))
    ax.set_xlim(-0.35,1.35)
    ax.set_ylim(-0.012,0.275)
    ax.set_yticks([0,.05,.10,.15,.20,.25])
    ax.set_ylabel(r'$R_A$' if not include_rsigma else r'$R_A$, $R_\sigma$')
    if include_rsigma:
        handles=[Line2D([0],[0],color=BLACK,marker='o',markerfacecolor='white',lw=0,markersize=5,label=r'$R_A$'),
                 Line2D([0],[0],color=LIGHT,marker='^',markerfacecolor='white',lw=0,markersize=5,label=r'$R_\sigma$')]
        ax.legend(handles=handles,ncol=2,loc='upper left',frameon=False,
                  handletextpad=.35,borderpad=0.1,columnspacing=1.0)


def decomposition(ax,paired):
    sub=paired[(paired.method=='planted_current') & (paired.intervention=='spectrum_only')]
    assert len(sub)==20, f'Expected 20 independent cases, got {len(sub)}'
    cols=[('contrast_log_K',r'$\Delta\log K$'),
          ('contrast_log_Kc_variance',r'$\Delta\log K_{c,A}$'),
          ('contrast_log_R_variance',r'$\Delta\log R_A$')]
    ax.axhline(0,color='0.6',lw=0.75,zorder=0)
    for i,(col,label) in enumerate(cols):
        a=sub[col].dropna().to_numpy(dtype=float)
        assert len(a)==20
        x=np.full(len(a),i)+RNG.uniform(-0.065,0.065,size=len(a))
        ax.scatter(x,a,s=12,facecolors='none',edgecolors='.6',linewidths=.6,zorder=2)
        median=np.median(a); qlo,qhi=np.quantile(a,[.25,.75])
        ax.vlines(i,qlo,qhi,color=BLACK,lw=3.1,zorder=4)
        ax.plot([i-.12,i+.12],[median,median],color=BLACK,lw=2.5,zorder=5)
        ax.scatter([i],[median],s=25,edgecolors=BLACK,facecolors='white',linewidths=1.2,zorder=6)
    ax.set_xticks([0,1,2],labels=[t for _,t in cols])
    ax.set_xlim(-.5,2.5)
    ax.set_ylim(-.71,.71)
    ax.set_yticks([-.6,-.3,0,.3,.6])
    ax.set_ylabel('Paired log contrast')


def make_figure(include_rsigma):
    publication_style()
    summary=pd.read_csv(DAT/'T05_method_summary.csv')
    normal=pd.read_csv(DAT/'T04_normal_spectrum_controls.csv')
    paired=pd.read_csv(DAT/'T05_all_paired_contrasts.csv')

    # Intentional 2-column width; original serif and inward tick conventions.
    fig,axs=plt.subplots(2,2,figsize=(7.05,5.15),facecolor='white')
    a,b,c,d=axs.flat
    plot_summary(a,summary,'median_angle_deg','angle_ci95_lo','angle_ci95_hi')
    a.set_ylim(-3,95); a.set_yticks([0,30,60,90]); a.set_ylabel(r'$\theta_\ast$ (deg)'); axis_style(a,'a')
    plot_summary(b,summary,'median_capture_population','capture_ci95_lo','capture_ci95_hi')
    b.axhline(1.0,color='.80',lw=.8,ls='--',zorder=0)
    b.set_ylim(.62,1.035); b.set_yticks([.65,.75,.85,.95,1.0])
    b.set_ylabel(r'$c_Q$'); axis_style(b,'b')
    normal_controls(c,normal,include_rsigma); axis_style(c,'c')
    decomposition(d,paired); axis_style(d,'d')

    legend = [Line2D([0],[0],color=color,marker=marker,
                     markerfacecolor='white' if method != 'baseline_fixed_M2' else GRAY,
                     markeredgecolor=color,markersize=5,linewidth=0,label=name)
              for method,color,marker,name in METHODS]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.018),
               ncol=3,frameon=False,handlelength=1.0,columnspacing=2.2,
               handletextpad=.4,fontsize=8.6)
    fig.subplots_adjust(left=.108,right=.983,bottom=.165,top=.929,wspace=.33,hspace=.52)
    name='RevFig02_RA_Rsigma' if include_rsigma else 'RevFig02_RA_only'
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return name


if __name__=='__main__':
    for include in [False,True]:
        name=make_figure(include)
        print('Created:',name)
