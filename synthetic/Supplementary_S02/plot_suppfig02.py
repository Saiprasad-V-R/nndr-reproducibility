# -*- coding: utf-8 -*-
"""Publication-style Supplementary Figure S2, nonlinear validation.

Run in Spyder with F5. This script reads only the included frozen V19 T09 data.
It deliberately preserves inadmissible reductions as missing and marks
sparsely admissible time windows without joining them by curves.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, FixedLocator
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

ORANGE = '#D55E00'    # M2, manuscript palette
GREEN = '#009E73'     # M3
BLUE = '#0B3C6D'      # statistical comparator
GRAY = '#84929F'
INK = '#23262B'
MID = '#747A81'
PALE = '#E9EEF1'

MODELS = [
    ('FHN_directed_spikes', 'FHN'),
    ('SL_directed_below_Hopf', r'SL$^{-}$'),
    ('SL_directed_above_Hopf', r'SL$^{+}$'),
    ('double_well_gradient', 'Gradient'),
]


def configure_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9.2,
        'axes.labelsize': 9.8,
        'xtick.labelsize': 8.6,
        'ytick.labelsize': 8.6,
        'legend.fontsize': 7.5,
        'axes.linewidth': 0.85,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 4.2, 'ytick.major.size': 4.2,
        'xtick.minor.size': 2.1, 'ytick.minor.size': 2.1,
        'xtick.major.width': .8, 'ytick.major.width': .8,
        'xtick.minor.width': .65, 'ytick.minor.width': .65,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'figure.dpi': 170, 'savefig.dpi': 550,
    })


def panel_label(ax, letter):
    ax.text(-.13, 1.035, f'({letter})', transform=ax.transAxes, ha='left', va='bottom',
            fontweight='bold', fontsize=11.2, clip_on=False, color=INK)


def format_axis(ax, logarithmic=False):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=4.2)
    ax.tick_params(which='minor', length=2.1)
    for spine in ax.spines.values():
        spine.set_linewidth(.85)
    ax.grid(False)
    if logarithmic:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1))
        ax.yaxis.set_minor_formatter(NullFormatter())


def legend(ax, **kwargs):
    leg = ax.legend(frameon=False, fontsize=7.45, handlelength=1.75,
                    handletextpad=.5, labelspacing=.3, borderpad=.15, **kwargs)
    return leg


def subject_bootstrap(sub, seed, replicates=1600):
    groups = [g['plane_error_deg'].dropna().to_numpy(float)
              for _,g in sub.groupby('subject_rep', sort=True)]
    groups = [x for x in groups if x.size]
    assert len(groups) == 8
    point = float(np.median(np.concatenate(groups)))
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates)
    for i in range(replicates):
        ids = rng.integers(0, len(groups), len(groups))
        boot[i] = np.median(np.concatenate([groups[j] for j in ids]))
    return point, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def panel_a(ax, w):
    fhn = w[w.family == MODELS[0][0]]
    g = fhn.groupby('window_center', sort=True)
    centers = np.array(sorted(fhn.window_center.unique()), dtype=float)
    ref = g['oracle_RA'].median().reindex(centers).to_numpy(float)
    ax.axvspan(23, 30, color=PALE, linewidth=0, zorder=0)
    ax.axhline(1, color='#969B9F', linestyle=':', linewidth=1, zorder=0)
    ax.plot(centers, ref, color=INK, marker='o', markersize=3.7,
            lw=1.8, markeredgewidth=0, label=r'Known $T_w$', zorder=5)

    annotations=[]
    for col, valid_col, color, marker, label, offset in [
        ('inferred_M2_RA', 'inferred_M2_RA_valid', ORANGE, 'o', 'M2', -1),
        ('inferred_M3_RA', 'inferred_M3_RA_valid', GREEN, '^', 'M3', 1),
    ]:
        med = g[col].median().reindex(centers).to_numpy(float)
        n = g[valid_col].sum().reindex(centers).to_numpy(int)
        # Draw lines only through consecutive, well-supported estimates;
        # do not connect isolated 1/8 or 4/8 points to stable segments.
        supported = np.where(n >= 6, med, np.nan)
        ax.plot(centers, supported, color=color, lw=1.5, marker=marker, ms=4.3,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.2,
                label=label, zorder=4)
        partial = (n > 0) & (n < 6) & np.isfinite(med)
        ax.scatter(centers[partial], med[partial], marker=marker, s=34,
                   edgecolors=color, facecolors='white', linewidths=1.35,
                   zorder=6)
        for tt, yy, nn in zip(centers[partial], med[partial], n[partial]):
            annotations.append((float(tt), float(yy), int(nn), label))
    # Annotate only the two sparse FHN locations. Place the labels in
    # available space below the peaks, away from the method legend.
    ax.text(31.4, 2.66, r'$1/8$', fontsize=7.25, color=MID, ha='center')
    ax.text(40.1, 2.79, r'$4/8,\;5/8$', fontsize=7.2, color=MID, ha='center')
    ax.set_xlim(5, 55)
    ax.set_ylim(0, 3.85)
    ax.set_xticks([6, 14, 22, 30, 38, 46, 54])
    ax.set_xlabel(r'Window center $t$')
    ax.set_ylabel(r'$R_A$')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    legend(ax, loc='upper left', ncol=3, bbox_to_anchor=(.015,.985))
    format_axis(ax)
    return annotations


def panel_b(ax, w):
    positions = np.arange(4)
    # Distinct scientific references, compared on identical windows.
    series = [
        ('stat_reference_relative_error', BLUE, 'o', r'$F_{\rm stat}$', -.095),
        ('exp_mean_J_relative_error', GRAY, 's', r'$e^{\Delta t\bar J}$', +.095),
    ]
    results = {}
    for col,color,marker,label,offset in series:
        medians=[]
        for i,(family,short) in enumerate(MODELS):
            vals = w.loc[w.family==family,col].dropna().to_numpy(float)
            q1,med,q3 = np.quantile(vals, [.25,.5,.75])
            ax.plot([i+offset,i+offset],[q1,q3], color=color, lw=2.1,
                    solid_capstyle='round', zorder=3)
            ax.plot(i+offset,med,marker=marker,ms=5.3,linestyle='none',
                    mfc='white' if marker=='o' else color,mec=color,mew=1.15,zorder=5)
            medians.append(med)
        results[col] = medians
    ax.set_xticks(positions)
    ax.set_xticklabels([s for _,s in MODELS])
    ax.set_xlim(-.48,3.48)
    ax.set_yscale('log')
    ax.set_ylim(6e-4,.92)
    ax.set_ylabel(r'$\delta_F$')
    legend(ax, handles=[
        Line2D([], [], color=BLUE, marker='o', mfc='white', mec=BLUE, lw=1, ms=4.7, label=r'$F_{\rm stat}$'),
        Line2D([], [], color=GRAY, marker='s', mfc=GRAY, mec=GRAY, lw=1, ms=4.7, label=r'$e^{\Delta t\bar J}$'),
    ], loc='upper left', ncol=2, bbox_to_anchor=(.02,.985))
    format_axis(ax, logarithmic=True)
    return results


def panel_c(ax, methods):
    x = np.arange(4)
    out={}
    for method,color,marker,offset in [('M2',ORANGE,'o',-.11),('M3',GREEN,'^',+.11)]:
        rows=[]
        for i,(family,short) in enumerate(MODELS):
            sub = methods[(methods.family==family)&(methods.method==method)]
            point,lo,hi = subject_bootstrap(sub,seed=42420+i+(100 if method=='M3' else 0))
            ax.errorbar(i+offset, point, yerr=[[point-lo],[hi-point]], fmt=marker,
                        color=color, mfc='white', mec=color, mew=1.2, markersize=5.3,
                        elinewidth=1.1, capsize=2.7, linestyle='none', zorder=4)
            rows.append([family,point,lo,hi])
        out[method]=rows
    ax.set_xticks(x)
    ax.set_xticklabels([name for _,name in MODELS])
    ax.set_xlim(-.48,3.48)
    ax.set_ylim(0,93)
    ax.set_ylabel(r'$\theta_{\max}(\widehat Q,Q_{T_w})$ ($^\circ$)')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    legend(ax, handles=[
        Line2D([],[],marker='o',color=ORANGE,mfc='white',mec=ORANGE,ls='none',label='M2'),
        Line2D([],[],marker='^',color=GREEN,mfc='white',mec=GREEN,ls='none',label='M3'),
    ], loc='lower left', ncol=2)
    format_axis(ax)
    return out


def panel_d(ax, methods):
    # Horizontal paired lollipops instead of a nearly empty tall count plot.
    counts={}
    yticks=np.arange(len(MODELS))[::-1]  # FHN on top
    for method,color,marker,yshift in [('M2',ORANGE,'o',+.12),('M3',GREEN,'^',-.12)]:
        counts[method]=[]
        for j,(family,short) in enumerate(MODELS):
            sub = methods[(methods.family==family)&(methods.method==method)]
            invalid=int((~sub.inferred_RA_valid.astype(bool)).sum())
            counts[method].append(invalid)
            yy=float(yticks[j]+yshift)
            ax.plot([0,invalid],[yy,yy],color=color,alpha=.55,lw=1.65,zorder=2)
            ax.plot([invalid],[yy],marker=marker,ls='none',mec=color,mfc='white',mew=1.2,
                    ms=5.5,zorder=3)
            ax.annotate(str(invalid),(invalid,yy),xytext=(5,0),textcoords='offset points',
                        ha='left', va='center',fontsize=7.8,color=color)
    ax.set_yticks(yticks)
    ax.set_yticklabels([short for _,short in MODELS])
    ax.set_xlim(-1.5, 52.5)
    ax.set_xticks([0,10,20,30,40,50])
    ax.set_ylim(-.55,3.55)
    ax.set_xlabel(r'Inadmissible $R_A$ estimates (of 104)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y',labelright=False)
    legend(ax, handles=[
        Line2D([],[],marker='o',color=ORANGE,mfc='white',mec=ORANGE,ls='none',label='M2'),
        Line2D([],[],marker='^',color=GREEN,mfc='white',mec=GREEN,ls='none',label='M3'),
    ], loc='lower right', ncol=2)
    format_axis(ax)
    return counts


def main():
    configure_style()
    w=pd.read_csv(DATA/'T09_window_truth_and_fit.csv')
    methods=pd.read_csv(DATA/'T09_all_method_results.csv')
    assert len(w)==416 and len(methods)==832
    assert all((w.family==name).sum()==104 for name,_ in MODELS)
    assert all(len(methods[(methods.family==name)&(methods.method==method)])==104
               for name,_ in MODELS for method in ['M2','M3'])

    fig,axs=plt.subplots(2,2,figsize=(7.45,5.10),constrained_layout=False)
    a,b,c,d=axs.ravel()
    sparse=panel_a(a,w)
    discrep=panel_b(b,w)
    angles=panel_c(c,methods)
    counts=panel_d(d,methods)
    for ax,name in zip(axs.ravel(),'abcd'): panel_label(ax,name)
    fig.subplots_adjust(left=.118,right=.984,bottom=.13,top=.95,wspace=.31,hspace=.35)
    output_pdf=OUT/'SuppFig02_Nonlinear_Tangent_POLISHED.pdf'
    output_png=OUT/'SuppFig02_Nonlinear_Tangent_POLISHED.png'
    fig.savefig(output_pdf,bbox_inches='tight',pad_inches=.045,facecolor='white')
    fig.savefig(output_png,dpi=550,bbox_inches='tight',pad_inches=.045,facecolor='white')
    plt.close(fig)

    assert counts['M2']==[44,0,0,7],counts
    assert counts['M3']==[42,0,0,0],counts
    grad=w[w.family=='double_well_gradient']
    adverse=int(((grad.inferred_M2_RA>1)&(grad.oracle_RA<=1)&grad.inferred_M2_RA_valid).sum())
    gate_accepted=int(((grad.inferred_M2_RA>1)&(grad.oracle_RA<=1)&grad.gate_pass).sum())
    assert adverse==1 and gate_accepted==1
    truth={
        'data_source': 'V19/T09 frozen independent nonlinear physics audit',
        'window_rows':len(w), 'method_rows':len(methods),
        'subjects_per_model':8,'windows_per_model':104,
        'invalid_RA_counts':counts,
        'sparsely_admissible_fhn_marks':sparse,
        'reference_median_errors':discrep,
        'subject_cluster_bootstrap_95pct':angles,
        'gradient_spurious_M2_RA_gt_one':adverse,
        'gradient_spurious_accepted_by_transferred_gate':gate_accepted,
        'simulations_rerun':False,
    }
    (ROOT/'VALIDATION.json').write_text(json.dumps(truth,indent=2))
    print('Generated', output_pdf.name, 'and', output_png.name)
    return truth

if __name__=='__main__':
    main()
