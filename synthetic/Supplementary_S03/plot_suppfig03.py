# -*- coding: utf-8 -*-
"""NNDR S3: planted time-varying response geometry (V19 T10).

Plot-only, standalone, Spyder F5. All numerical inputs are frozen copies
in ./data; no simulations are re-run and no source data are modified.
"""
from pathlib import Path
import hashlib
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
FIGURES = HERE / 'figures'
FIGURES.mkdir(exist_ok=True)

CASES = ['stationary', 'slow_rotation', 'fast_rotation',
         'threshold_crossing', 'noise_variance_jump']
CASE_LABELS = ['Stationary', 'Slow rot.', 'Fast rot.',
               'Crossing', 'Noise jump']
ORANGE = '#D55E00'  # manuscript M2
GREEN = '#009E73'   # manuscript M3
INK = '#25272A'
MUTED = '#8A9095'
PALE = '#E0E5E9'
METHOD_COL = {'M2': ORANGE, 'M3': GREEN}
METHOD_MARK = {'M2': 'o', 'M3': '^'}
OFF = {'M2': -.145, 'M3': .145}


def lighten(color, fraction=.62):
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(color))
    return tuple(c*(1-fraction) + fraction*np.ones(3))


def configure_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9.2,
        'axes.labelsize': 9.8,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.4,
        'legend.fontsize': 8.0,
        'axes.linewidth': .85,
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


def format_axis(ax):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=4.2)
    ax.tick_params(which='minor', length=2.1)
    for s in ax.spines.values():
        s.set_linewidth(.85)
    ax.grid(False)


def tag(ax, s):
    ax.text(-.10, 1.045, f'({s})', transform=ax.transAxes,
            ha='left', va='bottom', fontweight='bold', fontsize=11.1,
            color=INK, clip_on=False)


def row_layout(ax):
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(CASE_LABELS)
    ax.set_ylim(4.55, -.55)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())


def paired_dumbbells(ax, df, column, interval_data=None):
    """Paired M=1 (pale/open) and M=12 (full) medians, no invented CI.

    Panel (a) alone optionally includes M=12 subject-cluster bootstrap CIs.
    """
    for j, case in enumerate(CASES):
        for met in ['M2','M3']:
            ss = df[(df.case == case) & (df.method == met)]
            assert len(ss) == 2
            v1 = float(ss.loc[ss.M == 1, column].iloc[0])
            v12 = float(ss.loc[ss.M == 12, column].iloc[0])
            y = j + OFF[met]
            col = METHOD_COL[met]
            pale = lighten(col)
            ax.plot([v1, v12], [y, y], color=lighten(col,.44), lw=1.50,
                    solid_capstyle='round', zorder=2)
            ax.plot(v1, y, marker=METHOD_MARK[met], markersize=5.0,
                    mfc='white', mec=pale, mew=1.45, ls='none', zorder=4)
            ax.plot(v12, y, marker=METHOD_MARK[met], markersize=5.1,
                    mfc=col, mec=col, mew=1.0, ls='none', zorder=5)
            if interval_data is not None:
                r = interval_data[(interval_data.case == case)&
                                  (interval_data.M == 12)&
                                  (interval_data.method == met)].iloc[0]
                lo = float(r.cluster_bootstrap_95_lower)
                hi = float(r.cluster_bootstrap_95_upper)
                # compact source-provided subject-cluster 95% interval
                ax.plot([lo,hi], [y,y], color=col, lw=1.0, alpha=.78, zorder=3)
                ax.plot([lo,lo], [y-.055,y+.055], color=col, lw=.95, alpha=.78, zorder=3)
                ax.plot([hi,hi], [y-.055,y+.055], color=col, lw=.95, alpha=.78, zorder=3)
    row_layout(ax)
    format_axis(ax)


def panel_a(ax, smry, ci):
    paired_dumbbells(ax, smry, 'median_angle_mid_deg', interval_data=ci)
    ax.set_xlim(0,51)
    ax.set_xticks([0,10,20,30,40,50])
    ax.set_xlabel(r'$\operatorname{med}\,\theta_{\mathrm{mid}}$ ($^\circ$)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    tag(ax,'a')


def panel_b(ax, smry):
    paired_dumbbells(ax, smry, 'median_abs_R_err_center_valid')
    ax.set_xlim(0,.46)
    ax.set_xticks([0,.1,.2,.3,.4])
    ax.set_xlabel(r'$\operatorname{med}\,|\widehat R_A-R_{A,c}|$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    tag(ax,'b')


def panel_c(ax, smry):
    temp = smry.copy()
    temp['valid_share'] = 100.0*temp['valid']/temp['observations']
    paired_dumbbells(ax,temp,'valid_share')
    ax.set_xlim(60,103)
    ax.set_xticks([60,70,80,90,100])
    ax.axvline(100, color=PALE, ls=':', lw=1.1, zorder=0)
    ax.set_xlabel(r'Admissible $R_A$ (\%)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    tag(ax,'c')


def panel_d(ax, window, crossing):
    sub = window[(window.case=='threshold_crossing')&(window.M==12)]
    tt = np.sort(sub.t_center.unique())
    target = sub.groupby('t_center')['R_center'].median().reindex(tt).to_numpy(float)
    ax.plot(tt, target, color=INK, lw=1.9, marker='.', ms=4.0, zorder=6)
    for met in ['M2','M3']:
        d = sub[(sub.method==met)&(sub.status=='ok')]
        med = d.groupby('t_center')['R_hat'].median().reindex(tt).to_numpy(float)
        lo = d.groupby('t_center')['R_hat'].quantile(.25).reindex(tt).to_numpy(float)
        hi = d.groupby('t_center')['R_hat'].quantile(.75).reindex(tt).to_numpy(float)
        color=METHOD_COL[met]; marker=METHOD_MARK[met]
        # thin whiskers: subject-level descriptive IQR, not a CI
        ax.vlines(tt,lo,hi,colors=color,alpha=.35,lw=1.15,zorder=2)
        ax.plot(tt,med,color=color,lw=1.35,marker=marker,ms=4.5,
                markerfacecolor='white',markeredgewidth=1.1,zorder=4)
    ct = float(crossing.true_center_crossing.dropna().iloc[0])
    assert np.allclose(crossing.true_center_crossing.dropna(),ct)
    ax.axhline(1,color=MUTED,lw=.9,ls='--',zorder=1)
    ax.axvline(ct,color='#B7BBC0',lw=.85,ls=':',zorder=1)
    ax.set_xlim(22,375)
    ax.set_ylim(.26,2.85)
    ax.set_xticks([40,120,200,280,360])
    ax.set_yticks([.5,1.,1.5,2.,2.5])
    ax.set_xlabel(r'Window center $t$')
    ax.set_ylabel(r'$R_A$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    format_axis(ax)
    tag(ax,'d')
    # Compact key for line identity, separate from the encoding key below.
    handles=[
        Line2D([0],[0],color=INK,lw=1.8,marker='.',ms=5,label='Known'),
        Line2D([0],[0],color=ORANGE,lw=1.2,marker='o',mfc='white',ms=4,label='M2'),
        Line2D([0],[0],color=GREEN,lw=1.2,marker='^',mfc='white',ms=4,label='M3'),
    ]
    ax.legend(handles=handles,frameon=False,loc='upper left',ncol=3,
              fontsize=7.6,columnspacing=.75,handlelength=1.5,
              handletextpad=.35, borderaxespad=.32)


def build():
    configure_style()
    s=pd.read_csv(DATA/'T10_summary_by_case.csv')
    ci=pd.read_csv(DATA/'T10_subject_cluster_plane_angle_CI.csv')
    w=pd.read_csv(DATA/'T10_window_method_results.csv')
    cross=pd.read_csv(DATA/'T10_crossing_per_subject.csv')
    assert len(s)==20 and len(ci)==20 and len(w)==1080 and len(cross)==24
    assert set(s['M'])=={1,12} and set(s['method'])=={'M2','M3'}
    assert set(s['case'])==set(CASES)
    assert (s.valid+s.inadmissible == s.observations).all()
    assert (s[s.M==12].inadmissible == 0).all()
    assert w.loc[w.status!='ok','R_hat'].isna().all()
    assert (s.observations==54).all()

    fig,axs=plt.subplots(2,2,figsize=(7.65,5.52))
    a,b,c,d=axs.flat
    panel_a(a,s,ci)
    panel_b(b,s)
    panel_c(c,s)
    panel_d(d,w,cross)

    # Shared legend for paired method/regime encoding in a-c only.
    handles = [
        Line2D([],[],color=ORANGE,marker='o',mfc=ORANGE,ms=5,lw=1.5,label='M2'),
        Line2D([],[],color=GREEN,marker='^',mfc=GREEN,ms=5,lw=1.5,label='M3'),
        Line2D([],[],color=MUTED,marker='o',mfc='white',ms=5,lw=1.35,label=r'$M=1$ (pale/open)'),
        Line2D([],[],color=INK,marker='o',mfc=INK,ms=5,lw=1.35,label=r'$M=12$ (full colour)'),
    ]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.022),
               ncol=4,frameon=False,fontsize=8.0,
               handlelength=1.6,columnspacing=1.5,handletextpad=.4)
    fig.subplots_adjust(left=.18,right=.987,top=.952,bottom=.15,wspace=.41,hspace=.33)
    prefix=FIGURES/'SuppFig03_Planted_TimeVarying_REVISED'
    fig.savefig(str(prefix)+'.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(str(prefix)+'.png',dpi=550,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return prefix


def sha256(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


if __name__ == '__main__':
    stem=build()
    record={"data_source":"V19 T10, six independent subjects and 9 overlapping windows per subject per case/M/method", "figure":"S3 candidate, figure-only styling",
            "files":{str(p.relative_to(HERE)):sha256(p) for p in list(DATA.glob('*.csv'))+[Path(str(stem)+'.pdf'),Path(str(stem)+'.png')]},
            "checks":{"rows_summary":20,"window_rows":1080,"crossing_rows":24,"M12_invalid_fits":0}}
    (HERE/'manifest.json').write_text(json.dumps(record,indent=2))
    print('Wrote',stem)
