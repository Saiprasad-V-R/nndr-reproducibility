# -*- coding: utf-8 -*-
"""NNDR reviewer-facing coloured-forcing controls (plot-only; Spyder F5).

This figure combines two *distinct* synthetic protocols, without pooling them:
(a,b) frozen T06 negative controls from the V19 suite;
(c,d) exact aggregate counts reported in the 12 November 2026 coloured-forcing note.
The source note is bundled. No simulation outputs are reconstructed or invented.

Outputs RevFig04_RA_only.pdf/.png and RevFig04_RA_Rsigma.pdf/.png.
No figure title/subtitle, original serif fonts, inward major/minor ticks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)

ORANGE = '#D55E00'           # original manuscript M2
GREEN = '#009E73'            # original manuscript M3
BLACK = '#222222'
GRAY = '#707070'
PALE_ORANGE = '#E7B18C'
PALE_GREEN = '#90CABB'
PALE = '#B3B3B3'

CONDITIONS = [
    ('gaussian_VAR1', 'White'),
    ('coloured_isotropic', 'Isotropic\nmemory'),
    ('coloured_spatial', 'Spatial\nmemory'),
]
FAMILIES = [
    ('gaussian_VAR1', 'White'),
    ('isotropic_bursts', 'Iso. bursts'),
    ('common_mode_bursts', 'Common bursts'),
    ('coloured_isotropic', 'Iso. memory'),
    ('coloured_spatial', 'Spatial memory'),
    ('observation_noise', 'Obs. noise'),
]


def style():
    mpl.rcParams.update({
        'font.family':'serif',
        'font.serif':['Computer Modern Roman','Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'cm',
        'font.size':8.6,
        'axes.labelsize':9.0,
        'xtick.labelsize':7.9,
        'ytick.labelsize':7.9,
        'legend.fontsize':8.0,
        'axes.linewidth':.80,
        'xtick.major.width':.8,'ytick.major.width':.8,
        'xtick.minor.width':.6,'ytick.minor.width':.6,
        'xtick.major.size':4.,'ytick.major.size':4.,
        'xtick.minor.size':2.,'ytick.minor.size':2.,
        'xtick.direction':'in','ytick.direction':'in',
        'xtick.top':True,'ytick.right':True,
        'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.dpi':600, 'pdf.fonttype':42,'ps.fonttype':42,
    })


def clean(ax, letter, x=-.105):
    ax.minorticks_on()
    ax.tick_params(which='major',direction='in',top=True,right=True,length=4)
    ax.tick_params(which='minor',direction='in',top=True,right=True,length=2)
    for sp in ax.spines.values():sp.set_linewidth(.80)
    ax.grid(False)
    ax.text(x,1.055,f'({letter})',transform=ax.transAxes,fontsize=10.7,
            fontweight='bold',ha='left',va='bottom',clip_on=False)


def wilson(k,n):
    if not (isinstance(k,(int,np.integer)) and isinstance(n,(int,np.integer)) and 0<=k<=n and n>0):
        raise ValueError('Invalid counts for Wilson interval')
    z=1.959963984540054
    p=k/n
    d=1+z*z/n
    c=(p+z*z/(2*n))/d
    e=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,c-e),min(1.,c+e)


def check_t06_row(df, family, method, phase, criterion):
    sel=df[(df['family']==family)&(df['method']==method)&
           (df['phase']==phase)&(df['criterion']==criterion)]
    if len(sel)!=1:
        raise RuntimeError(f'T06 missing/duplicate row: {family} {method} {phase} {criterion}')
    row=sel.iloc[0]
    assert int(row['denominator'])==int(row['subjects_total'])==24
    return row


def panel_a(ax, refs, companion):
    # Physical F is normal: its R_A and R_sigma values are zero to numerical precision.
    xs=np.arange(len(CONDITIONS),dtype=float)
    # Analytic stationary population F_reg, not finite-trajectory estimates.
    m2=[];m3=[];r2=[];r3=[];phys=[]
    for key,_ in CONDITIONS:
        z=refs[(refs.family==key)&(refs.phase=='event')]
        d2=z[z.method=='M2'];d3=z[z.method=='M3']
        assert len(d2)==len(d3)==1
        d2=d2.iloc[0];d3=d3.iloc[0]
        assert bool(d2['fully_analytic_reference_available'])
        m2.append(float(d2['R_A_exact_effective_VAR1']))
        m3.append(float(d3['R_A_exact_effective_VAR1']))
        r2.append(float(d2['R_sigma_exact_effective_VAR1']))
        r3.append(float(d3['R_sigma_exact_effective_VAR1']))
        phys.append(float(d2['R_A_true_normal_operator']))
    assert np.all(np.isfinite(phys)) and np.max(np.abs(phys))<1e-10
    assert np.isfinite(m2).all() and np.isnan(m3[:2]).all() and np.isfinite(m3[2])
    ax.axhline(1.,lw=.8,ls='--',color='.65',zorder=0)
    ax.plot([-.12,2-.12], [0,0],color=BLACK,lw=1.1,ls=':',alpha=.65,zorder=1)
    ax.scatter(xs-.12, phys, s=37,marker='s',facecolors=BLACK,edgecolors=BLACK,
               linewidths=.7,zorder=5)
    ax.plot(xs, m2, color=ORANGE,lw=1.15,alpha=.92,zorder=2)
    ax.scatter(xs, m2, s=39,marker='o',facecolors='white',edgecolors=ORANGE,
               linewidths=1.35,zorder=6)
    ax.scatter(xs+.12, m3, s=44,marker='^',facecolors='white',edgecolors=GREEN,
               linewidths=1.35,zorder=6)
    if companion:
        # Only the defined population M3 marker is shown; no spurious zero substitute.
        ax.scatter(xs+.015, r2, s=29,marker='D',facecolors='white',edgecolors=PALE_ORANGE,
                   linewidths=1.1,zorder=4)
        ax.scatter(xs+.17, r3, s=31,marker='D',facecolors='white',edgecolors=PALE_GREEN,
                   linewidths=1.1,zorder=4)
        ax.set_ylim(-.15,2.40)
        ax.set_ylabel(r'$R_A$; pale: $R_\sigma$')
        ax.set_yticks([0,.5,1,1.5,2.0])
    else:
        ax.set_ylim(-.08,1.52)
        ax.set_ylabel(r'$R_A$')
        ax.set_yticks([0,.5,1,1.5])
    ax.set_xlim(-.42,2.42)
    ax.set_xticks(xs, labels=[label for _,label in CONDITIONS])
    ax.yaxis.set_minor_locator(MultipleLocator(.25))
    ax.xaxis.set_minor_locator(NullLocator())
    clean(ax,'a')


def panel_b(ax, summary):
    # Each family is 24 independent negative-control subjects; raw event exceedances.
    y0=np.arange(len(FAMILIES))
    for i,(family,_label) in enumerate(FAMILIES):
        for method,dy,color,marker in [('M2',-.13,ORANGE,'o'),('M3',+.13,GREEN,'^')]:
            row=check_t06_row(summary,family,method,'event','R_A')
            k=int(row['count_gt_1']);n=int(row['denominator'])
            lo,hi=wilson(k,n)
            assert abs(lo-row['ci95_wilson_low'])<5e-8
            assert abs(hi-row['ci95_wilson_high'])<5e-8
            y=i+dy
            ax.plot([lo,hi],[y,y],color=color,lw=1.12,alpha=.88,zorder=2)
            ax.plot([lo,lo],[y-.055,y+.055],color=color,lw=.83,zorder=2)
            ax.plot([hi,hi],[y-.055,y+.055],color=color,lw=.83,zorder=2)
            ax.scatter([k/n],[y],s=32,marker=marker,facecolors='white',edgecolors=color,
                       linewidths=1.15,zorder=4)
    ax.set_xlim(-.035,1.02)
    ax.set_ylim(-.48,len(FAMILIES)-.50)
    ax.set_yticks(y0,labels=[label for _,label in FAMILIES])
    ax.invert_yaxis()
    ax.set_xticks([0,.25,.5,.75,1.])
    ax.xaxis.set_minor_locator(MultipleLocator(.125))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'$\Pr(\widehat R_A>1)$')
    clean(ax,'b',x=-.075)


def panel_c(ax, counts, companion):
    # R28 normal F, N=12, pooled across rho/theta/T, 216 records per phi per method.
    phi=counts['phi'].to_numpy(dtype=float)
    assert np.allclose(phi,[0,.6,.85]) and (counts.n==216).all()
    for method,colour,marker,offset in [('M2',ORANGE,'o',-.021),('M3',GREEN,'^',+.021)]:
        ky=counts[f'{method}_RA_gt1'].to_numpy(dtype=int)
        kn=counts[f'{method}_RA_admissible'].to_numpy(dtype=int)
        assert np.all(ky<=kn)
        yy=ky/216
        xi=phi+offset
        # lines are used only along the controlled numeric persistence parameter.
        ax.plot(xi,yy,color=colour,lw=1.18,alpha=.93,zorder=2)
        for x,k,y in zip(xi,ky,yy):
            lo,hi=wilson(int(k),216)
            ax.vlines(x,lo,hi,color=colour,lw=1.08,alpha=.75,zorder=2)
            ax.hlines([lo,hi],x-.012,x+.012,color=colour,lw=.78,zorder=2)
            ax.scatter([x],[y],s=37,marker=marker,facecolors='white',edgecolors=colour,
                       linewidths=1.2,zorder=5)
        if companion:
            kr=counts[f'{method}_Rsigma_gt1'].to_numpy(dtype=int)
            y2=kr/216
            pale=PALE_ORANGE if method=='M2' else PALE_GREEN
            # Companion is descriptive and deliberately carries no extra interval whiskers.
            ax.plot(phi+offset,y2,ls=':',lw=1.,color=pale,alpha=.85,zorder=1)
            ax.scatter(phi+offset,y2,s=24,marker='D',facecolors='white',edgecolors=pale,
                       linewidths=1.05,zorder=3)
    ax.set_xlim(-.085,.93);ax.set_ylim(-.045,.51)
    ax.set_xticks([0,.3,.6,.85])
    ax.xaxis.set_minor_locator(MultipleLocator(.1))
    ax.set_yticks([0,.1,.2,.3,.4,.5])
    ax.yaxis.set_minor_locator(MultipleLocator(.05))
    ax.set_ylabel(r'$\Pr(\widehat R_A>1)$')
    ax.set_xlabel(r'Forcing persistence $\phi$')
    clean(ax,'c')


def panel_d(ax, counts):
    phi=counts['phi'].to_numpy(dtype=float)
    assert np.allclose(phi,[0,.6,.85])
    den=counts['n_valid'].to_numpy(dtype=int)
    ks=counts['reject'].to_numpy(dtype=int)
    assert np.array_equal(den,[96,96,96]) and np.array_equal(ks,[2,95,96])
    yy=ks/den
    ax.axhline(.05,color='.73',lw=.8,ls=':',zorder=0)
    ax.plot(phi,yy,color=BLACK,lw=1.2,zorder=2)
    for x,k,n,y in zip(phi,ks,den,yy):
        lo,hi=wilson(int(k),int(n))
        ax.vlines(x,lo,hi,color=BLACK,lw=1.05,zorder=2)
        ax.hlines([lo,hi],x-.013,x+.013,color=BLACK,lw=.8,zorder=2)
        ax.scatter([x],[y],s=39,marker='D',facecolors='white',edgecolors=BLACK,
                   linewidths=1.25,zorder=4)
    ax.set_xlim(-.085,.93);ax.set_ylim(-.045,1.065)
    ax.set_xticks([0,.3,.6,.85]);ax.xaxis.set_minor_locator(MultipleLocator(.1))
    ax.set_yticks([0,.2,.4,.6,.8,1.]);ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.set_ylabel('Residual-test rejection fraction')
    ax.set_xlabel(r'Forcing persistence $\phi$')
    clean(ax,'d')


def make_figure(companion=False):
    style()
    ref=pd.read_csv(DATA/'T06_large_ensemble_references.csv')
    summary=pd.read_csv(DATA/'T06_typeI_Wilson_summary.csv')
    counts=pd.read_csv(DATA/'R28_normalF_finite_record_counts_transcribed.csv')
    white=pd.read_csv(DATA/'R28_whiteness_T500_counts_transcribed.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.10,5.25),facecolor='white')
    a,b,c,d=axs.flat
    panel_a(a,ref,companion)
    panel_b(b,summary)
    panel_c(c,counts,companion)
    panel_d(d,white)
    legend=[
        Line2D([0],[0],ls=':',color=BLACK,marker='s',mfc=BLACK,ms=5,lw=1,label=r'Physical $F$'),
        Line2D([0],[0],ls='none',color=ORANGE,marker='o',mfc='white',ms=6,mew=1.2,label='M2'),
        Line2D([0],[0],ls='none',color=GREEN,marker='^',mfc='white',ms=6,mew=1.2,label='M3'),
    ]
    if companion:
        legend += [Line2D([0],[0],ls=':',color=PALE,marker='D',mfc='white',ms=5,
                          label=r'Companion $R_\sigma$')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.016),ncol=len(legend),
               frameon=False,fontsize=8.,handletextpad=.42,columnspacing=1.65,handlelength=1.3)
    fig.subplots_adjust(left=.114,right=.986,bottom=.160,top=.934,wspace=.48,hspace=.51)
    stem='RevFig04_RA_Rsigma' if companion else 'RevFig04_RA_only'
    fig.savefig(FIG/(stem+'.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(stem+'.png'),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return stem


if __name__=='__main__':
    for flag in (False,True):
        stem=make_figure(companion=flag)
        print('Created',stem)
