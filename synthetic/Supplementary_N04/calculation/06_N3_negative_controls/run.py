#!/usr/bin/env python3
"""T06 negative controls for NNDR R_A and R_sigma; standalone Spyder script.

No command-line flags.  Fixed random seeds; paired baseline/event innovations;
real M2/M3 extraction; invalid principal-log fits retained.  The screening
rules here are *preliminary descriptive checks* and not the five-test T08
calibrated local-linear adequacy gate.

"Population proxy" means an independent large-ensemble fit to the same
windowed/possibly misspecified observation protocol, NOT a mathematical
population limit.  Exact latent normality is checked separately.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,hashlib,sys,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import block_diag,solve_discrete_lyapunov
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared.nndr_thresholds import reduced_diagnostics,spectral_radius
from shared.nndr_planes import m2_multistart
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'config.json').read_text())
METHODS=('M2','M3')


def orthonormal(n,seed):
    z=np.random.default_rng(seed).normal(size=(n,n));q,r=np.linalg.qr(z)
    return q@np.diag(np.where(np.diag(r)<0,-1.,1.))


def rot_block(r,theta):
    return r*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])


def build_matrices(family,seed,cfg=CFG):
    n=cfg['N'];O=orthonormal(n,seed+41)
    d=np.linspace(0.88,0.40,n)
    normal0=O@np.diag(d)@O.T
    F0=normal0.copy();F1=F0.copy();A=F0.copy();B=np.zeros((n,n))
    beta=np.nan;dt=1.0;W=cfg['T_train']
    if family.startswith('weak_beta'):
        ss=family.split('_');beta=float(ss[1].replace('beta','').replace('p','.'))
        dt=float(ss[2].replace('dt',''));W=int(ss[3].replace('W',''))
        gamma=.045;freq=gamma*beta*np.linspace(0.6,1.4,n//2)
        F0=O@block_diag(*[rot_block(np.exp(-gamma*dt), float(w*dt)) for w in freq])@O.T
        F1=F0.copy()
    if family=='SL_below_Hopf':
        gamma=np.linspace(.06,.14,n//2);omega=np.linspace(.1,.7,n//2)
        F0=O@block_diag(*[rot_block(np.exp(-g),w) for g,w in zip(gamma,omega)])@O.T
        F1=F0.copy()
    if family in ('switch_commuting','switch_noncommuting'):
        if family=='switch_commuting':
            F1=O@np.diag(np.linspace(.70,.45,n))@O.T
        else:
            O1=orthonormal(n,seed+71)
            F1=O1@np.diag(np.linspace(.82,.48,n))@O1.T
    if family.startswith('VAR2'):
        aa=np.linspace(.62,.38,n)
        bb=np.linspace(.14,-.075,n)
        A=O@np.diag(aa)@O.T
        if family=='VAR2_commuting': B=O@np.diag(bb)@O.T
        else:
            O1=orthonormal(n,seed+131)
            B=O1@np.diag(bb)@O1.T
        F0=A;F1=A
        companion=np.block([[A,B],[np.eye(n),np.zeros((n,n))]])
        if spectral_radius(companion)>=1:
            raise AssertionError('Unstable VAR2 augmented operator')
    if family not in ('VAR2_commuting','VAR2_noncommuting'):
        for mat in (F0,F1):
            if np.linalg.norm(mat.T@mat-mat@mat.T,'fro')>1e-11:
                raise AssertionError('Generating one-step map is nonnormal')
            if np.linalg.svd(mat,compute_uv=False)[0]>=1:
                raise AssertionError('Generating one-step map not contractive')
    return {'F0':F0,'F1':F1,'O':O,'A':A,'B':B,'beta':beta,'delta_t':dt,'W':W}


def simulate(family,params,seed,phase,M=None,cfg=CFG):
    """State simulation; same seed yields paired shocks in baseline/event."""
    n=cfg['N'];M=int(cfg['M'] if M is None else M);T=int(params['W']);s=cfg['innovation_sd']
    rng=np.random.default_rng(seed);eta=rng.normal(size=(M,T,n))*s
    aux=rng.normal(size=(M,T,n))*s
    measured_noise=rng.normal(size=(M,T+1,n))*s
    state=np.zeros((M,n));previous=np.zeros_like(state);coloured=np.zeros_like(state)
    record=np.empty((M,T+1,n));record[:,0]=state
    O=params['O'];event=(phase=='event')
    for t in range(T):
        innov=eta[:,t,:].copy()
        if family in ('isotropic_bursts','common_mode_bursts') and event and T//3<=t<2*T//3:
            if family=='isotropic_bursts':innov*=4.0
            else:innov+=5.0*aux[:,t,0,None]*(np.ones(n)/np.sqrt(n))[None,:]
        if family.startswith('coloured'):
            if family=='coloured_isotropic' or not event:
                # Scalar AR innovation with isotropic covariance; commutes with F.
                coloured=.65*coloured+innov
            else:
                # Spatially anisotropic forcing not aligned to the true F eigenvectors.
                P=orthonormal(n,seed+932)
                B=P@np.diag(np.linspace(1.8,.3,n))@P.T
                coloured=.65*coloured+innov@B.T
            innov=coloured.copy()
        if family.startswith('VAR2'):
            nxt=state@params['A'].T+previous@params['B'].T+innov
            previous=state
        elif family=='cubic_saturation':
            y=state@O;ny=y*np.diag(O.T@params['F0']@O)[None,:]
            # cubic coordinate saturation in the normal eigenbasis
            ny-=(1.7 if event else .25)*y**3
            nxt=ny@O.T+innov
        elif family=='SL_below_Hopf':
            y=state@O;yn=y@ (O.T@params['F0']@O).T
            pairs=y.reshape(M,n//2,2)
            amp2=np.sum(pairs*pairs,axis=-1,keepdims=True)
            yn-=(1.1 if event else .3)*(amp2*pairs).reshape(M,n)
            nxt=yn@O.T+innov
        else:
            if family.startswith('switch') and event and t>=T//2:
                F=params['F1']
            else:F=params['F0']
            nxt=state@F.T+innov
        if not np.all(np.isfinite(nxt)) or np.max(abs(nxt))>1e6:
            raise ValueError('Unstable numerical realization '+family)
        state=nxt;record[:,t+1]=state
    if family=='observation_noise' and event:
        P=orthonormal(n,seed+513)
        B=P@np.diag(np.linspace(2.3,.15,n))@P.T
        record=record + (measured_noise@B.T)*1.5
    return record


def fit_map(data,cfg=CFG):
    xx=data[:,:-1,:].reshape(-1,data.shape[-1]);yy=data[:,1:,:].reshape(-1,data.shape[-1]);n=xx.shape[1]
    ss=xx.T@xx
    ridge=cfg['ridge_relative']*np.trace(ss)/n
    mat=np.linalg.solve(ss+ridge*np.eye(n),xx.T@yy).T
    resid=yy-xx@mat.T
    return mat,resid,ridge


def plane_m3(F):
    cc=F@F.T-F.T@F;w,v=np.linalg.eigh(cc)
    if np.max(abs(w))<1e-12:return None
    return v[:,[-1,0]]


def preliminary_adequacy(data,F,cfg=CFG):
    """Diagnostic-only prespecified gates; not the later calibrated T08 C5."""
    rr=data[:,1:,:]-data[:,:-1,:]@F.T
    r0=rr[:,:-1,:];r1=rr[:,1:,:]
    den=np.sqrt(np.sum(r0*r0,axis=(0,1))*np.sum(r1*r1,axis=(0,1)))
    cor=np.divide(np.sum(r0*r1,axis=(0,1)),den,out=np.zeros(rr.shape[-1]),where=den>1e-12)
    nlag=rr.shape[0]*(rr.shape[1]-1)
    lag=float(np.max(abs(cor)))
    gate_corr=lag > cfg['screen_lag_corr_multiplier']/np.sqrt(nlag)
    # Three equal temporal blocks detect a burst confined to the middle third.
    temporal_blocks=np.array_split(rr,3,axis=1)
    vblocks=[float(np.mean(q**2)) for q in temporal_blocks]
    vr=max(vblocks)/max(min(vblocks),1e-16)
    gate_var=vr>cfg['screen_variance_ratio_cutoff']
    return {'screen_lag_max_abs':lag,'screen_variance_ratio':vr,
            'screen_lag_reject':bool(gate_corr),'screen_variance_reject':bool(gate_var),
            'screen_pass':bool(not (gate_corr or gate_var))}


def diagnostics_on(F,method,seed,delta_t=1.0,cfg=CFG):
    if method=='M2':
        plane=m2_multistart(F,seed=seed,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])['Q']
    elif method=='M3':
        plane=plane_m3(F)
    else:raise ValueError(method)
    if plane is None:return {'status':'degenerate_plane','R_variance':np.nan,'R_sigma':np.nan,'sigma_max':np.nan}
    return reduced_diagnostics(plane.T@F@plane,delta_t=delta_t)


def wilson_interval(k,n,z=1.959963984540054):
    if n<=0:return (np.nan,np.nan)
    p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    h=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return max(0.,center-h),min(1.,center+h)


def permutation_paired_increase(diffs,seed,B):
    """One-sided paired sign-flip p-value for positive mean event-minus-baseline."""
    v=np.asarray(diffs,float);v=v[np.isfinite(v)]
    if len(v)<8:return np.nan
    obs=float(np.mean(v));rng=np.random.default_rng(seed)
    signs=rng.integers(0,2,size=(B,len(v)),dtype=np.int8)*2-1
    null=(signs@v)/len(v)
    return float((1+np.sum(null>=obs-1e-12))/(B+1))


def bh_adjust(pvalues):
    ar=np.asarray(pvalues,float);res=np.full(ar.shape,np.nan)
    ok=np.where(np.isfinite(ar))[0]
    if not len(ok):return res
    rank=ok[np.argsort(ar[ok])];x=ar[rank];q=np.minimum.accumulate((x*len(x)/np.arange(1,len(x)+1))[::-1])[::-1]
    res[rank]=np.clip(q,0,1);return res


def exact_population_map(family,params,phase,sim_seed,cfg=CFG):
    """Unregularized stationary least-squares population VAR1 projection.

    For switching, cubic and Stuart-Landau only a finite-window ensemble proxy
    is available. For nonstationary burst covariance, the conditional drift F
    remains the exact population least-squares minimizer (zero-mean forcing
    independent of the current state). The simulation starts at zero, so the
    large-ensemble finite-window proxy can differ from the stationary value.
    """
    F=params['F0'];n=len(F);s=cfg['innovation_sd']
    if family in ('gaussian_VAR1','isotropic_bursts','common_mode_bursts'):
        return F
    if family=='observation_noise':
        C=solve_discrete_lyapunov(F,s*s*np.eye(n))
        if phase=='baseline':return F
        P=orthonormal(n,sim_seed+513)
        B=P@np.diag(np.linspace(2.3,.15,n))@P.T
        Cobs=(1.5*s)**2*(B@B.T)
        return F@C@np.linalg.inv(C+Cobs)
    if family.startswith('coloured'):
        Phi=.65*np.eye(n)
        B=np.eye(n)
        if family=='coloured_spatial' and phase=='event':
            P=orthonormal(n,sim_seed+932)
            B=P@np.diag(np.linspace(1.8,.3,n))@P.T
        A_aug=np.block([[F,Phi],[np.zeros((n,n)),Phi]])
        U=np.vstack([s*B,s*B]);C=solve_discrete_lyapunov(A_aug,U@U.T)
        Cxx=C[:n,:n];Cxi_x=C[n:,:n]
        return (F@Cxx+Phi@Cxi_x)@np.linalg.inv(Cxx)
    if family.startswith('VAR2'):
        A=params['A'];B=params['B']
        aug=np.block([[A,B],[np.eye(n),np.zeros((n,n))]])
        Q=np.zeros((2*n,2*n));Q[:n,:n]=s*s*np.eye(n)
        C=solve_discrete_lyapunov(aug,Q)
        Cxx=C[:n,:n];Cprev_x=C[n:,:n]
        return (A@Cxx+B@Cprev_x)@np.linalg.inv(Cxx)
    if family.startswith('weak_beta'):
        return F
    return None


def collect(cfg=CFG):
    rows=[];pairrows=[];poprows=[];start=time.monotonic()
    for j,fam in enumerate(cfg['families']):
        for k in range(cfg['replicates']):
            subject=f'{fam}__{k:03d}';seed=cfg['seed_base']+j*100000+k*149
            p=build_matrices(fam,seed,cfg)
            subject_rows=[]
            for phase in ('baseline','event'):
                obs=simulate(fam,p,seed+191,phase,cfg=cfg)
                Fhat,resid,ridge=fit_map(obs,cfg)
                scr=preliminary_adequacy(obs,Fhat,cfg)
                for mi,met in enumerate(METHODS):
                    d=diagnostics_on(Fhat,met,seed+234+mi*73,p['delta_t'],cfg)
                    row={'family':fam,'seed':seed,'subject':subject,'phase':phase,'method':met,
                         'beta':p['beta'],'delta_t':p['delta_t'],'window_length':p['W'],
                         'fit_rho_full':spectral_radius(Fhat),'fit_sigma_full':float(np.linalg.svd(Fhat,compute_uv=False)[0]),
                         'true_rho_full_early':spectral_radius(p['F0']),
                         'true_sigma_full_early':float(np.linalg.svd(p['F0'],compute_uv=False)[0]),
                         'true_normal_commutator_early':float(np.linalg.norm(p['F0'].T@p['F0']-p['F0']@p['F0'].T,'fro')),
                         'true_normal_commutator_late':float(np.linalg.norm(p['F1'].T@p['F1']-p['F1']@p['F1'].T,'fro')),
                         'true_sigma_full_late':float(np.linalg.svd(p['F1'],compute_uv=False)[0]),
                         'fit_relative_regularizer':cfg['ridge_relative'],
                         'fit_absolute_regularizer':ridge,
                         'fit_status':d['status'],
                         'R_A':float(d.get('R_variance',np.nan)),
                         'R_sigma':float(d.get('R_sigma',np.nan)),
                         'K':float(d.get('K',np.nan)),
                         'Kc':float(d.get('Kc_variance',np.nan)),
                         'rho_reduced':float(d.get('rho',np.nan)),
                         'sigma_reduced':float(d.get('sigma_max',np.nan)),
                         'fit_aliased_near_nyquist':bool(d.get('full_rotation_warning',False)),
                         'RA_exceeds_1':bool(np.isfinite(d.get('R_variance',np.nan)) and d['R_variance']>1),
                         'Rsigma_exceeds_1':bool(np.isfinite(d.get('R_sigma',np.nan)) and d['R_sigma']>1),
                         **scr}
                    row['screened_RA_exceeds_1']=bool(row['screen_pass'] and row['RA_exceeds_1'])
                    row['screened_Rsigma_exceeds_1']=bool(row['screen_pass'] and row['Rsigma_exceeds_1'])
                    rows.append(row);subject_rows.append(row)
            for met in METHODS:
                b=next(v for v in subject_rows if v['method']==met and v['phase']=='baseline')
                e=next(v for v in subject_rows if v['method']==met and v['phase']=='event')
                pairrows.append({'family':fam,'subject':subject,'seed':seed,'method':met,
                                 'both_RA_admissible':bool(np.isfinite(b['R_A']) and np.isfinite(e['R_A'])),
                                 'both_Rsigma_admissible':bool(np.isfinite(b['R_sigma']) and np.isfinite(e['R_sigma'])),
                                 'RA_event_minus_baseline':e['R_A']-b['R_A'] if np.isfinite(b['R_A']) and np.isfinite(e['R_A']) else np.nan,
                                 'Rsigma_event_minus_baseline':e['R_sigma']-b['R_sigma'] if np.isfinite(b['R_sigma']) and np.isfinite(e['R_sigma']) else np.nan,
                                 'both_screen_pass':bool(b['screen_pass'] and e['screen_pass']),
                                 'RA_screened_contrast':e['R_A']-b['R_A'] if np.isfinite(b['R_A']) and np.isfinite(e['R_A']) and b['screen_pass'] and e['screen_pass'] else np.nan})
        # One large-sample population proxy per family with fixed reference seed.
        seed=cfg['population_reference_seed']+j*17003
        params=build_matrices(fam,seed,cfg)
        for phase in ('baseline','event'):
            sample=simulate(fam,params,seed+191,phase,M=cfg['population_reference_M'],cfg=cfg)
            Fpop,_,ridge=fit_map(sample,cfg)
            raw=exact_population_map(fam,params,phase,seed+191,cfg)
            for mi,method in enumerate(METHODS):
                dp=diagnostics_on(Fpop,method,seed+233+mi*23,params['delta_t'],cfg)
                dr=diagnostics_on(params['F0'],method,seed+233+mi*23,params['delta_t'],cfg)
                de=diagnostics_on(raw,method,seed+233+mi*23,params['delta_t'],cfg) if raw is not None else {'status':'not_available'}
                poprows.append({'family':fam,'phase':phase,'method':method,
                                'large_ensemble_M':cfg['population_reference_M'],
                                'large_ensemble_T':params['W'],
                                'reference_type':'large_sample_windowed_projection_not_population_limit',
                                'RA_proxy':float(dp.get('R_variance',np.nan)),
                                'Rsigma_proxy':float(dp.get('R_sigma',np.nan)),
                                'status_proxy':dp['status'],
                                'R_A_true_normal_operator':float(dr.get('R_variance',np.nan)),
                                'R_sigma_true_normal_operator':float(dr.get('R_sigma',np.nan)),
                                'R_A_exact_effective_VAR1':float(de.get('R_variance',np.nan)),
                                'R_sigma_exact_effective_VAR1':float(de.get('R_sigma',np.nan)),
                                'status_exact_effective_VAR1':de['status'],
                                'commutator_F_exact_effective_VAR1':float(np.linalg.norm(raw@raw.T-raw.T@raw,'fro')) if raw is not None else np.nan,
                                'commutator_Fproxy':float(np.linalg.norm(Fpop@Fpop.T-Fpop.T@Fpop,'fro')),
                                'Fproxy_spectral_radius':spectral_radius(Fpop),
                                'Fproxy_sigma':float(np.linalg.svd(Fpop,compute_uv=False)[0]),
                                'Fproxy_vs_exact_VAR1_fro':float(np.linalg.norm(Fpop-raw,'fro')) if raw is not None else np.nan,
                                'fully_analytic_reference_available':raw is not None})
        print(f'{j+1:02d}/{len(cfg["families"])} {fam} done',flush=True)
    return pd.DataFrame(rows),pd.DataFrame(pairrows),pd.DataFrame(poprows),time.monotonic()-start


def summarize(rows,pairs,pop,cfg=CFG):
    result=[];cohort=[]
    for (family,method),gg in rows.groupby(['family','method'],sort=False):
        ev=gg[gg.phase=='event'];bv=gg[gg.phase=='baseline'];n=len(ev)
        pair=pairs[(pairs.family==family)&(pairs.method==method)]
        for phase,phase_rows in [('baseline',bv),('event',ev)]:
            nr=int(np.isfinite(phase_rows.R_A).sum());ns=int(np.isfinite(phase_rows.R_sigma).sum())
            for criterion,col in [('R_A','RA_exceeds_1'),('R_sigma','Rsigma_exceeds_1'),('R_A_post_prelim_screen','screened_RA_exceeds_1'),('R_sigma_post_prelim_screen','screened_Rsigma_exceeds_1')]:
                if 'post_' in criterion:
                    total=int(phase_rows.screen_pass.sum())
                    count=int(phase_rows[col].sum())
                    censored=int(n-total)
                else:total=n;count=int(phase_rows[col].sum());censored=0
                lo,hi=wilson_interval(count,total)
                result.append({'family':family,'method':method,'phase':phase,'criterion':criterion,
                               'subjects_total':n,'subjects_pass_screen':int(phase_rows.screen_pass.sum()),
                               'denominator':total,'inadmissible_ratio':(n-nr if criterion.startswith('R_A') else n-ns),
                               'count_gt_1':count,'fraction_gt_1':count/total if total else np.nan,
                               'ci95_wilson_low':lo,'ci95_wilson_high':hi,
                               'censored_by_screen':censored,
                               'nominal_screen_reject_rate':1-float(phase_rows.screen_pass.mean())})
        for criterion,col in [('R_A','RA_event_minus_baseline'),('R_sigma','Rsigma_event_minus_baseline'),('R_A_post_prelim_screen','RA_screened_contrast')]:
            x=pair[col].dropna().to_numpy();pval=permutation_paired_increase(x,cfg['perm_seed']+len(cohort)*193,cfg['permutation_draws'])
            cohort.append({'family':family,'method':method,'criterion':criterion,'subjects_total':len(pair),
                           'complete_pairs':len(x),'mean_event_minus_baseline':float(np.mean(x)) if len(x) else np.nan,
                           'median_event_minus_baseline':float(np.median(x)) if len(x) else np.nan,
                           'p_one_sided_paired_signflip':pval,
                           'testable':bool(len(x)>=8)})
    cohort=pd.DataFrame(cohort)
    for criterion,idx in cohort.groupby('criterion').groups.items():
        cohort.loc[idx,'p_BH_within_criterion']=bh_adjust(cohort.loc[idx,'p_one_sided_paired_signflip'])
    return pd.DataFrame(result),cohort


def figures(rows,pairs,pop,summary,cohort,cfg=CFG):
    families=cfg['families'];y=np.arange(len(families))
    colours={'M2':'#2463ba','M3':'#db6f27'}
    fig,axes=plt.subplots(1,4,figsize=(17.4,10.1),sharey=True,layout='constrained')
    specs=[('R_A','(a) Delayed-variance false positives','$R_A>1$: proportion'),
           ('R_sigma','(b) One-step false positives','$R_\\sigma>1$: proportion'),
           ('contrast','(c) Paired event contrasts','Median $R_A$ event − baseline'),
           ('R_A_post_prelim_screen','(d) Preliminary screen','Fraction $R_A>1$ among retained')]
    for ax,(criterion,title,xlabel) in zip(axes,specs):
        for met in METHODS:
            offset=-.15 if met=='M2' else .15
            vals=[];el=[];eh=[]
            for family in families:
                if criterion=='contrast':
                    cc=cohort[(cohort.family==family)&(cohort.method==met)&(cohort.criterion=='R_A')].iloc[0]
                    vals.append(float(cc['median_event_minus_baseline']));el.append(0);eh.append(0)
                else:
                    ss=summary[(summary.family==family)&(summary.method==met)&(summary.phase=='event')&(summary.criterion==criterion)].iloc[0]
                    vals.append(float(ss['fraction_gt_1']));el.append(max(0.,float(ss.fraction_gt_1-ss.ci95_wilson_low)));eh.append(max(0.,float(ss.ci95_wilson_high-ss.fraction_gt_1)))
            vals=np.asarray(vals,float)
            if criterion=='contrast':
                ax.plot(vals,y+offset,'o',ms=4,color=colours[met],label=met)
            else:
                # NaNs from complete screening are marked separately.
                ok=np.isfinite(vals)
                ax.errorbar(vals[ok],(y+offset)[ok],xerr=[np.array(el)[ok],np.array(eh)[ok]],fmt='o',
                            ms=4.4,capsize=2.1,elinewidth=.85,color=colours[met],label=met)
        ax.set_title(title,loc='left',fontsize=10.5,fontweight='bold',pad=11)
        ax.set_xlabel(xlabel,fontsize=9)
        ax.set_yticks(y);ax.set_ylim(len(families)-.35,-.65)
        ax.grid(axis='x',alpha=.16);ax.tick_params(axis='y',length=0)
        if criterion=='contrast':
            ax.axvline(0,ls='--',lw=.85,color='grey');ax.set_xscale('symlog',linthresh=.02)
        else:ax.set_xlim(-.07,1.05)
    axes[0].set_yticklabels([f.replace('weak_','osc. ').replace('_',' ') for f in families],fontsize=8.8)
    for ax in axes[1:]:ax.tick_params(axis='y',labelleft=False)
    axes[0].legend(loc='lower right',ncol=2,fontsize=9,frameon=False)
    fig.suptitle('Fig. N3 — Negative controls under normal generators and VAR(1) misspecification',
                 fontsize=13.5,fontweight='bold')
    fig.savefig(OUT/'T06_FigN3_negative_controls.png',dpi=230,bbox_inches='tight')
    fig.savefig(OUT/'T06_FigN3_negative_controls.pdf',bbox_inches='tight');plt.close(fig)
    # Oscillatory sampling sensitivity: explicit distinctions by Δt and window.
    sel=summary[(summary.phase=='event')&(summary.criterion=='R_A')&(summary.family.str.startswith('weak_beta'))]
    fig,ax=plt.subplots(figsize=(8.6,4.7),layout='constrained')
    for met in METHODS:
        gg=sel[sel.method==met].copy()
        gg['beta']=gg.family.str.extract(r'beta(\d+(?:p\d+)?)')[0].str.replace('p','.').astype(float)
        gg['dt']=gg.family.str.extract(r'_dt(\d+)')[0].astype(int)
        gg['W']=gg.family.str.extract(r'_W(\d+)')[0].astype(int)
        for (dt,W),b in gg.groupby(['dt','W']):
            b=b.sort_values('beta');ax.plot(b.beta,b.fraction_gt_1,'o-',ms=5,lw=1.3,
                label=f'{met}, $\\Delta t$={dt}, W={W}')
    ax.set_xlabel(r'Nominal planted $\beta=\omega/\gamma$ (central frequency scale)')
    ax.set_ylabel(r'Event false-positive fraction ($R_A>1$)')
    ax.set_ylim(-.03,1.03);ax.set_xscale('log');ax.legend(fontsize=8,ncol=2)
    ax.grid(alpha=.15);ax.set_title('Multidimensional damped normal oscillators')
    fig.savefig(OUT/'T06_oscillator_sampling_sensitivity.png',dpi=230)
    fig.savefig(OUT/'T06_oscillator_sampling_sensitivity.pdf');plt.close(fig)


def main():
    t=time.monotonic();rows,pairs,pop,elapsed=collect(CFG)
    summary,cohort=summarize(rows,pairs,pop,CFG)
    for name,df in [('T06_subject_method_results',rows),('T06_paired_event_contrasts',pairs),
                    ('T06_large_ensemble_references',pop),('T06_typeI_Wilson_summary',summary),
                    ('T06_cohort_permutation_tests',cohort)]:
        df.to_csv(OUT/(name+'.csv'),index=False)
    figures(rows,pairs,pop,summary,cohort,CFG)
    z=summary[(summary.phase=='event')&(summary.criterion=='R_A')]
    largest=z.sort_values('fraction_gt_1',ascending=False).head(10)
    basic=cohort[cohort.criterion=='R_A']
    manifest={'module':'T06','seed_base':CFG['seed_base'],'parameters':CFG,'runtime_seconds':time.monotonic()-t,
             'n_subjects':int(len(CFG['families'])*CFG['replicates']),
             'n_phase_method_rows':int(len(rows)),
             'n_invalid_RA':int(rows.R_A.isna().sum()),'n_invalid_Rsigma':int(rows.R_sigma.isna().sum()),
             'n_event_RA_false_positives':int(rows[(rows.phase=='event')].RA_exceeds_1.sum()),
             'n_event_Rsigma_false_positives':int(rows[(rows.phase=='event')].Rsigma_exceeds_1.sum()),
             'n_bh_significant_RA_contrasts':int((basic.p_BH_within_criterion<.05).sum()),
             'largest_event_rates':largest[['family','method','fraction_gt_1']].to_dict('records'),
             'population_reference_scope':'One independently simulated 250-trajectory finite-window ensemble per family PLUS independent exact stationary effective VAR(1) for Gaussian VAR1, bursts, coloured forcing, observation noise, VAR2 and oscillatory controls. Finite-window switch/nonlinear case remains proxy only.',
             'screens':'Only two descriptive residual screens. T08 full 5-test calibration NOT done.',
             'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    (OUT/'T06_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'n_subjects':manifest['n_subjects'],'n_rows':len(rows),'invalid_RA':manifest['n_invalid_RA'],
                      'invalid_Rsigma':manifest['n_invalid_Rsigma'],'event_false_RA':manifest['n_event_RA_false_positives'],
                      'RA_BH_contrasts':manifest['n_bh_significant_RA_contrasts'],'runtime_seconds':round(manifest['runtime_seconds'],1)},indent=2))
    return rows,pairs,pop,summary,cohort

if __name__=='__main__':main()
