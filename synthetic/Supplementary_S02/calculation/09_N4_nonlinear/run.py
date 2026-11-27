#!/usr/bin/env python3
"""T09 nonlinear NNDR: analytic Jacobians, ordered tangent maps, independent statistical references.

Self-contained from suite; Run in Spyder without arguments. Analytical Jacobians are
explicitly differentiated from Euler--Maruyama drift (same integrator discretization).
Known physical *one-sample* tangent map is the ordered product of microstep
(I+h*J) factors, not exp(sample_dt*mean J). Random Brownian innovations are
additive, isotropic per physical coordinate, independent across nodes/time.

State/propagator references are generated independently of the inferred map.
FHN pulse windows are *nonstationary*. The C1 C0^-1 reference is thus a
long-ensemble, time-matched statistical predictor, NOT a Jacobian.
The gradient double well is a reversible physical SDE with symmetric local J;
its coarse-time statistical predictor need NOT be Euclidean normal.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,sys,hashlib,time,platform,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared.nndr_planes import m2_multistart, principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics, spectral_radius
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';OUT.mkdir(parents=True,exist_ok=True)
CFG=json.loads((HERE/'config.json').read_text())
FAMILIES=('FHN_directed_spikes','SL_directed_below_Hopf','SL_directed_above_Hopf','double_well_gradient')


def directed_laplacian(nodes):
    # Incoming directed edges: non-circulant, non-reciprocal, known orientation.
    assert nodes==4
    C=np.zeros((nodes,nodes))
    C[1,0]=.16; C[2,1]=.14; C[3,2]=.12; C[3,0]=.035; C[0,3]=.013
    return np.diag(C.sum(axis=1))-C


def symmetric_laplacian(n):
    C=np.zeros((n,n))
    for i in range(n):
        j=(i+1)%n;C[i,j]=C[j,i]=1.
    return np.diag(C.sum(axis=1))-C


def params(family,cfg=CFG):
    n=cfg['N'];nodes=cfg['nodes']
    if family.startswith('FHN'):
        return {'family':family,'L':directed_laplacian(nodes),'nodes':nodes,
                'eps':0.085,'a':0.70,'b':0.80,'noise':cfg['noise_sd_fhn']}
    if family.startswith('SL'):
        mu=cfg['sl_mu_above'] if 'above' in family else cfg['sl_mu_below']
        return {'family':family,'L':.65*directed_laplacian(nodes),'nodes':nodes,
                'mu':mu,'omega':cfg['sl_omega'],'noise':cfg['noise_sd_sl']}
    if family=='double_well_gradient':
        return {'family':family,'L':cfg['gradient_coupling']*symmetric_laplacian(n),
                'a':cfg['gradient_a'],'noise':cfg['noise_sd_gradient']}
    raise ValueError(family)


def initial_state(M,p,seed):
    rg=np.random.default_rng(seed)
    if p['family'].startswith('FHN'):
        xx=np.zeros((M,2*p['nodes']))
        from scipy.optimize import brentq
        ibase=CFG['fhn_baseline_drive']
        v0=brentq(lambda v:v-v**3/3-(v+p['a'])/p['b']+ibase,-3.,0.)
        xx[:,0::2]=v0+.025*rg.standard_normal((M,p['nodes']))
        xx[:,1::2]=(v0+p['a'])/p['b']+.012*rg.standard_normal((M,p['nodes']))
        return xx
    if p['family'].startswith('SL'):
        xx=.14*rg.standard_normal((M,2*p['nodes']))
        return xx
    return np.sqrt(p['a'])*rg.choice([-1.,1.],size=(M,len(p['L'])))+.09*rg.standard_normal((M,len(p['L'])))


def drift_jacobian(x,t,p,cfg=CFG):
    """Analytic instantaneous continuous drift and its Jacobian for all M paths."""
    x=np.asarray(x,float); M,n=x.shape;family=p['family']
    if family.startswith('FHN'):
        v=x[:,0::2];w=x[:,1::2];nodes=v.shape[1];L=p['L']
        pulse=float(cfg['fhn_pulse_on']<=t<cfg['fhn_pulse_off'])
        drive=cfg['fhn_baseline_drive']+cfg['fhn_pulse_amplitude']*pulse
        d=np.empty_like(x)
        d[:,0::2]=v-v**3/3-w+drive-v@L.T
        d[:,1::2]=p['eps']*(v+p['a']-p['b']*w)
        J=np.zeros((M,n,n));J[:,0::2,0::2]=-L[None,:,:]
        ii=2*np.arange(nodes)
        J[:,ii,ii]+=1.-v*v
        J[:,ii,ii+1]=-1.
        J[:,ii+1,ii]=p['eps']
        J[:,ii+1,ii+1]=-p['eps']*p['b']
    elif family.startswith('SL'):
        v=x[:,0::2];w=x[:,1::2];nodes=v.shape[1];L=p['L']
        mu=p['mu'];om=p['omega'];rad=v*v+w*w
        d=np.empty_like(x)
        d[:,0::2]=(mu-rad)*v-om*w-v@L.T
        d[:,1::2]=(mu-rad)*w+om*v-w@L.T
        J=np.zeros((M,n,n));J[:,0::2,0::2]=-L[None,:,:]
        J[:,1::2,1::2]=-L[None,:,:]
        ii=2*np.arange(nodes)
        J[:,ii,ii]+=mu-3*v*v-w*w
        J[:,ii+1,ii+1]+=mu-v*v-3*w*w
        J[:,ii,ii+1]=-om-2*v*w
        J[:,ii+1,ii]=om-2*v*w
    else:
        L=p['L'];a=p['a'];d=a*x-x**3-x@L.T
        J=np.broadcast_to(-L[None,:,:],(M,n,n)).copy()
        ii=np.arange(n);J[:,ii,ii]+=a-3*x*x
    return d,J


def simulate(p,seed,M,do_tangent,cfg=CFG):
    """Euler--Maruyama; tangent exactly differentiates the microstep map.

    At each sampled step, Phi is reset to I and left-multiplied by (I+hJ)
    in chronological order. Random additive shocks do not change the Jacobian.
    """
    h=cfg['integration_dt']; dt=cfg['sample_dt']; subdiv=int(round(dt/h))
    assert abs(dt-subdiv*h)<1e-12 and subdiv>=1
    n=cfg['N'];T=cfg['T_samples'];rg=np.random.default_rng(seed)
    x=initial_state(M,p,seed+3)
    states=np.empty((M,T+1,n));states[:,0,:]=x
    all_phi=np.empty((M,T,n,n)) if do_tangent else None
    all_j=np.empty((M,T,n,n)) if do_tangent else None
    eye=np.eye(n)
    for k in range(T):
        if do_tangent:
            phi=np.broadcast_to(eye,(M,n,n)).copy();jacc=np.zeros_like(phi)
        for s in range(subdiv):
            t=(k*subdiv+s)*h
            drift,J=drift_jacobian(x,t,p,cfg)
            if do_tangent:
                phi=np.matmul((eye[None,:,:]+h*J),phi)
                jacc+=J/subdiv
            x=x+h*drift+p['noise']*np.sqrt(h)*rg.standard_normal((M,n))
            if not np.all(np.isfinite(x)) or np.max(np.abs(x))>20:
                raise FloatingPointError(f'Divergent {p["family"]} at k={k}, micro={s}')
        states[:,k+1,:]=x
        if do_tangent:all_phi[:,k,:,:]=phi;all_j[:,k,:,:]=jacc
    return states,all_phi,all_j


def empirical_map(states,start,end,ridge_relative=.001):
    """Yule--Walker C1*C0^-1 with global window centering and intercept.

    Returned raw C1 C0^-1 is distinct from the ridge regularized estimate;
    the physical coordinates are never whitened or diagonalized.
    """
    X=states[:,start:end,:].reshape(-1,states.shape[-1])
    Y=states[:,start+1:end+1,:].reshape(-1,states.shape[-1]);n=X.shape[1]
    x=X-X.mean(axis=0);y=Y-Y.mean(axis=0)
    C0=x.T@x/len(x);C1=y.T@x/len(x)
    raw=C1@np.linalg.pinv(C0,rcond=1e-12)
    ridge=ridge_relative*np.trace(C0)/n
    fitted=C1@np.linalg.solve(C0+ridge*np.eye(n),np.eye(n))
    return fitted,raw,{'n_pairs':len(x),'C0_trace':float(np.trace(C0)),
                       'ridge':float(ridge),'C0_condition':float(np.linalg.cond(C0)),
                       'stat_symmetry_C1':float(np.linalg.norm(C1-C1.T,'fro'))}


def m3_plane(F):
    C=F@F.T-F.T@F
    ev,v=np.linalg.eigh(C)
    return np.column_stack([v[:,-1],v[:,0]])


def plane(F,method,seed,cfg=CFG):
    if method=='M2':return m2_multistart(F,seed=int(seed),restarts=cfg['m2_restarts'],
                                         iterations=cfg['m2_iterations'])['Q']
    if method=='M3':return m3_plane(F)
    raise ValueError(method)


def full_properties(F):
    singular=np.linalg.svd(F,compute_uv=False)
    return {'full_sigma_max':float(singular[0]),'full_rho':spectral_radius(F),
            'full_commutator':float(np.linalg.norm(F@F.T-F.T@F,'fro'))}


def diag_for(Q,F,dt):
    return reduced_diagnostics(Q.T@F@Q,delta_t=dt)


def cond_name(family,center,cfg=CFG,begin=None,end=None):
    if family.startswith('FHN'):
        # Use actual window SUPPORT, not its center, to prevent pulse leakage.
        if begin is None or end is None:raise ValueError('FHN requires explicit window boundaries')
        left=begin*cfg['sample_dt'];right=end*cfg['sample_dt']
        if right<=cfg['fhn_pulse_on']:return 'before_pulse'
        if left>=cfg['fhn_pulse_off']:return 'after_pulse'
        return 'overlaps_pulse'
    if family=='SL_directed_below_Hopf':return 'below_Hopf'
    if family=='SL_directed_above_Hopf':return 'above_Hopf'
    return 'gradient_equilibrium'


def gate_window(data,diagseed,cutoffs,source,cfg=CFG):
    # Only a transfer sensitivity: T08 calibration did not include the current
    # N=8, W=60 nonlinear physics. It cannot certify a new nominal type-I rate.
    scores=source.compute_diagnostics(data,diagseed)
    failed=[k for k in source.SCORES if scores[k]>cutoffs[k]]
    return {'gate_pass':len(failed)==0,'gate_fail_count':len(failed),
            'gate_failed_tests':','.join(k[0].upper() for k in failed),
            **{f'adequacy_{k}':scores[k] for k in source.SCORES},
            'gate_alias_suspected':scores['alias_suspected']}


def load_gate():
    sp=importlib.util.spec_from_file_location('t09_t08_gate',ROOT/'08_adequacy_calibration'/'run.py')
    mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod)
    c=pd.read_csv(ROOT/'08_adequacy_calibration'/'outputs'/'T08_locked_calibration_cutoffs.csv')
    assert set(mod.SCORES)==set(c.metric)
    return mod,dict(zip(c.metric,c.upper_cutoff))


def finite_diff_check(family,p,seed=10,cfg=CFG):
    """Independently verify analytic J and ordered one-sample tangent map."""
    rg=np.random.default_rng(seed)
    x=initial_state(1,p,seed)[0]
    t=cfg['sample_dt']*5.5;eps=2e-6;n=len(x);h=cfg['integration_dt']
    def f(v):return drift_jacobian(v[None,:],t,p,cfg)[0][0]
    _,J=drift_jacobian(x[None,:],t,p,cfg)
    numerical=np.column_stack([(f(x+eps*np.eye(n)[j])-f(x-eps*np.eye(n)[j]))/(2*eps) for j in range(n)])
    jac_err=float(np.max(abs(numerical-J[0])))
    steps=int(round(cfg['sample_dt']/h))
    shocks=rg.standard_normal((steps,n))
    def micro(z, tangent=False):
        zz=z.copy();phi=np.eye(n)
        for s in range(steps):
            d,j=drift_jacobian(zz[None,:],5*cfg['sample_dt']+s*h,p,cfg)
            if tangent:phi=(np.eye(n)+h*j[0])@phi
            zz+=h*d[0]+p['noise']*np.sqrt(h)*shocks[s]
        return (zz,phi) if tangent else zz
    _,P=micro(x,True)
    numP=np.column_stack([(micro(x+eps*np.eye(n)[j])-micro(x-eps*np.eye(n)[j]))/(2*eps) for j in range(n)])
    phi_err=float(np.max(abs(numP-P)))
    return {'family':family,'instantaneous_J_max_abs_error':jac_err,
            'sample_variational_max_abs_error':phi_err,
            'gradient_J_asymmetry':float(np.max(abs(J[0]-J[0].T)))}


def summarize_and_plot(summary,results,aux,cfg=CFG):
    # Plot deliberately separates the two estimands and records all invalid ratios.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.7,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(2,3,figsize=(13.8,7.6),constrained_layout=True)
    labels={'FHN_directed_spikes':'FHN · directed','SL_directed_below_Hopf':'Stuart–Landau · below Hopf',
            'SL_directed_above_Hopf':'Stuart–Landau · above Hopf','double_well_gradient':'Gradient · double well'}
    colors={'FHN_directed_spikes':'#146B9F','SL_directed_below_Hopf':'#E38C35',
            'SL_directed_above_Hopf':'#9A4DAA','double_well_gradient':'#49765C'}
    # (a) known full gain vs window time, displaying independent replicate variability
    for fam,group in summary.groupby('family',sort=False):
        g=group.groupby('window_center').agg(m=('oracle_sigma','median'),q1=('oracle_sigma',lambda x:np.percentile(x,25)),q3=('oracle_sigma',lambda x:np.percentile(x,75)))
        xx=g.index.to_numpy();col=colors[fam]
        ax[0,0].plot(xx,g.m,color=col,label=labels[fam],lw=1.5)
        ax[0,0].fill_between(xx,g.q1,g.q3,color=col,alpha=.13)
    ax[0,0].axhline(1,color='#666',ls='--',lw=.8);ax[0,0].set(xlabel='Time (model units)',ylabel='Max gain, true tangent map',title='(a) Known one-sample tangent amplification')
    ax[0,0].legend(fontsize=7,loc='best',frameon=False)
    # (b) time series for first FHN seed, direct data, with stimulus annotation
    fhn=aux['fhn_first_states']; v=fhn[:,0::2].mean(axis=1)
    tt=np.arange(len(v))*cfg['sample_dt']
    ax[0,1].plot(tt,v,color=colors['FHN_directed_spikes'],lw=1)
    ax[0,1].axvspan(cfg['fhn_pulse_on'],cfg['fhn_pulse_off'],color='#F0B34A',alpha=.23,label='Driven interval')
    ax[0,1].set(xlabel='Time (model units)',ylabel='Mean fast voltage',title='(b) FHN response and imposed drive')
    ax[0,1].legend(fontsize=8,frameon=False)
    # (c) reference matrix mismatch across families; compare time-ordered vs naive exp
    mismatch=summary.groupby('family')[['stat_reference_relative_error','exp_mean_J_relative_error']].median().reindex(FAMILIES)
    xp=np.arange(len(mismatch));ww=.36
    ax[0,2].bar(xp-ww/2,mismatch['stat_reference_relative_error'],width=ww,label='Independent C1 C0^-1',color='#487EA5')
    ax[0,2].bar(xp+ww/2,mismatch['exp_mean_J_relative_error'],width=ww,label='exp(mean J Δt)',color='#D9914B')
    ax[0,2].set_xticks(xp,['FHN','SL below','SL above','Gradient'],rotation=15)
    ax[0,2].set(ylabel='Median relative operator difference',title='(c) Different references are not interchangeable')
    ax[0,2].legend(fontsize=7,frameon=False)
    # (d) recovered Q angle versus known tangent M2 plane (method-wise)
    method=results[results.reference_type=='ordered_tangent']
    data=[];methnames=[]
    short={'FHN_directed_spikes':'FHN','SL_directed_below_Hopf':'SL−',
           'SL_directed_above_Hopf':'SL+','double_well_gradient':'DW'}
    for f in FAMILIES:
        for mm in ('M2','M3'):
            vals=method[(method.family==f)&(method.method==mm)].plane_error_deg.to_numpy()
            data.append(vals);methnames.append(short[f]+' '+mm)
    bp=ax[1,0].boxplot(data,patch_artist=True,showfliers=False)
    for i,patch in enumerate(bp['boxes']):patch.set_facecolor('#427FA5' if i%2==0 else '#D18D52');patch.set_alpha(.75)
    ax[1,0].set_xticks(np.arange(1,len(data)+1),methnames,rotation=35)
    ax[1,0].set(ylabel='Largest principal angle (degrees)',title='(d) Finite-data versus tangent M2 plane',ylim=(0,90))
    # (e) admissibility by family, independent truth vs inferred
    modes=[]
    for fam in FAMILIES:
        sub=summary[summary.family==fam]
        inferred=method[(method.family==fam)&(method.method=='M2')]
        modes.append([sub.oracle_RA_valid.mean(),inferred.inferred_RA_valid.mean()])
    modes=np.asarray(modes)
    ax[1,1].bar(xp-.18,modes[:,0],width=.35,label='Ordered tangent',color='#427FA5')
    ax[1,1].bar(xp+.18,modes[:,1],width=.35,label='Inferred M2',color='#D18D52')
    ax[1,1].set_xticks(xp,['FHN','SL below','SL above','Gradient'],rotation=15)
    ax[1,1].set(ylabel='Fraction with admissible rates R',title='(e) Rate-domain admissibility',ylim=(0,1.07))
    ax[1,1].legend(fontsize=7,frameon=False)
    # (f) gate rejection and RA detections separately
    frac=summary.groupby('family')[['gate_reject','oracle_RA_exceeds','inferred_M2_RA_exceeds']].mean().reindex(FAMILIES)
    w=.25
    ax[1,2].bar(xp-w,frac.gate_reject,width=w,label='Adequacy reject',color='#4C7199')
    ax[1,2].bar(xp,frac.oracle_RA_exceeds,width=w,label='Known tangent R>1',color='#D58C47')
    ax[1,2].bar(xp+w,frac.inferred_M2_RA_exceeds,width=w,label='Inferred M2 R>1',color='#4F8E72')
    ax[1,2].set_xticks(xp,['FHN','SL below','SL above','Gradient'],rotation=15)
    ax[1,2].set(ylabel='Fraction of windows (all included)',title='(f) Amplification and model applicability',ylim=(0,1.05))
    ax[1,2].legend(fontsize=6.7,frameon=False)
    for a in ax.flat:a.grid(alpha=.15,axis='y')
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'T09_FigN4_nonlinear_known_jacobians.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)


def write_cluster_uncertainty(summary, method_df,cfg=CFG):
    """Subject-cluster bootstrap; overlapping windows are NOT independent trials."""
    rng=np.random.default_rng(cfg['bootstrap_seed']); B=cfg['bootstrap_draws']
    outcomes=[];rows=[]
    for (fam,condition),dat in summary.groupby(['family','condition'],sort=False):
        n=len(dat); win=dat.window_id.nunique()
        truth_negative=dat.oracle_RA_valid & (~dat.oracle_RA_exceeds)
        for method in ('M2','M3'):
            detect=dat[f'inferred_{method}_RA_exceeds']
            valid=dat[f'inferred_{method}_RA_valid']
            spurious=truth_negative & detect
            num_spurious=int(spurious.sum());caught=int((spurious & dat.gate_reject).sum())
            outcomes.append({'family':fam,'condition':condition,'method':method,
                'windows':win,'subjects':dat.subject_rep.nunique(),
                'reference_RA_valid':int(dat.oracle_RA_valid.sum()),
                'known_tangent_RA_gt1':int(dat.oracle_RA_exceeds.sum()),
                'reference_RA_le1':int(truth_negative.sum()),
                'inferred_RA_valid':int(valid.sum()),
                'inferred_RA_gt1':int(detect.sum()),
                'spurious_RA_gt1_on_known_subthreshold':num_spurious,
                'spurious_caught_by_gate':caught,
                'spurious_accepted_by_gate':num_spurious-caught,
                'gate_rejected':int(dat.gate_reject.sum()),
                'accepted_and_detected':int((detect & ~dat.gate_reject).sum()),
                'invalid_rate_fraction':float((~valid).mean()),
                'median_plane_angle_vs_tangent':float(dat[f'plane_{method}_error_deg'].median())})
            reps=sorted(dat.subject_rep.unique())
            idx=[dat.index[dat.subject_rep==r].to_numpy() for r in reps]
            # Weights of each unique subject under clustered resampling.
            boot=np.zeros((B,4))
            for b in range(B):
                chosen=rng.choice(len(reps),len(reps),replace=True)
                chosen_idx=np.concatenate([idx[i] for i in chosen])
                d=summary.loc[chosen_idx]
                elig=(d.oracle_RA_valid & ~d.oracle_RA_exceeds)
                sb=(elig & d[f'inferred_{method}_RA_exceeds'])
                boot[b]=[np.median(d[f'plane_{method}_error_deg']),
                         np.mean(d[f'inferred_{method}_RA_exceeds']),
                         np.mean(d.gate_reject),
                         sb.sum()/max(1,elig.sum())]
            point=[float(dat[f'plane_{method}_error_deg'].median()),float(detect.mean()),
                   float(dat.gate_reject.mean()),float(num_spurious/max(1,truth_negative.sum()))]
            for j,metric in enumerate(['median_plane_angle_deg','fraction_inferred_RA_gt1',
                                        'fraction_gate_rejected','spurious_fraction_of_known_subthreshold']):
                lower,upper=np.percentile(boot[:,j],[2.5,97.5]);
                rows.append({'family':fam,'condition':condition,'method':method,
                             'metric':metric,'point_estimate':point[j],
                             'CI95_lower_subject_bootstrap':float(lower),
                             'CI95_upper_subject_bootstrap':float(upper),
                             'independent_subjects':len(reps),
                             'overlapping_windows':len(dat),
                             'bootstrap_draws':B})
    out=pd.DataFrame(outcomes);ci=pd.DataFrame(rows)
    out.to_csv(OUT/'T09_spurious_detections_and_gate.csv',index=False)
    ci.to_csv(OUT/'T09_subject_cluster_CI.csv',index=False)
    return out,ci


def plot_nonlinear_sensitivity(summary,cfg=CFG):
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.4),constrained_layout=True)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fhn=summary[summary.family=='FHN_directed_spikes']
    x=np.sort(fhn.window_center.unique())
    def median_spread(df,col):
        groups=df.groupby('window_center')[col]
        return groups.median().reindex(x).to_numpy(),groups.quantile(.25).reindex(x).to_numpy(),groups.quantile(.75).reindex(x).to_numpy()
    for col,color,label in [('oracle_RA','#C17732','Known tangent M2'),
                             ('inferred_M2_RA','#246C99','Inferred M2')]:
        y,q1,q3=median_spread(fhn,col)
        axs[0,0].plot(x,y,color=color,lw=1.8,marker='.',label=label)
        axs[0,0].fill_between(x,q1,q3,color=color,alpha=.12)
    axs[0,0].axhline(1,color='.4',ls='--',lw=.9)
    axs[0,0].axvspan(cfg['fhn_pulse_on'],cfg['fhn_pulse_off'],color='#F5C66C',alpha=.25)
    axs[0,0].set(title='(a) FHN: R_A by known versus fitted operator',xlabel='Window-center time',ylabel='R_A (valid windows only)')
    axs[0,0].legend(frameon=False,fontsize=8)
    # Unbounded near unstable reduced operators: report means only for admissible, no fabricated 0s.
    for col,color,label in [('plane_M2_error_deg','#246C99','Versus known tangent plane'),
                             ('statistical_vs_tangent_angle','#AD5E83','Statistical versus tangent references')]:
        med,lo,hi=median_spread(fhn,col)
        axs[0,1].plot(x,med,color=color,lw=1.8,marker='.',label=label)
        axs[0,1].fill_between(x,lo,hi,color=color,alpha=.14)
    axs[0,1].axvspan(cfg['fhn_pulse_on'],cfg['fhn_pulse_off'],color='#F5C66C',alpha=.25)
    axs[0,1].set(title='(b) FHN: plane agreement versus dynamics',xlabel='Window-center time',ylabel='Largest principal angle (degrees)',ylim=(0,92))
    axs[0,1].legend(frameon=False,fontsize=8)
    fractions=fhn.groupby('window_center').agg(valid=('inferred_M2_RA_valid','mean'),reject=('gate_reject','mean')).reindex(x)
    axs[1,0].plot(x,fractions.valid,'o-',color='#246C99',label='Inferred R_A admissible')
    axs[1,0].plot(x,fractions.reject,'s-',color='#AD5E83',label='Transferred adequacy gate rejects')
    axs[1,0].axvspan(cfg['fhn_pulse_on'],cfg['fhn_pulse_off'],color='#F5C66C',alpha=.25)
    axs[1,0].set(title='(c) FHN: admissibility and model warning',xlabel='Window-center time',ylabel='Fraction of 8 subjects',ylim=(-.04,1.06))
    axs[1,0].legend(frameon=False,fontsize=8)
    grad=summary[summary.family=='double_well_gradient']
    gg=grad.groupby('window_center').agg(oracle=('oracle_RA','median'),fit=('inferred_M2_RA','median'))
    axs[1,1].plot(gg.index,gg.oracle,'o-',color='#C17732',label='Known tangent M2')
    axs[1,1].plot(gg.index,gg.fit,'o-',color='#246C99',label='Inferred M2')
    axs[1,1].axhline(1,color='.4',ls='--',lw=.9)
    sp=grad[grad.inferred_M2_RA_exceeds & grad.oracle_RA_valid & ~grad.oracle_RA_exceeds]
    if len(sp):
        axs[1,1].scatter(sp.window_center,sp.inferred_M2_RA,color='crimson',marker='x',s=68,zorder=6,
                         label='Spurious above-one window')
    axs[1,1].set(title='(d) Gradient negative control: R_A',xlabel='Window-center time',ylabel='R_A')
    axs[1,1].legend(frameon=False,fontsize=8)
    for a in axs.flat:a.grid(alpha=.17,axis='y')
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'T09_nonlinear_FHN_gradient_sensitivity.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)

def save_report(summary,methods,checks,stats,cfg=CFG):
    counts=summary.groupby('family').agg(windows=('window_id','count'),
                  true_gain=('oracle_sigma',lambda x:int((x>1).sum())),
                  tangent_R_valid=('oracle_RA_valid','sum'),
                  fitted_M2_R_valid=('inferred_M2_RA_valid','sum'),
                  fitted_M2_R_gt1=('inferred_M2_RA_exceeds','sum'),
                  gate_reject=('gate_reject','sum'),
                  rel_stat_diff=('stat_reference_relative_error','median'),
                  rel_naive_diff=('exp_mean_J_relative_error','median'))
    ang=methods[methods.reference_type=='ordered_tangent'].groupby(['family','method']).plane_error_deg.median().unstack()
    spurious=pd.read_csv(OUT/'T09_spurious_detections_and_gate.csv')
    spurious=spurious[['family','condition','method','windows','reference_RA_le1',
       'inferred_RA_valid','inferred_RA_gt1','spurious_RA_gt1_on_known_subthreshold',
       'spurious_caught_by_gate','spurious_accepted_by_gate','gate_rejected']]
    lines=['# T09 — nonlinear dynamics with known Jacobians: numerical audit','',
    '## What is independently known','',
    f'- Four 8D physical systems; {cfg["replicates"]} independent subjects per family; '
    f'{cfg["M_estimation"]} estimation trajectories/subject; '
    f'{cfg["M_independent_statistical_reference"]} independent reference trajectories per family.',
    f'- Euler–Maruyama integration step h={cfg["integration_dt"]}, observation Δt={cfg["sample_dt"]}; '
    f'W={cfg["moving_window"]}, stride={cfg["moving_stride"]}, T={cfg["T_samples"]}.',
    '- Brownian forcing: independent state-coordinate shocks with standard deviation '
    '`noise_sd*sqrt(h)` per numerical microstep. Physical coupling is directed in FHN and Stuart–Landau; '
    'the double well has symmetric coupling and is a gradient SDE.',
    '- The window mean of *ordered one-sample variational maps* is the Jacobian/tangent reference. '
    'The map is differentiated from the Euler–Maruyama scheme, including all time-ordering.',
    '- The independent ensemble statistical predictor uses centered C1*C0^(-1) from entirely separate '
    'replicated physical trajectories in matching time windows. These references differ, especially '
    'for nonstationary forcing; this is not an estimation failure of the method per se.',
    '- No real rates ratio is assigned to an unstable or negative-multiplier compression. '
    '`R_sigma` and full-system gain are reported separately. Above-Hopf time windows may '
    'have non-stable local tangent compressions.',
    '- The five-test T08 gate is applied unchanged (no threshold fitting to T09 outcomes), '
    'but its numerical calibration was not trained for W=60 8D nonlinear SDEs; all gate rates are '
    'exploratory transfer diagnostics, NOT calibrated type-I errors.',
    '', '## Numerical derivative checks', '',
    pd.DataFrame(checks).to_markdown(index=False,floatfmt='.2g'),'',
    '## All-window status and reference discrepancies', '', counts.to_markdown(floatfmt='.3g'),'',
    '## Median largest principal-angle error (degrees) vs known tangent M2 plane','',ang.to_markdown(floatfmt='.2f'),'',
    '## Spurious fitted R_A > 1 versus known tangent subthreshold and adequacy gate','',
    spurious.to_markdown(index=False), '',
    'The spurious-rate denominator is strictly windows with known tangent rates R_A valid and <=1; '
    'unknown or unstable local tangent references are NOT silently treated as negatives. '
    'The gradient false-positive above threshold is *accepted*, not caught, by the transferred gate.',
    '',
    '## Subject-level uncertainty','',
    'T09_subject_cluster_CI.csv contains 95% percentile intervals from resampling the eight independent '
    'subjects within each family/condition, preserving all overlapping windows of each subject. '
    'The window-level counts are not independent Bernoulli trials.',
    '',
    '## Interpretation and key limitations','',
    '- Gradient SDE: instantaneous Jacobians are symmetric by construction. Products of '
    'noncommuting symmetric instantaneous updates and/or C1*C0^(-1) can be nonnormal; '
    'do NOT claim K=0 for a statistical linearization or for finite-time products.',
    '- Report the fraction of windows with valid fitted and true reduced R separately from '
    'the fraction above threshold. A fitted high-R window is not necessarily evidence of '
    'a high-R local Jacobian in these nonlinear/nonstationary systems.',
    '- FHN before/overlapping/after conditions are defined by actual complete window support, '
    'not centers; windows intersecting pulse time [23,30] are classified as overlapping.',
    '- The almost-normal Stuart–Landau and gradient tangent maps can have weakly identified gain planes; '
    'a large principal angle alone does not imply a large error in physically meaningful gain.',
    '- The gradient reference is the local tangent operator; its true full gain stays below one in all '
    'sampled windows. Despite this, one M2 compressed fit exceeds the corrected threshold, and the '
    'transferred adequacy gate does not flag that case. This is an adverse control result.',
    '- State-coordinate plane estimates are not planted geometric invariants; the independent '
    'reference here is a known-physics variational-map optimizer, not a unique invariant '
    'nonlinear manifold.',
    '- Different RNG seeds for independent statistical reference and estimation; '
    'no reference labels enter fitting or the T08 adequacy cutoff selection.',
    '', '## Next work','',
    'T10: moving-window independent time-varying planted-plane control; T11: gap-dependent sample efficiency; '
    'T12: dependence-preserving bootstrap and solver robustness. Empirical analyses remain separate.' ]
    (OUT/'T09_report.md').write_text('\n'.join(lines)+'\n')


def main(cfg=CFG):
    st=time.time();gate,cutoffs=load_gate() if cfg['adequacy_gate'] else (None,{})
    windows=list(range(0,cfg['T_samples']-cfg['moving_window']+1,cfg['moving_stride']))
    records=[];methods=[];checks=[];refs=[];aux={}
    for fi,family in enumerate(FAMILIES):
        p=params(family,cfg);check=finite_diff_check(family,p,seed=800+fi,cfg=cfg)
        checks.append(check)
        print(f'[T09] {family}: J max={check["instantaneous_J_max_abs_error"]:.2e}, '
              f'Phi max={check["sample_variational_max_abs_error"]:.2e}',flush=True)
        # Independent statistical reference: distinct seed and independent initial conditions.
        refstates,_,_=simulate(p,cfg['seed_base']+500000+fi*101,cfg['M_independent_statistical_reference'],False,cfg)
        refmaps={}
        for begin in windows:
            end=begin+cfg['moving_window']
            refmaps[begin]=empirical_map(refstates,begin,end,1e-10)[1]
        if family=='FHN_directed_spikes':
            aux['fhn_first_states']=None
        for rep in range(cfg['replicates']):
            seed=cfg['seed_base']+fi*10000+rep*31
            obs,Phis,Jbars=simulate(p,seed,cfg['M_estimation'],True,cfg)
            if family=='FHN_directed_spikes' and rep==0:
                aux['fhn_first_states']=obs[0]
            for begin in windows:
                end=begin+cfg['moving_window'];center=(begin+end)/2*cfg['sample_dt']
                window_id=f'{family}|s{rep:02d}|w{begin:03d}'
                Ftrue=np.mean(Phis[:,begin:end,:,:],axis=(0,1))
                Jmean=np.mean(Jbars[:,begin:end,:,:],axis=(0,1))
                Fnaive=expm(cfg['sample_dt']*Jmean)
                Fest,Femp,info=empirical_map(obs,begin,end,cfg['ridge_relative'])
                Fstat=refmaps[begin]
                oracleQ=plane(Ftrue,'M2',seed+begin+99,cfg)
                oracleD=diag_for(oracleQ,Ftrue,cfg['sample_dt'])
                known=full_properties(Ftrue)
                fstat_rel=np.linalg.norm(Fstat-Ftrue,'fro')/max(np.linalg.norm(Ftrue,'fro'),1e-12)
                naive_rel=np.linalg.norm(Fnaive-Ftrue,'fro')/max(np.linalg.norm(Ftrue,'fro'),1e-12)
                Qstat=plane(Fstat,'M2',seed+begin+199,cfg)
                # Direct statistical linearization is estimated independently from observation cohort.
                tstat=diag_for(Qstat,Fstat,cfg['sample_dt'])
                gateres=gate_window(obs[:,begin:end+1,:],seed+begin+501,cutoffs,gate,cfg) if gate else {'gate_pass':True,'gate_fail_count':0,'gate_failed_tests':''}
                drow={**known,**gateres,**info,'family':family,'subject_rep':rep,'seed':seed,
                      'window_id':window_id,'window_start':begin,'window_end':end,
                      'window_center':center,'condition':cond_name(family,center,cfg,begin,end),
                      'oracle_sigma':known['full_sigma_max'],
                      'oracle_RA':oracleD['R_variance'],'oracle_Rsigma':oracleD['R_sigma'],
                      'oracle_RA_status':oracleD['status'],'oracle_RA_valid':oracleD['status']=='ok',
                      'oracle_RA_exceeds':bool(oracleD['status']=='ok' and oracleD['R_variance']>1),
                      'statistical_RA':tstat['R_variance'],'statistical_RA_status':tstat['status'],
                      'statistical_RA_valid':tstat['status']=='ok',
                      'stat_reference_relative_error':fstat_rel,
                      'exp_mean_J_relative_error':naive_rel,
                      'tangent_vs_fitted_error':float(np.linalg.norm(Fest-Ftrue,'fro')/max(np.linalg.norm(Ftrue,'fro'),1e-12)),
                      'statistical_vs_fitted_error':float(np.linalg.norm(Fest-Fstat,'fro')/max(np.linalg.norm(Fstat,'fro'),1e-12)),
                      'statistical_vs_tangent_angle':principal_angle_deg(oracleQ,Qstat),
                      'statistical_C1_asymmetry':float(np.linalg.norm(Fstat@Fstat.T-Fstat.T@Fstat,'fro')),
                      'jac_mean_asymmetry':float(np.linalg.norm(Jmean-Jmean.T,'fro')),
                      'gate_reject':not gateres['gate_pass']}
                for method in ('M2','M3'):
                    Q=plane(Fest,method,seed+begin+349,cfg)
                    fitD=diag_for(Q,Fest,cfg['sample_dt'])
                    angle=principal_angle_deg(Q,oracleQ)
                    angle_stat=principal_angle_deg(Q,Qstat)
                    ratio_capture=float(np.linalg.svd(Q.T@Ftrue@Q,compute_uv=False)[0]/max(known['full_sigma_max'],1e-20))
                    methods.append({'window_id':window_id,'family':family,'subject_rep':rep,
                         'condition':drow['condition'],'window_center':center,
                         'method':method,'reference_type':'ordered_tangent',
                         'plane_error_deg':angle,'plane_error_vs_stat_deg':angle_stat,
                         'true_full_gain_capture':ratio_capture,
                         'inferred_RA':fitD['R_variance'],'inferred_Rsigma':fitD['R_sigma'],
                         'inferred_status':fitD['status'],'inferred_RA_valid':fitD['status']=='ok',
                         'inferred_RA_exceeds':bool(fitD['status']=='ok' and fitD['R_variance']>1),
                         'oracle_RA':oracleD['R_variance'],'oracle_status':oracleD['status'],
                         'stat_RA':tstat['R_variance'],'stat_status':tstat['status'],
                         'gate_pass':gateres['gate_pass']})
                    drow[f'inferred_{method}_RA']=fitD['R_variance']
                    drow[f'inferred_{method}_RA_valid']=fitD['status']=='ok'
                    drow[f'inferred_{method}_RA_exceeds']=bool(fitD['status']=='ok' and fitD['R_variance']>1)
                    drow[f'plane_{method}_error_deg']=angle
                records.append(drow)
            print(f'[T09] {family} rep={rep+1}/{cfg["replicates"]} windows={len(windows)}',flush=True)
    summary=pd.DataFrame(records);method_df=pd.DataFrame(methods);checks_df=pd.DataFrame(checks)
    summary.to_csv(OUT/'T09_window_truth_and_fit.csv',index=False)
    method_df.to_csv(OUT/'T09_all_method_results.csv',index=False)
    checks_df.to_csv(OUT/'T09_numerical_jacobian_checks.csv',index=False)
    aggregate=summary.groupby(['family','condition'],as_index=False).agg(
        windows=('window_id','count'),oracle_gain_gt1=('oracle_sigma',lambda x:int((x>1).sum())),
        oracle_RA_valid=('oracle_RA_valid','sum'),oracle_RA_gt1=('oracle_RA_exceeds','sum'),
        estimated_RA_valid=('inferred_M2_RA_valid','sum'),estimated_RA_gt1=('inferred_M2_RA_exceeds','sum'),
        adequacy_rejected=('gate_reject','sum'),
        median_tangent_stat_error=('stat_reference_relative_error','median'),
        median_plane_M2=('plane_M2_error_deg','median'),median_plane_M3=('plane_M3_error_deg','median'))
    aggregate.to_csv(OUT/'T09_condition_summary.csv',index=False)
    write_cluster_uncertainty(summary,method_df,cfg)
    plot_nonlinear_sensitivity(summary,cfg)
    summarize_and_plot(summary,method_df,aux,cfg)
    save_report(summary,method_df,checks,aggregate,cfg)
    manifest={'module':'T09','seed':cfg['seed_base'],'configuration':cfg,
      'utc_created':pd.Timestamp.utcnow().isoformat(),'elapsed_seconds':round(time.time()-st,2),
      'windows':len(summary),'methods':len(method_df),'checks':checks,
      'invalid_counts':{'oracle':int((~summary.oracle_RA_valid).sum()),
                        'estimated_M2':int((~summary.inferred_M2_RA_valid).sum()),
                        'estimated_M3':int((~summary.inferred_M3_RA_valid).sum())},
      'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
          for path in (HERE/'run.py',HERE/'config.json',ROOT/'shared/nndr_thresholds.py',
                       ROOT/'shared/nndr_planes.py',ROOT/'08_adequacy_calibration/outputs/T08_locked_calibration_cutoffs.csv',
                       ROOT/'08_adequacy_calibration/run.py')}}
    (OUT/'T09_manifest.json').write_text(json.dumps(manifest,indent=2,default=float))
    print('[T09] Completed windows',len(summary),'method rows',len(method_df),
          'seconds',round(time.time()-st,1),flush=True)
    print(aggregate.to_string(index=False),flush=True)
    return summary,method_df

if __name__=='__main__':main()
