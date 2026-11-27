#!/usr/bin/env python3
"""T08: five synthetic adequacy checks and outcome-blind calibration.

Run with no arguments from any working directory / Spyder. Uses the *original*
T06/T07 generator code and stored paired-subject seeds, never reconstructs
noise from a fitted result. The tests are:
 A. 20% chronological one-step held-out full/diagonal prediction MSE ratio;
 B. held-out lag covariance mismatch with Fhat^2, Fhat^4;
 C. training direct two-/four-step semigroup mismatch (and alias flags);
 D. squared-residual temporal dependence relative to shared-phase multivariate
    phase-randomized Fourier surrogates;
 E. 2D optimal-gain (singular-plane) residual portmanteau statistic.

Thresholds: 99th empirical percentile per statistic, learned *only* from
known correctly specified VAR(1) simulations at the original sampling rates
and lengths (T06/T07 Gaussian, short/oscillatory normal systems and unmodified
baseline recordings), on EVEN subject replicates. ODD replicates are held out for evaluation. No
R_A values, labels, outcomes or event contrasts enter cutoff calibration.

All raw and accepted-only detection rates are reported with their denominators;
 invalid fitted R_A is not silently recoded as a valid negative.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,importlib.util,json,hashlib,time,platform,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'config.json').read_text())

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

T06=load_module(ROOT/'06_N3_negative_controls'/'run.py','t08_t06_source')
T07=load_module(ROOT/'07_N3_positive_controls'/'run.py','t08_t07_source')
SCORES=['a_predictive_log_mse_ratio','b_lag_covariance_mismatch','c_semigroup_discrepancy',
        'd_energy_excess_vs_surrogate','e_projected_portmanteau']


def fit_ridge(x,y,lam_rel=.001):
    n=x.shape[1]
    xx=x.T@x
    ridge=lam_rel*np.trace(xx)/n
    return np.linalg.solve(xx+ridge*np.eye(n),x.T@y).T,float(ridge)


def diag_ridge(x,y,lam_rel=.001):
    ridge=lam_rel*np.sum(x*x)/x.shape[1]
    return np.sum(x*y,axis=0)/(np.sum(x*x,axis=0)+ridge)


def select_plane_svd(F):
    """Method-agnostic 2D subspace from the leading left/right singular pair."""
    u,s,vh=np.linalg.svd(F,full_matrices=False)
    v=vh[0].copy();r=u[:,0].copy()
    r-=v*np.dot(v,r)
    nr=np.linalg.norm(r)
    if nr<1e-9:
        # Degenerate normal case: use second orthonormal left vector.
        r=u[:,1]-v*np.dot(v,u[:,1]);nr=np.linalg.norm(r)
    return np.stack([v,r/max(nr,1e-16)],axis=1)


def lag_corr_energy(vec,L=6):
    """Largest absolute normalized correlation of the cross-channel energy."""
    z=np.asarray(vec,float)
    z=z-z.mean()
    den=np.dot(z,z)
    if den<1e-20:return 0.
    return float(max(abs(np.dot(z[:-h],z[h:])/den) for h in range(1,min(L,len(z)-2)+1)))


def shared_phase_surrogate(R,rng):
    """Random phase shared across variables at each frequency within trajectory.

    This preserves EACH multivariate trajectory's full cross-spectrum and
    auto-spectra, not merely its marginal spectral densities.
    """
    M,T,n=R.shape
    spec=np.fft.rfft(R,axis=1)
    angles=rng.uniform(-np.pi,np.pi,size=(M,spec.shape[1],1))
    phase=np.exp(1j*angles)
    phase[:,0,:]=1.
    if T%2==0: phase[:,-1,:]=1.
    return np.fft.irfft(spec*phase,n=T,axis=1)


def squared_residual_score(R,seed,B=15,L=6):
    obs=lag_corr_energy(np.mean(R**2,axis=(0,2)),L)
    rng=np.random.default_rng(int(seed))
    null=np.empty(B)
    for b in range(B):
        surrogate=shared_phase_surrogate(R,rng)
        null[b]=lag_corr_energy(np.mean(surrogate**2,axis=(0,2)),L)
    # Positive value means more volatility persistence than matched spectral null.
    return float(obs-np.mean(null)),float(obs),float(np.mean(null)),float(np.std(null,ddof=1)),float((1+(null>=obs).sum())/(B+1))


def portmanteau_projected(R,Q,L=6):
    projected=R@Q
    projected-=projected.mean(axis=(0,1),keepdims=True)
    m,t,d=projected.shape
    z=projected.reshape(-1,d)
    C0=(z.T@z)/(m*t)
    inv=np.linalg.pinv(C0,rcond=1e-10)
    val=0.
    for h in range(1,min(L,t-2)+1):
        a=projected[:,h:,:].reshape(-1,d)
        b=projected[:,:-h,:].reshape(-1,d)
        C=a.T@b/(m*(t-h))
        val+=float(np.trace(C.T@inv@C@inv))/(1-h/t)
    return float(m*t*val),float(1-chi2.cdf(m*t*val,df=4*L))


def alias_flag(F1,F2,F4):
    """Data-only spectral principal-branch flags, no access to true frequency."""
    Q=select_plane_svd(F1)
    angles=[]
    for h,Fh in [(1,F1),(2,F2),(4,F4)]:
        eig=np.linalg.eigvals(Q.T@Fh@Q)
        if max(abs(np.imag(eig)))>1e-6:
            phase=float(max(abs(np.angle(eig))))
            angles.append((h,phase/h,phase))
    if not angles:return False,np.nan
    nyquist=any(abs(np.pi-v[2])<.15 for v in angles)
    inconsistent=(len(angles)>1 and np.ptp([x[1] for x in angles])>.14)
    return bool(nyquist or inconsistent),float(max([x[1] for x in angles]))


def compute_diagnostics(data,diagseed,cfg=CFG):
    data=np.asarray(data,float)
    M,Tp1,n=data.shape;T=Tp1-1
    ntrain=int(np.floor(T*(1-cfg['heldout_fraction'])))
    X=data[:,:-1,:];Y=data[:,1:,:]
    xtr=X[:,:ntrain,:].reshape(-1,n);ytr=Y[:,:ntrain,:].reshape(-1,n)
    xva=X[:,ntrain:,:].reshape(-1,n);yva=Y[:,ntrain:,:].reshape(-1,n)
    F,lam=fit_ridge(xtr,ytr,cfg['ridge_relative'])
    diagonal=diag_ridge(xtr,ytr,cfg['ridge_relative'])
    err_full=np.mean((yva-xva@F.T)**2)
    err_diag=np.mean((yva-xva*diagonal)**2)
    score_a=float(np.log((err_full+1e-20)/(err_diag+1e-20)))
    score_b=[];score_c=[]
    mats={1:F}
    for h in cfg['lag_horizons']:
        Xh=data[:,:-h,:];Yh=data[:,h:,:]
        # Valid held-out k have both x[k] and x[k+h] in the held-out segment.
        xv=Xh[:,ntrain:,:].reshape(-1,n)
        yv=Yh[:,ntrain:,:].reshape(-1,n)
        C0=xv.T@xv/max(len(xv),1)
        Ch=yv.T@xv/max(len(xv),1)
        pred=np.linalg.matrix_power(F,h)@C0
        score_b.append(np.linalg.norm(Ch-pred,'fro')/max(np.linalg.norm(Ch,'fro'),1e-12))
        # independently estimated direct h-step map from training region
        xt=Xh[:,:max(0,ntrain-h+1),:].reshape(-1,n)
        yt=Yh[:,:max(0,ntrain-h+1),:].reshape(-1,n)
        Gh,_=fit_ridge(xt,yt,cfg['ridge_relative'])
        mats[h]=Gh
        score_c.append(np.linalg.norm(Gh-np.linalg.matrix_power(F,h),'fro')/max(np.linalg.norm(np.linalg.matrix_power(F,h),'fro'),1e-12))
    score_b=float(np.mean(score_b));score_c=float(np.mean(score_c))
    alias, freq=alias_flag(mats[1],mats[2],mats[4])
    # Residual diagnostics: whole record to catch a burst in the middle third.
    R=Y-X@F.T
    score_d,actual_energy,sur_energy,sur_sd,sur_p=squared_residual_score(R,diagseed,cfg['phase_surrogates'],cfg['energy_lags'])
    Q=select_plane_svd(F)
    score_e,pport=portmanteau_projected(R,Q,cfg['portmanteau_lags'])
    return {'a_predictive_log_mse_ratio':score_a,
            'b_lag_covariance_mismatch':score_b,
            'c_semigroup_discrepancy':score_c,
            'd_energy_excess_vs_surrogate':score_d,
            'e_projected_portmanteau':score_e,
            'a_val_mse_full':float(err_full),'a_val_mse_diagonal':float(err_diag),
            'd_energy_observed':actual_energy,'d_energy_surrogate_mean':sur_energy,
            'd_energy_surrogate_sd':sur_sd,'d_phase_surrogate_p':sur_p,
            'e_portmanteau_chi2_p_uncalibrated':pport,
            'alias_suspected':bool(alias),'estimated_principal_frequency':freq,
            'ridge_train':lam,'T_train_subwindow':ntrain,'T_test_subwindow':T-ntrain}


def subject_reps(s):
    return int(str(s).split('_')[-1])


def gather_t06(cfg=CFG):
    source=pd.read_csv(ROOT/'06_N3_negative_controls'/'outputs'/'T06_subject_method_results.csv')
    keys=source[['subject','seed','family','phase']].drop_duplicates()
    out=[]
    for ix,(subject,seed,family,phase) in enumerate(keys.itertuples(index=False,name=None)):
        pars=T06.build_matrices(family,int(seed),T06.CFG)
        obs=T06.simulate(family,pars,int(seed)+191,phase,cfg=T06.CFG)
        dx=compute_diagnostics(obs,seed_mix('T06',subject,phase),cfg)
        out.append({'dataset':'T06','record_key':f'T06|{subject}|{phase}','subject':subject,
                    'rep':subject_reps(subject),'family':family,'perturbation':family,
                    'spectral_type':'various','target_R':np.nan,'label':'normal','phase':phase,
                    'variance_mode':'shared_innovation','seed':seed,'known_clean_var1':family=='gaussian_VAR1',
                    **dx})
        if (ix+1)%200==0:print(f'[T08] T06 physical records {ix+1}/{len(keys)}',flush=True)
    return pd.DataFrame(out),source


def seed_mix(*parts):
    h=hashlib.sha256('|'.join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:8],'little')%(2**32-1)


def gather_t07(cfg=CFG):
    source=pd.read_csv(ROOT/'07_N3_positive_controls'/'outputs'/'T07_method_phase_results.csv')
    source=source[source.variance_mode=='shared_innovation'].copy()
    keys=source[['subject','rep','seed','spectral_type','target_R','perturbation','label','phase']].drop_duplicates()
    out=[];last_key=None;systems=None;shocks=None
    for ix,(subject,rep,seed,stype,target,pert,label,phase) in enumerate(keys.itertuples(index=False,name=None)):
        if subject!=last_key:
            systems=T07.build_models(stype,target,int(seed),T07.CFG)
            shocks=T07.random_shocks(int(seed)+191,T07.CFG)
            last_key=subject
        obs=T07.simulate(systems,label,phase,pert,shocks,int(seed)+191,T07.CFG['innovation_sd'],T07.CFG)
        dx=compute_diagnostics(obs,seed_mix('T07',subject,label,phase),cfg)
        out.append({'dataset':'T07','record_key':f'T07|{subject}|{label}|{phase}',
                    'subject':subject,'rep':int(rep),'family':pert,'perturbation':pert,
                    'spectral_type':stype,'target_R':target,'label':label,'phase':phase,
                    'variance_mode':'shared_innovation','seed':seed,'known_clean_var1':pert=='gaussian_VAR1',
                    **dx})
        if (ix+1)%400==0:print(f'[T08] T07 physical records {ix+1}/{len(keys)}',flush=True)
    return pd.DataFrame(out),source


def clean_var1_by_construction(meta):
    """Source PROTOCOL labels only; never fitted RA or detected-event outcomes.

    T06: normal Gaussian and time-invariant damped oscillatory VAR1 across both
    phases, plus the undisturbed baseline for burst, measurement and switching.
    T07: Gaussian across both phases, plus the undisturbed baseline for
    burst, observation noise, switching and higher-order event perturbations.
    Coloured-noise baseline and nonlinear saturation are not true white VAR1.
    """
    g=meta
    t6=(g.dataset=='T06')
    t7=(g.dataset=='T07')
    fam=g.perturbation.astype(str)
    baseline=(g.phase=='baseline')
    t6clean=(fam.eq('gaussian_VAR1')|fam.str.startswith('weak_beta')|
             (baseline & fam.isin(['isotropic_bursts','common_mode_bursts',
                                   'observation_noise','switch_commuting','switch_noncommuting'])))
    t7clean=(fam.eq('gaussian_VAR1')|
             (baseline & fam.isin(['isotropic_bursts','common_mode_bursts',
                                   'observation_noise','switching','VAR2'])))
    return (t6&t6clean)|(t7&t7clean)


def calibrate(meta,cfg=CFG):
    clean=meta[(meta.known_clean_var1)&(meta.rep%2==0)]
    assert len(clean)>=300
    cutoffs={s:float(np.quantile(clean[s],cfg['calibration_quantile_per_test'])) for s in SCORES}
    # Calibration does NOT inspect R_A, labels nor event outcomes.
    assert all(np.isfinite(list(cutoffs.values())))
    return cutoffs,clean


def apply_gate(meta,cutoffs):
    meta=meta.copy()
    for i,s in enumerate(SCORES):meta[f'fail_{chr(97+i)}']=meta[s]>cutoffs[s]
    columns=[f'fail_{chr(97+i)}' for i in range(len(SCORES))]
    meta['n_adequacy_fail']=meta[columns].sum(axis=1).astype(int)
    meta['adequacy_pass']=meta.n_adequacy_fail==0
    # alias is explicitly marked: a caution flag, not necessarily grounds for an
    # additional rejection; it was not calibrated as a sixth test.
    meta['evaluation_split']=np.where(meta.rep%2==0,'calibration_rep','heldout_rep')
    return meta


def attach_methods(meta,t06rows,t07rows):
    neg=t06rows.copy();neg['record_key']='T06|'+neg.subject+'|'+neg.phase
    neg['dataset']='T06';neg['perturbation']=neg['family'];neg['label']='normal'
    pos=t07rows.copy();pos['record_key']='T07|'+pos.subject+'|'+pos.label+'|'+pos.phase
    pos['dataset']='T07'
    methods=pd.concat([neg,pos],ignore_index=True,sort=False)
    keys=['record_key','adequacy_pass','n_adequacy_fail','evaluation_split','alias_suspected']+[f'fail_{chr(97+i)}' for i in range(5)]
    methods=methods.merge(meta[keys],on='record_key',how='left',validate='many_to_one')
    assert len(methods)==len(neg)+len(pos) and methods.adequacy_pass.notna().all()
    methods['raw_detect_RA']=methods['RA_exceeds_1'].astype(bool)
    methods['screened_detect_RA']=methods.raw_detect_RA & methods.adequacy_pass
    methods['raw_detect_Rsigma']=methods['Rsigma_exceeds_1'].astype(bool)
    methods['screened_detect_Rsigma']=methods.raw_detect_Rsigma & methods.adequacy_pass
    return methods


def wilson(k,n):return T06.wilson_interval(k,n)


def outcome_table(methods):
    out=[]
    ev=methods[(methods.phase=='event')&(methods.evaluation_split=='heldout_rep')].copy()
    for (dataset,pert,label,method),g in ev.groupby(['dataset','perturbation','label','method']):
        # In T07 positives R=0.5 and 1 are not truth-positive for R>1.
        truth=(g.label=='positive') & (g.target_R>1)
        if dataset=='T07' and label=='positive':
            g=g[truth].copy()
        if len(g)==0:continue
        n=len(g);passed=g[g.adequacy_pass];kraw=int(g.raw_detect_RA.sum());kpass=int(passed.raw_detect_RA.sum())
        lo,hi=wilson(kpass,len(passed));lo_total,hi_total=wilson(kpass,n)
        out.append({'dataset':dataset,'perturbation':pert,'label':label,'method':method,
                    'denominator_total':n,'n_adequacy_pass':len(passed),'n_screen_reject':n-len(passed),
                    'screen_reject_fraction':(n-len(passed))/n,
                    'raw_RA_detect_count':kraw,'raw_RA_detect_fraction':kraw/n,
                    'accepted_RA_detect_count':kpass,
                    'accepted_RA_detect_fraction_of_all':kpass/n,
                    'accepted_RA_detect_fraction_of_accepted':kpass/len(passed) if len(passed) else np.nan,
                    'accepted_RA_detect_ci95_accepted_lo':lo,'accepted_RA_detect_ci95_accepted_hi':hi,
                    'accepted_RA_detect_ci95_all_lo':lo_total,'accepted_RA_detect_ci95_all_hi':hi_total,
                    'inadmissible_RA_full':int(g.R_A.isna().sum()),
                    'inadmissible_RA_accepted':int(passed.R_A.isna().sum()),
                    'raw_Rsigma_detect_count':int(g.raw_detect_Rsigma.sum()),
                    'accepted_Rsigma_detect_count':int(passed.raw_detect_Rsigma.sum())})
    return pd.DataFrame(out)


def summary_by_test(meta):
    ev=meta[(meta.phase=='event')&(meta.evaluation_split=='heldout_rep')]
    out=[]
    for (dataset,family,label),g in ev.groupby(['dataset','perturbation','label']):
        row={'dataset':dataset,'perturbation':family,'label':label,'n':len(g),
             'gate_fail_n':int((~g.adequacy_pass).sum()),'gate_fail_rate':float((~g.adequacy_pass).mean()),
             'alias_flag_n':int(g.alias_suspected.sum())}
        for i,s in enumerate(SCORES):row['reject_'+chr(97+i)]=int(g[f'fail_{chr(97+i)}'].sum())
        out.append(row)
    return pd.DataFrame(out)



def grouped_bootstrap_intervals(meta,methods,seed=814210,B=2500):
    """Cluster bootstrap by Monte Carlo subject, never by time points/paired records."""
    rng=np.random.default_rng(seed)
    cases=[]
    clean=meta[(meta.evaluation_split=='heldout_rep')&meta.known_clean_var1].copy()
    clean['event_value']=(~clean.adequacy_pass).astype(int)
    cases.append(('correct_VAR1_any_false_rejection',clean,'event_value'))
    for dataset in ('T06','T07'):
        for method in ('M2','M3'):
            m=methods[(methods.dataset==dataset)&(methods.method==method)&(methods.phase=='event')&(methods.evaluation_split=='heldout_rep')]
            if dataset=='T07':
                norm=m[m.label=='normal']; pos=m[(m.label=='positive')&(m.target_R>1)]
                cases.extend([(f'T07_normal_raw_FPR_{method}',norm,'raw_detect_RA'),
                              (f'T07_normal_screened_FPR_{method}',norm,'screened_detect_RA'),
                              (f'T07_positive_raw_power_{method}',pos,'raw_detect_RA'),
                              (f'T07_positive_screened_per_all_{method}',pos,'screened_detect_RA')])
            else:
                cases.extend([(f'T06_normal_raw_FPR_{method}',m,'raw_detect_RA'),
                              (f'T06_normal_screened_FPR_{method}',m,'screened_detect_RA')])
    out=[]
    for name,df,col in cases:
        if len(df)==0:continue
        grouped=df.groupby(['dataset','subject'],dropna=False)[col].agg(['sum','count']).reset_index()
        sums=grouped['sum'].to_numpy(float);ns=grouped['count'].to_numpy(float)
        n=len(sums)
        draws=rng.integers(0,n,size=(B,n))
        values=sums[draws].sum(axis=1)/ns[draws].sum(axis=1)
        a,b=np.quantile(values,[.025,.975])
        wil_lo,wil_hi=wilson(int(np.sum(sums)),int(np.sum(ns)))
        out.append({'estimand':name,'n_subject_clusters':n,'n_records':len(df),
                    'fraction':float(np.sum(sums)/np.sum(ns)),
                    'cluster_bootstrap95_low':float(a),'cluster_bootstrap95_high':float(b),
                    'wilson95_low_descriptive':wil_lo,'wilson95_high_descriptive':wil_hi,
                    'bootstrap_draws':B})
    return pd.DataFrame(out)

def make_plots(summary,meta,cutoffs):
    ev=summary.copy()
    f,axs=plt.subplots(2,2,figsize=(13,9.2),layout='constrained')
    main=ev[(ev.dataset=='T07')].groupby(['label','method']).agg(
        n=('denominator_total','sum'),npass=('n_adequacy_pass','sum'),
        raw=('raw_RA_detect_count','sum'),after=('accepted_RA_detect_count','sum')).reset_index()
    ax=axs[0,0];xt=np.arange(4);entries=[('normal','M2'),('normal','M3'),('positive','M2'),('positive','M3')]
    for i,(lab,m) in enumerate(entries):
        q=main[(main.label==lab)&(main.method==m)].iloc[0]
        ax.bar(i-.18,q['raw']/q['n'],width=.34,facecolor='#3c6fa8',label='Unscreened / all' if i==0 else None)
        ax.bar(i+.18,q['after']/q['n'],width=.34,facecolor='#d47c38',label='Accepted detections / all' if i==0 else None)
    ax.set_xticks(xt,[f'{m}\n{lab}' for lab,m in entries]);ax.set_ylabel('Rate, full-cohort denominator');ax.set_ylim(0,1.03)
    ax.set_title('(a) Matched-normal false positives and positive power');ax.legend(fontsize=8)
    ax=axs[0,1]
    show=ev[(ev.method=='M2')&(ev.dataset=='T07')].groupby(['perturbation','label']).agg(n=('denominator_total','sum'),npass=('n_adequacy_pass','sum')).reset_index()
    perturb=list(T07.CFG['perturbations']);xx=np.arange(len(perturb))
    a=show[show.label=='normal'].set_index('perturbation');b=show[show.label=='positive'].set_index('perturbation')
    ax.bar(xx-.18,1-(a.loc[perturb,'npass']/a.loc[perturb,'n']),width=.36,color='#3c6fa8',label='Normal')
    ax.bar(xx+.18,1-(b.loc[perturb,'npass']/b.loc[perturb,'n']),width=.36,color='#d47c38',label='Non-normal')
    ax.set_xticks(xx,[s.replace('_','\n') for s in perturb],rotation=40,ha='right',fontsize=7)
    ax.set_ylabel('Rejected fraction');ax.set_ylim(0,1.03);ax.set_title('(b) Rejection by disturbance (T07)');ax.legend(fontsize=8)
    ax=axs[1,0]
    group=ev[(ev.dataset=='T06')&(ev.method=='M2')].copy()
    group=group.sort_values('screen_reject_fraction',ascending=False)
    sel=group.head(12)
    yy=np.arange(len(sel));ax.barh(yy,sel.screen_reject_fraction,color='#3c6fa8')
    ax.set_yticks(yy,[z.replace('_',' ') for z in sel.perturbation],fontsize=8);ax.invert_yaxis()
    ax.set_xlim(0,1.03);ax.set_xlabel('Rejected fraction');ax.set_title('(c) Highest rejection rates among negative controls')
    ax=axs[1,1]
    m=meta[(meta.evaluation_split=='heldout_rep')&(meta.phase=='event')]
    cats=['T06\nGaussian normal','T07\nGaussian normal','T07\nGaussian non-normal','T06\nSpatial coloured','T07\nSpatial coloured']
    masks=[(m.dataset=='T06')&(m.perturbation=='gaussian_VAR1'),
           (m.dataset=='T07')&(m.perturbation=='gaussian_VAR1')&(m.label=='normal'),
           (m.dataset=='T07')&(m.perturbation=='gaussian_VAR1')&(m.label=='positive'),
           (m.dataset=='T06')&(m.perturbation=='coloured_spatial'),
           (m.dataset=='T07')&(m.perturbation=='coloured_spatial')]
    matrix=np.array([[m[cond][f'fail_{c}'].mean() for c in 'abcde'] for cond in masks])
    im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(np.arange(5),['A pred','B lag','C semi','D energy','E white']);ax.set_yticks(np.arange(5),cats,fontsize=8)
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>.5 else '#202020',fontsize=8)
    ax.set_title('(d) Rejection probability by diagnostic');f.colorbar(im,ax=ax,fraction=.046,pad=.03)
    for ext in ('png','pdf'):f.savefig(OUT/f'T08_FigN3_five_adequacy_calibration.{ext}',dpi=210,bbox_inches='tight')
    plt.close(f)


def main():
    t0=time.time()
    source06=pd.read_csv(ROOT/'06_N3_negative_controls'/'outputs'/'T06_subject_method_results.csv')
    source07=pd.read_csv(ROOT/'07_N3_positive_controls'/'outputs'/'T07_method_phase_results.csv')
    source07=source07[source07.variance_mode=='shared_innovation'].copy()
    cached=OUT/'T08_replayed_diagnostics_cache.csv'
    if cached.exists() and CFG.get('reuse_replay_cache',False):
        meta=pd.read_csv(cached)
        meta06=meta[meta.dataset=='T06'];meta07=meta[meta.dataset=='T07']
    else:
        meta06,_=gather_t06()
        meta07,_=gather_t07()
        meta=pd.concat([meta06,meta07],ignore_index=True,sort=False)
        meta.to_csv(cached,index=False)
    assert meta.record_key.is_unique
    meta['known_clean_var1']=clean_var1_by_construction(meta)
    cutoffs,clean=calibrate(meta)
    screened=apply_gate(meta,cutoffs)
    methods=attach_methods(screened,source06,source07)
    tab=outcome_table(methods)
    pertest=summary_by_test(screened)
    phase_table=screened.groupby(['dataset','perturbation','label','phase','evaluation_split'],dropna=False).agg(
        n=('adequacy_pass','size'),n_pass=('adequacy_pass','sum'),alias_flag_n=('alias_suspected','sum')).reset_index()
    phase_table['n_rejected']=phase_table['n']-phase_table['n_pass']
    phase_table['rejected_fraction']=phase_table['n_rejected']/phase_table['n']
    phase_table.to_csv(OUT/'T08_baseline_event_rejection_by_family.csv',index=False)
    boot=grouped_bootstrap_intervals(screened,methods)
    boot.to_csv(OUT/'T08_subject_cluster_bootstrap_intervals.csv',index=False)
    screened.to_csv(OUT/'T08_record_level_adequacy.csv',index=False)
    methods.to_csv(OUT/'T08_all_method_results_with_gate.csv',index=False)
    tab.to_csv(OUT/'T08_holdout_power_FPR_and_rejection.csv',index=False)
    pertest.to_csv(OUT/'T08_holdout_diagnostic_rejection_by_family.csv',index=False)
    calib=pd.DataFrame([{'test':chr(65+i),'metric':s,'upper_cutoff':cutoffs[s],
                         'calibration_count':len(clean),'quantile':CFG['calibration_quantile_per_test'],
                         'calibration_data':'Known correctly specified T06/T07 VAR1; even-replicate; both spectra/lengths and clean baselines'} for i,s in enumerate(SCORES)])
    calib.to_csv(OUT/'T08_locked_calibration_cutoffs.csv',index=False)
    make_plots(tab,screened,cutoffs)
    hold=screened[screened.evaluation_split=='heldout_rep']
    clean_test=hold[hold.known_clean_var1]
    # Count physical records, not method rows.
    gauss_reject=int((~clean_test.adequacy_pass).sum())
    t07=tab[tab.dataset=='T07']
    log={'record_count':len(meta),'T06_records':len(meta06),'T07_records':len(meta07),
         'source_T06_method_rows':len(source06),'source_T07_method_rows':len(source07),
         'calibration_count':len(clean),'calibration_independent_subject_clusters':len(clean[['dataset','subject']].drop_duplicates()),
         'clean_heldout_n':len(clean_test),'clean_heldout_subject_clusters':len(clean_test[['dataset','subject']].drop_duplicates()),
         'clean_heldout_rejected_n':gauss_reject,
         'clean_heldout_rejection_rate':gauss_reject/len(clean_test),
         'heldout_records':len(hold),
         't07_power_M2_unscreened':float(t07[(t07.label=='positive')&(t07.method=='M2')].raw_RA_detect_count.sum()/t07[(t07.label=='positive')&(t07.method=='M2')].denominator_total.sum()),
         't07_power_M2_screened_per_all':float(t07[(t07.label=='positive')&(t07.method=='M2')].accepted_RA_detect_count.sum()/t07[(t07.label=='positive')&(t07.method=='M2')].denominator_total.sum()),
         't07_power_M3_unscreened':float(t07[(t07.label=='positive')&(t07.method=='M3')].raw_RA_detect_count.sum()/t07[(t07.label=='positive')&(t07.method=='M3')].denominator_total.sum()),
         't07_power_M3_screened_per_all':float(t07[(t07.label=='positive')&(t07.method=='M3')].accepted_RA_detect_count.sum()/t07[(t07.label=='positive')&(t07.method=='M3')].denominator_total.sum()),
         't07_fpr_M2_raw':float(t07[(t07.label=='normal')&(t07.method=='M2')].raw_RA_detect_count.sum()/t07[(t07.label=='normal')&(t07.method=='M2')].denominator_total.sum()),
         't07_fpr_M2_screened_per_all':float(t07[(t07.label=='normal')&(t07.method=='M2')].accepted_RA_detect_count.sum()/t07[(t07.label=='normal')&(t07.method=='M2')].denominator_total.sum()),
         'T06_spatial_coloured_reject':float(hold[(hold.dataset=='T06')&(hold.perturbation=='coloured_spatial')&(hold.phase=='event')].adequacy_pass.eq(False).mean()),
         'cutoffs':cutoffs,'runtime_seconds':time.time()-t0,'python':platform.python_version(),
         'numpy':np.__version__,'pandas':pd.__version__}
    sourcepaths=[ROOT/'06_N3_negative_controls'/'run.py',ROOT/'06_N3_negative_controls'/'config.json',ROOT/'06_N3_negative_controls'/'outputs'/'T06_subject_method_results.csv',ROOT/'07_N3_positive_controls'/'run.py',ROOT/'07_N3_positive_controls'/'config.json',ROOT/'07_N3_positive_controls'/'outputs'/'T07_method_phase_results.csv',HERE/'run.py',HERE/'config.json']
    sourcehash={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sourcepaths}
    log['source_sha256']=sourcehash
    (OUT/'T08_manifest.json').write_text(json.dumps(log,indent=2)+'\n')
    (OUT/'T08_source_manifest.json').write_text(json.dumps({'inputs_sha256':sourcehash,'independent_calibration_subject_clusters':log['calibration_independent_subject_clusters'],'replay_cache_opt_in_only':True},indent=2)+'\n')
    lines=['# T08 — Calibrated five-part local-linear model adequacy','',
           '## Study and calibration protocol','',
           f'- Replayed physical recordings exactly from T06 ({len(meta06)}) and T07 ({len(meta07)}) reference generators and original stored seeds; no re-use of preliminary two-rule gate.',
           '- A. Last 20% one-step mean-square predictive ratio, full fitted map relative to diagonal regularized VAR1; an upper-tail log ratio indicates prediction failure.',
           '- B. Held-out lag covariance discrepancy at lags 2 and 4 versus the powers of the trained VAR1 map.',
           '- C. Semigroup discrepancy: directly trained lag-2/lag-4 maps against powers of the one-step map. Complex principal-frequency alias suspicion is a *separate flag*, not an additional calibrated rejection rule.',
           '- D. Cross-channel mean squared residual energy autocorrelation, contrasted with multivariate Fourier phase-randomized surrogate trajectories. Each surrogate applies the SAME phase to all variables at a given frequency within a trajectory, preserving its cross-spectrum. We calibrate the observed-minus-surrogate-mean difference, not an asymptotic p-value.',
           '- E. Six-lag two-dimensional projected residual portmanteau. Projection is the leading right/left singular plane of the TRAINING operator, not outcome-selected M2/M3. This makes the gate method-agnostic; no claim of exact M2 projection.',
           f'- Lock all five upper cutoffs at empirical {CFG["calibration_quantile_per_test"]:.0%} quantiles among {len(clean)} correctly specified VAR1 records from {len(clean[["dataset","subject"]].drop_duplicates())} independent synthetic subject clusters (even replicate indices); includes Gaussian normal/nonnormal, short and oscillatory exact VAR1, and unperturbed baselines of burst/noise/switch/VAR2-event protocols. Within-subject label/phase records are correlated, and are not counted as independent clusters. No RA label or future event outcome used.',
           '- Evaluate on odd replicate indices only. Even-replicate cases are saved for audit but NOT counted in held-out power/FPR.',
           '- Five separately calibrated 1% upper-tail screens are combined with an OR rule (nominal Bonferroni familywise bound 5% under ideal calibration); the 1% empirical quantiles are finite-sample estimates, not exact finite-sample guarantees.',
           '- Invalid reduced R_A fits are unassessable and still included in full-cohort denominators. Failed model-adequacy windows are also unassessable, not automatically classified as true negatives.',
           '', '## Locked upper-tail cutoffs','',
           '| Test | Statistic | Cutoff |','|---|---|---:|']
    for i,s in enumerate(SCORES):lines.append(f'| {chr(65+i)} | `{s}` | {cutoffs[s]:.5g} |')
    lines+=['','## Held-out evaluation','',
            f'- Independent held-out physical records: {len(hold)}; correctly specified VAR1 recordings: {len(clean_test)}, jointly rejected {gauss_reject}/{len(clean_test)} ({gauss_reject/len(clean_test):.2%}). These are paired recordings from fewer independent subject clusters.',
            f'- T07 true positive (R_star>1) raw M2 / M3 detection: {log["t07_power_M2_unscreened"]:.2%} / {log["t07_power_M3_unscreened"]:.2%}; after screening detected per ALL: {log["t07_power_M2_screened_per_all"]:.2%} / {log["t07_power_M3_screened_per_all"]:.2%}.',
            f'- T07 matched-normal raw M2 false positive: {log["t07_fpr_M2_raw"]:.2%}; screened false positive per ALL: {log["t07_fpr_M2_screened_per_all"]:.2%}.',
            f'- T06 event spatially coloured forcing adequacy rejection: {log["T06_spatial_coloured_reject"]:.2%}.',
            '- Screened detection rate among accepted windows has a different denominator from detected-and-accepted rate among all. Both saved per condition and method.',
            '- Primary uncertainty intervals are formed by resampling independent subject clusters, never time samples or multiple paired records as independent. When zero events are observed, percentile-bootstrap bounds degenerate at zero, so descriptive Wilson binomial bounds are also given; neither is a universal error guarantee. See T08_subject_cluster_bootstrap_intervals.csv.',
            '- These estimates describe this synthetic design, not a universal validated rate on arbitrary empirical data. No empirical thresholds are tuned or tested here.',
            '', '## Full-cohort versus conditional-accepted denominators','',
            '- Detailed T08_holdout_power_FPR_and_rejection.csv reports RAW detections / all, accepted detections / all, and accepted detections / accepted, as distinct estimands. A high accepted-only power after rejecting most positives is NOT evidence of an effective operational detector.',
            '- T08_baseline_event_rejection_by_family.csv reports all baseline and event rejection fractions independently, with subject-level split labels.',
            '', '## Important limitations','',
            '- The gate includes a 2D leading-singular-plane portmanteau, not a recovered M2/M3 plane. The gate is shared by both inference methods and should be named accordingly in the rebuttal.',
            '- Validation is split by independent Monte Carlo replicate, not a test on entirely new physical matrix families. Generalization to new dynamics remains outstanding.',
            '- The five checks do not prove local linearity or physical causation; a model may pass yet still be misspecified. Aliasing and invalid estimated matrix logarithms require separate caution.',
            '- The entire waveform is available to tests B–E (A is strictly held out), including the post-event period. In observational use, screens must be recalibrated for record length and sampling.',
            '', '## Reproduction','',
            '`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 08_adequacy_calibration/run.py`',
            'Input: original T06/T07 scripts, configuration files, CSV result identifiers, and random seeds. Outputs include complete record-level tests, five locked cutoffs, method-level joined gates, held-out family power/rejection, figures, and tests.']
    (OUT/'T08_report.md').write_text('\n'.join(lines)+'\n')
    print('[T08] COMPLETE',json.dumps({k:v for k,v in log.items() if k not in ['cutoffs','python','numpy','pandas']},indent=2),flush=True)
    return log

if __name__=='__main__':main()
