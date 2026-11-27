#!/usr/bin/env python3
"""T01 / submitted Figure 5 regenerated using revision's rates-based threshold.

Exactly specified Eq. (44) fixed-spectrum operator family, N=100, M={1,25},
300 independent trajectory pairs per training realization, test length 1200.
Chronological 80/20 CV performed *within each training trajectory* and a
relative, trace-normalized ridge parameter is selected (revision R2.2).
Population reference is M2(true F), following the submitted Fig.5; this is not
an extraction-independent planted-plane benchmark (reserved for new Fig N1).
The plane correlation is the actual max over plane directions, unlike the
legacy g/||g|| shortcut. Invalid rate logarithms are reported missing, not clipped.
"""
from __future__ import annotations
import os,sys,json,time,hashlib,platform
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor,cho_solve
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from shared.nndr_synthetic_operators import fixed_spectrum_family,operator_at_kappa,original_multiplier_ratio
from shared.nndr_planes import m2_multistart,principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics,spectral_radius
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CFG=json.loads((HERE/'config.json').read_text())


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def simulate_trajectory(F,noise):
    """Return (T,N) states x1...xT with x0=0."""
    t,n=noise.shape
    X=np.empty((t,n),float)
    prev=np.zeros(n)
    for i in range(t):
        prev=F@prev+noise[i]
        X[i]=prev
    return X


def trajectories_to_pairs(trajectories):
    T,N=trajectories.shape[1:]
    X=np.concatenate([np.concatenate([np.zeros((1,N)),xx[:-1]],axis=0) for xx in trajectories],axis=0)
    Y=np.concatenate([xx for xx in trajectories],axis=0)
    return X,Y


def ridge_from_pairs(X,Y,nu_rel):
    """Input pairs row-major (n_pairs,N); same relative ridge definition throughout."""
    n=X.shape[1]
    G=X.T@X
    lam=float(nu_rel)*float(np.trace(G))/n
    coef=np.linalg.solve(G+lam*np.eye(n), X.T@Y)
    return coef.T,lam


def choose_ridge(trajectories,grid,frac=0.8):
    """Hold out final 20% of each trajectory; never cross trajectory boundaries."""
    T=trajectories.shape[1];cut=int(T*frac)
    xx0=np.concatenate([np.concatenate([np.zeros((1,trajectories.shape[-1])),xx[:-1]],axis=0) for xx in trajectories],axis=0)
    yy0=np.concatenate([xx for xx in trajectories],axis=0)
    ptr=np.arange(T)[None,:]+T*np.arange(len(trajectories))[:,None]
    tr=ptr[:,:cut].ravel();val=ptr[:,cut:].ravel()
    Xtr,Ytr=xx0[tr],yy0[tr]
    Xv,Yv=xx0[val],yy0[val]
    G=Xtr.T@Xtr;C=Xtr.T@Ytr
    scale=float(np.trace(G))/Xtr.shape[1]
    errors=[]
    for nu in grid:
        lam=float(nu)*scale
        pred=Xv @ np.linalg.solve(G+lam*np.eye(G.shape[0]),C)
        errors.append(float(np.mean((pred-Yv)**2)))
    j=int(np.argmin(errors))
    return float(grid[j]),float(errors[j]),np.asarray(errors)


def m1_legacy(F):
    """Reproduce M1 eigenbasis-SVD construction (defined for complex spectra too)."""
    eig,V=np.linalg.eig(F)
    U,ss,_=np.linalg.svd(V,full_matrices=False)
    z=U[:,-1]
    if np.iscomplexobj(z):
        z=z.real if np.linalg.norm(z.real)>=np.linalg.norm(z.imag) else z.imag
    v=np.asarray(z.real,float);v/=np.linalg.norm(v)
    u=F@v;u-=v*(v@u)
    if np.linalg.norm(u)<1e-13:
        return None
    u/=np.linalg.norm(u)
    return {'Q':np.column_stack([u,v]),'kappa_P':float(ss[0]/ss[-1])}


def m3_commutator(F):
    C=F@F.T-F.T@F
    e,V=np.linalg.eigh(C)
    if max(abs(e[0]),abs(e[-1]))<1e-13:
        return None
    return {'Q':np.column_stack([V[:,-1],V[:,0]]),'gap':float(min(e[-1]-e[-2],e[1]-e[0]))}


def max_plane_correlation(Q,states):
    """Maximum ABSOLUTE Pearson correlation with mean field over all plane directions.

    Multiple R = sqrt(cov(z,m)^T Cov(z)^{-1} cov(z,m) / var(m))
    is the exact maximizer (provided the covariance is nonsingular).
    """
    Z=np.asarray(states)@Q
    m=np.mean(states,axis=1)
    Zc=Z-Z.mean(axis=0);mc=m-m.mean()
    C=Zc.T@Zc
    c=Zc.T@mc
    if mc@mc<=1e-25:return float('nan')
    coef=np.linalg.pinv(C, rcond=1e-12)@c
    return float(np.sqrt(np.clip(c@coef/(mc@mc),0,1)))


def compute_case(F, Q, test_states):
    Gamma=Q.T@F@Q
    d=reduced_diagnostics(Gamma)
    return {'R':float(d['R_variance']),'R_sigma':float(d['R_sigma']),
            'K':float(d['K']),'rho_Gamma':float(d['rho']),
            'sigma_Gamma':float(d['sigma_max']),'status':d['status'],
            'plane_corr':max_plane_correlation(Q,test_states),
            'capture_fraction':float(d['sigma_max']/np.linalg.svd(F,compute_uv=False)[0])}


def plot(df):
    labels={'M1':'Eigenbasis M1','M2':'Optimization M2','M3':'Commutator M3'}
    color={'M1':'#377eb8','M2':'#d95f02','M3':'#138768'}
    marker={'M1':'s','M2':'o','M3':'^'}
    fig,ax=plt.subplots(2,2,figsize=(9.8,6.55),sharex=True)
    axes=ax.ravel();ref=df[(df.method=='TRUE')&(df.M==25)].sort_values('R_true')
    axes[0].plot(ref.R_true,ref.R_true,color='black',lw=1.25,label='M2 known-operator reference',zorder=2)
    axes[1].axhline(0,color='#444444',lw=.7)
    for M in [25,1]:
      for m in ['M1','M2','M3']:
        d=df[(df.M==M)&(df.method==m)].sort_values('R_true')
        kw={'color':color[m],'marker':marker[m],'ms':3.7 if M==25 else 3.0,
            'mfc':color[m] if M==25 else 'none',
            'lw':1.3 if M==25 else 0.0,
            'linestyle':'-' if M==25 else 'none',
            'alpha':1.0 if M==25 else 0.6,
            'label':labels[m] if M==25 else '_nolegend_'}
        axes[0].plot(d.R_true,d.R_hat,**kw)
        axes[1].plot(d.R_true,d.signed_error,**kw)
        axes[2].plot(d.R_true,d.principal_angle,**kw)
        axes[3].plot(d.R_true,d.c_plane,**kw)
    for i,a in enumerate(axes):
        a.axvline(1,color='#8b939b',linestyle='--',linewidth=.85)
        a.set_xscale('log');a.set_xlim(.01,None)
        a.tick_params(axis='both',direction='in',top=True,right=True,labelsize=8)
        a.grid(lw=.35,alpha=.15)
        a.text(-.115,1.04,'('+chr(97+i)+')',transform=a.transAxes,fontweight='bold',fontsize=10)
    axes[0].set_yscale('log');axes[0].set_ylim(bottom=1e-3)
    axes[1].set_yscale('symlog',linthresh=.5)
    axes[2].set_ylim(-2,93)
    axes[3].set_ylim(0,1.04)
    axes[0].set_ylabel(r'Inferred $R$ (rates-based)')
    axes[1].set_ylabel(r'$\widehat R-R_{\rm ref}$ (symlog)')
    axes[2].set_ylabel('Largest principal angle (degrees)')
    axes[3].set_ylabel(r'$c_{\rm plane}$ (exact maximum)')
    axes[2].set_xlabel(r'Known-operator $R_{\rm ref}$ (rates-based)')
    axes[3].set_xlabel(r'Known-operator $R_{\rm ref}$ (rates-based)')
    axes[0].legend(fontsize=7,frameon=False,loc='upper left')
    from matplotlib.lines import Line2D
    legend2=[Line2D([0],[0],marker='o',color='gray',label='M=25: filled, connected',markersize=4),
             Line2D([0],[0],marker='o',color='gray',linestyle='none',mfc='white',label='M=1: open',markersize=4)]
    axes[3].legend(handles=legend2,fontsize=7,frameon=False,loc='lower right')
    fig.tight_layout(pad=1.65,w_pad=1.5,h_pad=1.7)
    fig.savefig(OUT/'T01_fig05_revised_rates.pdf')
    fig.savefig(OUT/'T01_fig05_revised_rates.png',dpi=CFG['figure_dpi'])
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    start=time.time()
    conf=CFG;n=conf['N'];grid=conf['kappas']
    family=fixed_spectrum_family(n=n,rho=conf['rho'],delta_sigma=conf['delta_sigma'],seed=conf['family_seed'])
    rng=np.random.default_rng(conf['noise_train_seed'])
    bank=conf['sigma_noise']*rng.standard_normal((max(conf['M_values']),conf['T_train'],n))
    testbank=conf['sigma_noise']*np.random.default_rng(conf['noise_test_seed']).standard_normal((conf['T_test'],n))
    records=[];cv_records=[];legacy_rows=[]
    for ii,kappa in enumerate(grid):
        F=operator_at_kappa(family,kappa)
        ref=m2_multistart(F,seed=conf['m2_seed'],restarts=conf['m2_restarts'],iterations=conf['m2_iterations'])
        Qstar=ref['Q']
        states_test=simulate_trajectory(F,testbank)
        truth=compute_case(F,Qstar,states_test)
        old_R=original_multiplier_ratio(Qstar.T@F@Qstar)
        legacy_rows.append({'kappa':float(kappa),'R_legacy_multipliers':old_R,
                            'R_revised_rates':truth['R'],'revised_to_legacy_ratio':truth['R']/old_R})
        assert truth['status']=='ok',(kappa,truth)
        assert abs(spectral_radius(F)-conf['rho'])<1e-7
        trajectories=np.asarray([simulate_trajectory(F,noise) for noise in bank])
        for M in conf['M_values']:
            tr=trajectories[:M]
            X,Y=trajectories_to_pairs(tr)
            nu,cv_err,cv_grid=choose_ridge(tr,conf['cv_nu_rel'],conf['cv_train_frac'])
            Fhat,lam=ridge_from_pairs(X,Y,nu)
            ell=spectral_radius(Fhat)
            full_error=float(np.linalg.norm(Fhat-F,'fro')/np.linalg.norm(F,'fro'))
            cv_records.append({'kappa':kappa,'M':M,'nu_relative':nu,'lambda_absolute':lam,
                               'cv_mse':cv_err,'rho_fitted':ell,'F_rel_error':full_error})
            methods={'M1':m1_legacy(Fhat),
                     'M2':m2_multistart(Fhat,seed=conf['m2_seed'],restarts=conf['m2_restarts'],iterations=conf['m2_iterations']),
                     'M3':m3_commutator(Fhat)}
            for name,res in [('TRUE',{'Q':Qstar}),*methods.items()]:
                if res is None:
                    row={'method':name,'M':M,'kappa':kappa,'R_true':truth['R'],'R_hat':np.nan,'status':'plane_unavailable'}
                    records.append(row);continue
                Q=res['Q']
                if np.max(abs(Q.T@Q-np.eye(2)))>1e-8:raise AssertionError('Q not orthonormal')
                diag=compute_case(Fhat if name!='TRUE' else F,Q,states_test)
                if name=='TRUE':diag=truth
                a=float(principal_angle_deg(Q,Qstar))
                row={'method':name,'M':M,'kappa':kappa,'R_true':truth['R'],'R_sigma_true':truth['R_sigma'],
                     'R_hat':diag['R'],'R_sigma_hat':diag['R_sigma'],
                     'signed_error':diag['R']-truth['R'],
                     'principal_angle':a,'c_plane':diag['plane_corr'],
                     'status':diag['status'],'rho_fitted':ell,'F_rel_error':full_error,
                     'nu_relative':nu,'lambda_absolute':lam,
                     'K_hat':diag['K'],'rho_Gamma':diag['rho_Gamma'],
                     'sigma_Gamma':diag['sigma_Gamma'],'capture_fraction':diag['capture_fraction']}
                if name=='M2':
                    row['objective']=float(res['objective'])
                    row['restart_objective_spread']=float(np.max(res['restart_objectives'])-np.min(res['restart_objectives']))
                records.append(row)
                if diag['status']=='ok':
                    if not np.isfinite(diag['R']) or diag['R']<0:raise AssertionError('Invalid R')
                    if (diag['R_sigma']>1.0)!=(diag['sigma_Gamma']>1.0) and abs(diag['R_sigma']-1)>1e-7:
                        raise AssertionError('One-step threshold equivalence failed')
                if name!='TRUE' and diag['capture_fraction']>1+1e-8:
                    raise AssertionError('Compression gains exceed full operator')
        if ii%4==0 or ii==len(grid)-1:
            print(f'[{ii+1:02d}/{len(grid)}] kappa={kappa:.4g}; reference R={truth["R"]:.5g}; seconds={time.time()-start:.1f}',flush=True)
    df=pd.DataFrame(records);cv=pd.DataFrame(cv_records)
    df.to_csv(OUT/'T01_fig05_full_results.csv',index=False)
    cv.to_csv(OUT/'T01_fig05_ridge_cv.csv',index=False)
    pd.DataFrame(legacy_rows).to_csv(OUT/'T01_fig05_known_operator_old_vs_new.csv',index=False)
    plot(df)
    summ={
      'experiment':'T01 original manuscript Figure 5 updated rates-based R',
      'N':n,'M_values':conf['M_values'],'T_train':conf['T_train'],'T_test':conf['T_test'],
      'n_kappas':len(grid),'n_trajectories_fit':int(len(grid)*max(conf['M_values'])),
      'n_method_records':int(len(df)),'n_invalid_R':int(((df.method!='TRUE') & (df.status!='ok')).sum()),
      'invalid_status_counts':{str(k):int(v) for k,v in df[(df.method!='TRUE') & (df.status!='ok')].status.value_counts().items()},
      'R_reference_range':[float(df.R_true.min()),float(df.R_true.max())],
      'elapsed_sec':round(time.time()-start,2),
      'config_sha256':sha256(HERE/'config.json'),
      'script_sha256':sha256(HERE/'run.py'),
      'notes':['M2(true F) reference is extraction-dependent; not an independently planted physical plane.',
               'Heldout chronological CV per trajectory; trace-normalized ridge.',
               'Exact plane maximal correlation is multiple correlation, not raw covariance direction.',
               'No forced stable rescaling; status for unsupported Γ is preserved.']
    }
    for M in conf['M_values']:
        s=df[(df.M==M)&(df.method!='TRUE')]
        for method in ['M1','M2','M3']:
            w=s[s.method==method]
            summ[f'M{M}_{method}_median_principal_angle_deg']=float(w.principal_angle.median())
            summ[f'M{M}_{method}_median_abs_R_error']=float(w.signed_error.abs().median())
            summ[f'M{M}_{method}_valid_rate_fraction']=float((w.status=='ok').mean())
            summ[f'M{M}_{method}_median_c_plane']=float(w.c_plane.median())
    (OUT/'T01_fig05_results.json').write_text(json.dumps(summ,indent=2))
    report=['# T01 Fig. 5 — fixed operator family, revised rates-based R','',
     'The simulation holds the physical operator construction and the 29 historical κ settings fixed, with the original M=1/M=25 data budgets. The resulting R_ref is computed with the *rates* rather than with the *multipliers*.',
     '','## Scientific limits','',
     '- Reference Q_ref = M2(F_true), as in submitted Fig. 5. The reported M2 plane error measures operator-estimation error, not superiority of the extraction rule. That question is reserved for T02/N1.',
     '- Chronological 80/20 ridge validation, relative trace-normalized ridge, 12-start M2, and mathematically exact maximum correlation replace the legacy implementations.',
     '- M1 and M2 may produce negative-real reduced multipliers (or unstable compressions) from M=1 fits; unsupported rates are not relabelled by clipping.',
     '- The corrected R_ref=1 crossing moves to smaller κ; see old-versus-new reference table. The original operators are unchanged.',
     '- All plotted results use a single, common test realization for each operator and a paired noise bank across κ and M. Independent across-κ claims are not made.',
     '','## Results','',f'- κ points: {len(grid)}; N={n}; {summ["n_method_records"]} total result rows.',
     f'- Reference R range: {summ["R_reference_range"]}.',
     f'- Invalid reduced rate cases: {summ["n_invalid_R"]}, confined to M=1; status counts {summ["invalid_status_counts"]}.',
     '','|M|Method|Median largest principal angle (deg)|Median abs R error|Valid rates fraction|Median c_plane|','|---|---|---:|---:|---:|---:|']
    for M in conf['M_values']:
      for method in ['M1','M2','M3']:
        k=f'M{M}_{method}'
        report.append(f'|{M}|{method}|{summ[k+"_median_principal_angle_deg"]:.3f}|{summ[k+"_median_abs_R_error"]:.3f}|{summ[k+"_valid_rate_fraction"]:.3f}|{summ[k+"_median_c_plane"]:.3f}|')
    report += ['',f'Runtime: {summ["elapsed_sec"]:.1f} s.','',
    'Do not claim method superiority here: independent planted-plane tests and snapshot/SVD comparators remain T02.']
    (OUT/'T01_fig05_report.md').write_text('\n'.join(report)+'\n')
    cap=r'''\caption{Finite-data inference on a fixed-spectrum non-normal VAR(1) family. The known operator has $N=100$, $\rho(F)=0.95$, and isotropic Gaussian innovations of standard deviation $0.05$. At each of the 29 historical coupling values, $M\in\{1,25\}$ independent trajectories of 300 one-step transitions are generated and a ridge operator is fitted, with its trace-normalized regularization chosen by a chronological 80/20 split within each trajectory. Filled linked symbols denote $M=25$; open symbols denote $M=1$. The reference $Q_{\rm ref}$ is M2 applied to the known $F$ (and thus tests estimation accuracy, not extraction-rule optimality). The horizontal axis and (a) the inferred ratio use the continuous-time-rate variance-response threshold $R=K/K_c$, with the exact one-step diagnostic $R_\sigma$ retained in the data. (b) Signed ratio errors, displayed on a symmetric-log scale. (c) Largest principal angle to $Q_{\rm ref}$. (d) The exact plane-invariant maximum absolute Pearson correlation between the test mean field and any direction in the inferred plane. Unsupported reduced logarithms are not assigned clipped ratios.}'''
    (HERE/'Fig05_revised_caption.tex').write_text(cap+'\n')
    (OUT/'T01_fig05_manifest.json').write_text(json.dumps({'results_sha256':sha256(OUT/'T01_fig05_full_results.csv'),
      'pdf_sha256':sha256(OUT/'T01_fig05_revised_rates.pdf'),
      'png_sha256':sha256(OUT/'T01_fig05_revised_rates.png'),**summ},indent=2))
    print(json.dumps(summ,indent=2),flush=True)

if __name__=='__main__':main()
