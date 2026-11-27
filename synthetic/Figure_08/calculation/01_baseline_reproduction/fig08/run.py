#!/usr/bin/env python3
"""T01 submitted Figure 8: known embedded plane, broad spectral background.

One-command reproducibility: `python run.py` from any directory.
Frozen manuscript geometry: N=80, M=18, T=220, noise=.035, abs ridge=1e-4,
beta in [.25,.75,1.25,1.75,2.25], eight seeds, target R in [1,3,10,30].

Corrections: *rates* threshold Kc for fixed multipliers .945/.455, using the
supplied shared diagnostics; zero initial transients are kept. No forced
stability and no deletion of invalid fitted reduced eigenvalues. The planted
plane is an independent reference.  M2 uses multistart; M3 is a symmetric
commutator eigenspace. A random orthogonal change of coordinates is performed
for each operator; ridge is orthogonally equivariant, allowing simulation and
fitting in that basis without loss of generality.
"""
from __future__ import annotations
import os,sys,json,time,platform,hashlib
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared.nndr_thresholds import reduced_diagnostics, kc_variance_real, spectral_radius
from shared.nndr_planes import m2_multistart,principal_angle_deg
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CONFIG=json.loads((HERE/'config.json').read_text())

def embedded_multipliers(cfg=CONFIG):
    m=float(cfg['embedded_multiplier_mean']); d=float(cfg['embedded_multiplier_split'])
    a,b=m*(1+d),m*(1-d)
    assert 0<b<a<1
    return a,b

def threshold_values(cfg=CONFIG):
    a,b=embedded_multipliers(cfg)
    rates=sorted([-np.log(a),-np.log(b)])
    kc=kc_variance_real(*rates)
    delta_old=(a-b)/(a+b)
    kcold=np.sqrt(np.sqrt(1-delta_old**2)/(1-np.sqrt(1-delta_old**2)))
    return {'a':a,'b':b,'kc_rates':kc,'kc_old':kcold,
            'delta_rate':(rates[1]-rates[0])/(rates[1]+rates[0]),
            'delta_multiplier':delta_old}

def embedded_block(r, cfg=CONFIG):
    v=threshold_values(cfg)
    k=float(r)*v['kc_rates']
    off=k*(v['a']-v['b'])
    return np.array([[v['a'],off],[0.,v['b']]],float)

def background(beta,n=80, cfg=CONFIG):
    j=np.arange(1,int(n)-1,dtype=float)
    return float(cfg['background_scale'])*j**(-float(beta))

def orthogonal(n, seed):
    rng=np.random.default_rng(seed)
    O,R=np.linalg.qr(rng.standard_normal((n,n)))
    signs=np.where(np.diag(R)<0,-1.,1.)
    return O*signs

def operator(beta,r,seed_index,cfg=CONFIG):
    n=cfg['N']; B=embedded_block(r,cfg)
    diag=background(beta,n,cfg)
    O=orthogonal(n,cfg['seed_family_base']+100*int(seed_index)+int(100*beta))
    Fz=np.zeros((n,n),float)
    Fz[:2,:2]=B
    Fz[np.arange(2,n),np.arange(2,n)]=diag
    F=O@Fz@O.T
    Qstar=O[:,:2]
    # No extraction rule used in the reference plane, defined by O.
    Rstar=reduced_diagnostics(Qstar.T@F@Qstar)
    if Rstar['status']!='ok' or abs(Rstar['R_variance']-r)>1e-10:
        raise AssertionError('Embedded reference ratio calibration failed')
    return Fz,F,O,Qstar,Rstar

def simulate_z(Fz, ntraj, T, sd, seed):
    """Vectorized over independent trajectories, time chronological."""
    rng=np.random.default_rng(seed)
    n=Fz.shape[0]
    X=np.zeros((ntraj,T,n));Y=np.zeros_like(X)
    z=np.zeros((ntraj,n))
    a,b=Fz[0,0],Fz[1,1]
    c=Fz[0,1]
    rest=np.diag(Fz)[2:]
    for t in range(T):
        X[:,t,:]=z
        eta=sd*rng.standard_normal((ntraj,n))
        znew=np.empty_like(z)
        znew[:,0]=a*z[:,0]+c*z[:,1]+eta[:,0]
        znew[:,1]=b*z[:,1]+eta[:,1]
        znew[:,2:]=rest[None,:]*z[:,2:]+eta[:,2:]
        Y[:,t,:]=znew
        z=znew
    return X.reshape(-1,n),Y.reshape(-1,n)

def ridge_fit(X,Y,lam):
    S=X.T@X
    # Fixed absolute ridge from submitted experiment, no normalization.
    return np.linalg.solve(S+lam*np.eye(X.shape[1]),X.T@Y).T

def m3_plane(F):
    C=F@F.T-F.T@F
    w,V=np.linalg.eigh(C)
    return V[:,[np.argmax(w),np.argmin(w)]], w

def save_plots(df):
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    fig,aa=plt.subplots(2,2,figsize=(7.3,5.7));ax=aa.flat
    cfg=CONFIG
    for beta in cfg['beta_values']:
        Fz,_,O,_,_=operator(beta,3.,0)
        s=np.linalg.svd(Fz,compute_uv=False)
        ax[0].loglog(np.arange(1,len(s)+1),s/s[0],lw=1.15,label=rf'$\beta={beta:g}$')
    ax[0].set(xlabel='Singular-value index $j$',ylabel=r'$\sigma_j(F)/\sigma_1(F)$')
    ax[0].legend(fontsize=7,frameon=False)
    styles={'M2':('o','-'),'M3':('^','--')}
    summ=df.groupby(['method','beta']).agg(angle=('angle_deg','mean'),angle_sd=('angle_deg','std'),
         abslog=('abs_log10_error','mean'),abslog_sd=('abs_log10_error','std'),valid=('R_hat','count')).reset_index()
    for method in ['M2','M3']:
        d=summ[summ.method==method]
        mark,ls=styles[method]
        ax[1].errorbar(d.beta,d.angle,yerr=d.angle_sd/np.sqrt(32),marker=mark,ls=ls,
                       capsize=2,label=method)
        ax[2].errorbar(d.beta,d.abslog,yerr=d.abslog_sd/np.sqrt(d.valid),marker=mark,ls=ls,capsize=2,label=method)
        dd=df[(df.method==method)&df.R_hat.notna()]
        ax[3].scatter(dd.R_star,dd.R_hat,marker=mark,alpha=.48,s=14,label=method)
    ax[1].set(xlabel=r'Background exponent $\beta$',ylabel=r'Largest principal angle (deg)')
    ax[2].set(xlabel=r'Background exponent $\beta$',ylabel=r'$|\log_{10}(\widehat R/R_\star)|$')
    ax[3].plot([.65,50],[.65,50],'k:',lw=1,label='identity')
    ax[3].set(xscale='log',yscale='log',xlabel=r'Planted $R_\star$',ylabel=r'Inferred $\widehat R$')
    ax[3].set_xlim(.65,50);ax[3].set_ylim(.5,70)
    ax[1].legend(fontsize=8,frameon=False)
    for i,axis in enumerate(ax):
        axis.text(-.15,1.035,'('+chr(ord('a')+i)+')',fontweight='bold',transform=axis.transAxes,fontsize=10)
        axis.tick_params(direction='in',top=True,right=True)
    fig.tight_layout(pad=1.2,h_pad=1.4)
    fig.savefig(OUT/'T01_fig08_broad_spectrum_revised.pdf',bbox_inches='tight')
    fig.savefig(OUT/'T01_fig08_broad_spectrum_revised.png',dpi=260,bbox_inches='tight')
    plt.close(fig)

def compute(cfg=CONFIG):
    rows=[]; start=time.perf_counter()
    for beta in cfg['beta_values']:
        for target in cfg['R_targets_rates']:
            for seed in cfg['seed_indices']:
                Fz,F,O,Qstar,star=operator(beta,target,seed,cfg)
                data_seed=cfg['seed_data_base']+1000*int(seed)+int(100*beta)+int(target)
                X,Y=simulate_z(Fz,cfg['M'],cfg['T_train'],cfg['innovation_sd'],data_seed)
                Fzhat=ridge_fit(X,Y,cfg['ridge_abs'])
                # Equivalent to direct fit of X@O.T and Y@O.T by orthogonal invariance.
                Fhat=O@Fzhat@O.T
                norms=np.linalg.svd(Fhat,compute_uv=False)
                src_error=float(np.linalg.norm(Fhat-F,'fro')/np.linalg.norm(F,'fro'))
                for method in ['M2','M3']:
                    if method=='M2':
                        m=m2_multistart(Fhat,seed=seed,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])
                        Q=m['Q']; spread=np.ptp(m['restart_objectives']);objective=m['objective']
                    else:
                        Q,eigsC=m3_plane(Fhat);spread=np.nan;objective=np.nan
                    G=Q.T@Fhat@Q
                    est=reduced_diagnostics(G)
                    Rh=float(est['R_variance']); valid=est['status']=='ok' and np.isfinite(Rh) and Rh>0
                    rows.append({'beta':float(beta),'R_target':float(target),'R_star':float(star['R_variance']),
                       'seed_index':int(seed),'data_seed':int(data_seed),'method':method,
                       'angle_deg':principal_angle_deg(Q,Qstar),'R_hat':Rh if valid else np.nan,
                       'abs_log10_error':abs(np.log10(Rh/target)) if valid else np.nan,
                       'R_error':Rh-target if valid else np.nan,'R_status':est['status'],
                       'R_sigma_hat':float(est['R_sigma']),'R_sigma_star':float(star['R_sigma']),
                       'K_hat':float(est['K']),'K_star':float(star['K']),
                       'rhoGamma':float(est['rho']),'rhoFhat':spectral_radius(Fhat),
                       'sigmaGamma':float(est['sigma_max']),'sigmaFhat':float(norms[0]),
                       'capture_fraction':float(est['sigma_max']/norms[0]),
                       'full_relative_F_error':src_error,'m2_objective':objective,
                       'm2_restart_objective_range':spread})
        print(f'Finished beta={beta:g}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    return pd.DataFrame(rows),time.perf_counter()-start

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    v=threshold_values()
    print('Threshold convention:',v,flush=True)
    d,t=compute()
    d.to_csv(OUT/'T01_fig08_full_results.csv',index=False)
    save_plots(d)
    summary={'config':CONFIG,'thresholds':v,'elapsed_seconds':t,'n_rows':len(d),
        'n_unique_systems':len(d)//2,'invalid':int(d.R_hat.isna().sum()),
        'method_summaries':{},'beta_summaries':[]}
    for method,sub in d.groupby('method'):
        summary['method_summaries'][method]={'mean_angle_deg':float(sub.angle_deg.mean()),
           'median_angle_deg':float(sub.angle_deg.median()),
           'mean_absolute_log10_R_error':float(sub.abs_log10_error.mean()),
           'median_absolute_log10_R_error':float(sub.abs_log10_error.median()),
           'valid_count':int(sub.R_hat.notna().sum()),
           'median_capture_fraction':float(sub.capture_fraction.median()),
           'median_operator_relative_F_error':float(sub.full_relative_F_error.median())}
    for (method,beta),sub in d.groupby(['method','beta']):
        summary['beta_summaries'].append({'method':method,'beta':float(beta),
           'mean_angle_deg':float(sub.angle_deg.mean()),'mean_abs_log10_error':float(sub.abs_log10_error.mean()),
           'n_valid':int(sub.R_hat.notna().sum())})
    (OUT/'T01_fig08_summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'T01_fig08_run_manifest.json').write_text(json.dumps({'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
          'timing_seconds':t},indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k in ('n_rows','invalid','method_summaries','elapsed_seconds')},indent=2))

if __name__=='__main__':main()
