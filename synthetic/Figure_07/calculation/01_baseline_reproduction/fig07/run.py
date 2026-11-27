#!/usr/bin/env python3
"""T01 / submitted Fig. 7: single-record moving-window tracking, revised rates R.

Preserve: N=100, M=1, T=600, W=80, step=10, kappa 1 -> 200, rho=0.95,
noise std=.05 and the Eq.44 operator family.

Revisions vs legacy: principal matrix-log rates-based R; multi-start M2;
chronologically held-out, trace-relative ridge CV; no spectrum clipping,
no spectral stabilization; full invalidity and ridge diagnostics recorded.

Reference is M2 on the arithmetic mean of the TRUE operators in each
window, per submitted Eq. 49. This is an extraction-dependent reference,
not an independent planted physical plane (T10 tests that separately).
"""
from __future__ import annotations
import os, sys, time, json, hashlib, platform
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from shared.nndr_synthetic_operators import fixed_spectrum_family, operator_at_kappa
from shared.nndr_thresholds import reduced_diagnostics, spectral_radius
from shared.nndr_planes import m2_multistart, principal_angle_deg
HERE=Path(__file__).resolve().parent; OUT=HERE/'outputs'
CONFIG=json.loads((HERE/'config.json').read_text())


def make_schedule(T, start=1., stop=200.):
    return np.exp(np.linspace(np.log(start),np.log(stop),int(T)))


def simulate_rotated(fam, kappas, sigma_noise, seed):
    """Simulate exactly in U coordinates, avoiding 600 repeated matrix products.

    Since U is orthogonal and innovations are isotropic, this is equivalent
    to simulating x_{t+1}=F_t x_t+eps_t in physical coordinates.
    The independent U-coordinate innovations are the declared random source.
    """
    U=fam['U']; B=fam['B']; s0=fam['scales']; N=fam['n']; T=len(kappas)
    rng=np.random.default_rng(seed)
    eps=sigma_noise*rng.standard_normal((T,N))
    Z=np.zeros((T+1,N)); z=np.zeros(N)
    for t,k in enumerate(kappas):
        s=np.r_[s0,1./k]
        z=s*(B@(z/s))+eps[t]
        Z[t+1]=z
    return Z@U.T,eps


def exact_window_average(fam, kappa_window):
    """Exact time-average of F(kappa): its last column scales with kappa
    and last row scales with 1/kappa. Average of F != F(mean kappa).
    """
    U=fam['U']; B=fam['B']; s0=fam['scales']; N=fam['n']
    A=(s0[:,None]*B[:-1,:-1])/s0[None,:]
    top=np.mean(kappa_window)*s0*B[:-1,-1]
    left=np.mean(1./kappa_window)*B[-1,:-1]/s0
    H=np.empty((N,N),float)
    H[:-1,:-1]=A; H[:-1,-1]=top; H[-1,:-1]=left; H[-1,-1]=B[-1,-1]
    return U@H@U.T


def ridge_fit_pairs(x,y,nu_rel):
    n=x.shape[1]; G=x.T@x; C=x.T@y
    lam=float(nu_rel)*float(np.trace(G))/n
    return np.linalg.solve(G+lam*np.eye(n),C).T,float(lam)


def chronological_cv(x,y,relgrid,frac=.8):
    """Hold out LAST 20% of a single causal window; no shuffled pairs."""
    t=len(x); n=x.shape[1]; cut=max(2,min(t-2,int(t*frac)))
    xt,yt=x[:cut],y[:cut]; xv,yv=x[cut:],y[cut:]
    G=xt.T@xt; C=xt.T@yt; scale=float(np.trace(G))/n
    all_mse=[]
    for nu in relgrid:
        F=np.linalg.solve(G+float(nu)*scale*np.eye(n),C).T
        all_mse.append(float(np.mean((xv@F.T-yv)**2)))
    idx=int(np.argmin(all_mse))
    return float(relgrid[idx]),float(all_mse[idx]),np.asarray(all_mse)


def m3_commutator(F):
    C=F@F.T-F.T@F
    w,V=np.linalg.eigh(C)
    if max(abs(w[0]),abs(w[-1]))<1e-13:
        return None
    return {'Q':np.column_stack((V[:,-1],V[:,0])),'commutator_gap':float(min(w[-1]-w[-2],w[1]-w[0]))}


def diag(F,Q):
    G=Q.T@F@Q
    d=reduced_diagnostics(G)
    return {'R':float(d['R_variance']),'R_sigma':float(d['R_sigma']),
            'K':float(d['K']),'status':d['status'],
            'rhoG':float(d['rho']),'sigmaG':float(d['sigma_max']),
            'capture':float(d['sigma_max']/np.linalg.svd(F,compute_uv=False)[0]),
            'complex_pair':int(d.get('spectrum_type')=='complex')}


def compute(config=CONFIG,verbose=True):
    conf=config; n=conf['N']; T=conf['T']; W=conf['W'];step=conf['step']
    fam=fixed_spectrum_family(n=n,rho=conf['rho'],delta_sigma=conf['delta_sigma'],seed=conf['family_seed'])
    kappas=make_schedule(T,conf['kappa_start'],conf['kappa_end'])
    X,eps=simulate_rotated(fam,kappas,conf['sigma_noise'],conf['noise_seed'])
    ends=np.arange(W,T+1,step,dtype=int)
    records=[]; cv_rows=[]
    for wi,end in enumerate(ends):
        start=end-W
        x=X[start:end]; y=X[start+1:end+1]
        Fbar=exact_window_average(fam,kappas[start:end])
        nu,cv_error,errs=chronological_cv(x,y,conf['ridge_relative_grid'],conf['cv_train_fraction'])
        Fhat,lam=ridge_fit_pairs(x,y,nu)
        Fbar_ref=m2_multistart(Fbar,seed=conf['m2_seed'],restarts=conf['reference_m2_restarts'],iterations=conf['m2_iterations'])
        Qref=Fbar_ref['Q']; ref=diag(Fbar,Qref)
        methods={'M2':m2_multistart(Fhat,seed=conf['m2_seed']+end,restarts=conf['estimated_m2_restarts'],iterations=conf['m2_iterations']),
                 'M3':m3_commutator(Fhat)}
        true_fullrho=spectral_radius(Fbar); fitted_fullrho=spectral_radius(Fhat)
        if not np.allclose(Fbar,exact_window_average(fam,kappas[start:end]),rtol=1e-11,atol=1e-11):
            raise AssertionError('non-deterministic reference')
        for method,v in methods.items():
            d=diag(Fhat,v['Q']) if v is not None else dict(R=np.nan,R_sigma=np.nan,K=np.nan,status='no_commutator',rhoG=np.nan,sigmaG=np.nan,capture=np.nan,complex_pair=0)
            rec={'method':method,'t_end':int(end),'window_start':int(start),'pairs':int(W),
                 'kappa_end':float(kappas[end-1]),'kappa_window_mean':float(np.mean(kappas[start:end])),
                 'R_reference':ref['R'],'R_sigma_reference':ref['R_sigma'],'reference_status':ref['status'],
                 'R_hat':d['R'],'R_sigma_hat':d['R_sigma'],'K_hat':d['K'],'R_status':d['status'],
                 'signed_error':d['R']-ref['R'],'abs_error':abs(d['R']-ref['R']),
                 'principal_angle_deg':principal_angle_deg(Qref,v['Q']) if v is not None else np.nan,
                 'rho_F_true_mean':true_fullrho,'rho_Fhat':fitted_fullrho,
                 'rho_Gamma':d['rhoG'],'sigma_Gamma':d['sigmaG'],'capture_fraction':d['capture'],
                 'rel_error_Fhat_to_Fbar':float(np.linalg.norm(Fhat-Fbar,ord='fro')/np.linalg.norm(Fbar,ord='fro')),
                 'nu_relative':nu,'ridge_absolute':lam,'cv_mse':cv_error,
                 'rank_X':int(np.linalg.matrix_rank(x)),
                 'M2_objective_spread':float(np.ptp(v['restart_objectives'])) if method=='M2' and v is not None else np.nan,
                 'complex_pair':d['complex_pair']}
            records.append(rec)
        cv_rows.append({'t_end':end,'nu_relative':nu,'cv_mse':cv_error,'full_rho':fitted_fullrho})
        if verbose and (wi%10==0 or wi==len(ends)-1):
            a,b=records[-2:]
            print(f"window {wi+1}/{len(ends)} end={end} Rref={ref['R']:.4g} M2={a['R_hat']:.4g}({a['R_status']}) M3={b['R_hat']:.4g}({b['R_status']})",flush=True)
    return pd.DataFrame(records),pd.DataFrame(cv_rows),{'kappas':kappas,'states':X,'innovations_rotated':eps,'window_ends':ends}


def plot(df,path):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'mathtext.fontset':'dejavuserif','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(10.0,6.6),sharex=True)
    a,b,c,d=axs.ravel(); m2=df[df.method=='M2'].sort_values('t_end');m3=df[df.method=='M3'].sort_values('t_end')
    tend=m2.t_end.to_numpy()
    a.plot(tend,m2.kappa_end,color='0.45',lw=1.5,label='Instantaneous $\\kappa(t)$')
    a.plot(tend,m2.kappa_window_mean,color='#333333',ls='--',lw=1.6,label='Window mean')
    a.set_ylabel(r'Anisotropy parameter $\kappa$');a.legend(loc='upper left',fontsize=8,frameon=False)
    b.plot(tend,m2.R_reference,color='black',lw=1.7,marker='o',ms=2.3,label='Window-matched reference')
    b.axhline(1,color='0.5',ls=':',lw=1)
    b.set_ylabel(r'$R_{\mathrm{ref},W}$ (rates-based)')
    b.legend(loc='upper left',fontsize=8,frameon=False)
    c.plot(tend,m2.R_reference,color='black',lw=1.4,label='Reference')
    c.plot(tend,m2.R_hat,color='#d95f02',marker='o',ms=3,lw=1.05,label='M2 inferred')
    c.plot(tend,m3.R_hat,color='#008f66',marker='^',ms=3,lw=1.05,label='M3 inferred')
    c.axhline(1,color='0.5',ls=':',lw=1)
    c.set_ylabel(r'Inferred $R$ (rates-based)'); c.legend(loc='upper left',fontsize=8,frameon=False)
    d.axhline(0,color='0.5',lw=.8)
    d.plot(tend,m2.signed_error,color='#d95f02',marker='o',ms=3,lw=1,label='M2')
    d.plot(tend,m3.signed_error,color='#008f66',marker='^',ms=3,lw=1,label='M3')
    d.set_ylabel(r'$\widehat R-R_{\mathrm{ref},W}$')
    d.legend(loc='upper left',fontsize=8,frameon=False)
    for i,ax in enumerate(axs.ravel()):
        ax.text(-.095,1.04,'('+chr(97+i)+')',transform=ax.transAxes,fontweight='bold',fontsize=11)
        ax.grid(alpha=.18,lw=.45)
        ax.tick_params(direction='in',top=True,right=True)
        ax.set_xlim(tend.min(),tend.max())
    for ax in axs[1]:ax.set_xlabel(r'Window end $t_{\rm end}$')
    fig.tight_layout(pad=1.45,w_pad=1.4,h_pad=1.5)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=CONFIG['figure_dpi'],bbox_inches='tight')
    plt.close(fig)


def plot_diagnostics(df,path):
    fig,axes=plt.subplots(2,2,figsize=(10,6.3),sharex=True)
    for method,color in [('M2','#d95f02'),('M3','#008f66')]:
        d=df[df.method==method].sort_values('t_end')
        axes[0,0].plot(d.t_end,d.principal_angle_deg,color=color,label=method,marker='.',ms=4)
        axes[0,1].plot(d.t_end,d.rho_Fhat,color=color,label=method)
        axes[1,0].plot(d.t_end,d.capture_fraction,color=color,label=method)
        axes[1,1].plot(d.t_end,d.R_sigma_hat,color=color,label=method)
    axes[0,0].set_ylabel('Largest principal angle (degrees)')
    axes[0,1].set_ylabel(r'$\rho(\widehat F)$')
    axes[1,0].set_ylabel('One-step gain capture fraction')
    axes[1,1].set_ylabel(r'Exact one-step ratio $R_\sigma$')
    axes[0,1].axhline(1,color='0.5',ls='--',lw=.8)
    axes[1,1].axhline(1,color='0.5',ls='--',lw=.8)
    for ax in axes.ravel():ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    for ax in axes[-1]:ax.set_xlabel('Window end time')
    fig.tight_layout()
    fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.png'),dpi=CONFIG['figure_dpi']);plt.close(fig)


def make_summary(df):
    stats={'total_windows':int(df.t_end.nunique()),'total_method_rows':int(len(df)),'methods':{}}
    for method,g in df.groupby('method'):
        ok=(g.R_status=='ok') & np.isfinite(g.R_hat)
        good=g[ok]
        stats['methods'][method]={'valid_rate_windows':int(ok.sum()),'invalid_rate_windows':int((~ok).sum()),
                                  'status_counts':{str(k):int(v) for k,v in g.R_status.value_counts().items()},
                                  'median_abs_tracking_error_valid':float(good.abs_error.median()) if len(good) else None,
                                  'median_angle_deg_all':float(g.principal_angle_deg.median()),
                                  'median_full_operator_relative_error':float(g.rel_error_Fhat_to_Fbar.median()),
                                  'amplifying_valid_windows':int(((g.R_hat>1)&ok).sum()),
                                  'full_operator_unstable_windows':int((g.rho_Fhat>=1).sum()),
                                  'median_capture_fraction':float(g.capture_fraction.median())}
    return stats


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time()
    df,cv,raw=compute(CONFIG,verbose=True)
    df.to_csv(OUT/'T01_fig07_window_results.csv',index=False)
    cv.to_csv(OUT/'T01_fig07_cv_results.csv',index=False)
    np.savez_compressed(OUT/'T01_fig07_trajectory.npz',**raw)
    plot(df,OUT/'T01_fig07_timevarying_rates')
    plot_diagnostics(df,OUT/'T01_fig07_diagnostics')
    summary=make_summary(df)
    summary.update({'seconds':time.time()-start,'numpy':np.__version__,'python':platform.python_version(),'config':CONFIG})
    (OUT/'T01_fig07_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='config'},indent=2),flush=True)
    return summary

if __name__=='__main__':main()
