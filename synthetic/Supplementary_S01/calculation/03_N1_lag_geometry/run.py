#!/usr/bin/env python3
"""T03 N1c: rates-based lag invariance, independent spectrum/geometry changes,
and complex-rotation aliasing. Compare population F^tau with direct C_tau C0^-1
fits. No assumption of exact invariance for N>2 after plane re-selection.
"""
from __future__ import annotations
import os,sys,json,time,hashlib,platform
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.nndr_planes import m2_multistart,principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CFG=json.loads((HERE/'config.json').read_text())


def real_generator(g1,g2,k):
    """Fixed eigenvector geometry S[[1,-k],[0,1]]; eigenvalues -g1,-g2."""
    if not (g2>g1>0 and k>=0):raise ValueError('gamma ordering / k')
    return np.array([[-g1,k*(g1-g2)],[0.,-g2]])


def operator(g1,g2,k,n=2,seed=0):
    A=real_generator(g1,g2,k)
    if n==2:return expm(A),np.eye(2)
    rng=np.random.default_rng(seed)
    Z,R=np.linalg.qr(rng.standard_normal((n,n)))
    Z=Z*np.where(np.diag(R)>=0,1.,-1.)
    Fz=np.diag(np.r_[np.zeros(2),CFG['reference_background_radius']*np.arange(1,n-1,dtype=float)**(-.5)])
    Fz[:2,:2]=expm(A)
    return Z@Fz@Z.T,Z[:,:2]


def simulate(F,M,T,seed,sigma):
    rng=np.random.default_rng(seed)
    n=F.shape[0];z=np.zeros((M,n));a=np.empty((M,T+1,n));a[:,0]=z
    for t in range(T):
        z=z@F.T+sigma*rng.standard_normal((M,n))
        a[:,t+1]=z
    return a


def ridge_lag(data,tau,alpha):
    XX=data[:,:-tau,:].reshape(-1,data.shape[-1]);YY=data[:,tau:,:].reshape(-1,data.shape[-1])
    S=XX.T@XX;lam=float(alpha)*np.trace(S)/S.shape[0]
    return np.linalg.solve(S+lam*np.eye(S.shape[0]),XX.T@YY).T,len(XX)


def leading_structures(F):
    u,s,vh=np.linalg.svd(F,full_matrices=False)
    return u[:,0],vh[0],float(s[0])


def axis_angle(u,v):
    return float(np.degrees(np.arccos(np.clip(abs(float(u@v)),-1,1))))


def complex_generator(gamma=.07,omega=.72,s=1.8):
    B=np.array([[-gamma,-omega],[omega,-gamma]])
    D=np.diag([s,1/s])
    return D@B@np.linalg.inv(D)


def complex_alias_sweep(cfg=CFG):
    A=complex_generator(cfg['complex_rate_gamma'],cfg['complex_frequency_omega'],cfg['complex_nonnormal_similarity'])
    rows=[]
    for lag in cfg['lag_values']:
        F=expm(A*lag)
        eig=np.linalg.eigvals(F)
        d=reduced_diagnostics(F,delta_t=lag)
        # Branch condition for principal Log, independently known true omega.
        angle=float(cfg['complex_frequency_omega']*lag)
        alias=bool(abs(angle)>np.pi)
        # logm generator may yield positive/negative imaginary frequency after alias
        apparent_omega=float(abs(np.angle(eig[0]))/lag)
        rows.append({'lag':lag,'true_omega':float(cfg['complex_frequency_omega']),
                     'apparent_omega':apparent_omega,'alias':alias,
                     'R':float(d['R_variance']),'R_sigma':float(d['R_sigma']),
                     'K':float(d['K']),'status':d['status']})
    return pd.DataFrame(rows)


def run(cfg=CFG):
    rows=[];begin=time.perf_counter()
    cases=[]
    for i,gg in enumerate(cfg['spectra_varying_rates']):cases.append(('vary_spectrum_fixed_geometry',i,float(gg[0]),float(gg[1]),float(cfg['geometry_fixed'])))
    for i,k in enumerate(cfg['geometry_varying']):cases.append(('vary_geometry_fixed_spectrum',i,*map(float,cfg['spectrum_fixed_rates']),float(k)))
    for family,index,g1,g2,k in cases:
        # Exact population F^lag and its 2x2 true generator are independent of seed.
        for n in [2,cfg['N_embedded']]:
          for replicate in range(cfg['replicates']):
            seed=cfg['seed_base']+10000*cases.index((family,index,g1,g2,k))+101*n+replicate
            F,Qstar=operator(g1,g2,k,n=n,seed=seed)
            inp1=leading_structures(F)[1]
            base_true=None
            for M in cfg['M_values']:
                # Paired: the M1 trajectory is the first of M25.
                if M==cfg['M_values'][0]:
                    data_all=simulate(F,max(cfg['M_values']),cfg['T_train'],seed+19,cfg['noise_sd'])
                data=data_all[:M]
                ref=ref_E1=None
                for lag in cfg['lag_values']:
                    Ft=np.linalg.matrix_power(F,lag)
                    # Population extraction rule: M2 is the operative extraction
                    # except in the entire 2x2 ambient state where any plane is full.
                    qknown=m2_multistart(Ft,seed=seed+lag,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])['Q']
                    dtrue=reduced_diagnostics(qknown.T@Ft@qknown,delta_t=lag)
                    if lag==cfg['lag_values'][0]:
                        base_true=qknown
                    fhat,pairs=ridge_lag(data,lag,cfg['ridge_relative'])
                    qhat=m2_multistart(fhat,seed=seed+13*lag+M,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])['Q']
                    dhat=reduced_diagnostics(qhat.T@fhat@qhat,delta_t=lag)
                    u,v,G=leading_structures(Ft)
                    rows.append({'family':family,'case_index':index,'gamma1':g1,'gamma2':g2,'kappa_eigenvectors':k,
                        'N':n,'M':M,'replicate':replicate,'seed':seed,'lag':lag,'n_pairs':pairs,
                        'K_pop':float(dtrue['K']),'R_pop':float(dtrue['R_variance']),
                        'R_sigma_pop':float(dtrue['R_sigma']), 'pop_status':dtrue['status'],
                        'pop_plane_to_lag1_deg':principal_angle_deg(qknown,base_true),
                        'pop_plane_to_planted_deg':principal_angle_deg(qknown,Qstar),
                        'snapshot_input_rotation_deg':axis_angle(v,inp1),
                        'snapshot_gain_true':G,'snapshot_overlap_true':abs(float(u@v)),
                        'K_hat':float(dhat['K']),
                        'R_hat':float(dhat['R_variance']) if dhat['status']=='ok' else np.nan,
                        'R_sigma_hat':float(dhat['R_sigma']),
                        'est_status':dhat['status'],
                        'estimated_plane_to_population_deg':principal_angle_deg(qhat,qknown),
                        'estimated_plane_to_planted_deg':principal_angle_deg(qhat,Qstar),
                        'relative_F_lag_error':float(np.linalg.norm(fhat-Ft,'fro')/np.linalg.norm(Ft,'fro'))})
          print(f'{family} case={index} N={n}, elapsed {time.perf_counter()-begin:.1f}s',flush=True)
    return pd.DataFrame(rows),complex_alias_sweep(cfg),time.perf_counter()-begin


def plot(df,com):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(2,2,figsize=(9.2,6.5))
    # For each 2x2 spectrum-variation case, lag rate-threshold invariant exactly.
    sub=df[(df.N==2)&(df.family=='vary_spectrum_fixed_geometry')&(df.M==25)&(df.replicate==0)]
    for i,part in sub.groupby('case_index'):
        ax[0,0].plot(part.lag,part.R_pop,marker='o',label=rf'$(\gamma_1,\gamma_2)=({part.gamma1.iloc[0]:.2f},{part.gamma2.iloc[0]:.2f})$')
    ax[0,0].set(xlabel=r'Lag $\tau$',ylabel=r'Population $R$ (rates-based)',title='Exact two-dimensional invariance')
    ax[0,0].legend(fontsize=7,frameon=False)
    sub=df[(df.N==40)&(df.M==25)&(df.family=='vary_spectrum_fixed_geometry')]
    for idx,part in sub.groupby('case_index'):
        mids=part.groupby('lag').agg({'R_pop':'median','R_hat':'median'})
        ax[0,1].plot(mids.index,mids.R_pop,marker='o',label=f'Case {idx}: population')
        ax[0,1].plot(mids.index,mids.R_hat,ls='--',alpha=.65,label=f'Case {idx}: inferred')
    ax[0,1].set(xlabel=r'Lag $\tau$',ylabel=r'$R$ after plane re-selection',title='Embedded N=40, M=25')
    ax[0,1].legend(fontsize=6.5,frameon=False,ncol=2)
    sub=df[(df.N==40)&(df.M==25)&(df.family=='vary_geometry_fixed_spectrum')]
    for idx,part in sub.groupby('case_index'):
        d=part.groupby('lag').snapshot_input_rotation_deg.median()
        ax[1,0].plot(d.index,d.values,marker='o',label=rf'$\kappa={part.kappa_eigenvectors.iloc[0]:g}$')
    ax[1,0].set(xlabel=r'Snapshot horizon $\tau$',ylabel=r'Input-direction rotation from $\tau=1$ (deg)',title='Horizon-dependent optimal structures')
    ax[1,0].legend(fontsize=7,frameon=False)
    ax[1,1].plot(com.lag,com.apparent_omega,'o-',label='Principal-log inferred')
    ax[1,1].axhline(com.true_omega.iloc[0],color='black',ls=':',label='True frequency')
    ax[1,1].axvline(np.pi/com.true_omega.iloc[0],color='gray',ls='--',label='Principal-branch boundary')
    ax[1,1].set(xlabel=r'Sampling interval $\tau$',ylabel=r'Angular frequency (per unit time)',title='Complex-eigenvalue alias at coarse sampling')
    ax[1,1].legend(fontsize=7,frameon=False)
    for j,a in enumerate(ax.flat):
        a.text(-.125,1.035,f'({"abcd"[j]})',transform=a.transAxes,fontweight='bold',fontsize=10)
        a.grid(alpha=.18)
    fig.tight_layout(pad=1.4,h_pad=1.5)
    for e in ['png','pdf']:
        fig.savefig(OUT/f'T03_FigN1c_lag_geometry.{e}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def report(df,complexrows,elapsed):
    z=df[(df.N==2)&(df.M==25)]
    true_max=z.groupby(['family','case_index','replicate']).R_pop.agg(lambda x:float(x.max()-x.min()))
    eb=df[(df.N==40)&(df.M==25)]
    inv=int(df.R_hat.isna().sum())
    pop_invalid=int(df.R_pop.isna().sum())
    print('2x2 max exact R range across lag',true_max.max(),'embedded median true plane change',eb.pop_plane_to_lag1_deg.median())
    lines=['# T03 N1c — sampling-horizon, geometry and spectrum audit','',
           f'- 2D population maximum within-case R span across lags: **{true_max.max():.3g}**.',
           f'- Embedded N=40, M=25: median reselected known-operator plane rotation from lag 1: **{eb.pop_plane_to_lag1_deg.median():.3f}°**.',
           f'- Population reduced ratio invalid/unadmissible: **{pop_invalid}/{len(df)}** rows (five distinct spectrum/geometry--lag cases, each repeated over four orthogonal rotations and two M conditions), preserved.',
           f'- Inferred ratio invalid/unadmissible: **{inv}/{len(df)}** fits, preserved.',
           f'- Exact complex rotation principal-log alias flags: **{complexrows.alias.sum()}/{len(complexrows)}** sampling intervals.',
           '', '## Interpretation',
           'For a fixed full two-dimensional real-rate generator, K and rates-based R are invariant under exact positive-time sampling F^tau provided the principal log remains on the correct branch. In N>2, M2 can reselect a different plane at different lags; such ratios need not be invariant.',
           'Finite-sample lag covariance fits C_tau C0^-1 are computed directly from paired trajectories for each tau, with a trace-normalized ridge. Their lag-1 and lag-8 data counts differ (explicit in CSV). No exact finite-data invariance is presumed.',
           'The two interventions vary eigenvalues holding eigenvectors fixed and eigenvectors holding eigenvalues fixed; all full systems have stable spectra. Some full-stable N=40 matrices nevertheless produce unstable 2D M2 compressions even with no estimation noise; their R_pop is undefined and must not be extrapolated.',
           'Snapshot direction rotation is assessed in the same physical system.',
           'The complex-alias control holds a nonnormal generator fixed but estimates frequency using the principal logarithm at tau=1,2,4,8. The tau=8 point crosses the known Nyquist branch limit; apparent rates and threshold interpretation can change.',
           '', '## Correctness warning',
           'Do not cite exact 2D invariance as a statement about arbitrary higher-dimensional compressions. The complex Nyquist alias is a structural sampling limitation, not a finite-data estimation failure.',
           '',f'- Elapsed time: {elapsed:.2f} seconds. See T03_all_lags.csv and T03_complex_alias.csv for every value.']
    (OUT/'T03_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:6]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d,c,t=run()
    d.to_csv(OUT/'T03_all_lags.csv',index=False)
    c.to_csv(OUT/'T03_complex_alias.csv',index=False)
    plot(d,c);report(d,c,t)
    manifest={'n_rows':len(d),'n_complex':len(c),'n_invalid_R':int(d.R_hat.isna().sum()),'n_invalid_population_R':int(d.R_pop.isna().sum()),'config':CFG,
      'runtime_seconds':t,'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'python':platform.python_version(),'numpy':np.__version__}
    (OUT/'T03_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
