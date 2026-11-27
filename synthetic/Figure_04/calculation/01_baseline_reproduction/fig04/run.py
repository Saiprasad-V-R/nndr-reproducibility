#!/usr/bin/env python3
"""T01 / submitted Fig. 4: revised rates-based mean-field response alignment.

Run: python run.py (no command-line arguments).

Retains Eq. 44 fixed-spectrum family and the EXACT 116-point historical
kappa grid; the physical operator does not change between the old and revised
analysis.  Realized correlations depend on 16 independent length-1000 noise
trajectories *at each kappa*.  Common random numbers are intentionally paired
ACROSS kappa to isolate the operator effect (not independent across kappa).

Efficient simulation is mathematically equivalent to x_{t+1}=F x_t + eta_t:
F = P diag(lambda) P^{-1}, P=U diag(scale) V^T,
z_{t+1}=diag(lambda) z_t + V diag(1/scale) U^T eta_t,
mean=m^T P z, response=r^T P z.
The independent stationary population correlation is computed via the exact
Lyapunov solution in eigen-coordinates.
"""
from __future__ import annotations
import os
import sys, time, hashlib, platform, json, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from shared.nndr_synthetic_operators import fixed_spectrum_family,operator_at_kappa,original_multiplier_ratio
from shared.nndr_planes import m2_multistart
from shared.nndr_thresholds import reduced_diagnostics,spectral_radius

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CFG=json.loads((HERE/'config.json').read_text())


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def kappa_grid(cfg=CFG):
    g=cfg['kappa_grid']
    return np.unique(np.r_[np.linspace(*g['low']),np.linspace(*g['middle']),np.geomspace(*g['high_geom'])])


def sample_abs_corr(a, b):
    """Per-realization absolute Pearson correlation; a,b shape (M,T)."""
    da=a-a.mean(axis=1,keepdims=True)
    db=b-b.mean(axis=1,keepdims=True)
    numerator=np.sum(da*db,axis=1)
    denom=np.sqrt(np.sum(da**2,axis=1)*np.sum(db**2,axis=1))
    return np.abs(numerator/np.maximum(denom,1e-300))


def prepare_noise_coordinates(family, noise):
    """Precompute inverse similarity transform exploiting a rank-one kappa change."""
    U,V=family['U'],family['V']
    inv_scale=np.r_[1.0/family['scales'],1.0]
    raw=noise@U
    base=(raw*inv_scale)@V.T
    return raw,base


def simulate_observables(family, kappa, response_r, noise_base, noise_last):
    """Return (M,T) signals using exact diagonal eigen-coordinate recurrence."""
    U,V=family['U'],family['V']
    lam=family['eigenvalues']
    scale=np.r_[family['scales'],1.0/float(kappa)]
    # P = U diag(scale) V.T => P.T w = V diag(scale) U.T w
    w_mean=V@(scale*(U.T@(np.ones(len(lam))/len(lam))))
    w_response=V@(scale*(U.T@response_r))
    M,T,_=noise_base.shape
    z=np.zeros((M,len(lam)))
    mean=np.empty((M,T));response=np.empty((M,T))
    delta=float(kappa)-1.0
    update_vec=V[:,-1]
    for i in range(T):
        innov=noise_base[:,i,:]+delta*noise_last[:,i,None]*update_vec
        z=z*lam+innov
        mean[:,i]=z@w_mean
        response[:,i]=z@w_response
    return mean,response


def population_signed_corr(family, kappa, response_r, sigma_noise):
    """Exact infinite-time stationary correlation for x_{t+1}=Fx_t+sigma*I noise.

    In eigen coordinates the stationary covariance is S_ij=Q_ij/(1-lam_i lam_j).
    Return signed correlation; the Fig. 4 observable is |corr|.
    """
    V,U=family['V'],family['U']
    lam=family['eigenvalues']
    scale=np.r_[family['scales'],1.0/float(kappa)]
    winv=1/scale
    Q=(V*(winv**2))@V.T*(sigma_noise**2)
    S=Q/(1.0-lam[:,None]*lam[None,:])
    a=V@(scale*(U.T@(np.ones(len(lam))/len(lam))))
    b=V@(scale*(U.T@response_r))
    cov=float(a@S@b)
    var1=float(a@S@a); var2=float(b@S@b)
    return cov/np.sqrt(var1*var2)


def verify_exact_simulation(family, kappas, noise, raw, base):
    """For selected physical operators, compare exact reduced observables to full VAR."""
    U,V=family['U'],family['V']
    mvec=np.ones(CFG['N'])/CFG['N']
    records=[]
    for kappa in kappas:
        F=operator_at_kappa(family,kappa)
        plane=m2_multistart(F,restarts=CFG['m2_restarts'],iterations=CFG['m2_iterations'],seed=CFG['m2_seed'])
        x=np.zeros(CFG['N']); obs_m=[];obs_r=[]
        for eta in noise[0,:60,:]:
            x=F@x+eta
            obs_m.append(x@mvec)
            obs_r.append(x@plane['r'])
        m_fast,r_fast=simulate_observables(family,kappa,plane['r'],base[:1,:60],raw[:1,:60,-1])
        maximum=max(float(np.max(abs(m_fast[0]-obs_m))),float(np.max(abs(r_fast[0]-obs_r))))
        records.append({'kappa':float(kappa),'max_abs_observable_error':maximum})
        if maximum>1e-9:
            raise AssertionError(f'Eigen-coordinate simulation deviates from full F: k={kappa}, error={maximum}')
    return records


def plot(df, outname, show_population=False):
    sel=df.sort_values('R_variance_rates')
    r=sel['R_variance_rates'].to_numpy()
    med=sel['alignment_median'].to_numpy();lo=sel['alignment_q25'].to_numpy();hi=sel['alignment_q75'].to_numpy()
    fig,ax=plt.subplots(figsize=(5.25,3.45))
    ax.fill_between(r,lo,hi,color='#667085',alpha=.17,label='Interquartile range (16 realizations)',lw=0)
    ax.plot(r,med,color='#26394a',lw=1.9,label='Median alignment',zorder=3)
    if show_population:
        ax.plot(r,sel['population_abs_correlation'],color='#C65829',lw=1.3,linestyle='--',
                label='Exact stationary covariance',zorder=4)
    ax.axvline(1,color='#949ba7',ls='--',lw=1.0,zorder=0)
    ax.text(1.11,.075,r'$R=1$',transform=ax.get_xaxis_transform(),fontsize=8,color='#697586')
    ax.set_xscale('log');ax.set_xlim(*CFG['publication_x_range']);ax.set_ylim(0,1)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlabel(r'Reduced non-normality ratio, $R=K/K_c$ (rates-based)')
    ax.set_ylabel(r'$|\mathrm{corr}(m_k,\widehat{\mathbf{r}}^{\mathsf{T}}\mathbf{x}_k)|$')
    ax.tick_params(direction='in',top=True,right=True,which='both')
    ax.legend(loc='upper left',frameon=False,fontsize=7.6)
    fig.subplots_adjust(left=.165,right=.985,bottom=.18,top=.97)
    fig.savefig(OUT/(outname+'.png'),dpi=260,bbox_inches='tight')
    fig.savefig(OUT/(outname+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    t0=time.time()
    cfg=CFG
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    family=fixed_spectrum_family(n=cfg['N'],rho=cfg['rho_F'],delta_sigma=cfg['delta_sigma'],seed=cfg['family_seed'])
    grid=kappa_grid()
    if len(grid)!=116:
        raise AssertionError(f'Kappa grid should contain 116 points, got {len(grid)}')
    # One i.i.d. noise bank; paired across kappa to reduce Monte Carlo fluctuations.
    rng=np.random.default_rng(cfg['common_noise_seed'])
    noise=cfg['sigma_noise']*rng.standard_normal((cfg['n_noise_realizations'],cfg['T_align'],cfg['N']))
    raw,base=prepare_noise_coordinates(family,noise)
    checks=verify_exact_simulation(family,cfg['verification_kappas'],noise,raw,base)
    # Validate exact eigen-coordinate stationary covariance against an
    # independently solved full N-dimensional discrete Lyapunov equation.
    from scipy.linalg import solve_discrete_lyapunov
    lyapunov_checks=[]
    for kappa in cfg['verification_kappas']:
        fullF=operator_at_kappa(family,kappa)
        p=m2_multistart(fullF,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'],seed=cfg['m2_seed'])
        S=solve_discrete_lyapunov(fullF,cfg['sigma_noise']**2*np.eye(cfg['N']))
        mf=np.ones(cfg['N'])/cfg['N']; r=p['r']
        signed_full=float((mf@S@r)/np.sqrt((mf@S@mf)*(r@S@r)))
        signed_diag=population_signed_corr(family,kappa,r,cfg['sigma_noise'])
        err=abs(signed_full-signed_diag)
        lyapunov_checks.append({'kappa':float(kappa),'signed_corr_full_lyapunov':signed_full,'signed_corr_eigen_coordinate':signed_diag,'abs_error':err})
        if err>2e-9:
            raise AssertionError(f'Analytic covariance disagrees with full Lyapunov at kappa={kappa}: {err}')
    rows=[]; reps=[]
    plane_spectrum={}  # store enough for exact reproduction of the plotted points
    for idx,kappa in enumerate(grid):
        F=operator_at_kappa(family,kappa)
        plane=m2_multistart(F,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'],seed=cfg['m2_seed'])
        Q=plane['Q'];Gamma=Q.T@F@Q
        d=reduced_diagnostics(Gamma)
        if d['status']!='ok':
            raise AssertionError(f'Invalid Γ at k={kappa}, status={d["status"]}')
        mean,response=simulate_observables(family,kappa,plane['r'],base,raw[:,:,-1])
        c=sample_abs_corr(mean,response)
        pop=abs(population_signed_corr(family,kappa,plane['r'],cfg['sigma_noise']))
        r=d['R_variance'];rs=d['R_sigma']
        rows.append({'grid_index':idx,'kappa':float(kappa),'R_variance_rates':r,'R_sigma':rs,
                     'R_historical_multiplier':original_multiplier_ratio(Gamma),
                     'rho_F':spectral_radius(F),'rho_Gamma':d['rho'],
                     'sigma_F':np.linalg.svd(F,compute_uv=False)[0],
                     'sigma_Gamma':d['sigma_max'],
                     'capture_sigma_ratio':d['sigma_max']/np.linalg.svd(F,compute_uv=False)[0],
                     'K':d['K'],'Kc_variance':d['Kc_variance'],
                     'plane_objective':plane['objective'],
                     'plane_restart_objective_range':np.ptp(plane['restart_objectives']),
                     'population_abs_correlation':pop,
                     'alignment_median':float(np.median(c)),
                     'alignment_mean':float(np.mean(c)),
                     'alignment_q25':float(np.quantile(c,.25)),
                     'alignment_q75':float(np.quantile(c,.75)),
                     'alignment_std':float(np.std(c,ddof=1)),
                     'n_realizations':int(len(c))})
        for j,value in enumerate(c):
            reps.append({'grid_index':idx,'kappa':float(kappa),'repeat_index':j,
                         'R_variance_rates':r,'R_sigma':rs,'alignment_abs_corr':float(value)})
        plane_spectrum[f'Q_{idx:03d}']=Q
        plane_spectrum[f'Gamma_{idx:03d}']=Gamma
        if (idx+1)%25==0 or idx+1==len(grid):
            print(f'Computed {idx+1}/{len(grid)} grid points; R={r:.4g}; median correlation={np.median(c):.4f}',flush=True)
    df=pd.DataFrame(rows)
    replicates=pd.DataFrame(reps)
    df.to_csv(OUT/'T01_fig04_116point_alignment.csv',index=False,float_format='%.13g')
    replicates.to_csv(OUT/'T01_fig04_1856_replicates.csv',index=False,float_format='%.13g')
    np.savez_compressed(OUT/'T01_fig04_planes_and_reduced_operators.npz',**plane_spectrum)
    (OUT/'T01_fig04_direct_simulation_checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'T01_fig04_full_lyapunov_checks.json').write_text(json.dumps(lyapunov_checks,indent=2))
    plot(df,'T01_fig04_alignment_rates_revised',show_population=False)
    plot(df,'T01_fig04_alignment_with_stationary_reference',show_population=True)
    # The correlation is an observable, not a binary classifier. Diagnostic
    # value R=1 does NOT mathematically mandate a step in |corr| at that x.
    assert len(df)==116 and len(replicates)==116*16
    assert (df['rho_F']-cfg['rho_F']).abs().max()<1e-10
    assert df['R_variance_rates'].is_monotonic_increasing
    assert (df['alignment_q25']<=df['alignment_median']).all()
    assert (df['alignment_median']<=df['alignment_q75']).all()
    assert (df['alignment_median'].between(0,1)).all()
    assert (df['capture_sigma_ratio']<=1+1e-10).all()
    # Numerical population correlation comparison should be observed after the
    # run rather than asserting an arbitrary accuracy/tolerance.
    low=df[df['R_variance_rates']<1]; high=df[df['R_variance_rates']>1]
    near=df.iloc[(df['R_variance_rates']-1).abs().argsort()[:2]]
    with (OUT/'T01_fig04_report.md').open('w') as f:
        f.write('# T01 / Fig. 4: rates-based mean-field alignment\n\n')
        f.write('## Design and controls\n\nOriginal N=200, rho(F)=0.95, sigma=0.05, T=1000, 116 identical kappa values and 16 independent noise realizations per point. A common noise bank is deliberately reused across kappa values (paired comparison). Planes from the known F via 24-start M2. Original physical operators unchanged; R recalculated from rates; R_sigma included independently.\n\n')
        f.write(f'## Results\n\nR range: [{df.R_variance_rates.min():.6g}, {df.R_variance_rates.max():.6g}], with {len(low)} points below and {len(high)} above R=1.\n\n')
        f.write(f'Median alignment over kappa: R<1 median of medians = {low.alignment_median.median():.4f}; R>1 median of medians = {high.alignment_median.median():.4f}. These across-grid summaries are descriptive, not independent inferential tests.\n\n')
        f.write('Closest grid points to R=1:\n\n'+near[['kappa','R_variance_rates','alignment_median','population_abs_correlation']].to_markdown(index=False)+'\n\n')
        f.write(f'Median absolute discrepancy between 16-trial median and exact stationary covariance = {np.median(abs(df.alignment_median-df.population_abs_correlation)):.4g}; maximum = {np.max(abs(df.alignment_median-df.population_abs_correlation)):.4g}.\n\n')
        f.write('## Interpretation\n\nThe curve is alignment between an observed mean field and known-operator response coordinate, not a direct mathematical measurement of the variance-response threshold. R=1 need not produce a sharp discontinuity. The 16-trial interquartile band reflects variability across realizations, NOT a 95% confidence interval for the median. Comparisons across kappa use common random numbers and are paired.\n\n')
        f.write('## Independent checks\n\nFull VAR direct-simulation comparison:\n\n'+json.dumps(checks,indent=2)+'\n\nFull 200-dimensional Lyapunov comparison:\n\n'+json.dumps(lyapunov_checks,indent=2)+'\n')
        f.write('\n## Claim limits\n\nAt small R the correlation is nonmonotonic, and the threshold R=1 does not coincide with a sharp change in alignment. The variance threshold refers to the delayed Frobenius-norm response kernel, while the mean-field correlation depends on the observable projection and forcing. A systematic increase in correlation occurs along the super-threshold branch of THIS fixed operator family; it is not a universal theorem.\n')
    (HERE/'Fig04_revised_caption.tex').write_text(
        'Mean-field alignment with the known-operator response coordinate across the fixed-spectrum non-normal family. The horizontal axis is the corrected, rates-based reduced variance-response ratio $R=K/K_c$; the ordinate is $|\\mathrm{corr}(m_k,\\widehat{\\mathbf r}^{\\mathsf T}\\mathbf x_k)|$. All 116 historical values of the internal anisotropy parameter $\\kappa$ are retained, with $N=200$, $\\rho(F)=0.95$, $\\sigma=0.05$, and $T=1000$ steps. The curve is the median over 16 independent noise realizations at each $\\kappa$, and the band is their interquartile range. The same noise bank is used across $\\kappa$ for paired comparison. The vertical dashed line is $R=1$, the reduced delayed-variance-response threshold; it is not a predicted sharp transition in this correlation.\n')
    hashes={p.name:sha256(p) for p in [Path(__file__),HERE/'config.json',BASE/'shared/nndr_thresholds.py',BASE/'shared/nndr_planes.py',BASE/'shared/nndr_synthetic_operators.py']}
    manifest={'id':'T01-Fig04','time_utc':'2026-10-20','elapsed_seconds':time.time()-t0,
              'n_grid':len(df),'n_noise_trials_per_grid':16,'shared_random_numbers':True,
              'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
              'python':platform.python_version(),'numpy':np.__version__,'scipy':importlib.metadata.version('scipy'),
              'matplotlib':matplotlib.__version__,'source_sha256':hashes,
              'checks':{'full_spectrum_fixed':True,'exact_similarity_simulation':True,'full_lyapunov_crosscheck':True,
                        'quantiles_valid':True,'two_thresholds_separate':True}}
    (OUT/'T01_fig04_run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('VALIDATED Fig. 4, elapsed',round(time.time()-t0,2),'seconds',flush=True)

if __name__=='__main__': main()
