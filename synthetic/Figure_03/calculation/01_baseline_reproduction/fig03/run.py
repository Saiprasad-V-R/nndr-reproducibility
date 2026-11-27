#!/usr/bin/env python3
"""T01 / submitted Fig. 3 -- revised quantitative synthetic reconstruction.

Standalone: python run.py

Two scientific protocols are *deliberately separated*:
A) historical-fixed-kappa: keep exact Eq.44 operators selected by legacy
   multiplier-R, then recompute R on rates (correct original-paper revision).
B) corrected-R-matched: solve new kappas for R_rate=0.1,1,9.8,30.0;
   optional pedagogical redraw, and NOT the old operators.

All four systems in each protocol receive EXACTLY THE SAME innovation path.
M=1 is the primary time-series plot (as in submitted manuscript Fig. 3);
M=25 is recorded only as a distribution across 25 independent trajectories,
NOT as an ensemble-averaged single time series.

Known-operator planes are used, not finite-data inference. Consequently this
figure validates synthetic *response behaviour*, not M2/M3 estimation accuracy.
"""
from __future__ import annotations
import os,sys,platform,hashlib,json,importlib.metadata,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from shared.nndr_synthetic_operators import fixed_spectrum_family,operator_at_kappa,original_multiplier_ratio
from shared.nndr_planes import m2_multistart,principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics,spectral_radius

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CONFIG=json.loads((HERE/'config.json').read_text())


def checksum(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def analyze_operator(family,kappa):
    F=operator_at_kappa(family,kappa)
    m=m2_multistart(F,seed=CONFIG['m2_seed'],restarts=CONFIG['m2_restarts'],
                    iterations=CONFIG['m2_iterations'])
    Q=m['Q'];Gamma=Q.T@F@Q
    d=reduced_diagnostics(Gamma)
    if d['status']!='ok':
        raise AssertionError(f'Reduced spectrum invalid at kappa={kappa}: {d["status"]}')
    if not np.isclose(spectral_radius(F),CONFIG['rho_F'],rtol=0,atol=CONFIG['tolerances']['operator_spectrum']):
        raise AssertionError('Fixed eigenvalues violated')
    if np.linalg.norm(Q.T@Q-np.eye(2))>CONFIG['tolerances']['m2_orthogonality']:
        raise AssertionError('Q lost orthonormality')
    if d['sigma_max']>np.linalg.svd(F,compute_uv=False)[0]+1e-10:
        raise AssertionError('Reduced singular-value capture exceeds full operator')
    return F,m,d


def simulate_with_noise(F,noise):
    """Identical one-step update as manuscript Eq.(43), x0=0."""
    T,N=noise.shape
    x=np.zeros(N)
    X=np.empty((T,N))
    for k in range(T):
        x=F@x + noise[k]
        X[k,:]=x
    return X


def trajectory_statistics(X,plane):
    n=plane['n'];r=plane['r'];m=X.mean(axis=1)
    nc=X@n;rc=X@r
    def rms(z):return float(np.sqrt(np.mean(z*z)))
    return {'mean_field':m,'input':nc,'response':rc,
            'rms_mean_field':rms(m),'rms_input':rms(nc),
            'rms_response':rms(rc),'rms_response_to_input':rms(rc)/max(rms(nc),1e-14),
            'abs_correlation_mean_response':float(abs(np.corrcoef(m,rc)[0,1]))}


def evaluate_protocol(family,name,kappas,noise,noise_bank):
    entries=[];curves={};systems={};ensemble=[]
    for j,k in enumerate(kappas):
        F,m,d=analyze_operator(family,k)
        X=simulate_with_noise(F,noise)
        tr=trajectory_statistics(X,m)
        row={'protocol':name,'case':j+1,'kappa':float(k),
             'R_variance_rates':float(d['R_variance']),
             'R_sigma':float(d['R_sigma']),
             'K':float(d['K']),'Kc_variance':float(d['Kc_variance']),
             'R_legacy_multiplier_formula':original_multiplier_ratio(m['Q'].T@F@m['Q']),
             'R_original_single_start_legacy_code': (CONFIG['historical_R_multipliers'][j] if name=='historical_fixed_kappa' else float('nan')),
             'rho_F':float(spectral_radius(F)),
             'rho_Gamma':float(d['rho']),
             'sigma_F':float(np.linalg.svd(F,compute_uv=False)[0]),
             'sigma_Gamma':float(d['sigma_max']),
             'capture_sigma_ratio':float(d['sigma_max']/np.linalg.svd(F,compute_uv=False)[0]),
             'plane_objective_best':float(m['objective']),
             'plane_objective_spread':float(np.ptp(m['restart_objectives'])),
             'plane_objective_min':float(m['min_restart_objective']),
             'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
             **{key:val for key,val in tr.items() if not isinstance(val,np.ndarray)}}
        entries.append(row)
        curves[j+1]={key:tr[key] for key in ('mean_field','input','response')}
        systems[f'F_case{j+1}']=F
        systems[f'Q_case{j+1}']=m['Q']
        systems[f'Gamma_case{j+1}']=m['Q'].T@F@m['Q']
        for repeat in range(noise_bank.shape[0]):
            Xr=simulate_with_noise(F,noise_bank[repeat])
            sr=trajectory_statistics(Xr,m)
            ensemble.append({'protocol':name,'case':j+1,'kappa':float(k),
                             'repeat_index':repeat+1,
                             'rms_input':sr['rms_input'],
                             'rms_response':sr['rms_response'],
                             'rms_response_to_input':sr['rms_response_to_input'],
                             'abs_correlation_mean_response':sr['abs_correlation_mean_response']})
        print(f'{name} case {j+1}: k={k:.7g}; Rrate={d["R_variance"]:.6f}; '
              f'Rsigma={d["R_sigma"]:.6f}; rho(F)={spectral_radius(F):.6f}; '
              f'RMS response/input={tr["rms_response_to_input"]:.3f}',flush=True)
    return pd.DataFrame(entries),curves,systems,pd.DataFrame(ensemble)


def calibrated_kappas(family,targets):
    """Find new kappa values on [1,450] where REVISED rates R matches targets."""
    def f(k,target):
        F,m,d=analyze_operator(family,k)
        return d['R_variance']-target
    result=[]
    for target in targets:
        lo,hi=1.0,450.0
        if f(lo,target)*f(hi,target)>=0:
            raise ValueError('Target R not bracketed')
        k=brentq(f,lo,hi,args=(target,),xtol=1e-10,rtol=2e-11,maxiter=80)
        value=f(k,target)+target
        if abs(value-target)>CONFIG['tolerances']['recalibrated_target_absolute']:
            raise AssertionError('Calibrated target mismatch')
        result.append(k)
    return result


def make_plot(df,curves,filename,description):
    mpl=matplotlib
    mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
                         'mathtext.fontset':'dejavuserif','font.size':9,
                         'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'axes.linewidth':0.7,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(7.1,5.25),sharex=True,constrained_layout=False)
    colors=('#0072B2','#D55E00')
    ylabel=(r'Mean field $m_k$',r'Input $\widehat{\mathbf{n}}^{\mathsf{T}}\mathbf{x}_k$',
            r'Response $\widehat{\mathbf{r}}^{\mathsf{T}}\mathbf{x}_k$')
    for c in range(2):
        ids=[1,2] if c==0 else [3,4]
        for row,key in enumerate(('mean_field','input','response')):
            ax=axes[row,c]
            all_vals=[]
            for color,j in zip(colors,ids):
                r=df[df['case']==j].iloc[0]
                y=curves[j][key]
                all_vals.append(y)
                ax.plot(np.arange(len(y)),y,color=color,lw=.85,alpha=.88,
                        label=rf'$R={r["R_variance_rates"]:.3g}$')
            maximum=max(np.max(np.abs(z)) for z in all_vals)
            ax.set_ylim(-1.12*maximum,1.12*maximum)
            ax.axhline(0,color='.6',lw=.5,zorder=0)
            ax.set_xlim(0,CONFIG['T']-1)
            ax.tick_params(direction='in',top=True,right=True,length=3.5)
            if c==0: ax.set_ylabel(ylabel[row])
            if row==0:
                ax.legend(loc='upper center',bbox_to_anchor=(.5,1.31),
                          ncol=2,frameon=False,fontsize=8,handlelength=1.8,columnspacing=.8)
            if row==2: ax.set_xlabel(r'Time step $k$')
            if c==0:
                ax.text(-.21,1.03,'('+('a','b','c')[row]+')',transform=ax.transAxes,
                        fontsize=10,fontweight='bold')
    axes[0,0].set_title(r'Cases 1--2: smaller $\kappa$',pad=34,fontsize=9.5)
    axes[0,1].set_title(r'Cases 3--4: larger $\kappa$',pad=34,fontsize=9.5)
    fig.subplots_adjust(left=.12,right=.985,bottom=.095,top=.86,hspace=.23,wspace=.32)
    fig.savefig(OUT/(filename+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(filename+'.png'),dpi=240,bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    family=fixed_spectrum_family(n=CONFIG['N'],rho=CONFIG['rho_F'],
                   delta_sigma=CONFIG['delta_sigma'],seed=CONFIG['family_seed'])
    np.savez_compressed(OUT/'T01_fig03_family.npz',U=family['U'],V=family['V'],
                         eigenvalues=family['eigenvalues'],scales=family['scales'])
    rng=np.random.default_rng(CONFIG['common_noise_seed'])
    noise=CONFIG['sigma_noise']*rng.standard_normal((CONFIG['T'],CONFIG['N']))
    # Additional, separate common innovation bank, shared across four operators.
    extra=np.random.default_rng(CONFIG['extra_noise_seed']).standard_normal(
         (CONFIG['M_replicates']-1,CONFIG['T'],CONFIG['N']))
    noise_bank=np.concatenate([noise[None],CONFIG['sigma_noise']*extra],axis=0)
    np.savez_compressed(OUT/'T01_fig03_common_innovations.npz',M1=noise,M25=noise_bank)
    fixed=list(CONFIG['historical_kappas'])
    matched=calibrated_kappas(family,CONFIG['recalibrated_target_R_rates'])
    A,curvesA,systemsA,repA=evaluate_protocol(family,'historical_fixed_kappa',fixed,noise,noise_bank)
    B,curvesB,systemsB,repB=evaluate_protocol(family,'new_kappa_matched_to_rates_R',matched,noise,noise_bank)
    main_df=pd.concat([A,B],ignore_index=True)
    rep_df=pd.concat([repA,repB],ignore_index=True)
    main_df.to_csv(OUT/'T01_fig03_operator_and_single_trajectory_summary.csv',index=False,float_format='%.13g')
    rep_df.to_csv(OUT/'T01_fig03_25_trajectory_replicates.csv',index=False,float_format='%.13g')
    rep_df.groupby(['protocol','case']).agg(n=('rms_response_to_input','size'),
                       median_ratio=('rms_response_to_input','median'),
                       q25_ratio=('rms_response_to_input',lambda x:x.quantile(.25)),
                       q75_ratio=('rms_response_to_input',lambda x:x.quantile(.75)),
                       median_alignment=('abs_correlation_mean_response','median')).to_csv(
                       OUT/'T01_fig03_25_trajectory_summary.csv')
    for protocol,curves,sys in [('historical_fixed_kappa',curvesA,systemsA),
                                 ('new_kappa_matched_to_rates_R',curvesB,systemsB)]:
        np.savez_compressed(OUT/('T01_fig03_'+protocol+'_operators.npz'),**sys)
        np.savez_compressed(OUT/('T01_fig03_'+protocol+'_timeseries.npz'),
                            **{f'case{j}_{key}':values for j,obj in curves.items() for key,values in obj.items()})
    make_plot(A,curvesA,'T01_fig03_original_operators_revised_R',
              'Submitted Fig. 3 operators — corrected rates-based $R$')
    make_plot(B,curvesB,'T01_fig03_recalibrated_targets',
              'New operators recalibrated to $R=0.1,1,9.8,30$')

    # Assertions: same physical spectrum, changed classification; independent threshold check.
    assert (A['rho_F']-CONFIG['rho_F']).abs().max() < 1e-10
    assert bool(A['R_variance_rates'].iloc[1] > 1)
    assert bool(A['R_legacy_multiplier_formula'].iloc[1] > 1)
    assert bool(A['R_variance_rates'].iloc[0] < 1)
    assert all(np.isfinite(main_df['R_variance_rates']))
    assert all(np.isfinite(main_df['R_sigma']))
    assert all(main_df['capture_sigma_ratio']<=1+1e-12)
    assert np.allclose(B['R_variance_rates'],CONFIG['recalibrated_target_R_rates'],atol=.015,rtol=0)

    delta= A['R_variance_rates'].to_numpy()-A['R_legacy_multiplier_formula'].to_numpy()
    report=f'''# T01 / submitted Fig. 3: rates-based revision\n\n## Experimental scope\n\nEq. (43)-(44), N={CONFIG['N']}, spectral radius={CONFIG['rho_F']}, isotropic innovation sigma={CONFIG['sigma_noise']}, T={CONFIG['T']}. The four systems use a **common fixed white-noise realization** for the main M=1 trajectories (seed {CONFIG['common_noise_seed']}). The M=25 supporting analysis reports distribution statistics across individual realizations, not trajectories averaged in time or over participants.\n\nThe response plane is extracted from the **known generating operator** by multiple-restart M2 (not inferred from each short trajectory). This figure isolates true-operator geometry, not finite-data recovery.\n\n## Central scientific finding\n\nRetaining the four exact historical values of internal kappa and recomputing the rate-based R changes the cases from the submitted R approximately 0.1, 1, 9.8 and 30 to **{', '.join(f'{v:.4f}' for v in A['R_variance_rates'])}**. Case 2 is no longer at the threshold; it is above it. This is a correction to Fig. 3's labels and subcritical/supercritical grouping. One-step R_sigma is reported separately.\n\nThe operators' numerical spectral radii equal {CONFIG['rho_F']} at tolerance 1e-10. An independent set of systems matching the newly specified R targets is plotted separately and must not be described as a mere relabeling of the submitted systems.\n\n## Caveats\n\n- R>1 tests delayed increase in the continuous-time impulse-response variance kernel, not global system instability or stationary variance growth.\n- R_sigma>1 is the exact reduced one-step growth criterion; it need not coincide with R>1 at finite sampling.\n- The plotted response/input amplitude ratio depends on forcing and on the chosen orthonormal plane; it does not test the statistical power of estimating R from finite data.\n- These examples are not a randomized performance comparison of M2 and M3; see T02.\n- Do not infer that event amplitudes prove a mechanistic crossing.\n\n## Outputs\n\n- Figure A: `T01_fig03_original_operators_revised_R.pdf/png` (recommended for a faithful correction).\n- Figure B: `T01_fig03_recalibrated_targets.pdf/png` (clearly new illustrative parameter choices).\n- Quantitative table, all kappa values, full and compressed operator norms, 25-trajectory replicate summaries, common noise arrays, and operator matrices in outputs.\n\n## Runtime\n\n{time.time()-t0:.2f} seconds on this machine.\n'''
    (OUT/'T01_fig03_report.md').write_text(report)
    manifest={'id':'T01-Fig03','date_utc':'2026-10-20','elapsed_seconds':time.time()-t0,
        'python':platform.python_version(),'numpy':np.__version__,
        'scipy':importlib.metadata.version('scipy'),
        'matplotlib':matplotlib.__version__,
        'source_sha256':{p.name:checksum(p) for p in [Path(__file__),HERE/'config.json',
               BASE/'shared/nndr_thresholds.py',BASE/'shared/nndr_planes.py',
               BASE/'shared/nndr_synthetic_operators.py']},
        'common_noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
        'checks':{'stable_full_spectra':True,'same_noise_across_cases':True,
                  'corrected_second_case_above_threshold':True,
                  'matched_targets':True,'reduced_gain_not_overstated':True}}
    (OUT/'T01_fig03_run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nVALIDATION PASSED; elapsed seconds:',round(time.time()-t0,2),flush=True)


if __name__=='__main__': main()
