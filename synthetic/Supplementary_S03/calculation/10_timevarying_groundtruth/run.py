#!/usr/bin/env python3
"""T10: independently planted time-varying plane; not the M2(true F) reference.

No command-line arguments. Each subject has its own innovation stream. The
continuous and finite-window operators are distinct estimands. We retain invalid
principal-log outcomes and noncrossing trajectories, rather than interpolating.
"""
from __future__ import annotations
import sys,json,hashlib,platform,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.nndr_planes import m2_multistart,principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
OUT=HERE/'outputs'


def planted_basis(theta,n=10):
    """Exactly orthogonal moving 2-plane, in a fixed Euclidean metric."""
    Q=np.zeros((n,2))
    c,s=np.cos(theta),np.sin(theta)
    Q[0,0]=c;Q[2,0]=s
    Q[1,1]=c;Q[3,1]=s
    return Q


def orthogonal_embedding(theta,n=10):
    Q=planted_basis(theta,n)
    c,s=np.cos(theta),np.sin(theta)
    O=np.eye(n)
    O[0,0]=c;O[2,0]=s; O[0,2]=-s;O[2,2]=c
    O[1,1]=c;O[3,1]=s; O[1,3]=-s;O[3,3]=c
    assert np.allclose(O[:,:2],Q)
    return O


def scalar_schedule(case,t,T,conf=CFG):
    """Smooth rotation/coupling, or a true noise-only nonstationarity."""
    z=float(t)/max(1,T-1)
    if case=='slow_rotation':
        theta=conf['angle_max_slow_rad']*z
    elif case=='fast_rotation':
        theta=conf['angle_max_fast_rad']*z
    else:
        theta=0.2
    # Constant RA>1 in all cases except threshold crossing.
    b=0.86
    if case=='threshold_crossing':
        b=0.14+1.34*z
    noise=conf['noise_sd']*(2.6 if case=='noise_variance_jump' and t>=T//2 else 1.)
    return theta,b,noise


def operator(theta,b,n=10):
    O=orthogonal_embedding(theta,n)
    D=np.diag(np.r_[0.92,0.50,np.linspace(.23,.42,n-2)])
    D[0,1]=b
    return O@D@O.T


def truth_at(case,t,T,conf=CFG):
    theta,b,noise=scalar_schedule(case,t,T,conf)
    F=operator(theta,b,conf['N'])
    Q=planted_basis(theta,conf['N'])
    diag=reduced_diagnostics(Q.T@F@Q)
    return {'theta':theta,'b':b,'noise_sd':noise,'F':F,'Q':Q,
            'R':float(diag['R_variance']), 'R_sigma':float(diag['R_sigma']),
            'status':diag['status']}


def simulate(case,seed,M=1,conf=CFG):
    T,n=conf['T'],conf['N']
    rng=np.random.default_rng(int(seed))
    X=np.zeros((M,T+1,n),float)
    for t in range(T):
        th,b,sd=scalar_schedule(case,t,T,conf)
        F=operator(th,b,n)
        X[:,t+1,:]=X[:,t,:]@F.T+sd*rng.standard_normal((M,n))
    return X


def fit_ridge(X,end,W,nu):
    A=X[:,end-W:end,:].reshape(-1,X.shape[-1])
    B=X[:,end-W+1:end+1,:].reshape(-1,X.shape[-1])
    n=X.shape[-1]
    gram=A.T@A
    lam=float(nu)*np.trace(gram)/n
    return np.linalg.solve(gram+lam*np.eye(n),A.T@B).T, int(np.linalg.matrix_rank(A)),float(lam)


def m3(F):
    C=F@F.T-F.T@F
    d,v=np.linalg.eigh(C)
    return np.column_stack([v[:,-1],v[:,0]])


def projected_orientation(Q):
    """Recover planted rotation from the fitted plane projector, modulo pi."""
    P=Q@Q.T
    zc=float(P[0,0]-P[2,2]); zs=float(2*P[0,2])
    return float((0.5*np.arctan2(zs,zc))%np.pi),float(np.hypot(zc,zs))


def fixed_plane_diagnostics(F,Q):
    d=reduced_diagnostics(Q.T@F@Q)
    return float(d['R_variance']),float(d['R_sigma']),str(d['status'])


def true_reference(case,end,W,conf=CFG):
    """Instantaneous at window end, midpoint, and average of planted scalars.

    The mean is a descriptive local reference, not a ground-truth finite-window
    least-squares VAR operator or a matrix logarithm of averaged propagators.
    """
    mid=(end-W+end-1)//2
    instant=truth_at(case,end-1,conf['T'],conf)
    center=truth_at(case,mid,conf['T'],conf)
    times=[truth_at(case,t,conf['T'],conf) for t in range(end-W,end)]
    Fmean=np.mean([v['F'] for v in times],axis=0)
    # A direct projection onto the planted midpoint plane does not equal the
    # window effective VAR coefficient for rotating systems.
    rwin,rswin,stwin=fixed_plane_diagnostics(Fmean,center['Q'])
    return {'instant':instant,'center':center,'window_compressed_R':rwin,
            'window_compressed_R_sigma':rswin,'window_compressed_status':stwin,
            'Fmean':Fmean,'t_mid':mid}


def run_simulation(conf=CFG,verbose=True):
    rows=[]
    T,W,stride=conf['T'],conf['window'],conf['stride']
    for case in conf['cases']:
        for subj in range(conf['independent_subjects']):
            # Nested cohorts: M=1 uses the first record of the same M12 cohort.
            X=simulate(case,conf['seed_base']+1000*conf['cases'].index(case)+subj,
                       max(conf['M_values']),conf)
            for end in range(W,T+1,stride):
                ref=true_reference(case,end,W,conf)
                for M in conf['M_values']:
                    Fhat,rank,lam=fit_ridge(X[:M],end,W,conf['ridge_relative'])
                    method_planes={
                        'M2':m2_multistart(Fhat,seed=conf['seed_base']+end+101*subj+10007*conf['cases'].index(case),
                              restarts=conf['m2_restarts'],iterations=conf['m2_iterations'])['Q'],
                        'M3':m3(Fhat)
                    }
                    for method,Q in method_planes.items():
                        R,Rs,status=fixed_plane_diagnostics(Fhat,Q)
                        thhat, thsignal=projected_orientation(Q)
                        rot_rate=conf['angle_max_slow_rad']/(T-1) if case=='slow_rotation' else (conf['angle_max_fast_rad']/(T-1) if case=='fast_rotation' else np.nan)
                        lag=(ref['center']['theta']-thhat)/rot_rate if np.isfinite(rot_rate) else np.nan
                        rows.append({'case':case,'subject':subj,'M':M,'method':method,
                          't_end':end,'t_center':ref['t_mid'],'window':W,
                          'R_instant':ref['instant']['R'], 'R_center':ref['center']['R'],
                          'R_window_geometry':ref['window_compressed_R'],
                          'R_sigma_instant':ref['instant']['R_sigma'],
                          'R_sigma_center':ref['center']['R_sigma'],
                          'R_sigma_window_geometry':ref['window_compressed_R_sigma'],
                          'truth_window_status':ref['window_compressed_status'],
                          'R_hat':R,'R_sigma_hat':Rs,'status':status,
                          'angle_instant_deg':principal_angle_deg(Q,ref['instant']['Q']),
                          'angle_mid_deg':principal_angle_deg(Q,ref['center']['Q']),
                          'R_abs_err_center':abs(R-ref['center']['R']) if np.isfinite(R) else np.nan,
                          'R_abs_err_instant':abs(R-ref['instant']['R']) if np.isfinite(R) else np.nan,
                          'rank_X':rank,'ridge_abs':lam,
                          'Fhat_relerr_to_arithmetic_F':float(np.linalg.norm(Fhat-ref['Fmean'],'fro')/np.linalg.norm(ref['Fmean'],'fro')),
                          'theta_instant':ref['instant']['theta'],'theta_mid':ref['center']['theta'],
                          'theta_hat_projector':thhat,'projector_orientation_signal':thsignal,
                          'orientation_lag_steps':lag,
                          'b_center':ref['center']['b'],'noise_sd_instant':ref['instant']['noise_sd']})
            if verbose:print(f'{case}: subject {subj+1}/{conf["independent_subjects"]}',flush=True)
    return pd.DataFrame(rows)


def crossing_time(t,values,level=1):
    t=np.asarray(t,float); a=np.asarray(values,float)
    ok=np.isfinite(a)
    if np.sum(ok)<2:return np.nan
    # Only adjacent valid observations may define a crossing.
    for j in range(1,len(t)):
        if ok[j-1] and ok[j] and a[j-1]<level<=a[j]:
            return float(t[j-1]+(level-a[j-1])*(t[j]-t[j-1])/(a[j]-a[j-1]))
    return np.nan


def summary(df,conf=CFG):
    summ=[]; crossings=[]
    for (case,M,method),d in df.groupby(['case','M','method'],sort=False):
        valid=np.isfinite(d.R_hat.to_numpy())
        summ.append({'case':case,'M':M,'method':method,
            'observations':len(d),'valid':int(valid.sum()),'inadmissible':int((~valid).sum()),
            'median_angle_mid_deg':float(d.angle_mid_deg.median()),
            'median_angle_instant_deg':float(d.angle_instant_deg.median()),
            'median_abs_R_err_center_valid':float(d.R_abs_err_center.median()),
            'median_abs_R_err_instant_valid':float(d.R_abs_err_instant.median()),
            'detected_R_greater_1':int((d.R_hat>1).sum()),
            'truth_R_greater_1_center':int((d.R_center>1).sum()),
            'rel_F_error_median':float(d.Fhat_relerr_to_arithmetic_F.median()),
            'median_orientation_lag_steps':float(d.orientation_lag_steps.median())})
        if case=='threshold_crossing':
            for sub,g in d.groupby('subject'):
                g=g.sort_values('t_center')
                tc=crossing_time(g.t_center,g.R_center)
                te=crossing_time(g.t_center,g.R_hat)
                crossings.append({'subject':int(sub),'M':M,'method':method,
                    'true_center_crossing':tc,'estimated_center_crossing':te,
                    'timing_error':te-tc if np.isfinite(te) else np.nan,
                    'crossing_recovered':bool(np.isfinite(te)),
                    'n_invalid':int(g.R_hat.isna().sum())})
    return pd.DataFrame(summ),pd.DataFrame(crossings)


def plot(df,summ,cross,conf=CFG):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(13.5,7.0),layout='constrained')
    cases=conf['cases'];colors={'M2':'#1666a2','M3':'#df7a2e'}
    a=axs[0,0]
    for M,ls in [(1,'--'),(12,'-')]:
        for method in ['M2','M3']:
            g=df[(df.case=='slow_rotation')&(df.M==M)&(df.method==method)]
            m=g.groupby('t_center').angle_mid_deg.median()
            a.plot(m.index,m.values,label=f'{method}, M={M}',c=colors[method],ls=ls,lw=1.5)
    a.set(title='(a) Slowly rotating plane',xlabel='Window midpoint',ylabel='Median plane error (deg)');a.legend(fontsize=7,frameon=False)
    b=axs[0,1]
    for case in ['stationary','slow_rotation','fast_rotation']:
        for M in [1,12]:
            z=summ[(summ.case==case)&(summ.M==M)&(summ.method=='M3')]
            b.bar(f'{case.split("_")[0][:4]}\nM{M}',z.median_angle_mid_deg.iloc[0],
                  color={'stationary':'#6b8797','slow_rotation':'#2d8a91','fast_rotation':'#cd7352'}[case],
                  alpha=.45 if M==1 else 1)
    b.set(title='(b) Rotation speed and sample count',ylabel='Median M3 angle (deg)')
    c=axs[0,2]
    for M,ls in [(1,'--'),(12,'-')]:
        for method in ['M2','M3']:
            q=df[(df.case=='threshold_crossing')&(df.M==M)&(df.method==method)]
            m=q.groupby('t_center').R_hat.median()
            c.plot(m.index,m.values,c=colors[method],ls=ls,label=f'{method} M={M}',lw=1.5)
    tr=df[df.case=='threshold_crossing'].groupby('t_center').R_center.first()
    c.plot(tr.index,tr.values,'k-',lw=2,label='Planted instantaneous at midpoint')
    c.axhline(1,color='.5',ls=':',lw=1)
    c.set(title='(c) Planted threshold crossing',xlabel='Window midpoint',ylabel=r'$R_A$ (median, valid fits)',ylim=(0,4.4));c.legend(fontsize=7,frameon=False)
    d=axs[1,0]
    for M in [1,12]:
        for method in ['M2','M3']:
            z=summ[(summ.M==M)&(summ.method==method)]
            d.plot(np.arange(len(z)),z.valid/z.observations,marker='o',ls='--' if M==1 else '-',c=colors[method],label=f'{method} M={M}')
    d.set(xticks=np.arange(len(cases)),xticklabels=[x.replace('_','\n') for x in cases],ylim=(0,1.06),
         title='(d) Ratio admissibility',ylabel='Valid reduced fits / total');d.legend(fontsize=7,frameon=False)
    e=axs[1,1]
    for M in [1,12]:
        for method in ['M2','M3']:
            q=cross[(cross.M==M)&(cross.method==method)]
            e.scatter(np.full(len(q),M)+(-.2 if method=='M2' else .2),q.timing_error,
                      marker='o' if method=='M2' else 's',color=colors[method],alpha=.8)
    e.axhline(0,c='k',ls=':',lw=1)
    e.set(title='(e) Crossing time error: valid crossings only',xticks=[1,12],xlabel='Number of trajectories M',ylabel='Estimated minus planted time')
    f=axs[1,2]
    for case,color in [('stationary','#6b8797'),('noise_variance_jump','#a73764')]:
        for M,ls in [(1,'--'),(12,'-')]:
            q=df[(df.case==case)&(df.M==M)&(df.method=='M3')]
            m=q.groupby('t_center').R_hat.median()
            f.plot(m.index,m.values,c=color,ls=ls,label=f'{case.replace("_"," ")} M={M}')
    tru=df[df.case=='stationary'].groupby('t_center').R_center.first()
    f.axhline(tru.iloc[0],c='k',lw=1.5,label='Fixed planted $R_A$')
    f.set(title='(f) Noise-only change vs constant geometry',xlabel='Window midpoint',ylabel=r'Estimated $R_A$',ylim=(0,4.4));f.legend(fontsize=7,frameon=False)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'T10_Fig7_timevarying_planted.{ext}',dpi=220)
    plt.close(fig)


def cluster_intervals(df,conf=CFG):
    """Percentile cluster-bootstrap intervals; reweight complete subjects, not windows."""
    rng=np.random.default_rng(conf['bootstrap_seed'])
    rows=[]
    for (case,M,method),g in df.groupby(['case','M','method'],sort=False):
        subjects=list(g.subject.unique())
        values={s:g.loc[g.subject==s,'angle_mid_deg'].to_numpy() for s in subjects}
        empirical=np.median(np.concatenate(list(values.values())))
        sims=[]
        for _ in range(conf['bootstrap_draws']):
            chosen=rng.choice(subjects,size=len(subjects),replace=True)
            sims.append(float(np.median(np.concatenate([values[s] for s in chosen]))))
        lo,hi=np.quantile(sims,[.025,.975])
        rows.append({'case':case,'M':M,'method':method,'subjects':len(subjects),
             'pooled_median_angle_deg':float(empirical),
             'cluster_bootstrap_95_lower':float(lo),'cluster_bootstrap_95_upper':float(hi),
             'bootstrap_draws':conf['bootstrap_draws']})
    return pd.DataFrame(rows)


def write_report(df,summ,cross,ci,conf=CFG):
    alln=len(df); invalid=int((~np.isfinite(df.R_hat)).sum())
    lines=['# T10 independent time-varying planted-plane benchmark','',
      '## Methods','',
      f'N={conf["N"]}, T={conf["T"]}, window={conf["window"]}, stride={conf["stride"]}, independent subjects={conf["independent_subjects"]}, nested M={conf["M_values"]}.',
      'True plane is Q*(t)=[cos(theta)e1+sin(theta)e3, cos(theta)e2+sin(theta)e4],',
      'with exact orthogonal complement, true 2x2 upper triangular block, and a normal diagonal background.',
      'Instantaneous R_A is evaluated on the planted B(t), independently of M2/M3.',
      'The center reference uses B(t_center); the window-geometry reference is Q*(t_center)^T mean(F_t) Q*(t_center).',
      'The latter is NOT the population statistical least-squares window coefficient. No equivalence is assumed.',
      'All time-varying models are stable in eigenvalues, but switching products can amplify transiently.',
      'Noise jump varies innovation standard deviation alone, keeping the physical operator fixed.',
      'Fitting uses ridge normal equations to all window pairs pooled across M trajectories.',
      'For M=1 only 80 pairs identify 10 columns; numerical conditioning remains a limitation.',
      'M2 is multistart approximate optimization, not certified global; M3 uses extremal commutator eigenvectors.',
      '',f'Rows: {alln}; inadmissible RA: {invalid}; crossings inferred only from adjacent valid windows.',
      '', '## Summary by case, M and method','',summ.to_markdown(index=False,floatfmt='.3f'),'',
      '## Individual crossing timing and misses','',cross.to_markdown(index=False,floatfmt='.3f'),'',
      '## Subject-cluster bootstrap 95% intervals for plane-angle error','',
      ci.to_markdown(index=False,floatfmt='.3f'),'',
      '## Interpretation and limitations','',
      'A window estimate cannot recover an instantaneous operator without temporal-mixing bias, even with infinitely many trajectories.',
      'No R_A value is imputed for an invalid principal logarithm or unstable reduced spectrum.',
      'A crossing is reported as missing, rather than scoring zero timing error, if no two adjacent valid window results bracket 1.',
      'Within-subject window observations are overlapping and not independent; assess uncertainty at subject level.',
      'This benchmark does not substitute for T11 gap perturbation or T12 bootstrap calibration.', '']
    (OUT/'T10_report.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    df=run_simulation()
    summ,cross=summary(df)
    df.to_csv(OUT/'T10_window_method_results.csv',index=False)
    summ.to_csv(OUT/'T10_summary_by_case.csv',index=False)
    cross.to_csv(OUT/'T10_crossing_per_subject.csv',index=False)
    ci=cluster_intervals(df)
    ci.to_csv(OUT/'T10_subject_cluster_plane_angle_CI.csv',index=False)
    plot(df,summ,cross)
    write_report(df,summ,cross,ci)
    (OUT/'T10_manifest.json').write_text(json.dumps({
      'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'python':platform.python_version(),'numpy':np.__version__,
      'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
      'n_rows':len(df),'invalid_R':int(df.R_hat.isna().sum())},indent=2))
    print('TOTAL rows',len(df),'invalid',int(df.R_hat.isna().sum()))
    print(summ[['case','M','method','valid','median_angle_mid_deg','median_abs_R_err_center_valid']].to_string(index=False))
    print('Crossings:',cross.groupby(['M','method']).agg({'crossing_recovered':'sum','timing_error':'median'}))

if __name__=='__main__':main()
