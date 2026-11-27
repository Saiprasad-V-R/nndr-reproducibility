#!/usr/bin/env python3
"""T02: independently planted NNDR planes versus M1/M2/M3/SVD/snapshot/PCA/random.

Uses direct lagged snapshots (not Fhat^T), matched relative ridge across horizons,
and strictly no ground-truth information in horizon selection. Snapshot-hindsight
best PLANTED-PLANE angle is labeled as an optimistic, non-deployable bound.
The Eq44 family has NO independent planted plane: report known-F M2 reference
as a secondary diagnostic, never mix with the planted benchmark summary.
"""
from __future__ import annotations
import os,sys,json,time,hashlib,platform
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.nndr_planes import m2_multistart,principal_angle_deg
from shared.nndr_synthetic_operators import fixed_spectrum_family,operator_at_kappa
from shared.nndr_thresholds import reduced_diagnostics,spectral_radius,kc_variance_real
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CFG=json.loads((HERE/'config.json').read_text())


def b_target(r,cfg=CFG):
    a,b=cfg['embedded_a'],cfg['embedded_b']
    kr=kc_variance_real(-np.log(a),-np.log(b))
    return np.array([[a,float(r)*kr*(a-b)],[0.,b]])


def make_system(family,seed,cfg=CFG):
    """Orthogonally rotated known systems; Qstar is independent of the extractor.

    For the Eq44 family no planted Qstar exists, even when M2(F) is measurable.
    """
    n=int(cfg['N']); z=np.zeros((n,n)); rng=np.random.default_rng(seed)
    if family=='eq44_fixed_spectrum':
        ff=fixed_spectrum_family(n=n,rho=.95,delta_sigma=.05,seed=seed)
        F=operator_at_kappa(ff,cfg['eq44_kappa'])
        return {'F':F,'latent':None,'Qstar':None,'O':None,'reference':'known_operator_M2_only'}
    if family=='dominant_embedded': r=cfg['target_R_dominant']
    elif family=='competing_nonnormal_background':r=cfg['target_R_competing']
    elif family=='two_near_tied_blocks':r=cfg['target_R_two_blocks']
    else:raise ValueError(family)
    z[:2,:2]=b_target(r,cfg)
    rest=cfg['background_scale']*np.arange(1,n-1,dtype=float)**(-cfg['background_beta'])
    z[np.arange(2,n),np.arange(2,n)]=rest
    if family=='competing_nonnormal_background':
        z[2:4,2:4]=np.array([[.90,2.40],[0,.35]])
    if family=='two_near_tied_blocks':
        z[2:4,2:4]=np.array([[.92,2.85],[0,.40]])
    O,R=np.linalg.qr(rng.standard_normal((n,n)))
    O=O*np.where(np.diag(R)>=0,1.,-1.)
    Qstar=O[:,:2]
    F=O@z@O.T
    known=reduced_diagnostics(Qstar.T@F@Qstar)
    assert known['status']=='ok' and abs(known['R_variance']-r)<1e-8
    return {'F':F,'latent':z,'Qstar':Qstar,'O':O,'reference':'independent_planted_Qstar'}


def simulate(sys_, M, T, sigma, seed):
    rng=np.random.default_rng(seed)
    Fz=sys_['latent'] if sys_['latent'] is not None else sys_['F']
    n=Fz.shape[0]
    zz=np.zeros((M,n))
    out=np.empty((M,T+1,n));out[:,0]=zz
    # All cases generated in orthogonally equivalent coordinates (exactly
    # equivalent for isotropic innovations and trace-scaled ridge).
    for t in range(T):
        zz=zz@Fz.T+sigma*rng.standard_normal((M,n))
        out[:,t+1]=zz
    if sys_['O'] is not None:
        out=out@sys_['O'].T
    return out


def ridge_fit(X,Y,alpha):
    n=X.shape[1];S=X.T@X;lam=alpha*np.trace(S)/n
    return np.linalg.solve(S+lam*np.eye(n),X.T@Y).T,lam


def snapshot_pairs(trajectories,lag):
    if lag<1 or lag>=trajectories.shape[1]:raise ValueError('invalid lag')
    return (trajectories[:,:-lag,:].reshape(-1,trajectories.shape[-1]),
            trajectories[:,lag:,:].reshape(-1,trajectories.shape[-1]))


def svd_pair(F):
    u,s,vh=np.linalg.svd(F,full_matrices=False)
    return u[:,0],vh[0,:],float(s[0])


def plane_from_pair(u,v,rtol=1e-8):
    """Take the span, do not pretend u,v satisfy M2 orthogonality.

    Return None if the vectors are (nearly) collinear and a unique 2-plane
    is not determined. Such cases remain in the denominator with a flag.
    """
    u=np.asarray(u,float);v=np.asarray(v,float)
    u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
    ov=float(np.dot(u,v))
    r=v-ov*u
    nr=np.linalg.norm(r)
    return (np.column_stack([u,r/nr]) if nr>=rtol else None),abs(ov)


def m3(F):
    C=F@F.T-F.T@F
    w,V=np.linalg.eigh(C)
    if max(abs(w[-1]),abs(w[0]))<1e-12:return None,float('nan')
    return V[:,[-1,0]],float(min(w[-1]-w[-2],w[1]-w[0]))


def m1(F):
    eig,V=np.linalg.eig(F)
    U,s,_=np.linalg.svd(V,full_matrices=False)
    v=U[:,-1]
    if np.iscomplexobj(v):v=v.real if np.linalg.norm(v.real)>=np.linalg.norm(v.imag) else v.imag
    v=np.asarray(v.real,float);v/=np.linalg.norm(v)
    u=F@v-(v@F@v)*v
    if np.linalg.norm(u)<1e-12:return None
    return np.column_stack([u/np.linalg.norm(u),v])


def get_plane_metrics(Q,Fhat,F_true,trajectories,Qstar,Qpop,sgain,horizon_metrics):
    if Q is None:
        return dict(angle_planted_deg=np.nan,angle_knownM2_deg=np.nan,
                    R_hat=np.nan,R_status='degenerate_plane',R_sigma_hat=np.nan,
                    sigma_reduced=np.nan,capture_fraction=np.nan,
                    variance_fraction=np.nan,true_plane_one_step_capture=np.nan)
    d=reduced_diagnostics(Q.T@Fhat@Q)
    ref=Qstar if Qstar is not None else Qpop
    angle=principal_angle_deg(Q,ref)
    states=trajectories[:,1:,:].reshape(-1,F_true.shape[0])
    energy=np.linalg.norm(states,'fro')**2
    varfrac=float(np.linalg.norm(states@Q,'fro')**2/energy) if energy>1e-20 else np.nan
    st=d['status'];r=d['R_variance']
    return dict(angle_planted_deg=angle if Qstar is not None else np.nan,
                angle_knownM2_deg=angle if Qstar is None else np.nan,
                R_hat=float(r) if st=='ok' and np.isfinite(r) else np.nan,
                R_status=st,R_sigma_hat=float(d['R_sigma']),
                sigma_reduced=float(d['sigma_max']),
                capture_fraction=float(d['sigma_max']/sgain),
                variance_fraction=varfrac,
                true_plane_one_step_capture=float(np.linalg.svd(Q.T@F_true@Q,compute_uv=False)[0]/np.linalg.svd(F_true,compute_uv=False)[0]))


def calculate(cfg=CFG):
    rows=[];hrows=[];control=[];poprows=[];started=time.perf_counter()
    for family in cfg['families']:
      for rep in range(int(cfg['replicates'])):
        seed=int(cfg['seed_base']+1000*cfg['families'].index(family)+37*rep)
        s=make_system(family,seed,cfg)
        full=s['F'];n=full.shape[0]
        Qpop_known=m2_multistart(full,seed=seed+3,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])['Q']
        Qpop=Qpop_known if s['Qstar'] is None else None
        sample=simulate(s,max(cfg['M_values']),cfg['T_train'],cfg['noise_sd'],seed+17)
        # exact full-operator power norms (no fitted-operator substitution)
        true_g={t:float(np.linalg.svd(np.linalg.matrix_power(full,t),compute_uv=False)[0]) for t in cfg['horizons']}
        true_one=true_g[1]
        up,vp,_=svd_pair(full)
        qsp,_=plane_from_pair(up,vp,cfg['pair_rank_tolerance'])
        qm3p,_=m3(full)
        for name,q in [('M2_known',Qpop_known),('M3_known',qm3p),('SVD_known',qsp)]:
            if q is None:continue
            dr=reduced_diagnostics(q.T@full@q)
            poprows.append({'family':family,'rep':rep,'method':name,
                'angle_planted_deg':principal_angle_deg(q,s['Qstar']) if s['Qstar'] is not None else np.nan,
                'angle_knownM2_deg':principal_angle_deg(q,Qpop_known),
                'true_full_gain_T1':true_one,
                'true_full_gain_T5':true_g[5],
                'R_pop':float(dr['R_variance']) if dr['status']=='ok' else np.nan,
                'R_status':dr['status']})
        if s['Qstar'] is not None:
            B=s['Qstar'].T@full@s['Qstar']
            planted_R=reduced_diagnostics(B)['R_variance']
        else:planted_R=np.nan
        # a competing full-system amplifier is deliberately a non-planted mechanism
        if family.startswith('competing'):
            assert true_one>float(np.linalg.svd(s['Qstar'].T@full@s['Qstar'],compute_uv=False)[0])
        for M in cfg['M_values']:
            tr=sample[:M]
            X,Y=snapshot_pairs(tr,1)
            Fhat,ridge=ridge_fit(X,Y,cfg['ridge_relative'])
            u0,v0,sgain=svd_pair(Fhat)
            qsvd,svd_ov=plane_from_pair(u0,v0,cfg['pair_rank_tolerance'])
            q2d=m2_multistart(Fhat,seed=seed+M+29,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])
            q3,gap=m3(Fhat)
            q1=m1(Fhat)
            e,V=np.linalg.eigh(X.T@X);qpca=V[:,[-1,-2]]
            rng=np.random.default_rng(seed+M+1701)
            qrand=np.linalg.qr(rng.standard_normal((n,2)))[0]
            snapshots=[]
            rank1=int(np.linalg.matrix_rank(X,tol=max(X.shape)*np.finfo(float).eps*np.linalg.norm(X,ord=2)))
            for lag in cfg['horizons']:
                XX,YY=snapshot_pairs(tr,lag)
                Ah,nu=ridge_fit(XX,YY,cfg['ridge_relative'])
                uu,vv,gh=svd_pair(Ah)
                q,overlap=plane_from_pair(uu,vv,cfg['pair_rank_tolerance'])
                # Snapshot T=1 is exactly the singular-pair plane of Fhat,
                # provided we use identical predictor/response pairs and ridge.
                if lag==1:
                    if not np.allclose(Ah,Fhat,rtol=1e-10,atol=1e-10):raise AssertionError('snapshot T1 map differs')
                    if q is not None and qsvd is not None and principal_angle_deg(q,qsvd)>1e-5:
                        raise AssertionError('snapshot T1 and fitted SVD planes differ')
                v_truth=float(np.linalg.norm(np.linalg.matrix_power(full,lag)@vv))
                # Precision of the observable's inferred snapshot gain vs actual full gain.
                horiz=dict(family=family,rep=rep,seed=seed,M=M,T=lag,
                           n_pairs=len(XX),data_span_rank=int(np.linalg.matrix_rank(XX)),
                           Ghat=float(gh),G_true=float(true_g[lag]),
                           Gtrue_v=float(v_truth),fraction_best_gain=float(v_truth/true_g[lag]),
                           input_response_overlap=float(overlap),ridge_abs=float(nu),
                           planted_angle_deg=principal_angle_deg(q,s['Qstar']) if q is not None and s['Qstar'] is not None else np.nan,
                           knownM2_angle_deg=principal_angle_deg(q,Qpop) if q is not None and Qpop is not None else np.nan,
                           degenerate_pair=q is None)
                hrows.append(horiz)
                snapshots.append((lag,q,overlap,gh,horiz))
            # data-selected horizon: maximum inferred gain; no planted truth used
            selected=max(snapshots,key=lambda x:x[3])
            good=[v for v in snapshots if v[1] is not None]
            if s['Qstar'] is not None and good:
                best=min(good,key=lambda x:x[-1]['planted_angle_deg'])
            else:best=None
            collection=[('M1_eigenbasis',q1,None,None),('M2',q2d['Q'],None,None),
                        ('M3',q3,None,None),('SVD_leading_pair',qsvd,svd_ov,1),
                        ('PCA2',qpca,None,None),('random2',qrand,None,None),
                        ('snapshot_Tb',selected[1],selected[2],selected[0])]
            for method,Q,ov,tau in collection:
                metrics=get_plane_metrics(Q,Fhat,full,tr,s['Qstar'],Qpop,sgain,None)
                rows.append(dict(family=family,rep=rep,seed=seed,M=M,method=method,
                                 reference=s['reference'],N=n,n_pairs=len(X),data_span_rank=rank1,
                                 horizon_selected=tau,input_response_overlap=ov,
                                 R_star=planted_R,
                                 sigma_true_full=true_one, sigma_fitted_full=sgain,
                                 rho_true_full=spectral_radius(full),
                                 operator_relative_error=float(np.linalg.norm(Fhat-full,'fro')/np.linalg.norm(full,'fro')),
                                 m2_restarts_range=float(np.ptp(q2d['restart_objectives'])) if method=='M2' else np.nan,
                                 m3_gap=gap if method=='M3' else np.nan,**metrics))
            # special horizon statistics kept separate (avoid pretending hindsight is deployable)
            control.append(dict(family=family,rep=rep,M=M,reference=s['reference'],
                        chosen_horizon=selected[0],chosen_angle_deg=selected[-1]['planted_angle_deg'],
                        hindsight_best_horizon=best[0] if best else np.nan,
                        hindsight_best_angle_deg=best[-1]['planted_angle_deg'] if best else np.nan,
                        horizon_mean_angle_deg=np.nanmean([v[-1]['planted_angle_deg'] for v in good]) if s['Qstar'] is not None else np.nan,
                        T1_svd_plane_discrepancy_deg=principal_angle_deg(qsvd,snapshots[0][1]) if qsvd is not None and snapshots[0][1] is not None else np.nan))
        print(f'Completed {family}: {rep+1}/{cfg["replicates"]}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    return pd.DataFrame(rows),pd.DataFrame(hrows),pd.DataFrame(control),pd.DataFrame(poprows),time.perf_counter()-started


def plot(rows,hor,controls):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.5,'pdf.fonttype':42,'axes.labelsize':9})
    families=['dominant_embedded','competing_nonnormal_background','two_near_tied_blocks']
    labels=['Dominant planted block','Stronger competing block','Two near-tied blocks']
    methods=['M1_eigenbasis','M2','M3','SVD_leading_pair','snapshot_Tb','PCA2','random2']
    names=['M1','M2','M3','SVD pair','Snapshot $T_b$','PCA','Random']
    fig,aa=plt.subplots(2,2,figsize=(10.5,7.1))
    for j,M in enumerate(CFG['M_values']):
        ax=aa[0,j]
        dat=rows[(rows.M==M)&(rows.family.isin(families))]
        locs=np.arange(len(families));w=.105
        for k,method in enumerate(methods):
            y=[dat[(dat.family==fam)&(dat.method==method)].angle_planted_deg.median() for fam in families]
            ax.bar(locs+(k-3)*w,y,width=w*.97,label=names[k],alpha=.9)
        ax.set(xticks=locs,xticklabels=labels,ylabel='Median largest angle to $Q_*$ (deg)',ylim=(0,95),title=f'{M} trajectory' if M==1 else f'{M} trajectories')
        ax.tick_params(axis='x',labelrotation=10,labelsize=7)
        if j==1:ax.legend(fontsize=7,frameon=False,ncol=2,loc='upper right')
    ax=aa[1,0]
    hh=hor[(hor.M==1)&hor.family.isin(families)]
    for fam,label in zip(families,labels):
        part=hh[hh.family==fam]
        ax.plot(CFG['horizons'],[part[part['T']==t].planted_angle_deg.median() for t in CFG['horizons']],marker='o',label=label)
    ax.set(xscale='log',xticks=CFG['horizons'],xlabel='Snapshot horizon $T$',ylabel='Median planted-plane angle (deg)',ylim=(0,95))
    ax.set_xticklabels([str(x) for x in CFG['horizons']]);ax.legend(fontsize=7,frameon=False)
    ax=aa[1,1]
    dat=rows[(rows.family=='dominant_embedded')&(rows.M==1)]
    x=np.arange(len(methods));med=[dat[dat.method==m].capture_fraction.median() for m in methods]
    ax.bar(x,med)
    ax.set(xticks=x,xticklabels=names,ylim=(0,1.05),ylabel=r'Median $\sigma_{\max}(Q^\top\widehat FQ)/\sigma_{\max}(\widehat F)$',title='Captured one-step growth: dominant block')
    ax.tick_params(axis='x',rotation=30,labelsize=7)
    for i,axis in enumerate(aa.flat):
        axis.text(-.105,1.025,f'({chr(ord("a")+i)})',transform=axis.transAxes,fontweight='bold',fontsize=10)
        axis.grid(axis='y',alpha=.16,lw=.4)
    fig.tight_layout(pad=1.4,h_pad=2.0)
    for suff in ('png','pdf'):fig.savefig(OUT/f'T02_FigN1a_b_d_comparators.{suff}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    # Experimental interpretation warning: Eq44 has no independently planted plane.
    fig,ax=plt.subplots(figsize=(7.2,3.6))
    for M in CFG['M_values']:
        dd=rows[(rows.family=='eq44_fixed_spectrum')&(rows.M==M)]
        xx=np.arange(len(methods));val=[dd[dd.method==m].angle_knownM2_deg.median() for m in methods]
        ax.plot(xx,val,marker='o',label=f'M={M}')
    ax.set(xticks=xx,xticklabels=names,ylabel='Median angle to known-operator M2 plane (deg)',ylim=(0,95))
    ax.tick_params(axis='x',rotation=25);ax.legend(frameon=False)
    fig.tight_layout()
    for suff in ('png','pdf'):fig.savefig(OUT/f'T02_Eq44_method_relative_diagnostic.{suff}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    r,h,c,pop,t=calculate()
    r.to_csv(OUT/'T02_all_methods.csv',index=False)
    h.to_csv(OUT/'T02_snapshot_horizons.csv',index=False)
    c.to_csv(OUT/'T02_horizon_selection_controls.csv',index=False)
    pop.to_csv(OUT/'T02_population_methods.csv',index=False)
    plot(r,h,c)
    tab=r.groupby(['family','M','method'],dropna=False).agg(n=('rep','count'),
           median_planted_angle=('angle_planted_deg','median'),
           mean_planted_angle=('angle_planted_deg','mean'),
           median_relative_growth_capture=('capture_fraction','median'),
           valid_R=('R_hat','count'),median_R=('R_hat','median'),
           mean_variance_fraction=('variance_fraction','mean')).reset_index()
    rng=np.random.default_rng(730819)
    ci=[]
    for (family,M,method),grp in r.groupby(['family','M','method']):
        vals=grp.angle_planted_deg.dropna().to_numpy()
        if not len(vals):continue
        boots=np.median(vals[rng.integers(0,len(vals),size=(2000,len(vals)))],axis=1)
        ci.append({'family':family,'M':M,'method':method,'median_angle_deg':float(np.median(vals)),
            'CI95_low_deg':float(np.quantile(boots,.025)),'CI95_high_deg':float(np.quantile(boots,.975)),
            'n_replicates':int(len(vals))})
    pd.DataFrame(ci).to_csv(OUT/'T02_angle_bootstrap_intervals.csv',index=False)
    tab.to_csv(OUT/'T02_group_summary.csv',index=False)
    claims=dict(runtime_seconds=t,systems=int(len(r)//(len(CFG['M_values'])*7)),
        method_rows=len(r),horizon_rows=len(h),population_rows=len(pop),n_independent_planted=int((r.reference=='independent_planted_Qstar').sum()),
        T1_equals_SVD_max_error=float(c.T1_svd_plane_discrepancy_deg.max()),
        invalid_R_count=int(r.R_hat.isna().sum()),
        config=CFG,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        config_sha256=hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'T02_manifest.json').write_text(json.dumps(claims,indent=2))
    lines=['# T02: independent planted-plane comparator audit','',
        f'- Replicated physical systems: **{claims["systems"]}** (four families × {CFG["replicates"]} seeds).',
        f'- Paired M=1 and M=18 conditions; **{len(r)} method-rows**; {len(h)} raw snapshot-horizon fits.',
        f'- Invalid R diagnostics (including truly degenerate 2D planes): **{claims["invalid_R_count"]}/{len(r)}**.',
        f'- T=1 snapshot-vs-SVD numerical plane discrepancy: **{claims["T1_equals_SVD_max_error"]:.4g} degrees**.',
        '', '## Protocol',
        'Known Q* comes from the planted rotated 2×2 block, not M2/M3; Eq44 has no independent planted Q* and is excluded from planted-plane summaries.',
        'M1=eigenbasis-SVD rule from legacy Fig5. M2 uses multi-start projected alternating optimization. M3 uses the extreme commutator eigenvectors. PCA uses the two leading state-covariance PCs. Random planes use independent Gaussian QR.',
        'Lagged snapshot maps A_T are fitted directly from pairs (x_t,x_{t+T}) using a matched trace-normalized relative ridge, not by exponentiating Fhat. Tb maximizes inferred sigma_max(A_T) over 1,2,5,10,20,50 using the training data only.',
        'The T=1 full-data-span snapshot map agrees exactly with Fhat and hence its singular pair. For T>1 the direct empirical lag fits have shorter sample lengths.',
        'The hindsight-best snapshot angle uses Q* and is a NONDEPLOYABLE optimistic bound, printed only in T02_horizon_selection_controls.csv.',
        'Full-system gains are computed from the exact known operator F, independently of extractor planes. The captured one-step gain and explained state variance assess distinct observables.',
        'The embedded target block need not be the strongest mechanism in the competing/near-tie families. Selecting a competing amplifier can be correct for full gain but count as mismatch to planted Q*.',
        'Population extraction controls (exact true F with no finite-data estimation) are in T02_population_methods.csv; seed-level bootstrap intervals for median planted-plane angles are in T02_angle_bootstrap_intervals.csv. Bootstrap intervals have only eight distinct seeds.',
        '', '## Summary: median angle to independently planted plane (degrees)', '',
        '| Family | M | M2 | M3 | SVD pair | Snapshot selected | PCA |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for f in ['dominant_embedded','competing_nonnormal_background','two_near_tied_blocks']:
        for M in CFG['M_values']:
            d=tab[(tab.family==f)&(tab.M==M)].set_index('method')
            vals=[d.loc[m,'median_planted_angle'] for m in ['M2','M3','SVD_leading_pair','snapshot_Tb','PCA2']]
            lines.append('| '+f+' | '+str(M)+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['','## Scope/qualifications','- This is the N1a,b,d **synthetic planted comparator component only**. T03 provides N1c lag invariance and T04 provides N1e factorial baselines; neither is complete here.','- No claim that the direct lagged snapshot ridge here duplicates the specialized regularizers of the two papers cited by Reviewer 1. That literature-specific sensitivity test remains outstanding.','- Eight paired seeds give descriptive Monte Carlo distributions, not an asymptotic population guarantee. Consult row-level CSVs for all estimates, excluded ratios, and horizon stability.', '']
    (OUT/'T02_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:10]));print('Completed:',t,'seconds',flush=True)

if __name__=='__main__':main()
