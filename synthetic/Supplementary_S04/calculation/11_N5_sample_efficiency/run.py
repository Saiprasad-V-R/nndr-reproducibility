#!/usr/bin/env python3
"""T11 N5: planted gap perturbation and original Figure 6 sample-efficiency audit.

Run in Spyder or from any cwd without arguments. The existing T01 original
Figure 6 simulation is NOT overwritten or silently relabelled. M3 is compared
against an independent planted commutator-extremal plane, whereas the T01
M3 angles in the historical scan use an M2(true-F) reference, so a rigorous
Davis-Kahan guarantee cannot be checked on those historical angles.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, json, time, hashlib, platform, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from shared.nndr_planes import m2_multistart, principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics, spectral_radius
CFG=json.loads((HERE/'config.json').read_text())
OUT=HERE/'outputs';OUT.mkdir(parents=True,exist_ok=True)


def commutator(F):
    return F@F.T - F.T@F


def extremal_plane(C):
    eig, vec=np.linalg.eigh(C)
    return vec[:,[-1,0]], float(min(eig[-1]-eig[-2],eig[1]-eig[0])), eig


def shear_commutator_magnitude(a,b,k):
    return abs(k)*np.hypot(k,a-b)


def shear_at_commutator_fraction(frac, a,b, c_primary):
    """Solve |k| sqrt(k^2+(a-b)^2)=frac*c_primary exactly."""
    if frac==0.0: return 0.0
    target=frac*c_primary
    delta=a-b
    z=(-delta**2+np.sqrt(delta**4+4*target**2))/2
    return float(np.sqrt(max(0,z)))


def operator_and_truth(n,frac,seed,cfg=CFG):
    """Independent planted plane with controlled commutator spectral gap.

    Orthogonal similarity preserves both the spectrum and C's eigenvalues.
    Competing block changes *only* the second block's shear: all generating
    eigenvalues and the planted leading 2D plane stay the same.
    """
    r=np.random.default_rng(seed)
    z=r.normal(size=(n,n)); O,R=np.linalg.qr(z)
    O=O*np.where(np.diag(R)>=0.,1.,-1.)
    a,b=map(float,cfg['primary_eigenvalues']);k=float(cfg['primary_shear'])
    ac,bc=map(float,cfg['competitor_eigenvalues'])
    cp=shear_commutator_magnitude(a,b,k)
    kc=shear_at_commutator_fraction(float(frac),ac,bc,cp)
    B1=np.array([[a,k],[0.,b]])
    B2=np.array([[ac,kc],[0.,bc]])
    background=np.linspace(.09,.35,n-4)
    latent=block_diag(B1,B2,np.diag(background))
    F=(O@latent)@O.T
    C=commutator(F)
    Q,gap,lam=extremal_plane(C)
    Qstar=O[:,:2]
    # The commutator extreme vectors from the leading 2x2 block span Qstar.
    assert principal_angle_deg(Q,Qstar)<1e-4
    assert abs(gap-(1.-frac)*cp)<1e-9
    assert abs(spectral_radius(F)-a)<2e-12
    return dict(F=F,Qstar=Qstar,Qcomm=Q,C=C,gap=gap,
                comm_eigenvalues=lam,shear_competitor=kc,shear_primary=k,
                full_sigma=float(np.linalg.svd(F,compute_uv=False)[0]),
                R_planted=float(reduced_diagnostics(Qstar.T@F@Qstar)['R_variance']))


def generate_max_trajectories(F,M,T,sd,seed):
    r=np.random.default_rng(seed)
    x=np.zeros((M,F.shape[0]))
    states=np.empty((M,T+1,F.shape[0]));states[:,0,:]=0
    for t in range(T):
        x=x@F.T+sd*r.standard_normal((M,F.shape[0]))
        states[:,t+1,:]=x
    return states


def fit_from_states(states, ridge_rel):
    M,Tp,n=states.shape
    T=Tp-1
    x=states[:,:-1,:].reshape(M*T,n)
    y=states[:,1:,:].reshape(M*T,n)
    A=x.T@x
    lam=ridge_rel *float(np.trace(A)/n)
    Fhat=np.linalg.solve(A+lam*np.eye(n),x.T@y).T
    # First transition x0=0 is structurally uninformative, every trajectory.
    informative_n=M*(T-1)
    rank=int(np.linalg.matrix_rank(x))
    return Fhat,lam,rank,informative_n


def dk_plane_bound(gap,comm_error):
    """Rigorous largest-angle bound for the top/bottom eigendirections.

    Let C symmetric and C_hat=C+D, delta=||D||_2. If delta<gap,
    each fitted simple-extreme eigenvector has orthogonal-complement norm
    <=delta/(gap-delta) by the residual equation and Weyl. For their span,
    sin(theta_max) <= sqrt(2)*delta/(gap-delta). If delta>=gap,
    only sin(theta_max)<=1 is asserted (vacuous bound).
    """
    if gap<=0 or comm_error>=gap: return 1.
    return float(min(1.,np.sqrt(2.)*comm_error/(gap-comm_error)))


def check_one(F,Fhat,Qstar,gap):
    E=Fhat-F
    eps=float(np.linalg.norm(E,2)); fop=float(np.linalg.norm(F,2))
    dC=commutator(Fhat)-commutator(F)
    dc=float(np.linalg.norm(dC,2))
    # Exact polynomial perturbation bound including the E E^T terms.
    rhs=4*fop*eps+2*eps**2
    if dc>rhs+1e-10: raise AssertionError('commutator perturbation bound violated')
    Qhat,_,_=extremal_plane(commutator(Fhat))
    deg=float(principal_angle_deg(Qstar,Qhat))
    sin_actual=float(np.sin(np.radians(deg)))
    sin_bound=dk_plane_bound(gap,dc)
    if sin_actual>sin_bound+1e-8:
        raise AssertionError(f'DK violation: sin={sin_actual} bound={sin_bound} delta/gap={dc/gap}')
    return dict(operator_error_op=eps, operator_relative_fro_error=float(np.linalg.norm(E,'fro')/np.linalg.norm(F,'fro')),
                commutator_perturbation_op=dc, commutator_perturbation_upper_bound=rhs,
                DK_first_order_proxy=2*dc/gap, DK_certified_sin_bound=sin_bound,
                DK_nonvacuous=bool(sin_bound<1.-1e-10),
                DK_actual_sin=sin_actual, M3_angle_deg=deg,
                DK_holds=bool(sin_actual<=sin_bound+1e-8),
                ratio_comm_error_gap=dc/gap, ratio_op_error_gap=eps/gap)


def generate_rows(cfg=CFG):
    rows=[]
    max_M=max(s['M'] for s in cfg['sampling_regimes']);max_T=max(s['T'] for s in cfg['sampling_regimes'])
    t0=time.monotonic()
    for n in cfg['N_values']:
        for frac in cfg['competitor_commutator_fraction']:
            for rep in range(cfg['independent_seeds']):
                seed=int(cfg['seed_base']+100000*n+rep)
                truth=operator_and_truth(n,frac,seed,cfg)
                # Same innovation tensor nested across M and T, no seed leakage.
                st=generate_max_trajectories(truth['F'],max_M,max_T,cfg['innovation_sd'],seed+123)
                for settings in cfg['sampling_regimes']:
                    m,t=settings['M'],settings['T']
                    fit,lam,rank,inform=fit_from_states(st[:m,:t+1,:],cfg['relative_ridge'])
                    diag=check_one(truth['F'],fit,truth['Qstar'],truth['gap'])
                    qm=m2_multistart(fit,seed=seed+1023,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])['Q']
                    dM2=reduced_diagnostics(qm.T@fit@qm)
                    dM3=reduced_diagnostics(extremal_plane(commutator(fit))[0].T@fit@extremal_plane(commutator(fit))[0])
                    row=dict(N=n,competitor_fraction=frac,seed_index=rep,subject_id=f'N{n}_c{frac:g}_seed{rep}',
                        regime=settings['name'],M=m,T_train=t,n_pairs=inform,n_transitions=m*t,
                        identifiable_unregularized=bool(inform>=n),data_matrix_rank=rank,
                        full_operator_rho=spectral_radius(truth['F']),full_operator_sigma=truth['full_sigma'],
                        R_planted=truth['R_planted'],commutator_gap=truth['gap'],
                        ratio_n_to_N=float(inform/n),ratio_n_to_N2=float(inform/(n*n)),
                        m2_angle_deg=principal_angle_deg(truth['Qstar'],qm),
                        m2_status=dM2['status'],m3_status=dM3['status'],
                        m2_R=float(dM2['R_variance']),m3_R=float(dM3['R_variance']),
                        ridge_lambda=lam,m2_restart_spread=float('nan'),**diag)
                    rows.append(row)
        print(f'T11 progress: N={n}: {len(rows)} rows, {time.monotonic()-t0:.1f} s',flush=True)
    return pd.DataFrame(rows)


def clustered_ci(x,rng,n=600):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if len(x)<2:return (np.nan,np.nan)
    loc=rng.integers(0,len(x),size=(n,len(x)))
    med=np.median(x[loc],axis=1)
    return tuple(map(float,np.quantile(med,[.025,.975])))


def make_summary(df,cfg=CFG):
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    rows=[]
    for (reg,frac),g in df.groupby(['regime','competitor_fraction'],sort=False):
        def ci(col):
            # Independent replicates are operator subject IDs, not individual observations.
            vals=g.groupby('subject_id')[col].first().values
            return clustered_ci(vals,rng,cfg['bootstrap_draws'])
        cilo,cihi=ci('M3_angle_deg')
        rows.append(dict(regime=reg,competitor_fraction=frac,systems=len(g),
            median_M3_angle_deg=float(g.M3_angle_deg.median()),angle_CI95_low=cilo,angle_CI95_high=cihi,
            median_M2_angle_deg=float(g.m2_angle_deg.median()),
            median_op_error=float(g.operator_error_op.median()),
            median_fro_error=float(g.operator_relative_fro_error.median()),
            median_comm_gap=float(g.commutator_gap.median()),
            median_comm_ratio=float(g.ratio_comm_error_gap.median()),
            nonvacuous_DK=int(g.DK_nonvacuous.sum()),
            valid_M3_R=int(g.m3_status.eq('ok').sum()),
            informative_n=int(g.n_pairs.iloc[0]),N_values=','.join(map(str,sorted(g.N.unique())))))
    return pd.DataFrame(rows)


def legacy_audit(cfg=CFG):
    path=ROOT/cfg['legacy_fig06_csv']
    d=pd.read_csv(path)
    # Compare actual commutator spectral gaps from source Eq (44) operators.
    from shared.nndr_synthetic_operators import fixed_spectrum_family,operator_at_kappa
    lookup={}
    out=[]
    for (_,r) in d.iterrows():
        key=(int(r.N),round(float(r.kappa),11))
        if key not in lookup:
            fam=fixed_spectrum_family(n=key[0],rho=.95,delta_sigma=.05,seed=1+key[0])
            F=operator_at_kappa(fam,float(r.kappa))
            C=commutator(F);Q,g,eigs=extremal_plane(C)
            lookup[key]=(float(g),float(np.linalg.norm(F,2)))
        real_gap,normF=lookup[key]
        # This *does not* prove a DK bound for historical M3 versus M2(F).
        out.append(dict(axis=r.axis,x=r.x,N=r.N,M=r.M,T_train=r.T_train,target_R=r.target_R,
            method=r.method,angle_to_M2_population_deg=r.angle_deg,
            n_pairs=r.n_pairs,n_ge_N=bool(r.n_pairs>=r.N),
            gap_C_recomputed=real_gap,gap_C_saved=r.population_commutator_gap,
            comm_error_op=r.commutator_perturbation_op,
            delta_over_gap=r.commutator_perturbation_op/real_gap if real_gap>0 else np.nan,
            historical_2delta_over_gap=r.DK_bound,
            relative_Fro_error=r.operator_error_F,operator_error_op=r.operator_error_op,
            R_fit_valid=bool(r.valid),R_fit_status=r.status))
    return pd.DataFrame(out)


def draw_main(df, summary):
    cols={'0.0':'#2686a9','0.5':'#f0a153','0.9':'#c25d62','0.98':'#715ba1'}
    fig,axes=plt.subplots(2,3,figsize=(12.7,8.1),layout='constrained')
    for frac in CFG['competitor_commutator_fraction']:
        dd=df[df.competitor_fraction==frac]
        col=cols[str(frac)]
        # a: gap control
        one=dd[(dd.N==32)&(dd.regime=='M16_T240')]
        axes[0,0].scatter([frac],[one.commutator_gap.mean()],s=70,c=col,edgecolor='white',zorder=4)
        # b: sample size ratio
        gg=dd.groupby('regime',sort=False).agg({'ratio_n_to_N':'median','M3_angle_deg':'median'}).reset_index()
        gg=gg.sort_values('ratio_n_to_N')
        axes[0,1].plot(gg.ratio_n_to_N,gg.M3_angle_deg,'o-',ms=4,lw=1.5,color=col,label=f'comp. {frac:.2g}')
        # c: DK vs actual; ratios above 10 are explicitly marked on right edge.
        in_view=dd.ratio_comm_error_gap<=10
        axes[0,2].scatter(dd.loc[in_view,'ratio_comm_error_gap'],
                          dd.loc[in_view,'DK_actual_sin'],s=9,c=col,alpha=.45)
        if (~in_view).any():
            axes[0,2].scatter(np.full(int((~in_view).sum()),10.),
                              dd.loc[~in_view,'DK_actual_sin'],s=18,c=col,
                              marker='>',alpha=.35)
        # d: operator error vs M3
        axes[1,0].scatter(dd.operator_relative_fro_error,dd.M3_angle_deg,s=12,c=col,alpha=.45)
    ax=axes[0,0]
    a,b=CFG['primary_eigenvalues'];cp=shear_commutator_magnitude(a,b,CFG['primary_shear'])
    ff=np.linspace(0,.99,90);ax.plot(ff,cp*(1-ff),color='0.35',ls=':',lw=1.2,label='analytical gap')
    ax.set(xlabel='Competing commutator strength / primary',ylabel=r'$g_C$ (extreme-eigenvalue gap)',title='(a) Controlled commutator gap');ax.legend(frameon=False,fontsize=8)
    axes[0,1].set(xscale='log',ylim=(-2,92),xlabel=r'Informative transitions / dimension, $n/N$',ylabel='Median M3 plane error (degrees)',title='(b) Information and gap jointly matter')
    axes[0,1].axvline(1,color='.45',ls=':',lw=1);axes[0,1].legend(frameon=False,fontsize=7,ncol=2)
    x=np.logspace(-3,.2,200)
    y=np.array([dk_plane_bound(1,xx) for xx in x])
    axes[0,2].plot(x,y,'k-',lw=1.3,label='certified bound')
    axes[0,2].plot(x,np.minimum(1,2*x),'k:',lw=1,label='first-order proxy')
    axes[0,2].set(xscale='log',xlim=(.007,13),ylim=(-.03,1.04),xlabel=r'$\|\widehat C-C\|_2/g_C$',ylabel=r'$\sin(\theta_{\max})$',title='(c) Davis–Kahan control')
    nclipped=int((df.ratio_comm_error_gap>10).sum())
    axes[0,2].text(.97,.05,f'$>{10}$: {nclipped} fits (right triangles)',
                   transform=axes[0,2].transAxes,ha='right',fontsize=7,color='.3')
    axes[0,2].legend(frameon=False,fontsize=7)
    axes[1,0].set(xlabel=r'$\|\widehat F-F\|_F/\|F\|_F$',ylabel='M3 plane angle (degrees)',title='(d) Operator error alone is insufficient',ylim=(-2,92))
    cc=summary.copy()
    for frac in CFG['competitor_commutator_fraction']:
        c=cc[cc.competitor_fraction==frac];col=cols[str(frac)]
        pairs=[]
        for reg in [v['name'] for v in CFG['sampling_regimes']]:
            rs=c[c.regime==reg]
            if len(rs): pairs.append((reg,float(rs.nonvacuous_DK.iloc[0])/int(rs.systems.iloc[0])))
        axes[1,1].plot(range(len(pairs)),[y for _,y in pairs],'.-',ms=7,color=col,label=f'comp. {frac:.2g}')
    axes[1,1].set_xticks(range(len(CFG['sampling_regimes'])),[v['name'].replace('_','\n') for v in CFG['sampling_regimes']],fontsize=7)
    axes[1,1].set(ylabel='Fraction with informative DK bound',ylim=(-.05,1.05),title='(e) Certified bounds often become vacuous')
    axes[1,1].legend(frameon=False,fontsize=7,ncol=2)
    rr=cc[(cc.regime=='M16_T240')]; axes[1,2].axis('off')
    lines=['(f) High-data regime: median across 6 seeds per N', '',
           'Competing fraction    median angle    DK informative']
    for frac in CFG['competitor_commutator_fraction']:
        row=rr[rr.competitor_fraction==frac]
        lines.append(f'       {frac:5.2f}                  {row.median_M3_angle_deg.iloc[0]:6.2f}°          {int(row.nonvacuous_DK.iloc[0]):2d}/{int(row.systems.iloc[0]):2d}')
    lines+=['','Population M3 plane = independently planted plane.',
           'No N²-sample identifiability claim is made.',
           'For unstable/invalid compressed rates, R is omitted,',
           'but plane-angle and perturbation errors are retained.']
    axes[1,2].text(.03,.96,'\n'.join(lines),ha='left',va='top',fontsize=8.5,linespacing=1.6,family='monospace')
    fig.suptitle('Fig. N5 | Sample efficiency and commutator spectral separation',fontsize=15,weight='bold')
    for kind in ['png','pdf']:
        fig.savefig(OUT/f'T11_FigN5_gap_sample_efficiency.{kind}',dpi=230,bbox_inches='tight')
    plt.close(fig)


def draw_legacy(legacy):
    d=legacy[legacy.method=='M3']
    fig,axs=plt.subplots(1,3,figsize=(12.5,3.85),layout='constrained')
    colors={0.2:'#2686a9',1.0:'#d27d31',10.0:'#538c54'}
    for tar in (0.2,1.0,10.0):
        a=d[(d.axis=='N')&(d.target_R==tar)].sort_values('N')
        axs[0].plot(a.N,a.angle_to_M2_population_deg,'o-',label=rf'$R_A\approx{tar:g}$',color=colors[tar],ms=4)
        axs[1].plot(a.N,a.gap_C_recomputed,'s-',label=rf'$R_A\approx{tar:g}$',color=colors[tar],ms=4)
        axs[2].plot(a.N,a.delta_over_gap,'o-',label=rf'$R_A\approx{tar:g}$',color=colors[tar],ms=4)
    axs[0].set(ylabel='M3 angle to M2(known F), degrees',title='(a) Submitted Fig. 6 reference')
    axs[1].set(ylabel=r'Commutator extreme gap $g_C$',yscale='log',title='(b) Exact known-operator gap')
    axs[2].set(ylabel=r'$\|\widehat C-C\|_2/g_C$',yscale='log',title='(c) Perturbation relative to gap')
    for ax in axs:
        ax.set_xlabel('Network dimension N');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Original Fig. 6 dimension scan reanalyzed without changing its data',fontsize=12,weight='bold')
    for kind in ['png','pdf']:
        fig.savefig(OUT/f'T11_original_Fig06_gap_reanalysis.{kind}',dpi=230,bbox_inches='tight')
    plt.close(fig)


def run():
    t0=time.monotonic()
    with threadpool_limits(limits=1):
        d=generate_rows()
        summary=make_summary(d)
        legacy=legacy_audit()
        # Explicit mathematical domain/count assertions.
        assert d.DK_holds.all()
        assert np.all(d.commutator_perturbation_op<=d.commutator_perturbation_upper_bound+1e-10)
        assert len(d)==len(CFG['N_values'])*len(CFG['competitor_commutator_fraction'])*CFG['independent_seeds']*len(CFG['sampling_regimes'])
        assert np.max(abs(legacy.gap_C_recomputed-legacy.gap_C_saved))<1e-7
        d.to_csv(OUT/'T11_full_gap_sample_rows.csv',index=False,float_format='%.12g')
        summary.to_csv(OUT/'T11_cluster_bootstrap_summary.csv',index=False,float_format='%.10g')
        legacy.to_csv(OUT/'T11_original_Fig06_reanalysis.csv',index=False,float_format='%.12g')
        draw_main(d,summary);draw_legacy(legacy)
    extras={
      'controlled_physical_operators':len(d[['N','competitor_fraction','seed_index']].drop_duplicates()),
      'independent_base_subjects':len(d[['N','seed_index']].drop_duplicates()),
      'controlled_fits':len(d),'original_Fig06_method_rows':len(legacy),
      'DK_inequality_passed':bool(d.DK_holds.all()),
      'DK_nonvacuous':int(d.DK_nonvacuous.sum()),
      'n_lt_N':int((d.n_pairs<d.N).sum()),
      'M3_R_inadmissible':int(d.m3_status.ne('ok').sum()),
      'M2_R_inadmissible':int(d.m2_status.ne('ok').sum()),
      'max_comm_inequality_ratio':float(np.max(d.commutator_perturbation_op/np.maximum(d.commutator_perturbation_upper_bound,1e-15))),
      'max_legacy_gap_discrepancy':float(np.max(abs(legacy.gap_C_recomputed-legacy.gap_C_saved))) }
    for frac in CFG['competitor_commutator_fraction']:
        a=d[(d.competitor_fraction==frac)&(d.regime=='M16_T240')]
        extras['high_data_fraction_'+str(frac)]={'M3_median_angle':float(a.M3_angle_deg.median()),'M2_median_angle':float(a.m2_angle_deg.median()),'DK_nonvacuous_count':int(a.DK_nonvacuous.sum()),'n':len(a)}
    extras['duration_sec']=time.monotonic()-t0
    (OUT/'T11_summary.json').write_text(json.dumps(extras,indent=2))
    report=f'''# T11 — Fig. N5: Sample-efficiency and commutator-gap validation\n\n## Study and provenance\n\nNew controlled experiment: {extras['independent_base_subjects']} independent base operator/innovation subjects, {extras['controlled_physical_operators']} physical operators (four paired commutator strengths per base subject), {extras['controlled_fits']} finite-data fits. N={CFG['N_values']}; competing commutator fractions {CFG['competitor_commutator_fraction']}; 6 sample-size/horizon regimes, including M=25, T=3/4 and M=1, and {CFG['independent_seeds']} independent seeds per physical design. The same orthogonal embedding and innovations are reused at all four competing commutator strengths for each base subject, as well as nested across sampling regimes. Per-condition bootstrap resamples independent base subjects, not overlapping pairs. Existing T01 Figure 6's {len(legacy)} stored method rows are reused **without simulation rerun** and separately audited.\n\n## Exact perturbation facts\n\nLet E=F_hat-F and C(F)=FF^T-F^TF. Then C(F+E)-C(F)=FE^T+EF^T+EE^T-F^TE-E^TF-E^TE, so ||delta C||_2 <=4||F||_2||E||_2+2||E||_2^2. For simple extreme commutator eigenvalues separated by g_C from the nearest interior eigenvalues, the M3 plane has\n\n    sin(theta_max) <= min(1, sqrt(2)*delta/(g_C-delta)), delta<g_C; else <=1.\n\nThis conservative bound follows by applying the residual equation to each extreme eigenvector and combining both with a Frobenius bound. The often-plotted 2*delta/g_C is a *first-order proxy*, **not asserted as a certified global bound**.\n\nAll {len(d)} controlled fits satisfy both the commutator polynomial bound and the conservative Davis-Kahan inequality. Only {extras['DK_nonvacuous']} give a nonvacuous (<1) certified bound; the remaining fits do **not** contradict perturbation theory. Population M3 exactly spans an independent planted Qstar because the primary block has larger extreme commutator eigenvalue than the competing block. The ensemble deliberately decreases g_C while keeping all eigenvalues fixed.\n\n## Empirical outcomes\n\n- Cases with informative n=M(T-1)<N: {extras['n_lt_N']}/{len(d)}. This affects unregularized full-operator identifiability. It is not an N^2 threshold.\n- Inadmissible rate-based ratios: M2 {extras['M2_R_inadmissible']}/{len(d)}; M3 {extras['M3_R_inadmissible']}/{len(d)}. Plane angles include **all** fits regardless of R admissibility.\n- Maximum observed relative difference in separately recomputed original Fig. 6 commutator gaps: {extras['max_legacy_gap_discrepancy']:.3g} absolute. Original Fig. 6 M3 angles are to M2(known F), so **must not** be used as a Davis-Kahan test against population M3 eigenvectors.\n\n### High-information subset M16,T240\n\n| Competitor fraction | Median M3 angle (degrees) | Median M2 angle (degrees) | Nonvacuous D-K bounds / 18 |\n|---|---:|---:|---:|\n'''
    for frac in CFG['competitor_commutator_fraction']:
        z=extras['high_data_fraction_'+str(frac)]
        report+=f'| {frac:g} | {z["M3_median_angle"]:.3f} | {z["M2_median_angle"]:.3f} | {z["DK_nonvacuous_count"]}/{z["n"]} |\n'
    report+='''\n## Scientific qualifications\n\nThe scatter plot in panel (c) marks perturbation ratios >10 at the right boundary as triangles, retaining all such cases visually. A full operator with n>=N informative transitions need not yield a well-estimated commutator plane if its g_C is small. Conversely, plane recovery may succeed when the Frobenius error of F_hat is non-negligible, if the extreme commutator gap is large enough. The mathematical inequality is deliberately conservative, and its frequent vacuity is not a finding of failed recovery. M2 is a nonconvex numerical extractor, and a Davis-Kahan theorem for the symmetric commutator does not automatically apply to its optimized plane. Re-estimation under the same stochastic data is not an independent validation replicate; all interval denominators count seeded operators. The exact original Figure 6 N=50–400, M=1 insets and T=3/4 were previously generated under T01 and are **analyzed here, not newly simulated**.\n\n## Files\n\n`T11_full_gap_sample_rows.csv` (all fitted diagnostics and exclusions), `T11_cluster_bootstrap_summary.csv` (6 independent seeded operator replicates per N × condition, clustered by subject), `T11_original_Fig06_reanalysis.csv` (existing original scan), `T11_FigN5_gap_sample_efficiency.pdf`, `T11_original_Fig06_gap_reanalysis.pdf`, `T11_summary.json`.\n'''
    (OUT/'T11_report.md').write_text(report)
    manifest={'module':'T11','status':'numerically_executed','config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_T01_fig06_sha256':hashlib.sha256((ROOT/CFG['legacy_fig06_csv']).read_bytes()).hexdigest(),
              'python':platform.python_version(),'numpy':np.__version__,'time_sec':extras['duration_sec'],
              'seed_base':CFG['seed_base'],'bootstrap_seed':CFG['bootstrap_seed'],'fitted_rows':len(d),
              'original_fig06_rows':len(legacy),'all_DK_pass':bool(d.DK_holds.all()),
              'output_files':sorted(p.name for p in OUT.iterdir() if p.is_file())}
    (OUT/'T11_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('T11 END:',json.dumps(extras,indent=2),flush=True)
    return d,summary,legacy

if __name__=='__main__':run()
