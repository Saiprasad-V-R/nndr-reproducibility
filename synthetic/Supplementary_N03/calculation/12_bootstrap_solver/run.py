#!/usr/bin/env python3
"""T12: dependence-preserving bootstrap interval coverage / M2 solver audit.

RUN in Spyder or Python without CLI arguments. Generates independent VAR(1)
subjects from six pre-specified operator families. Stratified moving-block
resampling uses contiguous paired (x_t,x_t+1) observations and resamples
independent trajectories as clusters, rather than pretending each time point
is independent. A *fixed planted-plane oracle* is explicitly separated from
*data-selected M2*, whose population target is the M2 plane of known F.

This is an empirical finite-design calibration, not a universal coverage proof.
Bias-corrected percentile = BC, not BCa; the bootstrap does not estimate
acceleration and no coverage claim is made for invalid / unidentifiable fits.
"""
from __future__ import annotations
import os
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[v]='1'
import sys,json,time,hashlib,platform,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.nndr_planes import m2_multistart,principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics,spectral_radius

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CFG=json.loads((HERE/'config.json').read_text())
METRICS={'R_A':'R_variance','R_sigma':'R_sigma','sigma_max':'sigma_max'}

def orthogonal(seed,n):
    rng=np.random.default_rng(seed)
    q,r=np.linalg.qr(rng.standard_normal((n,n)))
    return q*np.where(np.diag(r)>=0,1.,-1.)

def cblock(s):
    r,theta=.78,.5
    return np.array([[r*np.cos(theta), -r*np.sin(theta)*np.exp(s)],
                     [r*np.sin(theta)*np.exp(-s),r*np.cos(theta)]])

def planted_latent(family,n=6):
    if n<6:raise ValueError('n must be at least 6')
    bg=[.23,.33,.41,.14] if n==6 else np.linspace(.14,.41,n-2).tolist()
    if family=='real_near_normal': b= np.array([[.88,.03],[0.,.50]])
    elif family=='real_strong':b=np.array([[.88,.80],[0.,.50]])
    elif family=='complex_weak':b=cblock(.20)
    elif family=='complex_strong':b=cblock(1.50)
    elif family=='repeated_real':b=np.array([[.82,.35],[0.,.82]])
    elif family=='near_tied_commutator':
        k=.72;a,b=.89,.4;ac,bc=.83,.48
        c=k*np.hypot(k,a-b);frac=.98
        z=(-(ac-bc)**2+np.sqrt((ac-bc)**4+4*(frac*c)**2))/2
        k2=float(np.sqrt(z))
        return block_diag(np.array([[a,k],[0.,b]]),np.array([[ac,k2],[0.,bc]]),*([.15,.25] if n==6 else np.linspace(.12,.35,n-4)))
    else:raise ValueError(family)
    return block_diag(b,*bg)

def make_subject(family,seed,n=6):
    B=planted_latent(family,n)
    O=orthogonal(seed,B.shape[0])
    F=O@B@O.T
    Qstar=O[:,:2]
    dstar=reduced_diagnostics(Qstar.T@F@Qstar)
    assert dstar['status']=='ok'
    assert spectral_radius(F)<1
    return {'F':F,'B':B,'O':O,'Qstar':Qstar,'oracle_truth':dstar}

def simulate_subject(F,M,T,burnin,noise_sd,seed):
    rng=np.random.default_rng(seed)
    X=np.zeros((M,F.shape[0]))
    for _ in range(burnin):
        X=X@F.T+noise_sd*rng.standard_normal(X.shape)
    states=np.empty((M,T+1,F.shape[0]));states[:,0]=X
    for t in range(T):
        X=X@F.T+noise_sd*rng.standard_normal(X.shape)
        states[:,t+1]=X
    return states

def sufficient_statistics(states):
    x=states[:,:-1,:];y=states[:,1:,:]
    return np.einsum('mti,mtj->ij',x,x),np.einsum('mti,mtj->ij',y,x)

def fit_gram(xx,yx,ridge_rel):
    n=len(xx)
    lam=ridge_rel*np.trace(xx)/n
    return np.linalg.solve(xx+lam*np.eye(n),yx.T).T

def block_length(T):
    return max(7,int(round(1.5*T**(1/3))))

def block_grams(states,L):
    """Exact paired-block Grams of lengths L and terminal remainder R.

    An independently sampled trajectory contributes precisely T paired
    transitions, and no block concatenation boundary is treated as a pair.
    """
    x=states[:,:-1,:];y=states[:,1:,:];M,T,n=x.shape
    if not(1<=L<=T):raise ValueError('Block length out of range')
    R=T-(math.ceil(T/L)-1)*L
    xx=np.einsum('mti,mtj->mtij',x,x)
    yx=np.einsum('mti,mtj->mtij',y,x)
    def sums(z,length):
        pref=np.concatenate([np.zeros((M,1,n,n)),np.cumsum(z,axis=1)],axis=1)
        return pref[:,length:,:,:]-pref[:,:-length,:,:]
    return {'full':(sums(xx,L),sums(yx,L)),
            'tail':(sums(xx,R),sums(yx,R)),'lengths':(L,R)}

def bootstrap_fit(blocks,rng,M,T,L,ridge_rel):
    """Resample trajectory labels then within-trajectory paired blocks.

    Exactly M*T transition pairs enter the bootstrap fit; trajectories are
    resampled whole at the outer level, and blocks stay within one trajectory.
    """
    (bx,by)=blocks['full'];(tx,ty)=blocks['tail']
    K=int(math.ceil(T/L));R=T-(K-1)*L
    assert blocks['lengths']==(L,R)
    picks=rng.integers(0,M,size=M)
    starts=rng.integers(0,T-L+1,size=(M,K-1))
    last=rng.integers(0,T-R+1,size=M)
    Gx=bx[picks[:,None],starts].sum(axis=(0,1))+tx[picks,last].sum(axis=0)
    Gy=by[picks[:,None],starts].sum(axis=(0,1))+ty[picks,last].sum(axis=0)
    return fit_gram(Gx,Gy,ridge_rel)

def bootstrap_raw_pair_indices(M,T,L,seed):
    """Reproduce the transition indices used by one bootstrap draw."""
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,M,size=M)
    K=int(math.ceil(T/L));R=T-(K-1)*L
    starts=rng.integers(0,T-L+1,size=(M,K-1))
    last=rng.integers(0,T-R+1,size=M)
    indices=[]
    for i in range(M):
        idx=np.concatenate([np.arange(s,s+L) for s in starts[i]]+[np.arange(last[i],last[i]+R)])
        indices.append((picks[i],idx))
    return indices

def truth_for_m2(B,cfg=CFG):
    # Strong population M2 numerical reference on unrotated latent map.
    best=m2_multistart(B,seed=cfg['base_seed']+909,restarts=cfg['solver_reference_starts'],iterations=400)
    return best

def numeric_diag(Q,F):
    d=reduced_diagnostics(Q.T@F@Q)
    return {k:float(d.get(v,np.nan)) for k,v in METRICS.items()},d['status']

def bc_bounds(samples,point,level):
    """BC percentile interval, without acceleration; NOT a BCa interval."""
    x=np.asarray(samples,float)
    p=(float(np.sum(x<point))+.5*float(np.sum(x==point)))/len(x)
    p=float(np.clip(p,.5/len(x),1-.5/len(x)))
    z0=norm.ppf(p);za=norm.ppf((1-level)/2)
    alphas=norm.cdf([2*z0+za,2*z0-za])
    return np.quantile(x,alphas)

def interval(samples,point,level,method):
    x=np.asarray(samples,float)
    x=x[np.isfinite(x)]
    if len(x)<2 or not np.isfinite(point):return np.array([np.nan,np.nan])
    if method=='percentile':return np.quantile(x,[(1-level)/2,1-(1-level)/2])
    if method=='bias_corrected':return bc_bounds(x,point,level)
    raise ValueError(method)

def wilson(k,n,level=.95):
    if n<=0:return (float('nan'),float('nan'))
    z=norm.ppf((1+level)/2);ph=k/n
    mid=(ph+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n)
    return (float(mid-half),float(mid+half))

def autocorrelation_ess(states,Q,Lmax=15):
    x=states[:,:-1,:]@Q[:,0]
    # Center per trajectory, estimate positive-lag IAT, stop at first <=0.
    M,T=x.shape
    cen=x-x.mean(axis=1,keepdims=True)
    var=float(np.sum(cen**2))
    if var<1e-20:return float('nan')
    positive=[]
    for lag in range(1,min(Lmax,T//4)+1):
        r=float(np.sum(cen[:,:-lag]*cen[:,lag:])/var)*(T/(T-lag))
        if r<=0:break
        positive.append(r)
    return float(M*T/(1+2*sum(positive)))

def solver_reference(F,starts=24,seed=0):
    """Independent strong numerical lower bound for M2 objective.

    Optimize one unit vector v; optimal u is the unit projection of Fv
    perpendicular to v. Numerical multistart L-BFGS-B does NOT certify a
    global upper bound.
    """
    F=np.asarray(F,float)
    A=F.T@F;H=(F+F.T)/2;n=len(F)
    rng=np.random.default_rng(seed)
    best=-np.inf;bestv=None
    def obj_grad(w):
        wn=float(np.linalg.norm(w))
        if wn<1e-13:return 1e3,np.zeros(n)
        v=w/wn;hv=H@v;h=float(v@hv)
        q=float(v@(A@v)-h*h)
        grad=2*(A@v)-4*h*hv
        grad=(grad-v*(v@grad))/wn
        return -q,-grad
    for _ in range(starts):
        initial=rng.normal(size=n);initial/=np.linalg.norm(initial)
        res=minimize(obj_grad,initial,method='L-BFGS-B',jac=True,
                     options={'maxiter':240,'gtol':2e-10,'ftol':1e-14})
        val=float(np.sqrt(max(0,-res.fun)))
        if val>best:best,bestv=val,res.x/np.linalg.norm(res.x)
    u=F@bestv;u-=bestv*(bestv@u);u/=np.linalg.norm(u)
    Q=np.column_stack([u,bestv])
    return {'objective':float(best),'Q':Q}

def assess_fit(Fhat,Qstar,center_seed,restarts,iterations):
    Qest=m2_multistart(Fhat,seed=center_seed,restarts=restarts,iterations=iterations)
    dor,status_or=numeric_diag(Qstar,Fhat)
    dsel,status_sel=numeric_diag(Qest['Q'],Fhat)
    return {'oracle':dor,'selected_M2':dsel}, {'oracle':status_or,'selected_M2':status_sel},Qest

def run(cfg=CFG):
    OUT.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic();rows=[];cirecs=[];solver_rows=[];boot_rows=[];refs=[]
    family_ref={fam:truth_for_m2(planted_latent(fam),cfg) for fam in cfg['families']}
    for fam in cfg['families']:
        B=planted_latent(fam);ref=family_ref[fam]
        independent=solver_reference(B,starts=cfg['solver_reference_starts'],seed=cfg['base_seed']+124)
        best_reference=max(ref['objective'],independent['objective'])
        Qpop=ref['Q'] if ref['objective']>=independent['objective'] else independent['Q']
        truthpop,statuspop=numeric_diag(Qpop,B)
        dstar,status_star=numeric_diag(np.eye(cfg['N'])[:,:2],B)
        refs.append(dict(family=fam,N=cfg['N'],truth_oracle_R_A=dstar['R_A'],truth_oracle_R_sigma=dstar['R_sigma'],
             truth_oracle_sigma_max=dstar['sigma_max'],truth_M2_R_A=truthpop['R_A'],
             truth_M2_R_sigma=truthpop['R_sigma'],truth_M2_sigma_max=truthpop['sigma_max'],
             population_M2_status=statuspop,reference_objective=best_reference,
             numerical_reference_agreement=abs(ref['objective']-independent['objective']),
             truth_plane_angle_to_planted=principal_angle_deg(Qpop,np.eye(cfg['N'])[:,:2])))
        for ireg,reg in enumerate(cfg['sampling_regimes']):
            M,T=reg['M'],reg['T'];L=block_length(T)
            for i in range(cfg['independent_subjects']):
                subj_seed=cfg['base_seed']+100000*cfg['families'].index(fam)+1000*ireg+i
                subject=make_subject(fam,subj_seed)
                F=subject['F'];Qstar=subject['Qstar'];Qpop_rot=subject['O']@Qpop
                st=simulate_subject(F,M,T,cfg['burnin'],cfg['innovation_sd'],subj_seed+713)
                xx,yx=sufficient_statistics(st)
                Fhat=fit_gram(xx,yx,cfg['ridge_relative'])
                orig,status,Qcenter=assess_fit(Fhat,Qstar,subj_seed+99,
                                  cfg['m2_center_restarts'],cfg['m2_iterations'])
                bs_gram=block_grams(st,L)
                rng=np.random.default_rng(subj_seed+991)
                vals={mode:{v:[] for v in METRICS} for mode in ('oracle','selected_M2')}
                status_count={'oracle':{},'selected_M2':{}}
                b_angles=[];b_obj=[]
                for b in range(cfg['bootstrap_draws']):
                    fb=bootstrap_fit(bs_gram,rng,M,T,L,cfg['ridge_relative'])
                    d,st_codes,qb=assess_fit(fb,Qstar,subj_seed+202000+b,
                            cfg['m2_bootstrap_restarts'],cfg['m2_iterations'])
                    for mode in vals:
                        status_count[mode][st_codes[mode]]=status_count[mode].get(st_codes[mode],0)+1
                        for metric in METRICS:vals[mode][metric].append(d[mode][metric])
                    b_angles.append(principal_angle_deg(qb['Q'],Qpop_rot))
                    b_obj.append(qb['objective'])
                    # Every draw retained for reproducing CIs, including invalid ratios.
                    boot_rows.append(dict(family=fam,regime=reg['name'],subject=i,
                        subject_id=f'{fam}_{reg["name"]}_{i}',draw=b,
                        oracle_status=st_codes['oracle'],selected_status=st_codes['selected_M2'],
                        **{f'{mode}_{metric}':d[mode][metric] for mode in vals for metric in METRICS},
                        selected_angle_to_population_deg=b_angles[-1],selected_M2_objective=b_obj[-1]))
                true={'oracle':dstar,'selected_M2':truthpop}
                qdiff=principal_angle_deg(Qcenter['Q'],Qpop_rot)
                effective=autocorrelation_ess(st,Qstar)
                for mode in vals:
                    for metric in METRICS:
                        bvals=np.asarray(vals[mode][metric],float)
                        nvalid=int(np.isfinite(bvals).sum())
                        point=orig[mode][metric]
                        pop=true[mode][metric]
                        condition=(nvalid >= math.ceil(cfg['bootstrap_minimum_admissible_fraction']*cfg['bootstrap_draws'])
                                and np.isfinite(point) and np.isfinite(pop))
                        for lev in cfg['levels']:
                            for meth in cfg['ci_methods']:
                                bounds=interval(bvals,point,lev,meth) if condition else [np.nan,np.nan]
                                covered=bool(np.isfinite(bounds[0]) and bounds[0]<=pop<=bounds[1])
                                cirecs.append(dict(family=fam,regime=reg['name'],M=M,T=T,
                                    subject=i,subject_id=f'{fam}_{reg["name"]}_{i}',
                                    mode=mode,metric=metric,level=lev,method=meth,
                                    nominal=lev,population_truth=pop,point=point,
                                    lower=bounds[0],upper=bounds[1],width=bounds[1]-bounds[0],
                                    bias=point-pop,covered=covered,estimable=bool(condition),
                                    bootstrap_admissible=nvalid,bootstrap_total=cfg['bootstrap_draws'],
                                    status=status[mode],ESS_heuristic=effective,
                                    transitions=M*T,informative_transitions=M*T,block_length=L,
                                    independent_trajectories=M))
                row=dict(family=fam,regime=reg['name'],M=M,T=T,subject=i,subject_id=f'{fam}_{reg["name"]}_{i}',
                    spectral_kind='complex' if 'complex' in fam else ('repeated_real' if 'repeated' in fam else 'real'),
                    true_population_R_A=dstar['R_A'],true_population_M2_R_A=truthpop['R_A'],
                    point_oracle_R_A=orig['oracle']['R_A'],point_selected_R_A=orig['selected_M2']['R_A'],
                    oracle_status=status['oracle'],selected_status=status['selected_M2'],
                    selected_plane_angle_deg=qdiff,selected_objective=Qcenter['objective'],
                    solver_restart_obj_range=float(np.ptp(Qcenter['restart_objectives'])),
                    bootstrap_median_plane_angle_deg=float(np.median(b_angles)),
                    bootstrap_q90_plane_angle_deg=float(np.quantile(b_angles,.9)),
                    bootstrap_median_objective=float(np.median(b_obj)),
                    ESS_heuristic=effective,block_length=L)
                rows.append(row)
                # Audit 24 independent starts of the selected *fitted* operator.
                if i<cfg['solver_audit_per_family']:
                    strong=solver_reference(Fhat,starts=cfg['solver_reference_starts'],
                                            seed=subj_seed+800)
                    numer_ref=max(strong['objective'],Qcenter['objective'])
                    qs=[];objs=[]
                    for r in range(cfg['solver_single_start_repeats']):
                        ss=m2_multistart(Fhat,seed=subj_seed+5000+r,restarts=1,
                                          iterations=cfg['m2_iterations'])
                        qs.append(ss['Q']);objs.append(ss['objective'])
                    gaps=np.maximum(0,numer_ref-np.asarray(objs))
                    angles=[principal_angle_deg(q,qs[int(np.argmax(objs))]) for q in qs]
                    solver_rows.append(dict(family=fam,regime=reg['name'],subject=i,
                        subject_id=f'{fam}_{reg["name"]}_{i}',M=M,T=T,
                        reference_objective=numer_ref,
                        strong_reference_objective=strong['objective'],
                        best_center_objective=Qcenter['objective'],
                        single_start_best=float(np.max(objs)),single_start_median=float(np.median(objs)),
                        single_start_max_gap=float(np.max(gaps)),
                        single_start_median_gap=float(np.median(gaps)),
                        single_start_fraction_within_1pct=float(np.mean(gaps<=.01*max(numer_ref,1e-10))),
                        max_angle_between_single_start_planes_deg=float(np.max(angles)),
                        median_angle_between_single_start_planes_deg=float(np.median(angles)),
                        center_gap_to_strong_reference=max(0,numer_ref-Qcenter['objective']),
                        selected_plane_angle_to_population_deg=qdiff))
            print(f'T12 {fam}, {reg["name"]}: {cfg["independent_subjects"]} subjects, elapsed {time.monotonic()-t0:.1f}s',flush=True)
    return (pd.DataFrame(rows),pd.DataFrame(cirecs),pd.DataFrame(solver_rows),
            pd.DataFrame(boot_rows),pd.DataFrame(refs))

def summarize(ci,cfg=CFG):
    out=[]
    keys=['family','regime','mode','metric','level','method']
    for tags,g in ci.groupby(keys,sort=False):
        n=len(g);est=int(g.estimable.sum());cov=int(g.covered.sum())
        lo,hi=wilson(cov,est) if est else (np.nan,np.nan)
        ulo,uhi=wilson(cov,n)
        record=dict(zip(keys,tags))
        out.append(dict(record,n_subjects=n,n_estimable=est,n_nonestimable=n-est,
                        n_covered=cov,conditional_coverage=cov/est if est else np.nan,
                        unconditional_coverage=cov/n,
                        conditional_Wilson95_low=lo,conditional_Wilson95_high=hi,
                        unconditional_Wilson95_low=ulo,unconditional_Wilson95_high=uhi,
                        median_CI_width=float(g.width.median()),
                        mean_bias=float(g.bias.mean()),median_bias=float(g.bias.median()),
                        median_ESS_heuristic=float(g.ESS_heuristic.median()),
                        mean_valid_boot_frac=float((g.bootstrap_admissible/g.bootstrap_total).mean())))
    return pd.DataFrame(out)

def plot_figures(summary,ci,solver,subjects,refs):
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':220})
    # Main 2x2 report: coverage of RA and R_sigma, nonestimability, solver gap.
    fig,axs=plt.subplots(2,2,figsize=(12.1,8.3),constrained_layout=True)
    famnames={'real_near_normal':'Real weak','real_strong':'Real strong','complex_weak':'Complex weak',
              'complex_strong':'Complex strong','repeated_real':'Repeated real','near_tied_commutator':'Near-tied'}
    regimes=[r['name'] for r in CFG['sampling_regimes']]
    plotdata=summary[(summary.metric=='R_A')&(summary.level==.95)&(summary.method=='percentile')].copy()
    ax=axs[0,0]
    for i,mode in enumerate(['oracle','selected_M2']):
        d=plotdata[plotdata['mode']==mode].groupby('regime',sort=False).agg({'n_covered':'sum','n_subjects':'sum','n_estimable':'sum'}).reindex(regimes)
        ax.plot(np.arange(len(regimes)),d.n_covered/d.n_subjects,'o-',label='Fixed planted plane' if i==0 else 'Selected M2')
    ax.axhline(.95,color='grey',ls=':',label='Nominal 95%')
    ax.set_xticks(range(len(regimes)),regimes);ax.set_ylim(-.03,1.05)
    ax.set_ylabel('Unconditional 95% interval coverage');ax.set_title('(a) Variance-response threshold $R_A$')
    ax.legend(fontsize=8)
    ax=axs[0,1]
    for met,ls in [('R_A','-'),('R_sigma','--'),('sigma_max',':')]:
        d=summary[(summary['mode']=='selected_M2')&(summary.metric==met)&(summary.level==.95)&(summary.method=='percentile')].groupby('regime').agg({'n_nonestimable':'sum','n_subjects':'sum'}).reindex(regimes)
        ax.plot(np.arange(3),d.n_nonestimable/d.n_subjects,marker='o',ls=ls,label=met)
    ax.set_xticks(range(3),regimes);ax.set_ylim(-.03,1.05)
    ax.set_title('(b) M2: intervals not estimable');ax.set_ylabel('Fraction of subjects');ax.legend(fontsize=8)
    ax=axs[1,0]
    for j,mode in enumerate(['oracle','selected_M2']):
        d=plotdata[(plotdata.regime=='M1_T240')&(plotdata['mode']==mode)].set_index('family').reindex(CFG['families'])
        xs=np.arange(len(d))+(-.16 if j==0 else .16)
        ax.bar(xs,d.unconditional_coverage,width=.3,label='Fixed planted plane' if j==0 else 'Selected M2',alpha=.8)
    ax.axhline(.95,ls=':',color='grey')
    ax.set_xticks(range(len(CFG['families'])),[famnames[f] for f in CFG['families']],rotation=24,ha='right')
    ax.set_ylabel('Unconditional 95% coverage');ax.set_ylim(0,1.02)
    ax.set_title('(c) Dependence on spectral family, $M=1,T=240$');ax.legend(fontsize=8)
    ax=axs[1,1]
    groups=solver.groupby('family').agg({'single_start_fraction_within_1pct':'mean',
                    'median_angle_between_single_start_planes_deg':'median',
                    'single_start_median_gap':'median'}).reindex(CFG['families'])
    ax.bar(range(len(groups)),groups.single_start_fraction_within_1pct,color='#4A77A8',alpha=.83)
    ax.set_xticks(range(len(groups)),[famnames[f] for f in CFG['families']],rotation=24,ha='right')
    ax.set_ylim(0,1.02);ax.set_title('(d) M2 single-start objective stability')
    ax.set_ylabel('Fraction within 1% of strong numerical reference')
    for p in axs.flat:p.grid(alpha=.13,axis='y')
    fig.savefig(OUT/'T12_FigS_bootstrap_solver.png',bbox_inches='tight')
    fig.savefig(OUT/'T12_FigS_bootstrap_solver.pdf',bbox_inches='tight')
    plt.close(fig)
    # Supplement: percentile vs BC and CI widths, plane-angle instability.
    fig,axs=plt.subplots(1,3,figsize=(14.4,4.0),constrained_layout=True)
    groups=summary[(summary.metric=='R_A')&(summary.level==.95)]
    ax=axs[0]
    for method in CFG['ci_methods']:
        for mode,ls in [('oracle','-'),('selected_M2','--')]:
            d=groups[(groups.method==method)&(groups['mode']==mode)].groupby('regime').agg({'n_covered':'sum','n_subjects':'sum'}).reindex(regimes)
            ax.plot(range(3),d.n_covered/d.n_subjects,marker='o',ls=ls,
                    label=f'{"Fixed" if mode=="oracle" else "Selected"}, {method}')
    ax.axhline(.95,color='gray',ls=':');ax.set_xticks(range(3),regimes)
    ax.set_ylim(0,1.03);ax.set_title('(a) 95% $R_A$: percentile vs BC')
    ax.set_ylabel('Unconditional coverage');ax.legend(fontsize=7)
    ax=axs[1]
    d=groups[(groups.method=='percentile')&(groups['mode']=='oracle')]
    for fam in CFG['families']:
        z=d[d.family==fam].set_index('regime').reindex(regimes)
        ax.plot(range(3),z.median_CI_width,marker='o',label=famnames[fam])
    ax.set_yscale('symlog',linthresh=.02);ax.set_xticks(range(3),regimes)
    ax.set_title('(b) Fixed-plane $R_A$ interval width');ax.legend(fontsize=6)
    ax=axs[2]
    solver.boxplot(column='median_angle_between_single_start_planes_deg',by='family',ax=ax,rot=35,grid=False)
    ax.set_title('(c) Different starts can select different planes');ax.set_xlabel('')
    ax.set_ylabel('Median angle to best-start plane (degrees)')
    ax.set_xticklabels([famnames[f] for f in CFG['families']],rotation=25,ha='right',fontsize=7)
    fig.suptitle('')
    fig.savefig(OUT/'T12_bootstrap_solver_sensitivity.png',bbox_inches='tight')
    fig.savefig(OUT/'T12_bootstrap_solver_sensitivity.pdf',bbox_inches='tight')
    plt.close(fig)

def write_report(summary,ci,solver,subjects,refs,cfg=CFG):
    d=summary[(summary.metric=='R_A')&(summary.level==.95)&(summary.method=='percentile')]
    lines=['# T12 — Dependence-preserving bootstrap calibration and M2 solver audit','',
           '## Design','',
           f'- {len(subjects)} independent synthetic subject-recordings; {len(refs)} spectral families; '
           f'{cfg["bootstrap_draws"]} moving-block resamples per record; '
           f'{len(solver)} solver-reference audits with {cfg["solver_single_start_repeats"]} single starts each.',
           f'- Trajectory-stratified bootstrap resamples whole independent trajectories and contiguous blocks of (x_t,x_(t+1)) pairs. '
           'Block boundaries are NOT interpreted as valid transitions. The terminal block is trimmed so exactly M*T transition pairs are used; this is a block-pair bootstrap.',
           '- Stationary burn-in precedes all estimation windows; no artificial independence of adjacent transitions.',
           '- Oracle: planted Qstar fixed in all bootstrap draws; population truth from Qstar.T @ F @ Qstar.',
           '- Selected M2: reoptimized in every bootstrap; population truth uses high-start M2 of exact F, not the planted plane unless they coincide.',
           '- M2 numerical strong reference uses independent multistart L-BFGS-B on the single-vector reduced objective, plus high-start M2. It is NOT a global optimum certificate.',
           '- Percentile and *bias-corrected percentile* (BC, not BCa) evaluated at nominal 90% and 95%. BC uses z0 only, no acceleration.',
           '- A ratio interval is estimable only if the point target is admissible and >=90% of bootstrap replicates yield that ratio; invalids are not clipped.',
           '- Primary coverage denominator is all independent subjects; missing/inadmissible intervals count as noncoverage. Conditional coverage and Wilson bounds also reported.',
           '- Heuristic effective sample size uses a positive-sequence autocorrelation estimate of the fixed planted response projection; not an exact likelihood or independent-sample count.',
           '', '## Unconditional 95% percentile CI coverage for R_A', '',
           '| Sampling | Method | Covered / all subjects | Estimable | Unconditional | Conditional |',
           '|---|---|---:|---:|---:|---:|']
    for reg in [r['name'] for r in cfg['sampling_regimes']]:
        for mode in ['oracle','selected_M2']:
            g=d[(d.regime==reg)&(d['mode']==mode)]
            k,n,e=int(g.n_covered.sum()),int(g.n_subjects.sum()),int(g.n_estimable.sum())
            lines.append(f'| {reg} | {mode} | {k}/{n} | {e}/{n} | {k/n:.3f} | {k/e:.3f} |' if e else f'| {reg} | {mode} | {k}/{n} | {e}/{n} | {k/n:.3f} | NA |')
    lines += ['','## Solver sensitivity', '',
              f'- Median single-start objective-hit fraction within 1% of independent strong numerical reference: {solver.single_start_fraction_within_1pct.median():.3f}.',
              f'- Median over audited datasets of the maximum single-start objective gap: {solver.single_start_max_gap.median():.5g}.',
              f'- Median single-start plane spread relative to best-start plane: {solver.median_angle_between_single_start_planes_deg.median():.2f} degrees.',
              '','## Interpretation','',
              'Bootstrap coverage is specific to this finite ensemble and the stated block length / ridge / fit protocol. '
              'Strongly non-normal regimes, repeated spectra and nearly tied commutator modes may produce nonregular estimates; standard percentile/BC intervals should not be advertised as universally calibrated. '
              'M2 restarts are a numerical uncertainty diagnostic; close objectives do not guarantee the same physical plane when the landscape is nearly degenerate.',
              '','## Files','',
              '- `T12_subject_point_estimates.csv` — subject-level estimates, statuses, ESS and plane angles.',
              '- `T12_bootstrap_draws.csv` — **every bootstrap replicate** including invalid R statuses.',
              '- `T12_interval_results.csv` — every nominal CI and coverage indicator.',
              '- `T12_coverage_by_family_regime.csv` — conditional/unconditional coverage and Wilson bounds.',
              '- `T12_solver_audit.csv` — independent numerical strong-reference comparisons.',
              '- `T12_known_population_references.csv` — each planted versus selected-M2 population truth.',
              '- Two figures PNG/PDF, code, config, tests, run manifest.','']
    (OUT/'T12_report.md').write_text('\n'.join(lines))

def dimension_sensitivity(cfg=CFG):
    """Independent N=6,12,24 control, with same oracle/M2 coverage protocol.

    Eight? The exact number is in config: six subjects per spectral family,
    dimension and sampling regime. This supplemental study uses fewer draws
    than the main 288-record calibration and is exploratory, not pooled with it.
    """
    rows=[];ref_cache={}
    specs=cfg['dimension_sensitivity']
    for n in specs['N_values']:
        for fam in specs['families']:
            B=planted_latent(fam,n)
            ref=m2_multistart(B,seed=cfg['base_seed']+903,restarts=22,iterations=240)
            ref_cache[(n,fam)]=ref['Q']
            truth_or,_=numeric_diag(np.eye(n)[:,:2],B)
            truth_sel,_=numeric_diag(ref['Q'],B)
            for regime in specs['sampling_regimes']:
                M,T=regime['M'],regime['T'];L=block_length(T)
                for j in range(specs['independent_subjects']):
                    seed=cfg['base_seed']+n*100000+cfg['families'].index(fam)*2000+M*100+j
                    subject=make_subject(fam,seed,n=n)
                    states=simulate_subject(subject['F'],M,T,cfg['burnin'],cfg['innovation_sd'],seed+71)
                    xx,yx=sufficient_statistics(states)
                    f=fit_gram(xx,yx,cfg['ridge_relative'])
                    center,st,_=assess_fit(f,subject['Qstar'],seed+119,
                        specs['m2_restarts'],cfg['m2_iterations'])
                    blocks=block_grams(states,L)
                    rng=np.random.default_rng(seed+881)
                    values={'oracle':[], 'selected_M2':[]}
                    for b in range(specs['bootstrap_draws']):
                        fb=bootstrap_fit(blocks,rng,M,T,L,cfg['ridge_relative'])
                        dd,ss,_=assess_fit(fb,subject['Qstar'],seed+55000+b,
                            specs['m2_restarts'],cfg['m2_iterations'])
                        for mode in values:values[mode].append(dd[mode]['R_A'])
                    for mode,true in [('oracle',truth_or['R_A']),('selected_M2',truth_sel['R_A'])]:
                        samples=np.asarray(values[mode],float)
                        valid=int(np.isfinite(samples).sum())
                        point=center[mode]['R_A']
                        feasible=(valid>=math.ceil(.9*specs['bootstrap_draws'])
                                 and np.isfinite(point) and np.isfinite(true))
                        bounds=interval(samples,point,.95,'percentile') if feasible else [np.nan,np.nan]
                        rows.append(dict(N=n,family=fam,regime=regime['name'],M=M,T=T,
                            subject=j,subject_id=f'N{n}_{fam}_{regime["name"]}_{j}',
                            method=mode,population_R_A=true,estimate_R_A=point,
                            valid_bootstrap=valid,total_bootstrap=specs['bootstrap_draws'],
                            status=st[mode],estimable=feasible,ci95_lo=bounds[0],ci95_hi=bounds[1],
                            covers=bool(np.isfinite(bounds[0]) and bounds[0]<=true<=bounds[1])))
            print(f'T12 dimension sensitivity N={n}, family={fam}',flush=True)
    return pd.DataFrame(rows)

def plot_dimension_sensitivity(df):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'savefig.dpi':220})
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.8),sharex=True,sharey=True,constrained_layout=True)
    for i,(fam,name) in enumerate([('real_strong','Strong real spectrum'),
                                   ('near_tied_commutator','Near-tied commutator modes')]):
        for j,(reg,label) in enumerate([('M1_T90','One trajectory, 90 steps'),
                                        ('M8_T240','Eight trajectories, 240 steps')]):
            ax=axs[i,j]
            dd=df[(df['family']==fam)&(df['regime']==reg)]
            for method,ls,color in [('oracle','-','#3477A8'),('selected_M2','--','#DC8637')]:
                g=dd[dd['method']==method].groupby('N').covers.mean()
                ax.plot(g.index,g.values,ls=ls,color=color,marker='o',lw=1.7,ms=5,
                        label='Fixed planted plane' if method=='oracle' else 'Selected M2')
            ax.axhline(.95,ls=':',color='gray',lw=1,label='Nominal 95%')
            ax.set_title(f'{name}; {label}',fontsize=9)
            ax.set_xticks([6,12,24]);ax.grid(alpha=.15,axis='y')
            if j==0:ax.set_ylabel('Unconditional 95% CI coverage')
            if i==1:ax.set_xlabel('Full operator dimension $N$')
            ax.set_ylim(-.025,1.055)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=8)
    fig.suptitle('Exploratory dimension sensitivity: six independent subjects per cell',fontsize=11)
    fig.savefig(OUT/'T12_dimension_sensitivity.png',bbox_inches='tight')
    fig.savefig(OUT/'T12_dimension_sensitivity.pdf',bbox_inches='tight')
    plt.close(fig)

def main():
    st=time.monotonic()
    subjects,ci,solver,bootstrap,refs=run()
    summary=summarize(ci)
    names=[('T12_subject_point_estimates.csv',subjects),('T12_interval_results.csv',ci),
           ('T12_solver_audit.csv',solver),('T12_bootstrap_draws.csv',bootstrap),
           ('T12_known_population_references.csv',refs),('T12_coverage_by_family_regime.csv',summary)]
    for name,d in names:d.to_csv(OUT/name,index=False)
    plot_figures(summary,ci,solver,subjects,refs)
    write_report(summary,ci,solver,subjects,refs)
    dim=dimension_sensitivity()
    dim.to_csv(OUT/'T12_dimension_sensitivity.csv',index=False)
    dim_summary=dim.groupby(['N','family','regime','method'],as_index=False).agg(
        subjects=('subject','size'),estimable=('estimable','sum'),covered=('covers','sum'),
        mean_bias=('estimate_R_A',lambda v:float(np.nanmean(v))))
    dim_summary['unconditional_coverage']=dim_summary.covered/dim_summary.subjects
    dim_summary.to_csv(OUT/'T12_dimension_summary.csv',index=False)
    plot_dimension_sensitivity(dim)
    with open(OUT/'T12_report.md','a') as report:
        report.write('\n## Exploratory dimension sensitivity (separate cohort)\n\n')
        report.write(f'{len(dim)//2} additional independent subject-recordings across N=6,12,24; ')
        report.write('two spectrum families, two sampling regimes, six subjects/cell, 60 bootstraps per recording. ')
        report.write('These smaller independent cohorts are NOT pooled with the 288-record main calibration; ')
        report.write('each cell has wide coverage uncertainty. See T12_dimension_summary.csv and PDF.\n')
    manifest={'module':'T12','UTC_date':'2026-11-04','config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
              'subjects':len(subjects),'bootstrap_draws':len(bootstrap),'interval_rows':len(ci),
              'solver_reference_rows':len(solver),'dimension_sensitivity_records':len(dim)//2,
              'dimension_sensitivity_fits':len(dim),'python':platform.python_version(),
              'numpy':np.__version__,'duration_seconds':time.monotonic()-st,
              'ratios_undefined_are_explicitly_flagged':True,'tested_families':CFG['families']}
    (OUT/'T12_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('T12 COMPLETED',json.dumps(manifest,indent=2),flush=True)
    return subjects,ci,solver,bootstrap,refs,summary

if __name__=='__main__':main()
