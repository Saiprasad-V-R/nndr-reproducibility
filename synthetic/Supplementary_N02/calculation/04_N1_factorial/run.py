#!/usr/bin/env python3
"""T04 — N1e: independent spectrum/geometry factorial classification.

Definition of the intervention: spectral changes modify decay rates but hold
non-orthogonal eigenvector shear k fixed; geometric changes vary k but hold
all real eigenvalues fixed. The label is set *before* any data or fit.

Features are always differences of two operator estimates on the same base
system, not latent generator parameters. Five grouped CV folds are fitted
using training systems only. Pair-cluster bootstrap of out-of-fold predictions.

Two regimes: known operator / independent planted plane (oracle diagnostic),
and finite-data fitted operator / unsupervised M2 plane (deployed benchmark).
No hidden clipping of invalid rate-log fits, no outcome-based thresholds.
"""
from __future__ import annotations
import os,sys,json,time,hashlib,platform
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.linalg import logm
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score
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
METHODS={
    'NNDR (K, Δ)': ('K','rate_split'),
    'NNDR (K, Kc)': ('K','Kc'),
    'Eigenvectors (κ(P), ρ)': ('eigen_cond','rho'),
    'Singular gain (σ₁, ρ)': ('sigma','rho'),
    'Henrici (dF/||F||, ρ)': ('df_norm','rho'),
    'Snapshot (max G, ρ)': ('Gmax','rho'),
    'Commutator (norm, ρ)': ('commutator_norm','rho'),
    'Kreiss grid (Kreiss, ρ)': ('kreiss_grid','rho'),
    'Abscissa (ω, ρ)': ('numerical_abscissa','rho')
}


def orthogonal(n,rng):
    O,R=np.linalg.qr(rng.standard_normal((n,n)))
    return O*np.where(np.diag(R)>=0,1.,-1.)


def rate_block(g1,g2,k):
    if not (0<g1<g2 and k>=0):raise ValueError('rates/shear invalid')
    a,b=np.exp(-g1),np.exp(-g2)
    return np.array([[a,k*(a-b)],[0.,b]])


def rate_generator(g1,g2,k):
    return np.array([[-g1,k*(g2-g1)],[0.,-g2]])


def system(anchor,kind='baseline'):
    n=anchor['N'];g1,g2,k=anchor['g1'],anchor['g2'],anchor['k0']
    if kind=='geometry':k=anchor['k1']
    if kind=='spectrum':g1,g2=anchor['g1_new'],anchor['g2_new']
    if kind=='normal':k=0.
    B=rate_block(g1,g2,k)
    latent=np.diag(np.r_[np.zeros(2),anchor['background']])
    latent[:2,:2]=B
    F=anchor['O']@latent@anchor['O'].T
    return {'F':F,'latent':latent,'Qstar':anchor['O'][:,:2],
            'block':B,'g1':g1,'g2':g2,'k':k,
            'O':anchor['O'],'background':anchor['background']}


def make_anchor(seed,n):
    rng=np.random.default_rng(seed)
    # Stable, separated real eigenvalues, with nuisance spectral variation.
    g1=float(rng.uniform(.07,.22));ratio=float(rng.uniform(2.4,6.8))
    g2=g1*ratio
    k0=float(np.exp(rng.uniform(np.log(1.6),np.log(9.0))))
    k1=float(k0*np.exp(rng.choice([-1,1])*rng.uniform(.44,1.15)))
    g1_new=float(g1*np.exp(rng.choice([-1,1])*rng.uniform(.30,.90)))
    g2_new=float(g2*np.exp(rng.choice([-1,1])*rng.uniform(.27,.74)))
    # Enforce separation without changing the geometry k.
    if g2_new<=1.6*g1_new:g2_new=1.6*g1_new
    bg=np.sort(rng.uniform(.07,.46,n-2))  # diagonal stable normal background
    # Strict distinctness of background eigenvalues avoids a degenerate eigenbasis.
    bg=np.linspace(.06,.42,n-2)+rng.uniform(-.003,.003,n-2)
    bg=np.clip(bg,.04,.44)
    return {'N':n,'seed':seed,'g1':g1,'g2':g2,'k0':k0,'k1':k1,
            'g1_new':g1_new,'g2_new':g2_new,'O':orthogonal(n,rng),'background':bg}


def simulate_estimate(sys_,seed,cfg=CFG):
    rng=np.random.default_rng(seed)
    M,T=int(cfg['M']),int(cfg['T_train']);n=sys_['F'].shape[0]
    latent=sys_['latent'];X=np.zeros((M,n));XX=np.zeros((n,n));YX=np.zeros((n,n))
    for _ in range(T):
        Y=X@latent.T+cfg['noise_sd']*rng.standard_normal((M,n))
        XX+=X.T@X;YX+=Y.T@X
        X=Y
    lam=float(cfg['ridge_relative']*np.trace(XX)/n)
    fhat_latent=np.linalg.solve(XX+lam*np.eye(n),YX.T).T
    O=sys_['O']
    return O@fhat_latent@O.T,lam


def kernel_derivative(g1,g2,k,t):
    """Exact derivative of ||exp(L t)||_F^2 for planted rate block."""
    t=np.asarray(t,float)
    a,b=np.exp(-g1*t),np.exp(-g2*t)
    return -2*g1*a*a-2*g2*b*b + 2*k*k*(a-b)*(-g1*a+g2*b)


def kernel_truth(g1,g2,k,grid=800):
    # Upper bound gives enough time for delayed amplification and avoids
    # declaring infinitesimal negative tail as false positive.
    t=np.linspace(.0001,12/min(g1,g2),grid)
    d=kernel_derivative(g1,g2,k,t)
    # More accurate around positive maximum by local refinement.
    idx=int(np.argmax(d))
    if 0<idx<len(t)-1:
        from scipy.optimize import minimize_scalar
        res=minimize_scalar(lambda v:-float(kernel_derivative(g1,g2,k,v)),
                            bounds=(t[idx-1],t[idx+1]),method='bounded')
        peak=max(float(d[idx]),-float(res.fun))
    else:peak=float(d[idx])
    return peak,bool(peak>1e-8)


def full_departure(F):
    eig=np.linalg.eigvals(F)
    d2=np.linalg.norm(F,'fro')**2-float(np.sum(np.abs(eig)**2))
    return float(np.sqrt(max(0.,d2)))


def kreiss_approx(F,cfg=CFG):
    """Explicit *lower bound* grid to Kreiss constant, not supremum proof.

    z=r exp(i theta), r>1. Include 1 (r->infty limit).
    """
    n=F.shape[0];I=np.eye(n);best=1.
    for r in cfg['kreiss_radii']:
        for theta in np.linspace(0,2*np.pi,int(cfg['kreiss_n_angles']),endpoint=False):
            z=r*np.exp(1j*theta)
            smallest=float(np.linalg.svd(z*I-F,compute_uv=False)[-1])
            if smallest>0:
                best=max(best,float((r-1)/smallest))
    return best


def full_metrics(F,known_generator=None,cfg=CFG):
    ev,V=np.linalg.eig(F)
    vn=V/np.maximum(1e-15,np.linalg.norm(V,axis=0,keepdims=True))
    cond=float(np.linalg.cond(vn))
    eigabs=np.abs(ev);rho=float(np.max(eigabs))
    smax=float(np.linalg.svd(F,compute_uv=False)[0]);df=full_departure(F)
    C=F@F.T-F.T@F
    comm=float(np.linalg.norm(C,'fro')/max(np.linalg.norm(F,'fro')**2,1e-18))
    G=[]
    for t in cfg['growth_horizons']:
        G.append(float(np.linalg.svd(np.linalg.matrix_power(F,int(t)),compute_uv=False)[0]))
    if known_generator is not None:
        L=known_generator
        omega=float(np.linalg.eigvalsh((L+L.T)/2)[-1]);abscissa_status='true_generator'
    elif np.all(eigabs>1e-12) and not np.any((np.abs(ev.imag)<1e-9)&(ev.real<0)):
        L=logm(F)
        if np.max(np.abs(L.imag))<1e-7*max(1.,np.max(np.abs(L.real))):
            L=np.real(L)
            omega=float(np.linalg.eigvalsh((L+L.T)/2)[-1]);abscissa_status='ok'
        else:omega=float('nan');abscissa_status='complex_principal_log'
    else:omega=float('nan');abscissa_status='no_real_principal_log'
    return {'rho':rho,'sigma':smax,'eigen_cond':cond,
            'df_norm':float(df/np.linalg.norm(F,'fro')),
            'Gmax':max(G),'max_G_horizon':int(cfg['growth_horizons'][np.argmax(G)]),
            'G4':G[cfg['growth_horizons'].index(4)],
            'commutator_norm':comm,'kreiss_grid':kreiss_approx(F,cfg),
            'numerical_abscissa':omega,'abscissa_status':abscissa_status,
            'full_one_step_gain':bool(smax>1+1e-10)}


def reduced_metrics(F,Q):
    d=reduced_diagnostics(Q.T@F@Q)
    rates=np.linalg.eigvals(Q.T@F@Q)
    if d['status']=='ok':
        log_rate=np.log(np.abs(rates));delta=float(abs(log_rate[0]-log_rate[1]))
        return {'K':float(d['K']),'Kc':float(d['Kc_variance']),
                'R':float(d['R_variance']),'R_sigma':float(d['R_sigma']),
                'rate_split':delta,'reduced_status':'ok',
                'rho_reduced':float(d['rho']),'sigma_reduced':float(d['sigma_max'])}
    return {'K':np.nan,'Kc':np.nan,'R':np.nan,'R_sigma':d['R_sigma'],
            'rate_split':np.nan,'reduced_status':d['status'],
            'rho_reduced':float(d['rho']),'sigma_reduced':float(d['sigma_max'])}


def evaluate(sys_,mode,seed,cfg=CFG):
    if mode=='population':
        F=sys_['F']; Q=sys_['Qstar']
        n=sys_['F'].shape[0]
        L=np.diag(np.r_[np.zeros(2),np.log(sys_['background'])]);L[:2,:2]=rate_generator(sys_['g1'],sys_['g2'],sys_['k'])
        L=sys_['O']@L@sys_['O'].T
        reg=np.nan
    elif mode=='inferred':
        F,reg=simulate_estimate(sys_,seed,cfg)
        Q=m2_multistart(F,seed=seed+199,restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])['Q']
        L=None
    else:raise ValueError(mode)
    full=full_metrics(F,L,cfg);red=reduced_metrics(F,Q)
    return {**full,**red,'plane_angle_deg':principal_angle_deg(Q,sys_['Qstar']),
            'ridge_lambda':reg,'fit_rel_Ferr':float(np.linalg.norm(F-sys_['F'],'fro')/np.linalg.norm(sys_['F'],'fro')),
            'mode':mode}


def field_diffs(a,b,fields):
    vals=[]
    for f in fields:
        u,v=a.get(f,np.nan),b.get(f,np.nan)
        if not (np.isfinite(u) and np.isfinite(v)):
            vals.append(np.nan);continue
        if f=='numerical_abscissa':
            vals.append(abs(float(v-u)))
        else:
            vals.append(float(abs(np.log(max(float(v),1e-12)/max(float(u),1e-12)))))
    return vals


def classify(raw,mode,cfg=CFG,methods=METHODS):
    """Five group-disjoint OOF folds; equal 2-feature count per method."""
    sub=raw[raw['mode']==mode]
    outputs=[];foldrows=[]
    for method,fields in methods.items():
        x=sub[[f'delta_{f}' for f in fields]].to_numpy(float)
        mask=np.all(np.isfinite(x),axis=1)
        s=sub.loc[mask].copy();X=x[mask];y=s.label.to_numpy(int);groups=s.anchor_id.to_numpy()
        if len(np.unique(groups))<cfg['cv_folds'] or len(np.unique(y))<2:continue
        p=np.full(len(s),np.nan);pred=np.full(len(s),-1);fold_ids=np.full(len(s),-1)
        for fold,(train,test) in enumerate(GroupKFold(n_splits=cfg['cv_folds']).split(X,y,groups)):
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight='balanced',solver='liblinear',random_state=cfg['seed']))
            model.fit(X[train],y[train]);p[test]=model.predict_proba(X[test])[:,1]
            pred[test]=model.predict(X[test]);fold_ids[test]=fold;
            foldrows.append({'mode':mode,'method':method,'fold':fold,
                             'n_train':len(train),'n_test':len(test),'accuracy':accuracy_score(y[test],pred[test]),
                             'auc':roc_auc_score(y[test],p[test]),'train_groups':len(np.unique(groups[train])),
                             'test_groups':len(np.unique(groups[test]))})
        result=s[['anchor_id','N','label','change']].copy()
        result['mode']=mode;result['method']=method;result['pred']=pred;result['proba']=p
        result['is_correct']=(pred==y).astype(int);result['fold']=fold_ids
        outputs.append(result)
    return pd.concat(outputs,ignore_index=True),pd.DataFrame(foldrows)


def boot_metrics(pred,cfg=CFG):
    """Paired, anchor-cluster bootstrap of OOF predictions; array-only resampling."""
    from scipy.stats import rankdata
    rng=np.random.default_rng(cfg['seed']+1234)
    g=pred.anchor_id.to_numpy(); gids=np.unique(g)
    idxgroups=[np.flatnonzero(g==k) for k in gids]
    yall=pred.label.to_numpy(int);pr=pred.pred.to_numpy(int)
    pp=pred.proba.to_numpy(float)
    B=int(cfg['bootstrap_replicates'])
    accs=np.empty(B);aucs=np.full(B,np.nan)
    for j in range(B):
        selected=rng.integers(0,len(gids),size=len(gids))
        ix=np.concatenate([idxgroups[i] for i in selected])
        yy=yall[ix];prob=pp[ix]
        accs[j]=float(np.mean(yy==pr[ix]))
        n1=int(yy.sum());n0=len(yy)-n1
        if n1 and n0:
            rank=rankdata(prob,method='average')
            aucs[j]=float((rank[yy==1].sum()-n1*(n1+1)/2)/(n0*n1))
    aucs=aucs[np.isfinite(aucs)]
    return dict(accuracy=float(pred.is_correct.mean()),AUC=float(roc_auc_score(pred.label,pred.proba)),
                accuracy_CI_low=float(np.quantile(accs,.025)),accuracy_CI_high=float(np.quantile(accs,.975)),
                AUC_CI_low=float(np.quantile(aucs,.025)),AUC_CI_high=float(np.quantile(aucs,.975)))


def make_normal_controls(anchors,cfg=CFG):
    rows=[]
    for n in cfg['dimensions']:
        for a in [x for x in anchors if x['N']==n][:cfg['normal_controls_per_dimension']]:
            s=system(a,'normal')
            for mode in ['population','inferred']:
                r=evaluate(s,mode,a['seed']+900000,cfg)
                rows.append({'anchor_id':a['seed'],'N':n,'mode':mode,
                             'rho':r['rho'],'sigma':r['sigma'],'R':r['R'],
                             'R_sigma':r['R_sigma'],'K':r['K'],
                             'reduced_status':r['reduced_status'],
                             'plane_angle_deg':r['plane_angle_deg'],
                             'df_norm':r['df_norm']})
    return pd.DataFrame(rows)


def common_support_scoring(pairs,cfg=CFG):
    """Refit all comparisons on the *exact same* eligible paired rows.

    Do not compare AUC/accuracy across differently screened denominators.
    No validation-label leakage: filtering uses availability of features only.
    """
    method_names=['NNDR (K, Δ)','NNDR (K, Kc)',
                  'Eigenvectors (κ(P), ρ)','Singular gain (σ₁, ρ)',
                  'Henrici (dF/||F||, ρ)','Snapshot (max G, ρ)',
                  'Kreiss grid (Kreiss, ρ)']
    methods={k:METHODS[k] for k in method_names}
    cols=sorted(set(x for ff in methods.values() for x in ff))
    sub=pairs[pairs['mode']=='inferred'].copy()
    keep=np.all(np.isfinite(sub[[f'delta_{k}' for k in cols]].to_numpy(float)),axis=1)
    s=sub.loc[keep].copy()
    prediction,folds=classify(s,'inferred',cfg,methods=methods)
    summaries=[]
    for method,z in prediction.groupby('method'):
        m=boot_metrics(z,cfg)
        summaries.append({'method':method,'n_pairs':len(z),
                          'n_unique_anchors':z.anchor_id.nunique(),**m})
    summary=pd.DataFrame(summaries)
    # Paired bootstrap of accuracy *differences*: each anchor contributes all
    # its valid interventions, with identical rows for every method.
    wide=prediction.pivot(index=['anchor_id','change'],columns='method',values='is_correct').reset_index()
    ref='NNDR (K, Δ)';rng=np.random.default_rng(cfg['seed']+5610)
    gr=wide.anchor_id.to_numpy();gid=np.unique(gr)
    indexes=[np.flatnonzero(gr==g) for g in gid]
    bootstrap_index=np.random.default_rng(cfg['seed']+5610).integers(0,len(gid),size=(cfg['bootstrap_replicates'],len(gid)))
    paired=[]
    for method in method_names:
        if method==ref:continue
        delta=(wide[method]-wide[ref]).to_numpy(float)
        boots=[]
        for choice in bootstrap_index:
            ix=np.concatenate([indexes[i] for i in choice])
            boots.append(np.mean(delta[ix]))
        paired.append({'baseline_method':method,'reference_method':ref,
                       'paired_accuracy_difference':float(np.mean(delta)),
                       'CI_low':float(np.quantile(boots,.025)),
                       'CI_high':float(np.quantile(boots,.975)),
                       'n_pairs':len(delta)})
    return prediction,folds,summary,pd.DataFrame(paired)


def repeat_stability(anchors,cfg=CFG):
    """Independent stochastic repeats at fixed F, without changing the operator."""
    rows=[]
    for n in cfg['dimensions']:
        chosen=[a for a in anchors if a['N']==n][:cfg['repeat_estimation_anchors_per_dimension']]
        for a in chosen:
            s=system(a,'baseline')
            r1=evaluate(s,'inferred',a['seed']+39,cfg)
            r2=evaluate(s,'inferred',a['seed']+778,cfg)
            for metric in ['K','Kc','R','eigen_cond','rho','sigma','Gmax','df_norm','commutator_norm','kreiss_grid']:
                x,y=r1.get(metric,np.nan),r2.get(metric,np.nan)
                # sample standard deviation/mean for TWO independent estimates
                cv=float(abs(x-y)/(np.sqrt(2)*(abs(x)+abs(y))/2)) if np.isfinite(x) and np.isfinite(y) and abs(x)+abs(y)>1e-12 else np.nan
                rows.append({'anchor_id':a['seed'],'N':n,'metric':metric,'estimate1':x,'estimate2':y,
                             'two_repeat_CV':cv,'status1':r1['reduced_status'],
                             'status2':r2['reduced_status']})
    return pd.DataFrame(rows)


def calculate(cfg=CFG):
    OUT.mkdir(exist_ok=True)
    rows=[];pairs=[];controls=[];anchors=[];t0=time.perf_counter()
    for n in cfg['dimensions']:
      for ix in range(cfg['anchors_per_dimension']):
        seed=int(cfg['seed']+n*10000+ix*101)
        a=make_anchor(seed,n);anchors.append(a)
        baseline=system(a,'baseline')
        for kind in ['baseline','geometry','spectrum']:
            s=system(a,kind)
            for mode in ['population','inferred']:
                s.update({'O':a['O'],'background':a['background']})
                m=evaluate(s,mode,seed+{'baseline':39,'geometry':40,'spectrum':41}[kind],cfg)
                peak,yes=kernel_truth(s['g1'],s['g2'],s['k'],cfg['numerical_kernel_grid'])
                rows.append({'anchor_id':seed,'N':n,'change':kind,'mode':mode,
                             'true_k':s['k'],'true_g1':s['g1'],'true_g2':s['g2'],
                             'true_R':reduced_diagnostics(s['block'])['R_variance'],
                             'kernel_peak_derivative':peak,'kernel_growth_truth':yes,
                             'true_K':s['k'],
                             'true_full_sigma':float(np.linalg.svd(s['F'],compute_uv=False)[0]),**m})
        if (ix+1)%10==0:
            print(f'N={n} completed {ix+1}/{cfg["anchors_per_dimension"]}, elapsed {time.perf_counter()-t0:.1f}s',flush=True)
    values=pd.DataFrame(rows)
    cols=list(set(f for fs in METHODS.values() for f in fs))
    for (seed,mode),ss in values.groupby(['anchor_id','mode']):
        base=ss[ss.change=='baseline'].iloc[0]
        for change in ['geometry','spectrum']:
            post=ss[ss.change==change].iloc[0]
            record={'anchor_id':seed,'mode':mode,'N':int(base.N),'change':change,
                    'label':int(change=='geometry'),'true_R0':base.true_R,'true_R1':post.true_R,
                    'true_K0':base.true_K,'true_K1':post.true_K,
                    'true_g1_0':base.true_g1,'true_g1_1':post.true_g1,
                    'true_g2_0':base.true_g2,'true_g2_1':post.true_g2,
                    'status0':base.reduced_status,'status1':post.reduced_status}
            for col in cols:
                record[f'delta_{col}']=field_diffs(base,post,[col])[0]
            pairs.append(record)
    pairs=pd.DataFrame(pairs)
    normal=make_normal_controls(anchors,cfg)
    common_preds,common_folds,common_summary,paired_diffs=common_support_scoring(pairs,cfg)
    repeats=repeat_stability(anchors,cfg)
    preds=[];folds=[]
    for mode in ['population','inferred']:
        a,b=classify(pairs,mode,cfg);preds.append(a);folds.append(b)
    preds=pd.concat(preds,ignore_index=True);folds=pd.concat(folds,ignore_index=True)
    summaries=[]
    for (mode,method),q in preds.groupby(['mode','method']):
        m=boot_metrics(q,cfg)
        summaries.append({'mode':mode,'method':method,'n_pairs':len(q),
                          'n_unique_anchors':q.anchor_id.nunique(),
                          'n_excluded':len(pairs[pairs['mode']==mode])-len(q),**m})
    summary=pd.DataFrame(summaries)
    return values,pairs,normal,preds,folds,summary,common_preds,common_folds,common_summary,paired_diffs,repeats,time.perf_counter()-t0


def make_plot(values,pairs,normal,summary,common_summary,cfg=CFG):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(2,2,figsize=(10.5,7.7)); a,b,c,d=ax.flat
    # One-step and delayed-variance cross-over are DISTINCT criteria.
    q=values[(values['mode']=='population')&(values.change!='baseline')]
    a.scatter(q.true_R,q.kernel_peak_derivative,c=np.where(q.change=='geometry','C0','C1'),
              s=15,alpha=.5)
    a.axhline(0,ls=':',color='black',lw=.9);a.axvline(1,ls='--',color='gray',lw=1)
    a.set(xscale='log',xlabel=r'Planted $R_A$ (rates)',ylabel=r'Maximum $d\|e^{Lt}\|_F^2/dt$',
          title='Delayed-variance kernel crossing (oracle)')
    # score comparison from actual, finite-data estimators.
    ss=common_summary.sort_values('accuracy')
    y=np.arange(len(ss));b.barh(y,ss['accuracy'],color='#5084a2',alpha=.85)
    b.errorbar(ss['accuracy'],y,
               xerr=[ss['accuracy']-ss['accuracy_CI_low'],ss['accuracy_CI_high']-ss['accuracy']],
               fmt='none',ecolor='#202020',capsize=2,elinewidth=.7)
    b.set_yticks(y,ss['method'],fontsize=7);b.axvline(.5,ls=':',color='gray')
    b.set(xlim=(0,1.035),xlabel='Grouped held-out classification accuracy',
          title=f'Common-support test (n={int(ss.n_pairs.iloc[0])} paired changes)')
    # paired feature separation, without design labels as predictors.
    pp=pairs[pairs['mode']=='inferred']
    for i,label in enumerate(['geometry','spectrum']):
        x=pp[pp.change==label]
        c.scatter(x['delta_K'],x['delta_rate_split'],s=15,alpha=.48,
                  color=f'C{i}',label=label)
    c.set(xscale='log',yscale='log',xlabel=r'|change in log inferred $K$|',
          ylabel=r'|change in log inferred $\Delta$|',
          title='Finite-data feature differences (all pairs)')
    c.legend(frameon=False,fontsize=8)
    # oracle vs fitted, and normal spectral controls.
    for method in ['NNDR (K, Δ)','Eigenvectors (κ(P), ρ)','Singular gain (σ₁, ρ)',
                   'Henrici (dF/||F||, ρ)','Snapshot (max G, ρ)']:
        x=summary[summary.method==method]
        if len(x)==2:
            xp=x.set_index('mode');d.plot([0,1],[xp.loc['population','accuracy'],xp.loc['inferred','accuracy']],
                                         'o-',label=method,ms=4,lw=1.3)
    d.axhline(.5,ls=':',color='gray')
    d.set(xticks=[0,1],xticklabels=['Known (n=180)','Fitted (varying n)'],ylim=(0,1.04),
          ylabel='Held-out accuracy',title='Cost of operator/plane estimation')
    d.legend(loc='lower left',fontsize=6.7,frameon=False)
    for i,axes in enumerate(ax.flat):
        axes.text(-.10,1.04,f'({"abcd"[i]})',transform=axes.transAxes,fontsize=11,fontweight='bold')
        axes.grid(alpha=.15)
    fig.subplots_adjust(left=.12,right=.98,top=.92,bottom=.09,hspace=.44,wspace=.34)
    for ext in ['pdf','png']:
        fig.savefig(OUT/f'T04_FigN1e_factorial.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def report(values,pairs,normal,summary,common_summary,paired_diffs,repeats,elapsed,cfg=CFG):
    v=values[values['mode']=='population']
    eligible=v[np.abs(v.true_R-1)>.06]
    consistency=float((eligible.kernel_growth_truth==(eligible.true_R>1)).mean())
    lines=['# T04 — N1e factorial spectrum-vs-geometry benchmark','',
           '## Design',
           f'- {cfg["anchors_per_dimension"]} independent anchors at N={cfg["dimensions"]}; every anchor yields a geometry-only and a spectrum-only change.',
           '- Intervention labels are specified by the model generator, before simulating or fitting. No planted k or decay rate enters the classifier.',
           f'- M={cfg["M"]}, T={cfg["T_train"]}; paired baseline/post design; relative ridge alpha={cfg["ridge_relative"]}; isotropic innovation sd={cfg["noise_sd"]}.',
           '- All classifiers use exactly two measured feature changes, StandardScaler + logistic regression, 5-fold group CV by anchor.',
           f'- Cluster bootstrap: {cfg["bootstrap_replicates"]} anchor resamples of out-of-fold predictions, 95% percentile intervals.',
           '- Population results use independently planted Q*, whereas fitted results use unconstrained multistart M2.',
           '- Kreiss constant is a lower bound via an explicit polar resolvent grid, NOT an optimized global supremum.',
           f'- Runtime: {elapsed:.1f} seconds.', '',
           '## Mathematical controls',
           f'- Independent numerical delayed-kernel-growth truth matches R_A>1 in {100*consistency:.2f}% of {len(eligible)} non-borderline planted cases (|R−1|>.06).',
           f'- Population normal-spectrum controls: {len(normal[normal["mode"]=="population"])}; exact R=0 for all nondegenerate spectral controls.',
           f'- Fitted normal controls: {len(normal[normal["mode"]=="inferred"])}; nonzero inferred R may be caused by finite-data noise or extraction.',
           f'- Estimated reduced R inadmissible in {int((values[values["mode"]=="inferred"].reduced_status!="ok").sum())}/{len(values[values["mode"]=="inferred"])} operator fits (not clipped).',
           '', '## Matched two-feature classification (held-out anchor groups)',
           '| Mode | Features | Accuracy [95% CI] | AUC [95% CI] | Pairs | Excluded |',
           '|---|---|---|---|---:|---:|']
    for _,r in summary.sort_values(['mode','accuracy'],ascending=[True,False]).iterrows():
        lines.append(f'| {r["mode"]} | {r["method"]} | {r["accuracy"]:.3f} [{r["accuracy_CI_low"]:.3f}, {r["accuracy_CI_high"]:.3f}] | {r["AUC"]:.3f} [{r["AUC_CI_low"]:.3f}, {r["AUC_CI_high"]:.3f}] | {int(r["n_pairs"])} | {int(r["n_excluded"])} |')
    lines+=['','## Strict common-support comparison (refitted on identical rows)',
            'All listed methods are independently retrained/tested on the SAME eligible rows and same group-held-out folds. This is the fair primary comparison; the larger table above shows method-specific coverage.',
            '| Features | Accuracy [95% CI] | AUC [95% CI] | Pairs |',
            '|---|---|---|---:|']
    for _,r in common_summary.sort_values('accuracy',ascending=False).iterrows():
        lines.append(f'| {r["method"]} | {r["accuracy"]:.3f} [{r["accuracy_CI_low"]:.3f}, {r["accuracy_CI_high"]:.3f}] | {r["AUC"]:.3f} [{r["AUC_CI_low"]:.3f}, {r["AUC_CI_high"]:.3f}] | {int(r["n_pairs"])} |')
    lines+=['','### Paired accuracy differences on common support (method minus NNDR K,Δ)',
            '| Baseline | Difference [95% cluster-bootstrap CI] |',
            '|---|---:|']
    for _,r in paired_diffs.sort_values('paired_accuracy_difference',ascending=False).iterrows():
        lines.append(f'| {r["baseline_method"]} | {r["paired_accuracy_difference"]:+.3f} [{r["CI_low"]:+.3f}, {r["CI_high"]:+.3f}] |')
    sub=values[values['mode']=='inferred'];ok=sub[sub.reduced_status=='ok'];
    rcorrect=(ok.R>1)==ok.kernel_growth_truth
    sigflag=(sub.R_sigma>1)==(sub.sigma_reduced>1)
    true_growth=sub.true_full_sigma>1
    pred_growth=sub.sigma>1
    lines+=['','## Estimand-specific validation',
            f'- Delayed-variance response crossing: {int(rcorrect.sum())}/{len(ok)} correct among admissible M2-inferred R, with {len(sub)-len(ok)} inadmissible fits reported separately.',
            f'- Reduced one-step R_sigma criterion agrees with sigma(reduced)>1 in {int(sigflag.sum())}/{len(sub)} rows (including invalid reduced R separately); the equivalence is applicable only for spectrally stable reduced maps.',
            f'- Full-operator one-step gain (sigma(Fhat)>1 versus true sigma(F)>1): {int((pred_growth==true_growth).sum())}/{len(sub)} correctly classified; distinct from reduced R_sigma.',
            f'- Fitted normal matched-spectrum controls falsely exceed R_A>1 in {int((normal[(normal["mode"]=="inferred")].R>1).sum())} of {int((normal["mode"]=="inferred").sum())} operators; their population R is zero.',
            '','## Independent finite-data estimator stability (two noise realizations)',
            'Sample CV = |x1−x2|/[sqrt(2)·(|x1|+|x2|)/2], computed on identical physical operators. Only valid pairs enter each metric median; report counts.',
            '| Metric | Median two-realization CV | Valid anchors |',
            '|---|---:|---:|']
    for metric,z in repeats.groupby('metric'):
        lines.append(f'| {metric} | {z.two_repeat_CV.median():.3f} | {int(z.two_repeat_CV.notna().sum())}/{len(z)} |')
    lines+=['','## Interpretation','- K and the rates splitting have distinct mechanistic meanings, but conventional eigenvector-condition-number and spectral-radius features may classify this *controlled* design equally well. Do not claim universal superiority.',
            '- The single-plane 2D rate threshold is checked against the independently evaluated derivative of the exact impulse-variance kernel, not merely against itself.',
            '- One-step gain and the reduced R_sigma are distinct from the delayed-variance response threshold. Record false negatives of reduced planes explicitly.',
            '- The classifier discriminates *which factor was changed in a paired experiment*. It does not infer causation from a single observational record.',
            '- Empirical subject-held-out event discrimination is a separate, currently unfinished task.','',
            '## Files','- T04_operator_metrics.csv: each known/fitted baseline and intervention with all metrics and kernel truth.',
            '- T04_intervention_features.csv: two-feature changes, labels and admissibility.',
            '- T04_out_of_fold_predictions.csv: exact held-out predictions per model.',
            '- T04_cv_summary.csv: per-method coverage, NOT matched denominators.',
            '- T04_common_support_summary.csv and T04_paired_accuracy_differences.csv: matched-denominator primary benchmark.',
            '- T04_repeated_estimation_cv.csv: independent finite-data stability comparisons.',
            '- T04_normal_spectrum_controls.csv: population and fitted normal-spectrum checks.',
            '- T04_FigN1e_factorial.pdf/.png: candidate Fig. N1e.','']
    (OUT/'T04_report.md').write_text('\n'.join(lines))


def main():
    cfg=CFG;OUT.mkdir(exist_ok=True)
    values,pairs,normal,preds,folds,summary,common_preds,common_folds,common_summary,paired_diffs,repeats,elapsed=calculate(cfg)
    for name,df in [('T04_operator_metrics',values),('T04_intervention_features',pairs),
                    ('T04_normal_spectrum_controls',normal),('T04_out_of_fold_predictions',preds),
                    ('T04_cv_folds',folds),('T04_cv_summary',summary),
                    ('T04_common_support_predictions',common_preds),('T04_common_support_folds',common_folds),
                    ('T04_common_support_summary',common_summary),
                    ('T04_paired_accuracy_differences',paired_diffs),
                    ('T04_repeated_estimation_cv',repeats)]:
        df.to_csv(OUT/(name+'.csv'),index=False)
    make_plot(values,pairs,normal,summary,common_summary,cfg);report(values,pairs,normal,summary,common_summary,paired_diffs,repeats,elapsed,cfg)
    meta={'config':cfg,'runtime_seconds':elapsed,'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
          'n_operators':len(values),'n_interventions':len(pairs),
          'n_predictions':len(preds),'n_normal_controls':len(normal),
          'sha256_config':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()}
    (OUT/'T04_manifest.json').write_text(json.dumps(meta,indent=2))
    print('COMMON SUPPORT\n',common_summary[['method','accuracy','AUC','n_pairs']].to_string(index=False));print('COVERAGE-CONTROL\n',summary[['mode','method','accuracy','AUC','n_pairs','n_excluded']].to_string(index=False))
    print(f'DONE T04: {len(values)} operator rows, {len(pairs)} paired rows, {elapsed:.1f}s',flush=True)

if __name__=='__main__':main()
