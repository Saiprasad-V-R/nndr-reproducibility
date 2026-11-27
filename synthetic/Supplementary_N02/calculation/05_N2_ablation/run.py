#!/usr/bin/env python3
"""T05: planted plane/geometry/threshold ablation, full vs reduced capture.

No CLI options.  Applies controlled paired changes to known stable latent
operators, then fits each phase using the same observation and ridge protocols.
R_A is delayed continuous-time variance-response threshold; R_sigma is the
one-step threshold.  Their labels are *not* interchangeable.
"""
from __future__ import annotations
import os,sys,json,hashlib,platform,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm,logm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared.nndr_thresholds import reduced_diagnostics,spectral_radius
from shared.nndr_planes import m2_multistart,principal_angle_deg
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'config.json').read_text())
METHODS=['M2','M3','PCA2','baseline_fixed_M2','baseline_fixed_M3','planted_current','planted_frozen']


def two_block(a,b,k):
    """Fixed-eigenvector non-normal block P diag(a,b) P^-1 (K=k)."""
    return np.array([[a,k*(a-b)],[0.0,b]],dtype=float)


def build_pair(family,intervention,seed,cfg=CFG):
    n=cfg['N'];r=np.random.default_rng(seed)
    O,sign=np.linalg.qr(r.normal(size=(n,n)))
    O=O*np.where(np.diag(sign)>=0,1.,-1.)
    a,b=cfg['base_eigenvalues'];k=cfg['base_K'];a2,b2=cfg['event_spectrum_eigenvalues']
    latent=np.diag(np.r_[np.zeros(2),cfg['background_scale']*np.arange(1,n-1,dtype=float)**(-cfg['background_beta'])])
    latent[:2,:2]=two_block(a,b,k)
    if family=='competing_amplifier':
        latent[2:4,2:4]=np.array(cfg['competing_block'],float)
    latent_e=latent.copy()
    if intervention=='geometry_only':latent_e[:2,:2]=two_block(a,b,cfg['event_K'])
    elif intervention=='spectrum_only':latent_e[:2,:2]=two_block(a2,b2,k)
    elif intervention=='plane_rotation':pass
    elif intervention=='combined':latent_e[:2,:2]=two_block(a2,b2,cfg['event_K'])
    elif intervention=='normal_control':latent[:2,:2]=np.diag([a,b]);latent_e[:2,:2]=np.diag([a2,b2])
    else:raise ValueError(intervention)
    O_event=O.copy()
    if intervention in ['plane_rotation','combined']:
        # A genuine change of the planted 2-plane: rotate its first axis into
        # an orthogonal background direction; O remains orthogonal.
        th=cfg['plane_rotation_rad'];V=np.eye(n)
        V[0,0]=V[5,5]=np.cos(th);V[0,5]=-np.sin(th);V[5,0]=np.sin(th)
        O_event=O@V
    Fbase=O@latent@O.T;Fevent=O_event@latent_e@O_event.T
    if max(spectral_radius(Fbase),spectral_radius(Fevent))>=1:raise AssertionError('Unstable generating operator')
    return [(Fbase,latent,O[:,:2],O), (Fevent,latent_e,O_event[:,:2],O_event)]


def make_data(latent,O,M,T,sd,seed):
    rng=np.random.default_rng(seed);n=latent.shape[0];x=np.zeros((M,n))
    arr=np.empty((M,T+1,n),dtype=float);arr[:,0]=x
    for t in range(T):
        x=x@latent.T+sd*rng.normal(size=(M,n));arr[:,t+1]=x
    return arr@O.T


def fit_operator(data,regularizer):
    X=data[:,:-1,:].reshape(-1,data.shape[-1]);Y=data[:,1:,:].reshape(-1,data.shape[-1])
    S=X.T@X;alpha=regularizer*np.trace(S)/S.shape[0]
    return np.linalg.solve(S+alpha*np.eye(len(S)),X.T@Y).T,alpha


def plane_m3(A):
    C=A@A.T-A.T@A;w,V=np.linalg.eigh(C)
    if max(abs(w[0]),abs(w[-1]))<1e-12:return None
    return V[:,[-1,0]]


def plane_pca(data):
    x=data[:,:-1,:].reshape(-1,data.shape[-1]);w,V=np.linalg.eigh(x.T@x)
    return V[:,[-1,-2]]


def diag_on_plane(A,Q):
    if Q is None:return {'status':'degenerate_plane'}
    return reduced_diagnostics(Q.T@A@Q)


def row_metrics(Ahat,Atrue,Q,Qstar,group,seed,phase,method,intervention,family):
    d=diag_on_plane(Ahat,Q)
    full_sigma=float(np.linalg.svd(Atrue,compute_uv=False)[0]);fit_full_sigma=float(np.linalg.svd(Ahat,compute_uv=False)[0])
    # For the *population* compression use the same estimated plane Q, so it
    # isolates model-fitting error from plane selection.
    pop=diag_on_plane(Atrue,Q)
    keys=['K','Kc_variance','R_variance','R_sigma','R_reactivity','dF','dF_generator','rho','sigma_max']
    row={'subject':group,'seed':seed,'phase':phase,'intervention':intervention,'family':family,'method':method,
         'angle_to_planted_deg':principal_angle_deg(Q,Qstar) if Q is not None else np.nan,
         'plane_population_status':pop.get('status','degenerate_plane'),
         'fit_status':d.get('status','degenerate_plane'),
         'full_rho_true':spectral_radius(Atrue),'full_sigma_true':full_sigma,
         'full_sigma_fitted':fit_full_sigma,
         'full_amplifies_true':bool(full_sigma>1+1e-10),
         'full_amplifies_fitted':bool(fit_full_sigma>1+1e-10),
         'fit_operator_relative_fro_error':float(np.linalg.norm(Ahat-Atrue,'fro')/np.linalg.norm(Atrue,'fro'))}
    for key in keys:
        row[key+'_fit']=float(d.get(key,np.nan));row[key+'_population_same_plane']=float(pop.get(key,np.nan))
    row['c_Q_fit']=row['sigma_max_fit']/fit_full_sigma if np.isfinite(row['sigma_max_fit']) else np.nan
    row['c_Q_population']=row['sigma_max_population_same_plane']/full_sigma if np.isfinite(row['sigma_max_population_same_plane']) else np.nan
    row['missed_full_amplification_population']=bool(full_sigma>1+1e-10 and np.isfinite(row['sigma_max_population_same_plane']) and row['sigma_max_population_same_plane']<=1+1e-10)
    row['missed_full_amplification_fit']=bool(full_sigma>1+1e-10 and np.isfinite(row['sigma_max_fit']) and row['sigma_max_fit']<=1+1e-10)
    row['false_fitted_gain_when_true_full_nonamplifying']=bool(full_sigma<=1+1e-10 and np.isfinite(row['sigma_max_fit']) and row['sigma_max_fit']>1+1e-10)
    row['stable_full_unstable_compression_population']=bool(spectral_radius(Atrue)<1 and np.isfinite(row['rho_population_same_plane']) and row['rho_population_same_plane']>=1)
    return row


def calculate(cfg=CFG):
    rows=[];reference=[];pairs=[];timings=[];start=time.monotonic()
    specs=[(f,t) for f in cfg['families'] for t in cfg['interventions']]
    if cfg['include_normal_control']:specs.append(('isolated_planted','normal_control'))
    for j,(family,intervention) in enumerate(specs):
      for rep in range(cfg['replicates']):
        seed=int(cfg['seed_base']+j*1000+rep*29)
        subject=f'{family}__{intervention}__{rep:03d}'
        systems=build_pair(family,intervention,seed,cfg)
        obs=[]
        for phase,(F,latent,Qstar,O) in zip(['baseline','event'],systems):
            # CRN across phases; physical intervention is the only intended change.
            data=make_data(latent,O,cfg['M'],cfg['T_train'],cfg['noise_sd'],seed+51)
            Fhat,ridge=fit_operator(data,cfg['ridge_relative'])
            q2=m2_multistart(Fhat,seed=seed+(0 if phase=='baseline' else 13)+137,
                            restarts=cfg['m2_restarts'],iterations=cfg['m2_iterations'])['Q']
            q3=plane_m3(Fhat);qpca=plane_pca(data)
            obs.append(dict(F=F,Fhat=Fhat,Qstar=Qstar,q2=q2,q3=q3,pca=qpca,ridge=ridge))
            true_d=diag_on_plane(F,Qstar)
            reference.append({'subject':subject,'seed':seed,'family':family,'intervention':intervention,'phase':phase,
                              'K_star':true_d.get('K',np.nan),'Kc_star':true_d.get('Kc_variance',np.nan),
                              'R_star':true_d.get('R_variance',np.nan),'Rsigma_star':true_d.get('R_sigma',np.nan),
                              'Rreact_star':true_d.get('R_reactivity',np.nan),'dF_map_star':true_d.get('dF',np.nan),
                              'dF_generator_star':true_d.get('dF_generator',np.nan),'Rstar_status':true_d['status'],
                              'full_rho':spectral_radius(F),'full_sigma':float(np.linalg.svd(F,compute_uv=False)[0]),
                              'planted_plane_sigma':float(np.linalg.svd(Qstar.T@F@Qstar,compute_uv=False)[0]),
                              'ridge_abs':ridge})
        qbase=obs[0];q0=qbase['Qstar']
        for phase,cur in zip(['baseline','event'],obs):
            choices={'M2':cur['q2'],'M3':cur['q3'],'PCA2':cur['pca'],
                     'baseline_fixed_M2':qbase['q2'],'baseline_fixed_M3':qbase['q3'],
                     'planted_current':cur['Qstar'],'planted_frozen':q0}
            for method,Q in choices.items():
                rows.append(row_metrics(cur['Fhat'],cur['F'],Q,cur['Qstar'],subject,seed,phase,method,intervention,family))
        pair={'subject':subject,'seed':seed,'family':family,'intervention':intervention,
              'true_plane_rotation_deg':principal_angle_deg(obs[0]['Qstar'],obs[1]['Qstar'])}
        for method in METHODS:
            a=[v for v in rows[-2*len(METHODS):] if v['method']==method]
            assert len(a)==2
            base=next(v for v in a if v['phase']=='baseline');ev=next(v for v in a if v['phase']=='event')
            record=dict(pair,method=method,base_status=base['fit_status'],event_status=ev['fit_status'],
                        base_angle=base['angle_to_planted_deg'],event_angle=ev['angle_to_planted_deg'],
                        base_c_Q_population=base['c_Q_population'],event_c_Q_population=ev['c_Q_population'])
            for key in ['K','Kc_variance','R_variance','R_sigma','R_reactivity','sigma_max']:
                va=base[key+'_fit'];vb=ev[key+'_fit']
                record['base_'+key]=va;record['event_'+key]=vb
                record['diff_'+key]=vb-va if np.isfinite(va) and np.isfinite(vb) else np.nan
                record['contrast_log_'+key]=(float(np.log(vb/va)) if np.isfinite(va) and np.isfinite(vb) and va>1e-10 and vb>1e-10 else np.nan)
            a=record['contrast_log_R_variance'];b=record['contrast_log_K'];c=record['contrast_log_Kc_variance']
            record['contrast_identity_residual']=a-(b-c) if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) else np.nan
            pairs.append(record)
      print(f'{family}/{intervention}: {cfg["replicates"]} paired subjects; {time.monotonic()-start:.1f}s',flush=True)
    return pd.DataFrame(rows),pd.DataFrame(reference),pd.DataFrame(pairs),time.monotonic()-start


def ci_group(values,seed=901,reps=1000):
    """Percentile interval across *subject* rows, not individual time steps."""
    arr=np.asarray(values,float);arr=arr[np.isfinite(arr)]
    if len(arr)==0:return (np.nan,np.nan)
    rg=np.random.default_rng(seed);samples=rg.integers(0,len(arr),size=(reps,len(arr)))
    med=np.median(arr[samples],axis=1)
    return tuple(float(x) for x in np.quantile(med,[.025,.975]))


def summarise(rows,refs,pairs,cfg=CFG):
    result=[]
    for (intervention,method),p in rows[rows.phase=='event'].groupby(['intervention','method']):
        a=p['angle_to_planted_deg'].to_numpy(float)
        b=p['c_Q_population'].to_numpy(float)
        valid=(p.fit_status=='ok')
        al,au=ci_group(a,seed=100+len(result),reps=cfg['bootstrap_reps'])
        bl,bu=ci_group(b,seed=700+len(result),reps=cfg['bootstrap_reps'])
        result.append(dict(intervention=intervention,method=method,n_subjects=len(p),valid_R_n=int(valid.sum()),
            median_angle_deg=float(np.nanmedian(a)),angle_ci95_lo=al,angle_ci95_hi=au,
            median_capture_population=float(np.nanmedian(b)),capture_ci95_lo=bl,capture_ci95_hi=bu,
            full_growth_n=int(p.full_amplifies_true.sum()),
            missed_population_n=int(p.missed_full_amplification_population.sum()),
            missed_fit_n=int(p.missed_full_amplification_fit.sum()),
            unstable_compression_population_n=int(p.stable_full_unstable_compression_population.sum()),
            median_R_fit=float(p.R_variance_fit.median()),median_R_population=float(p.R_variance_population_same_plane.median())))
    return pd.DataFrame(result)


def estimand_errors(rows,refs):
    merged=rows.merge(refs[['subject','phase','K_star','Kc_star','R_star','Rsigma_star','Rreact_star','dF_map_star','dF_generator_star']],on=['subject','phase'],validate='many_to_one')
    for est,truth in [('K_fit','K_star'),('Kc_variance_fit','Kc_star'),('R_variance_fit','R_star'),
                      ('R_sigma_fit','Rsigma_star'),('R_reactivity_fit','Rreact_star'),
                      ('dF_fit','dF_map_star'),('dF_generator_fit','dF_generator_star')]:
        merged['error_'+est]=merged[est]-merged[truth]
        merged['absolute_error_'+est]=abs(merged['error_'+est])
    return merged


def decompositions(pairs):
    entries=[]
    for (inter,method),p in pairs.groupby(['intervention','method']):
        v=p.dropna(subset=['contrast_log_K','contrast_log_Kc_variance','contrast_log_R_variance'])
        cK=float(v.contrast_log_K.mean()) if len(v) else np.nan
        cKc=float(v.contrast_log_Kc_variance.mean()) if len(v) else np.nan
        cR=float(v.contrast_log_R_variance.mean()) if len(v) else np.nan
        entries.append({'intervention':inter,'method':method,'n_pairs':len(p),'n_log_admissible':len(v),
                        'cK_mean_log':cK,'cKc_mean_log':cKc,'cR_mean_log':cR,
                        'decomp_residual':cR-cK+cKc if len(v) else np.nan})
    return pd.DataFrame(entries)


def plot(summary,pairs,rows,refs):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.1,7.1),layout='constrained')
    interventions=['geometry_only','spectrum_only','plane_rotation','combined']
    labels=['Geometry only','Spectrum only','Plane rotation','Combined']
    shown=['M2','M3','PCA2','baseline_fixed_M2','planted_current','planted_frozen']
    names=['M2','M3','PCA','Frozen M2','Planted evolving','Planted frozen']
    colors=plt.cm.tab10(np.arange(len(shown)))
    ax=axes[0,0];step=.12;x=np.arange(len(interventions))
    for j,(method,label) in enumerate(zip(shown,names)):
        vals=[float(summary[(summary.intervention==s)&(summary.method==method)].median_angle_deg.iloc[0]) for s in interventions]
        ax.bar(x+(j-2.5)*step,vals,step*.93,label=label,color=colors[j],alpha=.9)
    ax.set(xticks=x,xticklabels=labels,ylabel='Median event-plane error (degrees)',ylim=(0,95));ax.tick_params(axis='x',rotation=12,labelsize=8)
    ax.text(-.13,1.02,'(a)',transform=ax.transAxes,fontweight='bold')
    ax=axes[0,1]
    for j,(method,label) in enumerate(zip(shown,names)):
        vals=[float(summary[(summary.intervention==s)&(summary.method==method)].median_capture_population.iloc[0]) for s in interventions]
        ax.plot(x,vals,marker='o',lw=1.3,ms=3.5,color=colors[j],label=label)
    ax.axhline(1,color='0.5',ls=':',lw=.9);ax.set(xticks=x,xticklabels=labels,ylim=(0,1.06),ylabel=r'Known-operator capture $\sigma_{\max}(Q^TFQ)/\sigma_{\max}(F)$')
    ax.tick_params(axis='x',rotation=12,labelsize=8);ax.text(-.13,1.02,'(b)',transform=ax.transAxes,fontweight='bold')
    ax=axes[1,0]
    # True-planted spectrum-only contrast to highlight threshold movement vs K.
    q=refs[refs.intervention=='spectrum_only'];p=q.pivot(index='subject',columns='phase',values=['K_star','Kc_star','R_star'])
    for j,(key,label) in enumerate([('K_star',r'$K$'),('Kc_star',r'$K_c$'),('R_star',r'$R_A$')]):
        vals=np.log(p[(key,'event')]/p[(key,'baseline')]).to_numpy(float)
        ax.bar(j,np.median(vals),color=['#2978B5','#DE7D3E','#558A45'][j],width=.65)
        ax.plot(np.full(len(vals),j),vals,'k.',ms=3,alpha=.35)
    ax.axhline(0,color='0.4',lw=.8);ax.set(xticks=np.arange(3),xticklabels=[r'$\Delta\log K$',r'$\Delta\log K_c$',r'$\Delta\log R_A$'],ylabel='Exact planted spectrum intervention contrast')
    ax.text(-.13,1.02,'(c)',transform=ax.transAxes,fontweight='bold')
    ax=axes[1,1]
    sub=summary[summary.intervention.isin(interventions)]
    a=np.arange(len(interventions));w=.15
    for j,(meth,name) in enumerate([('M2','M2'),('M3','M3'),('PCA2','PCA'),('baseline_fixed_M2','Frozen M2')]):
        y=[]
        for inter in interventions:
            ss=sub[(sub.method==meth)&(sub.intervention==inter)].iloc[0]
            y.append(100*ss.missed_population_n/max(ss.full_growth_n,1))
        ax.bar(a+(j-1.5)*w,y,width=w*.94,label=name,color=colors[[0,1,2,3][j]])
    ax.set(xticks=a,xticklabels=labels,ylabel='Missed full-system one-step gain (%)',ylim=(0,105))
    ax.tick_params(axis='x',rotation=12,labelsize=8)
    ax.text(-.13,1.02,'(d)',transform=ax.transAxes,fontweight='bold')
    handles,legends=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='outside lower center',ncol=6,fontsize=8,frameon=False)
    out=OUT/'T05_FigN2ab_plane_threshold_ablation'
    fig.savefig(out.with_suffix('.png'),dpi=200);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)



def plot_stratified(rows):
    """Do not pool planted-block and competing-amplifier error distributions."""
    shown=['M2','M3','PCA2','baseline_fixed_M2','planted_current']
    interventions=['geometry_only','spectrum_only','plane_rotation','combined']
    fig,axs=plt.subplots(2,2,figsize=(10,6.3),layout='constrained')
    sub=rows[(rows.phase=='event') & (rows.intervention.isin(interventions))]
    for i,family in enumerate(['isolated_planted','competing_amplifier']):
        part=sub[sub.family==family]
        for j,(col,ylab) in enumerate([('angle_to_planted_deg','Median event plane angle (deg)'),
                                     ('c_Q_population',r'Known F gain capture $c_Q$')]):
            ax=axs[i,j];x=np.arange(len(interventions));wd=.15
            for k,method in enumerate(shown):
                values=[float(part[(part.intervention==inter)&(part.method==method)][col].median()) for inter in interventions]
                ax.bar(x+(k-2)*wd,values,width=.14,label=method if method!='baseline_fixed_M2' else 'Frozen M2')
            ax.set(xticks=x,xticklabels=['Geometry','Spectrum','Rotation','Combined'],ylabel=ylab,
                   title=('Isolated planted block' if i==0 else 'Competing full-system amplifier'))
            ax.tick_params(axis='x',rotation=12)
            ax.text(-.13,1.03,f'({chr(97+i*2+j)})',transform=ax.transAxes,fontweight='bold')
            if j==0:ax.set_ylim(0,95)
            else:ax.set_ylim(0,1.05)
    hh,ll=axs[0,0].get_legend_handles_labels();fig.legend(hh,ll,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
    out=OUT/'T05_family_stratified_ablation'
    fig.savefig(out.with_suffix('.png'),dpi=200);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)


def make_report(rows,refs,pairs,summary,cfg,secs):
    rq=refs.pivot(index='subject',columns='phase',values=['K_star','Kc_star','R_star'])
    s=rq.loc[[i for i in rq.index if 'spectrum_only' in i]]
    delta={k:float(np.median(np.log(s[(k,'event')]/s[(k,'baseline')]))) for k in ['K_star','Kc_star','R_star']}
    residual=pairs.contrast_identity_residual.to_numpy(float);good=residual[np.isfinite(residual)]
    tot=len(rows);valid=(rows.fit_status=='ok').sum();pop_invalid=(rows.plane_population_status!='ok').sum()
    f=summary[(summary.intervention=='plane_rotation')&(summary.method=='baseline_fixed_M2')].iloc[0]
    g=summary[(summary.intervention=='plane_rotation')&(summary.method=='M2')].iloc[0]
    txt=f'''# T05 — Fig. N2a,b: plane selection and diagnostic ablation\n\n**Design.** {cfg['replicates']} paired subjects per family/intervention, two planted families (isolated and competing amplifiers), four causal interventions plus isolated spectrum-matched normal controls; N={cfg['N']}, M={cfg['M']}, T={cfg['T_train']}, isotropic innovation SD={cfg['noise_sd']}, same noise seeds baseline/event and relative ridge={cfg['ridge_relative']}. Every planted system has a stable full-system spectrum. Real-spectrum planted K is eigenvector-conditioned coupling; Qstar is specified by the simulator, independent of M2/M3.\n\nTotal independent paired subjects: {len(refs)//2}. Operator phases: {len(refs)}. Plane-method phase rows: {tot}; valid inferred R_A: {valid}/{tot}; invalid known-operator projected R_A: {pop_invalid}/{tot}. Inadmissible fits are never clipped or forced stable.\n\n## Three independent estimands\n\n1. **Planted delayed-variance diagnostic:** R_A=K/Kc from the known 2x2 block. This is not a claim about the full operator.\n2. **One-step full-system amplification:** sigma_max(F)>1, independently calculated on the full matrix. The compression capture fraction c_Q=sigma_max(Q^T F Q)/sigma_max(F), bounded by one for orthonormal Q.\n3. **Plane recovery:** principal angle to planted Qstar; when a competing amplifier is stronger, an M2/M3 plane may be a valid amplifier of F without recovering Qstar.\n\nThe corrected **R_A** concerns delayed continuous-time impulse-variance response; **R_sigma** concerns the discrete one-step gain and is checked independently. The reactivity ratio K/K_NA uses the generator-domain threshold (in output `R_reactivity`). K, Kc, R_A, R_sigma, dF(Fitted_Gamma), dF(generator) are retained individually.\n\n## Mathematical and causal checks\n\nPlanted spectrum-only change holds eigenvectors (K) fixed and changes decay rates. Median log effects: c_K={delta['K_star']:+.6f}, c_Kc={delta['Kc_star']:+.6f}, c_R={delta['R_star']:+.6f}, c_K-c_Kc={delta['K_star']-delta['Kc_star']:+.6f}. This is an exact identity for positive *per-system* log ratios, and therefore for averages of log contrasts on their common admissible support; it is not an identity for median(R_event)/median(R_base) in general.\n\nMaximum per-subject exact log-contrast decomposition residual on eligible fitted rows = {np.max(np.abs(good)) if len(good) else np.nan:.3e}; eligible fitted pairs = {len(good)}/{len(pairs)}. This must not be extrapolated to a median-based primary contrast.\n\nRotation-only: baseline-frozen M2 event median planted-plane error {f.median_angle_deg:.2f} deg versus event-refitted M2 {g.median_angle_deg:.2f} deg; true planted R and full gain do not change under orthogonal rotation.\n\n## Full versus reduced limitations\n\nFor full-spectrum stable F, a compression can have spectral radius >=1: recorded `stable_full_unstable_compression_population`. A reduction may also miss true full-operator gain, where sigma_max(F)>1 but sigma_max(Q^T F Q)<=1; these rows remain in all denominators. The true capture ratios use the *known* full F, not Fhat, to avoid circularity.\n\nThis experiment is a **synthetic ablation**, not a proof of superiority on empirical events or of a causal real-world mechanism. Correlation among log contrasts in an event cohort must be evaluated separately from this constructed intervention. Exact R identity does not establish empirical predictivity.\n\nSee `T05_method_summary.csv`, `T05_all_phase_methods.csv`, `T05_all_paired_contrasts.csv`, `T05_planted_reference.csv` for all raw values, exclusions and independent-subject 95% percentile bootstrap intervals (n={cfg['replicates']} per design cell, or 2n where two families pool). Runtime {secs:.2f}s.\n'''
    (OUT/'T05_report.md').write_text(txt)
    return txt


def main():
    begin=time.monotonic();r,ref,p,elapsed=calculate()
    sm=summarise(r,ref,p)
    estimand=estimand_errors(r,ref);decomp=decompositions(p)
    estimand.to_csv(OUT/'T05_estimand_errors_all_rows.csv',index=False)
    decomp.to_csv(OUT/'T05_log_decomposition_by_method.csv',index=False)
    r.to_csv(OUT/'T05_all_phase_methods.csv',index=False)
    ref.to_csv(OUT/'T05_planted_reference.csv',index=False)
    p.to_csv(OUT/'T05_all_paired_contrasts.csv',index=False)
    sm.to_csv(OUT/'T05_method_summary.csv',index=False)
    plot(sm,p,r,ref);plot_stratified(r);report=make_report(r,ref,p,sm,CFG,elapsed)
    manifest={'module':'T05','seed_base':CFG['seed_base'],'config':CFG,'n_paired_subjects':len(ref)//2,
              'n_method_rows':len(r),'valid_rates_rows':int((r.fit_status=='ok').sum()),
              'n_population_invalid_rows':int((r.plane_population_status!='ok').sum()),
              'runtime_seconds':float(time.monotonic()-begin),'python':platform.python_version(),
              'numpy':np.__version__,'pandas':pd.__version__,
              'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()}
    (OUT/'T05_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('TOTAL',len(ref)//2,'pairs',len(r),'method rows','valid',manifest['valid_rates_rows'],'runtime',manifest['runtime_seconds'],flush=True)
    print(report[:1000]);return r,ref,p,sm

if __name__=='__main__':main()
