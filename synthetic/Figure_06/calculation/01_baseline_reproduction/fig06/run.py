#!/usr/bin/env python3
"""Regenerate submitted Fig. 6 with corrected rates-based R and explicit failure flags.

Original grids (M,N,T), fixed rho=0.95, and Gaussian forcing are retained.
The operator's kappa is *recalibrated* for the three specified R regimes,
because the submitted multiplier-based definition was incorrect; unlike Figs.
3--5's fixed-operator comparisons, these are NEW physical operators.
Population reference is M2(F), not a planted extraction-independent plane.
Multistart M2; time-ordered train/validation within each trajectory; relative
ridge; no forced stabilization; no fabricated values for invalid logarithms.
"""
from __future__ import annotations
import os,sys,json,time,csv,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
BASE=P.parents[1]
sys.path.insert(0,str(BASE))
from shared.nndr_synthetic_operators import fixed_spectrum_family, operator_at_kappa
from shared.nndr_planes import m2_multistart, principal_angle_deg
from shared.nndr_thresholds import reduced_diagnostics, spectral_radius
CFG=json.loads((P/'config.json').read_text())
OUT=P/'outputs'
OUT.mkdir(parents=True,exist_ok=True)

def hashfile(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def m3(F):
    C=F@F.T-F.T@F
    ev,V=np.linalg.eigh(C)
    if abs(ev[0])+abs(ev[-1])<1e-11: return None
    return np.column_stack([V[:,-1],V[:,0]])

def operator_bank(n):
    return fixed_spectrum_family(n=n,rho=CFG['rho'],delta_sigma=CFG['delta_sigma'],seed=CFG['family_seed_offset']+n)

@lru_cache(None)
def target_operator(n, target):
    family=operator_bank(n)
    def eval_R(k,high_quality=False):
        F=operator_at_kappa(family,float(k))
        q=m2_multistart(F,seed=0,restarts=CFG['reference_restarts'] if high_quality else 8,
                        iterations=CFG['reference_iterations'] if high_quality else 220)['Q']
        R=reduced_diagnostics(q.T@F@q)['R_variance']
        return float(R)
    # Find first monotone-ish bracket without assuming exact kappa power law.
    grid=np.geomspace(1,2000,24)
    rv=np.asarray([eval_R(float(k)) for k in grid])
    brackets=[]
    for i in range(len(grid)-1):
        if np.isfinite(rv[i]) and np.isfinite(rv[i+1]) and (rv[i]-target)*(rv[i+1]-target)<=0:
            brackets.append((float(grid[i]),float(grid[i+1])))
    if brackets:
        a,b=brackets[0]
        k=float(brentq(lambda z:np.log(eval_R(np.exp(z))/target),np.log(a),np.log(b),xtol=2e-4))
        k=float(np.exp(k))
    else:
        valid=np.isfinite(rv)&(rv>0)
        idx=np.argmin(abs(np.log(rv[valid]/target)))
        k=float(grid[np.where(valid)[0][idx]])
    F=operator_at_kappa(family,k)
    result=m2_multistart(F,seed=0,restarts=CFG['reference_restarts'],iterations=CFG['reference_iterations'])
    Q=result['Q']
    dd=reduced_diagnostics(Q.T@F@Q)
    if dd['status']!='ok':raise RuntimeError(f'Population R invalid: n={n} target={target} {dd["status"]}')
    assert abs(spectral_radius(F)-CFG['rho'])<1e-6
    return {'F':F,'Q':Q,'kappa':k,'R':float(dd['R_variance']), 'R_sigma':float(dd['R_sigma']),
            'objective_spread':float(np.ptp(result['restart_objectives']))}

def make_states(F,noise):
    """States x1...xT from x0=0, shape (M,T,N), paired without cross-trajectory transitions."""
    M,T,n=noise.shape
    x=np.zeros((M,n));state=np.empty_like(noise)
    for t in range(T):
        x=x@F.T+noise[:,t,:]
        state[:,t,:]=x
    return state

def split_pairs(states):
    m,t,n=states.shape
    X=np.concatenate([np.zeros((m,1,n)),states[:,:-1,:]],axis=1)
    return X,states

def ridge_fit(G,H,nu):
    n=G.shape[0];lam=float(nu)*float(np.trace(G))/n
    return np.linalg.solve(G+lam*np.eye(n),H).T,lam

def fit_with_cv(states):
    X,Y=split_pairs(states)
    M,T,n=X.shape
    cut=max(2,int(np.floor(T*CFG['cv_fraction']))) if T>=3 else 1
    cut=min(cut,T-1)
    Xtr=np.ascontiguousarray(X[:,:cut].reshape(-1,n));Ytr=np.ascontiguousarray(Y[:,:cut].reshape(-1,n))
    Xv=np.ascontiguousarray(X[:,cut:].reshape(-1,n));Yv=np.ascontiguousarray(Y[:,cut:].reshape(-1,n))
    G=Xtr.T@Xtr;H=Xtr.T@Ytr
    scale=max(float(np.trace(G))/n,1e-12)
    errors=[]
    eye=np.eye(n)
    for nu in CFG['ridge_relative_grid']:
        pred=Xv@np.linalg.solve(G+nu*scale*eye,H)
        errors.append(float(np.mean((pred-Yv)**2)))
    i=int(np.argmin(errors));nu=float(CFG['ridge_relative_grid'][i])
    XF=X.reshape(-1,n); YF=Y.reshape(-1,n)
    FF,lam=ridge_fit(XF.T@XF,XF.T@YF,nu)
    return FF,nu,lam,float(errors[i]),int(M*(T-1))

def one_condition(axis,x,n,m,t,target,ref,bank):
    F,Qref=ref['F'],ref['Q']
    states=make_states(F,bank[:m,:t,:n])
    Fhat,nu,lam,cv_mse,npairs=fit_with_cv(states)
    err=Fhat-F
    errop=float(np.linalg.norm(err,ord=2))
    errf=float(np.linalg.norm(err,ord='fro')/np.linalg.norm(F,ord='fro'))
    rhohat=spectral_radius(Fhat)
    comm_F=F@F.T-F.T@F
    c_ref=np.linalg.eigvalsh(comm_F)
    gap=float(min(c_ref[1]-c_ref[0],c_ref[-1]-c_ref[-2]))
    comhat=Fhat@Fhat.T-Fhat.T@Fhat
    errcomm=float(np.linalg.norm(comhat-comm_F,ord=2))
    bound=float(2*errcomm/gap) if gap>0 else np.inf
    rows=[]
    rmethods={
      'M2':m2_multistart(Fhat,seed=0,restarts=CFG['m2_restarts'],iterations=CFG['m2_iterations'])['Q'],
      'M3':m3(Fhat)}
    for method,Q in rmethods.items():
      d=reduced_diagnostics(Q.T@Fhat@Q) if Q is not None else {}
      valid=d.get('status')=='ok' and np.isfinite(d.get('R_variance',np.nan)) and d['R_variance']>0
      Re=float(d['R_variance']) if valid else np.nan
      signed_log=float(np.log10(Re/ref['R'])) if valid else np.nan
      R_sig=float(d.get('R_sigma',np.nan))
      angle=principal_angle_deg(Qref,Q) if Q is not None else np.nan
      rows.append(dict(axis=axis,x=x,N=n,M=m,T_train=t,target_R=target,kappa=ref['kappa'],
        R_ref=ref['R'],R_hat=Re,R_sigma=R_sig,valid=int(valid),status=d.get('status','plane_unavailable'),
        signed_log_ratio=signed_log,absolute_log_ratio=abs(signed_log),
        angle_deg=angle,operator_error_F=errf,operator_error_op=errop,rho_Fhat=rhohat,
        nu_rel=nu,ridge_lambda=lam,cv_mse=cv_mse,n_pairs=npairs,
        population_commutator_gap=gap,commutator_perturbation_op=errcomm,
        DK_bound=bound,method=method))
    return rows

def summary_by_condition(rows):
    # Single random realization per condition: do not call medians over conditions confidence intervals.
    return [{k:r[k] for k in ('axis','x','N','M','T_train','target_R','R_ref','kappa','method','status','valid','R_hat','R_sigma','angle_deg','signed_log_ratio','absolute_log_ratio','operator_error_F','operator_error_op','rho_Fhat','nu_rel','ridge_lambda','cv_mse','n_pairs','population_commutator_gap','commutator_perturbation_op','DK_bound')} for r in rows]

def writecsv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plots(rows):
    colors={0.2:'#3977aa',1.0:'#d87221',10.0:'#22944e'}
    fig,ax=plt.subplots(3,2,figsize=(10.2,10.15),layout='constrained')
    for row,(axis,xv,topnote) in enumerate([('M',CFG['M_values'],'$N=100$, $T_{train}=200$'),
            ('N',CFG['N_values'],'$M=25$, $T_{train}=200$'),
            ('T',CFG['T_values'],'$M=25$, $N=100$')]):
        L,R=ax[row]
        for tar in CFG['reference_R_targets']:
            for method in ['M2','M3']:
                rr=[r for r in rows if r['axis']==axis and r['target_R']==tar and r['method']==method]
                rr=sorted(rr,key=lambda r:r['x'])
                style='-' if method=='M2' else '--';marker='o' if method=='M2' else '^'
                xx=[r['x'] for r in rr]
                L.plot(xx,[r['angle_deg'] for r in rr],color=colors[tar],ls=style,marker=marker,ms=3.5,lw=1.25,
                       label=f'{method}, $R_{{ref}}\\approx{tar:g}$')
                R.plot(xx,[r['absolute_log_ratio'] for r in rr],color=colors[tar],ls=style,marker=marker,ms=3.5,lw=1.25)
        ref=sorted([r for r in rows if r['axis']==axis and r['target_R']==1.0 and r['method']=='M2'],key=lambda r:r['x'])
        tw=L.twinx()
        tw.plot([r['x'] for r in ref],[r['operator_error_F'] for r in ref],color='0.55',linestyle=':',lw=1.2,alpha=.8)
        tw.set_ylabel(r'$\|\hat F-F\|_F/\|F\|_F$',fontsize=8,color='0.45')
        tw.tick_params(axis='y',labelsize=7,colors='0.45')
        L.set_ylabel('Largest plane angle (degrees)')
        R.set_ylabel(r'$|\log_{10}(\hat R/R_{ref})|$')
        R.axhline(np.log10(2),ls=':',c='0.45',lw=.9)
        R.text(.98,np.log10(2)+.045,'factor-of-two error',transform=R.get_yaxis_transform(),ha='right',fontsize=7,color='.38')
        L.text(.04,.95,topnote,transform=L.transAxes,ha='left',va='top',fontsize=8)
        L.set_ylim(bottom=0,top=95)
        if axis in ('M','T'):
            L.set_xscale('log');R.set_xscale('log')
        L.set_xticks(xv);R.set_xticks(xv)
        if axis in ('M','T'):
            L.set_xticklabels([str(z) for z in xv]);R.set_xticklabels([str(z) for z in xv])
        L.set_xlabel({'M':'Number of trajectories $M$','N':'State dimension $N$','T':'Training horizon $T_{train}$'}[axis])
        R.set_xlabel({'M':'Number of trajectories $M$','N':'State dimension $N$','T':'Training horizon $T_{train}$'}[axis])
        for a in (L,R):
            a.tick_params(direction='in',top=True,right=True,labelsize=7)
            a.grid(alpha=.1)
        L.text(-.13,1.03,f'({chr(97+2*row)})',transform=L.transAxes,fontweight='bold',fontsize=10)
        R.text(-.13,1.03,f'({chr(98+2*row)})',transform=R.transAxes,fontweight='bold',fontsize=10)
    ax[0,0].legend(loc='center right',fontsize=7,ncol=2,frameon=False)
    # Inset single-trajectory results, R_ref near one.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    for row,axis in [(1,'N'),(2,'T')]:
        aa=inset_axes(ax[row,0],width='40%',height='36%',loc='upper right',borderpad=1)
        bb=inset_axes(ax[row,1],width='40%',height='36%',loc='upper right',borderpad=1)
        data=sorted([r for r in rows if r['axis']==axis+'_M1' and r['method']=='M2'],key=lambda r:r['x'])
        data3=sorted([r for r in rows if r['axis']==axis+'_M1' and r['method']=='M3'],key=lambda r:r['x'])
        for method,rr in [('M2',data),('M3',data3)]:
            xx=[r['x'] for r in rr];ls='-' if method=='M2' else '--'
            aa.plot(xx,[r['angle_deg'] for r in rr],color=colors[1.0],ls=ls,lw=1)
            bb.plot(xx,[r['absolute_log_ratio'] for r in rr],color=colors[1.0],ls=ls,lw=1)
        aa.set_title('Single trajectory, $M=1$',fontsize=6);bb.set_title('Single trajectory, $M=1$',fontsize=6)
        nvalid=sum(r['valid'] for r in (data+data3))
        bb.text(.04,.95,f'valid $R$: {nvalid}/{len(data+data3)}',transform=bb.transAxes,fontsize=6,va='top',ha='left',bbox=dict(fc='white',ec='none',alpha=.8))
        for a in (aa,bb):
            a.tick_params(labelsize=6,direction='in',top=True,right=True)
            if axis=='T':a.set_xscale('log')
    for suf in ('pdf','png'):
        fig.savefig(OUT/f'T01_fig06_scaling_revised.{suf}',dpi=CFG['figure_dpi'])
    plt.close(fig)

def main():
    start=time.time()
    rows=[];records=[]
    with threadpool_limits(limits=1):
        # Common underlying noise across target regimes and nested scans.
        for axis in ('M','N','T','N_M1','T_M1'):
            vals=CFG['M_values'] if axis=='M' else CFG['N_values'] if axis.startswith('N') else CFG['T_values']
            regimes=[1.0] if axis.endswith('_M1') else CFG['reference_R_targets']
            for x in vals:
                n=CFG['M_scan_N'] if axis=='M' else int(x) if axis.startswith('N') else CFG['T_scan_N']
                m=int(x) if axis=='M' else 1 if axis.endswith('_M1') else CFG['N_scan_M'] if axis=='N' else CFG['T_scan_M']
                t=int(x) if axis.startswith('T') else CFG['M_scan_T'] if axis=='M' else CFG['N_scan_T']
                # Inset N: fixed Ttrain; inset T: fixed N.
                seed=CFG['training_seed'] + 100003*n + 31*(1 if axis.endswith('_M1') else 0)
                rng=np.random.default_rng(seed)
                noise=CFG['sigma_noise']*rng.standard_normal((m,t,n))
                for tar in regimes:
                    ref=target_operator(n,float(tar))
                    row=one_condition(axis,x,n,m,t,float(tar),ref,noise)
                    rows.extend(row)
                print(f'{axis}={x} : {len(regimes)} reference regimes; elapsed {time.time()-start:.1f}s',flush=True)
    writecsv(OUT/'T01_fig06_full_results.csv',summary_by_condition(rows))
    with open(OUT/'T01_fig06_target_kappas.csv','w',newline='') as f:
        w=csv.DictWriter(f,['N','target_R','achieved_R','kappa','reference_Rsigma','M2_restart_range'])
        w.writeheader()
        for n in CFG['N_values']:
            for target in CFG['reference_R_targets']:
                r=target_operator(n,float(target))
                w.writerow({'N':n,'target_R':target,'achieved_R':r['R'],'kappa':r['kappa'],
                            'reference_Rsigma':r['R_sigma'],'M2_restart_range':r['objective_spread']})
    plots(rows)
    validity={str(ax):{} for ax in ('M','N','T','N_M1','T_M1')}
    for ax in validity:
        rr=[r for r in rows if r['axis']==ax]
        validity[ax]={'total':len(rr),'valid':sum(r['valid'] for r in rr),'status_counts':{s:sum(r['status']==s for r in rr) for s in sorted({r['status'] for r in rr})}}
    diag=[r for r in rows if r['axis']=='N' and r['target_R']==1. and r['method']=='M3']
    summary={'elapsed_sec':round(time.time()-start,1),'conditions':len(rows)//2,'method_rows':len(rows),'fig6_grid':'full manuscript 8 M × 8 N × 9 horizons plus M1 insets','validity':validity,
             'reference_recalibrated':True,'corrected_original_false_claim':'n_pairs >= N suffices generically for unregularized identifiability, not N²; error advantage is gap-dependent',
             'N_scan_R1':[{k:r[k] for k in ('N','R_ref','angle_deg','operator_error_F','operator_error_op','population_commutator_gap','DK_bound')} for r in diag],
             'source_hashes':{'run_py':hashfile(P/'run.py'),'config':hashfile(P/'config.json')},
             'notes':['One stochastic realization per scan setting; no Monte Carlo CI claimed.',
                      'Population M2 ground truth is extraction-dependent; N1 independent planted ground truth remains pending.',
                      'N/T M=1 panels are plotted as insets, with invalid R preserved as missing.',
                      'Theoretical Davis-Kahan bound can be vacuous and is reported without clipping.',
                      'Effective nonzero-predictor count is M*(T-1); initial x0=0 transitions do not provide independent predictor information.']}
    (OUT/'T01_fig06_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'method_rows':len(rows),'validity':validity,'seconds':summary['elapsed_sec']},indent=2),flush=True)
if __name__=='__main__': main()
