#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NNDR seizure EEG Stage 5: VAR(1/2/4) adequacy + event/subject contrasts.

RUN IN SPYDER: put this script in the folder containing chb01_03.edf, or in
its parent CHB-MIT data folder; run without command-line arguments.

Scope:
  - fixed 16-channel cross-patient scalp montage (21-channel pilot optional).
  - 0.5--15 Hz filter at 256 Hz, then polyphase anti-aliased decimation to 32 Hz.
  - 10/20/40 s windows at 2 s hop, baseline endpoints [-180,-40] s,
    strictly-ictal windows distinguished from mixed windows.
  - Original V4 full-window M2/M3 NNDR ratios R_A and R_sigma, with complex
    eigenpairs admitted and negative real multipliers excluded from R_A.
  - Separate HELD-OUT model-adequacy comparison of VAR(1), VAR(2), VAR(4),
    with identical chronological targets, validation-chosen ridge, a held-out
    test set, 1-,2-,4-,8-step forecasts and test residual statistics.
  - Additional V4-fixed ridge/shrink VAR(1), trained only on pre-test data.
  - Optional common-phase-null spotchecks of squared-residual dependence.
  - One event per event-level contrast; one patient per inferential unit.
    chb01 and chb21 are the SAME patient. Do NOT infer from overlapped windows.

Important: the original V4 R_A / R_sigma are NOT applied to VAR(p>1); those
models live in a lag-augmented state space and require a separate derivation.
The five synthetic-reference thresholds are NOT EEG screening criteria.

Requires: numpy, pandas, scipy, mne. No EDF/subject results are fabricated if
recordings are missing. Outputs are written to ./seizure_stage5_var_adequacy_output/.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, svdvals
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, resample_poly

HERE = Path(__file__).resolve().parent
OUT = HERE / 'seizure_stage6_cohort_output'

# DEFAULT: run pilot, then add recordings across patients here.
# These are documented annotated seizure EDFs, NOT invented events.
TARGET_FILES = () # Stage 6 scans old 141-file inventory below
# chb02_16+.edf has an event after 2972 s, providing a complete baseline.
# chb02_16.edf has onset 130 s, too early for a full -180..-40 s baseline;
# its incomplete baseline would be flagged and excluded from subject contrasts.
# To analyze additional EDFs, add filenames AND official chbXX-summary.txt.
# The known intervals below are used ONLY if a summary file is unavailable.
KNOWN_PHYSIONET_EVENTS = {
    'chb01_03.edf': ((2996., 3036.),),
    'chb01_04.edf': ((1467., 1494.),),
    'chb01_15.edf': ((1732., 1772.),),
    'chb02_16.edf': ((130., 212.),),
    'chb02_16+.edf': ((2972., 3053.),),
}

FS_NATIVE, FS = 256., 32.
WINDOW_SECONDS = (10,)
STEP_SECONDS = 2
BASELINE_END_REL = (-180.,-40.)
FILTER_LOW, FILTER_HIGH = .5, 15.
NOTCH_HZ = 60.
V4_RIDGE_REL = .003
V4_SHRINK_ALPHA = .05
M2_RESTARTS, M2_ITERS = 12, 200
ORDERS = (1,2,4)
RIDGE_GRID = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
FORECAST_HORIZONS = (2,4,8)
TEST_TRAIN_FRAC, TEST_VALID_FRAC = .60, .20
PHASE_NULL_SPOTCHECK = False
PHASE_NULL_REPS = 49
PHASE_NULL_ORDER = (1,4)
RNG_SEED = 17011
EPS = 1e-12
TOL = 1e-10

# Explicit V4 21-channel analysis. Across subjects, missing channels cause
# a recorded skip; do not silently change montage/dimension across patients.
CANONICAL_21 = (
    'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2',
    'FZ-CZ','CZ-PZ','T7-FT9','FT9-FT10','FT10-T8'
)


def as_base_channel(name):
    return re.sub(r'-[01]$', '', name.strip().upper().removeprefix('EEG ').replace('-REF','').replace('-LE','').replace(' ',''))


def subject_key(recording):
    m = re.search(r'(chb\d{2})', Path(recording).name, flags=re.I)
    if not m:
        raise ValueError(f'Not a recognized CHB-MIT recording: {recording}')
    case = m.group(1).lower()
    return 'chb01' if case == 'chb21' else case


def file_case(recording):
    m = re.search(r'(chb\d{2})', Path(recording).name, flags=re.I)
    return m.group(1).lower() if m else None


def resolve_target(name):
    # Avoid an expensive EDF-directory-wide traversal unless the file is named.
    candidates = []
    for root in (HERE, HERE.parent):
        p = root/name
        if p.is_file():
            candidates.append(p)
        for c in sorted(root.glob(f'chb*/{name}')):
            if c.is_file(): candidates.append(c)
    unique = list(dict.fromkeys(str(p.resolve()) for p in candidates))
    if not unique: return None, 'EDF_NOT_FOUND'
    if len(unique)>1: return None, f'AMBIGUOUS_MULTIPLE_EDFS: {unique}'
    return Path(unique[0]), 'found'


def parse_summary_text(source):
    """Read official CHB-MIT summary (including Seizure 1 Start Time)."""
    text = Path(source).read_text(encoding='utf-8', errors='replace')
    sections = re.split(r'(?=\bFile Name\s*:)', text, flags=re.I)
    out = {}
    for sec in sections:
        fm = re.search(r'File Name\s*:\s*([^\s]+\.edf)', sec, flags=re.I)
        if not fm: continue
        starts = re.findall(r'Seizure(?:\s+\d+)?\s+Start Time\s*:\s*(\d+(?:\.\d+)?)\s*seconds', sec, flags=re.I)
        ends = re.findall(r'Seizure(?:\s+\d+)?\s+End Time\s*:\s*(\d+(?:\.\d+)?)\s*seconds', sec, flags=re.I)
        nmatch = re.search(r'Number of Seizures in File\s*:\s*(\d+)',sec,flags=re.I)
        if len(starts) != len(ends):
            raise ValueError(f'Unmatched seizure endpoints in {fm.group(1)} of {source}')
        if nmatch and int(nmatch.group(1)) != len(starts):
            raise ValueError(f'Annotation count disagreement for {fm.group(1)} in {source}')
        out[fm.group(1).lower()] = [(float(a),float(b)) for a,b in zip(starts,ends)]
    return out


def get_annotations(edf):
    case = file_case(edf.name)
    probes = [edf.parent/f'{case}-summary.txt', edf.parent.parent/f'{case}-summary.txt', HERE/f'{case}-summary.txt']
    for source in probes:
        if source.is_file():
            ann = parse_summary_text(source)
            if edf.name.lower() not in ann:
                raise ValueError(f'{edf.name} absent from provided {source}; cannot infer seizure times')
            return ann[edf.name.lower()], str(source)
    if edf.name.lower() in KNOWN_PHYSIONET_EVENTS:
        return list(KNOWN_PHYSIONET_EVENTS[edf.name.lower()]), 'explicit_physionet_fallback_in_script'
    raise ValueError(f'Missing official {case}-summary.txt for {edf.name}: annotations required')


def read_eeg(edf):
    import mne
    # V4 preprocessing parity. Suppress only MNE duplicate-name notification;
    # the original and canonical selection is retained in the manifest.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Channel names are not unique.*')
        raw = mne.io.read_raw_edf(str(edf),preload=False,verbose=False)
    native = float(raw.info['sfreq'])
    if not np.isclose(native,FS_NATIVE):
        raise ValueError(f'Expected 256-Hz CHB-MIT EEG, received {native}')
    found = {}
    duplicate = []
    for name in raw.ch_names:
        base = as_base_channel(name)
        if base in found:
            duplicate.append({'name':name,'base':base,'first_name':found[base]})
        else:
            found[base] = name
    missing = [name for name in CANONICAL_21 if name not in found]
    if missing:
        raise ValueError(f'Cannot construct requested fixed montage; missing: {missing}')
    selected = [found[base] for base in CANONICAL_21]
    x = raw.get_data(picks=selected)
    duration_sec = raw.n_times/native
    b,a = iirnotch(NOTCH_HZ,Q=30.,fs=native)
    x = filtfilt(b,a,x,axis=1)
    sos = butter(5,[FILTER_LOW,FILTER_HIGH],btype='bandpass',fs=native,output='sos')
    x = sosfiltfilt(sos,x,axis=1)
    y = resample_poly(x,up=1,down=8,axis=1).T.copy()
    return y, {'edf':str(edf),'native_fs':native,'fs':FS,
               'band':'0.5-15_preAA_to32Hz','all_edf_channels':raw.ch_names,
               'selected_names':selected,'selected_canonical':list(CANONICAL_21),
               'deduplicated_channels':duplicate,'n_channels':len(selected),
               'duration_sec':duration_sec}


def departure(A):
    a,b,c,d=np.asarray(A,float).ravel()
    disc=(a-d)**2+4*b*c
    return float(abs(b-c) if disc >= 0 else np.hypot(a-d,b+c))


def ratio_metrics(G,dt):
    eig=np.linalg.eigvals(G)
    rho=float(max(abs(eig)))
    sigma=float(svdvals(G)[0])
    complex_pair=bool(max(abs(eig.imag))>1e-9*max(1.,rho))
    ans=dict(R_A=np.nan,R_sigma=np.nan,K=np.nan,Kc_A=np.nan,
             complex_pair=complex_pair,rho_G=rho,sigma_G=sigma,
             log_admissible=False,reason='')
    if not rho < 1.-TOL:
        ans['reason']='reduced_unstable'
        return ans
    one_step_margin=float(np.prod(1.-abs(eig)**2))
    if one_step_margin>EPS:
        ans['R_sigma']=float(departure(G)/np.sqrt(one_step_margin))
    gap=abs(eig[0]-eig[1])
    if gap>1e-8:
        ans['K']=float(departure(G)/gap)
    if min(abs(eig))<=1e-12:
        ans['reason']='singular_reduced_map'
        return ans
    if not complex_pair and min(eig.real)<=0:
        ans['reason']='negative_real_multiplier'
        return ans
    L_complex=logm(G)/dt
    if max(abs(L_complex.imag.ravel()))>1e-8*max(1.,float(max(abs(L_complex.real.ravel())))):
        ans['reason']='nonreal_principal_log'
        return ans
    L=L_complex.real
    mu=np.linalg.eigvals(L)
    if max(mu.real)>=-TOL:
        ans['reason']='nondecaying_generator'
        return ans
    if complex_pair:
        gamma=float(-np.mean(mu.real))
        omega=float(max(abs(mu.imag)))
        denom=2*np.sqrt(gamma*(np.hypot(gamma,omega)+gamma))
    else:
        dec=np.sqrt(-mu.real)
        denom=float(dec.sum()*np.sqrt(2*np.prod(dec)))
    if denom<=EPS:
        ans['reason']='response_threshold_margin_too_small'
        return ans
    ans['R_A']=float(departure(L)/denom)
    ans['log_admissible']=True
    if np.isfinite(ans['K']) and ans['R_A']>EPS:
        ans['Kc_A']=float(ans['K']/ans['R_A'])
    return ans


def fit_v4_full_window(X):
    """Same centered full-window ridge and trace shrinkage as V4."""
    Y=X-X.mean(axis=0,keepdims=True)
    A,B=Y[:-1],Y[1:]
    n=Y.shape[1]
    C=A.T@A
    ridge=V4_RIDGE_REL*float(np.trace(C))/n
    F0=np.linalg.solve(C+ridge*np.eye(n),A.T@B).T
    F=(1.-V4_SHRINK_ALPHA)*F0+V4_SHRINK_ALPHA*np.trace(F0)/n*np.eye(n)
    return F


def method2(F):
    rng=np.random.default_rng(0)  # V4 initialization parity
    n=F.shape[0]
    best_val,best_Q=-np.inf,None
    for _ in range(M2_RESTARTS):
        v=rng.normal(size=n)
        v/=np.linalg.norm(v)
        for _ in range(M2_ITERS):
            z=F@v
            z-=v*(v@z)
            if np.linalg.norm(z)<EPS:break
            u=z/np.linalg.norm(z)
            z=F.T@u
            z-=u*(u@z)
            if np.linalg.norm(z)<EPS:break
            nv=z/np.linalg.norm(z)
            if min(np.linalg.norm(v-nv),np.linalg.norm(v+nv))<1e-9:
                v=nv
                break
            v=nv
        z=F@v
        z-=v*(v@z)
        val=np.linalg.norm(z)
        if val>best_val and val>EPS:
            best_val=float(val)
            best_Q=np.linalg.qr(np.column_stack((z/val,v)))[0][:,:2]
    return best_Q,best_val


def method3(F):
    C=F@F.T-F.T@F
    evals,V=np.linalg.eigh(C)
    if max(abs(evals[0]),abs(evals[-1]))<EPS:
        return None,np.nan
    gap=float(min(evals[-1]-evals[-2],evals[1]-evals[0])) if len(evals)>2 else np.nan
    return np.linalg.qr(np.column_stack((V[:,-1],V[:,0])))[0][:,:2],gap


def window_labels(start,end,onset,offset,all_events):
    """No baseline may lie within another annotated seizure or its guard."""
    rel_end=end-onset
    overlap=max(0.,min(end,offset)-max(start,onset))
    other_conflict=any(not (end<=a-40 or start>=b+40)
                       for a,b in all_events if (a,b)!=(onset,offset))
    baseline=(BASELINE_END_REL[0]-1e-8 <= rel_end <= BASELINE_END_REL[1]+1e-8)
    if baseline and other_conflict:label='baseline_contaminated_skip'
    elif baseline:label='baseline'
    elif start>=onset-1e-8 and end<=offset+1e-8:label='strict_ictal'
    elif overlap>0 and end<=offset+1e-8:label='mixed_onset'
    elif overlap>0 and end>offset+1e-8:label='mixed_postictal'
    elif abs(end-onset)<1e-8:label='pre_onset_boundary'
    elif rel_end<0:label='other_preictal'
    else:label='other_postictal'
    return label,overlap/(end-start)


def generate_window_specs(n_samples,onset,offset,all_events,window_sec):
    W=int(round(FS*window_sec))
    hop=int(round(FS*STEP_SECONDS))
    st=max(0,int(round((onset+BASELINE_END_REL[0]-window_sec)*FS)))
    en=min(n_samples,int(round((offset+40.)*FS)))
    for beg in range(st,en-W+1,hop):
        start=beg/FS
        end=(beg+W)/FS
        label,occupancy=window_labels(start,end,onset,offset,all_events)
        # Bounded event-centered analysis, no statement of independent windows.
        yield beg,beg+W,start,end,label,occupancy


def lag_design(Z,p,targets):
    # Predict Z[t] from [Z[t-1],...,Z[t-p]] in a fixed predictor ordering.
    X=np.concatenate([Z[targets-k] for k in range(1,p+1)],axis=1)
    return X,Z[targets]


def ridge_fit(X,Y,relative):
    n=X.shape[1]
    C=X.T@X
    ridge=relative*np.trace(C)/n
    coef=np.linalg.solve(C+ridge*np.eye(n),X.T@Y).T
    return coef


def acf_scalar(v,maxlag=8):
    v=np.asarray(v,float)
    if len(v)<=maxlag+1 or np.std(v)<1e-10:return 0.
    vals=[]
    for k in range(1,maxlag+1):
        s=np.corrcoef(v[:-k],v[k:])[0,1]
        if np.isfinite(s):vals.append(abs(float(s)))
    return max(vals,default=0.)


def residual_dependence(E):
    """Scale-free TEST residual covariance and squared-norm autocorrelation."""
    if len(E)<12: return np.nan,np.nan
    C0=E.T@E/len(E)
    den=float(np.linalg.norm(C0,'fro'))+EPS
    errors=[]
    for k in range(1,min(8,len(E)//3)+1):
        Ck=E[k:].T@E[:-k]/(len(E)-k)
        errors.append(np.linalg.norm(Ck,'fro')/den)
    return float(max(errors)), float(acf_scalar(np.sum(E*E,axis=1)))


def recursive_forecast_mse(Z,coef,p,test_targets,h,intercept=None):
    # All anchors and targets are strictly inside the common chronological
    # test period. Observed data are used for the past, not future steps.
    anchors=test_targets[test_targets+h-1 <= test_targets[-1]]
    if len(anchors)==0:return np.nan
    pred=[]
    if intercept is None:intercept=np.zeros(Z.shape[1])
    for t in anchors:
        hist=[Z[t-k].copy() for k in range(1,p+1)]
        for _ in range(h):
            nxt=coef@np.concatenate(hist)+intercept
            hist=[nxt]+hist[:-1]
        pred.append(hist[0])
    truth=Z[anchors+h-1]
    return float(np.mean((np.asarray(pred)-truth)**2))


def fit_order_models(X_raw):
    """Time-ordered 60/20/20 validation, common targets across orders.

    No random shuffling. Ridge chosen from validation, model refitted on
    train+validation, test never used to choose ridge or order.
    """
    T,n=X_raw.shape
    max_order=max(ORDERS)
    all_targets=np.arange(max_order,T)
    n_all=len(all_targets)
    n_train=int(np.floor(TEST_TRAIN_FRAC*n_all))
    n_valid=int(np.floor(TEST_VALID_FRAC*n_all))
    if n_train<2*max_order+5 or n_valid<10 or n_all-n_train-n_valid<10:
        raise ValueError(f'Window too short for chronological split: T={T}')
    tr=all_targets[:n_train]
    va=all_targets[n_train:n_train+n_valid]
    te=all_targets[n_train+n_valid:]
    # Mean/std ONLY from observed train period, not validation or test.
    mu=X_raw[:tr[-1]+1].mean(axis=0)
    scale=X_raw[:tr[-1]+1].std(axis=0)
    scale=np.maximum(scale,1e-10)
    Z=(X_raw-mu)/scale
    trainval=np.concatenate((tr,va))
    # Common diagonal VAR(1) benchmark across the same held-out target rows.
    diag_X,diag_Y=lag_design(Z,1,trainval)
    d_ridge=V4_RIDGE_REL*np.sum(diag_X**2,axis=0)
    diag_coef=np.sum(diag_X*diag_Y,axis=0)/(np.sum(diag_X**2,axis=0)+d_ridge+EPS)
    baseline_test_X,baseline_test_Y=lag_design(Z,1,te)
    baseline_pred=baseline_test_X*diag_coef
    baseline_mse=float(np.mean((baseline_pred-baseline_test_Y)**2))
    result=[]
    for order in ORDERS:
        Xtr,Ytr=lag_design(Z,order,tr)
        Xva,Yva=lag_design(Z,order,va)
        Xtv,Ytv=lag_design(Z,order,trainval)
        Xt,Yt=lag_design(Z,order,te)
        validation=[]
        for ridge in RIDGE_GRID:
            coeff=ridge_fit(Xtr,Ytr,ridge)
            validation.append(float(np.mean((Xva@coeff.T-Yva)**2)))
        pick=int(np.argmin(validation))
        best_ridge=float(RIDGE_GRID[pick])
        coeff=ridge_fit(Xtv,Ytv,best_ridge)
        pred=Xt@coeff.T
        residual=Yt-pred
        mse=float(np.mean(residual**2))
        whit,sqacf=residual_dependence(residual)
        row=dict(model=f'VAR{order}',order=order,
                 cv_ridge=best_ridge,valid_mse=validation[pick],
                 test_mse=mse,test_log_mse_vs_diagonal=float(np.log((mse+EPS)/(baseline_mse+EPS))),
                 test_resid_lag_cov_max=whit,test_resid_sqacf_max=sqacf,
                 n_train=len(tr),n_validation=len(va),n_test=len(te),
                 baseline_diag_test_mse=baseline_mse,selected_on='validation_only')
        for h in FORECAST_HORIZONS:
            row[f'test_recursive_h{h}_mse']=recursive_forecast_mse(Z,coeff,order,te,h)
        result.append(row)
    # V4 fixed hyperparameters, but fitted using only the pre-test observations.
    # Convert F fitted in physical units to the same standardized coordinates.
    fit_segment=X_raw[:va[-1]+1]
    F=fit_v4_full_window(fit_segment)
    mu_fit=fit_segment.mean(axis=0)
    c=(F*scale[None,:])/scale[:,None]
    # V4 uses its own train+validation centering; retain its intercept when
    # expressing that model in the TRAIN-only standardized coordinates.
    intercept=((mu_fit-mu)-F@(mu_fit-mu))/scale
    Xt,Yt=lag_design(Z,1,te)
    res=Yt-(Xt@c.T+intercept)
    mse=float(np.mean(res**2))
    whit,sqacf=residual_dependence(res)
    fixed=dict(model='VAR1_V4fixed',order=1,cv_ridge=V4_RIDGE_REL,
               valid_mse=np.nan,test_mse=mse,
               test_log_mse_vs_diagonal=float(np.log((mse+EPS)/(baseline_mse+EPS))),
               test_resid_lag_cov_max=whit,test_resid_sqacf_max=sqacf,
               n_train=len(tr),n_validation=len(va),n_test=len(te),
               baseline_diag_test_mse=baseline_mse,
               selected_on='V4_fixed_no_test_tuning')
    for h in FORECAST_HORIZONS:
        fixed[f'test_recursive_h{h}_mse']=recursive_forecast_mse(Z,c,1,te,h,intercept)
    result.append(fixed)
    # Independent selection by validation loss, not test loss.
    cv_winner=min((r for r in result if r['model'] != 'VAR1_V4fixed'),
                  key=lambda r:r['valid_mse'])['model']
    for r in result:
        r['validation_selected_model']=cv_winner
    return result


def common_phase_randomized(X,rng):
    centered=X-X.mean(axis=0,keepdims=True)
    Z=np.fft.rfft(centered,axis=0)
    n=len(centered)
    last=Z.shape[0]-1 if n%2==0 else Z.shape[0]
    if last>1:
        phases=rng.uniform(0.,2*np.pi,size=last-1)
        Z[1:last]*=np.exp(1j*phases)[:,None]
    return np.fft.irfft(Z,n=n,axis=0).real


def phase_null_checks(X,window_specs,event_id,subject,recording):
    if not PHASE_NULL_SPOTCHECK:return []
    # Deterministic predeclared middle windows, once per event and label.
    rng=np.random.default_rng(RNG_SEED+event_id)
    out=[]
    for group in ('baseline','strict_ictal'):
        candidates=[s for s in window_specs if s['window_sec']==10 and s['label']==group]
        if not candidates:continue
        chosen=candidates[len(candidates)//2]
        w=X[chosen['start_idx']:chosen['end_idx']]
        actual=fit_order_models(w)
        null_by_model={f'VAR{order}':[] for order in PHASE_NULL_ORDER}
        for _ in range(PHASE_NULL_REPS):
            z=common_phase_randomized(w,rng)
            null_result={r['model']:r for r in fit_order_models(z)}
            for model in null_by_model:
                null_by_model[model].append(null_result[model]['test_resid_sqacf_max'])
        for model,null_values in null_by_model.items():
            a=next(r for r in actual if r['model']==model)
            null=np.asarray(null_values,float)
            obs=a['test_resid_sqacf_max']
            out.append(dict(recording=recording,subject=subject,event_id=event_id,
                label=group,window_sec=10,model=model,n_null=len(null),
                observed_sqacf=obs,null_median=float(np.median(null)),
                null_q95=float(np.quantile(null,.95)),
                upper_tail_p_exploratory=float((1+np.sum(null>=obs))/(len(null)+1)),
                note='Exploratory common-phase surrogate spotcheck, not EEG adequacy certification'))
    return out


def summarize_event_adequacy(df):
    records=[]
    for keys,g in df.groupby(['recording','subject','event_id','window_sec','label','model']):
        recording,subj,event_id,w,label,model=keys
        if label not in ('baseline','strict_ictal','mixed_onset'):continue
        records.append(dict(recording=recording,subject=subj,event_id=event_id,
             window_sec=w,label=label,model=model,n_windows=len(g),
             median_test_mse=float(g.test_mse.median()),
             median_recursive_h4_mse=float(g.test_recursive_h4_mse.median()),
             median_test_resid_lag_cov_max=float(g.test_resid_lag_cov_max.median()),
             median_test_resid_sqacf_max=float(g.test_resid_sqacf_max.median()),
             median_test_log_mse_vs_diagonal=float(g.test_log_mse_vs_diagonal.median())))
    return pd.DataFrame(records)


def order_pairwise_comparison(order_df):
    """P1 vs P2/P4 on IDENTICAL held-out targets and windows."""
    keys=['recording','subject','event_id','window_sec','start_s','end_s','label']
    p=order_df.pivot_table(index=keys,columns='model',
         values=['test_mse','test_recursive_h4_mse','test_resid_lag_cov_max'],
         aggfunc='first')
    p.columns=[f'{v}_{m}' for v,m in p.columns]
    p=p.reset_index()
    if 'test_mse_VAR1' not in p: return pd.DataFrame()
    for alt in ('VAR2','VAR4','VAR1_V4fixed'):
        for metric in ('test_mse','test_recursive_h4_mse','test_resid_lag_cov_max'):
            ref=f'{metric}_VAR1';other=f'{metric}_{alt}'
            if ref in p and other in p:
                p[f'{metric}_relative_reduction_{alt}_vs_VAR1']=(p[ref]-p[other])/(p[ref]+EPS)
    return p


def summarize_nndr_event(ratio_df):
    result=[]
    keys=['recording','subject','event_id','window_sec','method']
    for group,g in ratio_df.groupby(keys):
        recording,subj,ev,w,method=group
        b=g[g.label=='baseline']
        it=g[g.label=='strict_ictal']
        for measure in ('R_A','R_sigma'):
            ba=pd.to_numeric(b[measure],errors='coerce').dropna()
            ic=pd.to_numeric(it[measure],errors='coerce').dropna()
            bm=float(ba.median()) if len(ba) else np.nan
            im=float(ic.median()) if len(ic) else np.nan
            logratio=float(np.log2(im/bm)) if bm>0 and im>0 else np.nan
            full_baseline=bool(len(b)==71)  # 2 s endpoints, [-180,-40] inclusive
            result.append(dict(recording=recording,subject=subj,event_id=ev,
                window_sec=w,method=method,measure=measure,
                baseline_n_total=len(b),ictal_n_total=len(it),
                baseline_n_expected=71,complete_baseline_window=full_baseline,
                baseline_n_admissible=len(ba),ictal_n_admissible=len(ic),
                baseline_median=bm,strict_ictal_median=im,
                strict_minus_baseline=im-bm,
                log2_strict_over_baseline=logratio,
                valid_event_contrast=bool(np.isfinite(logratio) and full_baseline),
                inferential_unit='patient, NOT windows'))
    return pd.DataFrame(result)


def summarize_recording_and_subject(event_df):
    """Pre-declared hierarchy: event medians -> recording ratio -> subject median.

    When multiple seizures occur in a recording, each seizure first yields
    its own baseline and strict-ictal medians. Within a recording, calculate
    median event median and median baseline median, THEN their log2 ratio.
    Finally take the median of these recording-level log2 ratios per subject.
    """
    eligible=event_df[event_df.valid_event_contrast].copy()
    if eligible.empty:return pd.DataFrame(),pd.DataFrame()
    r=[]
    keys=['subject','recording','window_sec','method','measure']
    for group,g in eligible.groupby(keys):
        subject,rec,w,method,metric=group
        bm=float(g.baseline_median.median())
        im=float(g.strict_ictal_median.median())
        ratio=float(np.log2(im/bm)) if bm>0 and im>0 else np.nan
        r.append(dict(subject=subject,recording=rec,window_sec=w,
            method=method,measure=metric,n_valid_seizure_events=len(g),
            recording_median_event_baseline=bm,
            recording_median_event_ictal=im,
            recording_log2_ratio=ratio,
            note='One recording-level contrast regardless of event count'))
    recording_df=pd.DataFrame(r)
    rows=[]
    for (subj,w,method,metric),g in recording_df.groupby(
        ['subject','window_sec','method','measure']):
        rows.append(dict(subject=subj,window_sec=w,method=method,measure=metric,
              n_recordings=int(g.recording.nunique()),
              n_valid_seizure_events=int(g.n_valid_seizure_events.sum()),
              subject_median_recording_log2_ratio=float(g.recording_log2_ratio.median()),
              n_subjects_independent=1,
              note='One subject contributes one unit; chb01 and chb21 merged'))
    return recording_df,pd.DataFrame(rows)

def compare_with_v4(ratio_df):
    candidates=(HERE/'seizure_reciprocal_channel_ablation_v4_output'/'seizure_window_diagnostics_v4.csv',
                HERE.parent/'seizure_reciprocal_channel_ablation_v4_output'/'seizure_window_diagnostics_v4.csv')
    source=next((x for x in candidates if x.is_file()),None)
    if source is None:return pd.DataFrame(),{'status':'V4_REFERENCE_NOT_FOUND'}
    v4=pd.read_csv(source)
    v4=v4[v4.band=='0.5-15_preAA_to32Hz']
    newer=ratio_df[ratio_df.recording=='chb01_03.edf']
    if newer.empty:return pd.DataFrame(),{'status':'NO_PILOT_RECORDING_TO_COMPARE'}
    fields=['method','window_sec','start_s','end_s']
    chk=newer.merge(v4,on=fields,suffixes=('_v5','_v4'),how='inner',validate='one_to_one')
    for col in ('R_A','R_sigma','K','sigma_G'):
        chk[f'abs_diff_{col}']=abs(chk[f'{col}_v5']-chk[f'{col}_v4'])
    stats={'status':'checked','n_matched':int(len(chk)),'n_new_pilot_rows':int(len(newer)),
       'max_abs_diff_RA':float(chk.abs_diff_R_A.max(skipna=True)) if len(chk) else None,
       'max_abs_diff_Rsigma':float(chk.abs_diff_R_sigma.max(skipna=True)) if len(chk) else None}
    return chk,stats


def self_test():
    # Cross-platform numerical and event bookkeeping checks.
    L=np.array([[-1.,4.],[-1.,-1.]])
    d=ratio_metrics(expm(.1*L),.1)
    assert abs(d['R_A']-.833839455377)<1e-6
    assert abs(d['R_sigma']-1.487539508537)<1e-6
    assert ratio_metrics(np.diag([-.8,.7]),.1)['reason']=='negative_real_multiplier'
    assert np.isfinite(ratio_metrics(np.diag([-.8,.7]),.1)['R_sigma'])
    assert subject_key('chb21_01.edf') == subject_key('chb01_03.edf')
    example='''File Name: chb01_04.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 1467 seconds\nSeizure End Time: 1494 seconds\nFile Name: test_two_events.edf\nNumber of Seizures in File: 2\nSeizure 1 Start Time: 100 seconds\nSeizure 1 End Time: 120 seconds\nSeizure 2 Start Time: 200 seconds\nSeizure 2 End Time: 230 seconds\n'''
    import tempfile
    with tempfile.TemporaryDirectory() as dtemp:
        p=Path(dtemp)/'summary.txt';p.write_text(example)
        assert parse_summary_text(p)['chb01_04.edf']==[(1467.,1494.)]
        assert parse_summary_text(p)['test_two_events.edf']==[(100.,120.),(200.,230.)]
    for w,num in ((10,16),(20,11),(40,1)):
        specs=list(generate_window_specs(3600*int(FS),2996.,3036.,[(2996.,3036.)],w))
        assert sum(s[4]=='baseline' for s in specs)==71
        assert sum(s[4]=='strict_ictal' for s in specs)==num
    # Verify held-out VAR(2) does better for known VAR(2) data; this is NOT EEG.
    rng=np.random.default_rng(12)
    z=np.zeros((1400,5))
    noise=rng.normal(scale=.3,size=z.shape)
    for t in range(2,len(z)):
        z[t]=.03*z[t-1]+.75*z[t-2]+noise[t]
    fit=fit_order_models(z)
    err={r['model']:r['test_mse'] for r in fit}
    assert err['VAR2']<.75*err['VAR1'],err
    assert all(r['n_test']==len(z[max(ORDERS):])-int(.6*(len(z)-4))-int(.2*(len(z)-4)) for r in fit)
    # Exact one-step equivalence for any stable reduced G.
    checked=0
    for _ in range(1000):
        G=rng.normal(size=(2,2))
        G*=.8/max(1.,max(abs(np.linalg.eigvals(G))))
        q=ratio_metrics(G,.1)
        if np.isfinite(q['R_sigma']) and abs(q['R_sigma']-1)>1e-6:
            assert (q['R_sigma']>1)==(q['sigma_G']>1)
            checked+=1
    assert checked>=900
    return {'status':'passed','random_stable_one_step_equivalence_checks':checked,
        'var2_synthetic_test_mse':err['VAR2'],'var1_synthetic_test_mse':err['VAR1'],
        'event_label_counts_validated':True,'multiple_seizure_summary_parser_validated':True,
        'no_EDF_data_used_in_self_test':True}


def analyze_recording(edf):
    events,source=get_annotations(edf)
    if len(events)==0:
        raise ValueError('No annotated seizure in EDF; no event baseline contrast')
    x,manifest=read_eeg(edf)
    manifest['subject']=subject_key(edf.name)
    manifest['case']=file_case(edf.name)
    manifest['annotation_source']=source
    manifest['events']=events
    models,ratios,phase=[],[],[]
    subj=manifest['subject']
    for k,(onset,offset) in enumerate(events,1):
        event_id=k
        if onset<0 or offset<=onset or offset>=len(x)/FS:
            raise ValueError(f'Event outside EDF: {(onset,offset)}, length={len(x)/FS}')
        spots=[]
        print(f'  {edf.name} event {k}/{len(events)}: {onset:g}--{offset:g}s',flush=True)
        for w in WINDOW_SECONDS:
            for beg,end,start,end_s,label,occupancy in generate_window_specs(len(x),onset,offset,events,w):
                if label=='baseline_contaminated_skip': continue
                Xi=x[beg:end]
                meta=dict(recording=edf.name,subject=subj,case=file_case(edf.name),
                     event_id=event_id,onset_s=onset,offset_s=offset,
                     window_sec=w,step_sec=STEP_SECONDS,
                     start_s=start,end_s=end_s,label=label,ictal_occupancy=occupancy,
                     n_channels=Xi.shape[1])
                details=fit_order_models(Xi)
                for dr in details:models.append(dict(**meta,**dr))
                F=fit_v4_full_window(Xi)
                rho_F=float(max(abs(np.linalg.eigvals(F))))
                sigma_F=float(svdvals(F)[0])
                for method in ('M2','M3'):
                    Q,obj=method2(F) if method=='M2' else method3(F)
                    if Q is None:
                        ratios.append(dict(**meta,method=method,rho_F=rho_F,sigma_F=sigma_F,
                             plane_ok=False,R_A=np.nan,R_sigma=np.nan,
                             log_admissible=False,reason='plane_undefined'))
                        continue
                    metrics=ratio_metrics(Q.T@F@Q,1/FS)
                    if metrics['sigma_G']>sigma_F+1e-8:
                        raise AssertionError('Compression bound violated')
                    if np.isfinite(metrics['R_sigma']) and abs(metrics['R_sigma']-1)>1e-7:
                        assert (metrics['R_sigma']>1)==(metrics['sigma_G']>1)
                    ratios.append(dict(**meta,method=method,rho_F=rho_F,sigma_F=sigma_F,
                        capture=metrics['sigma_G']/(sigma_F+EPS),plane_ok=True,
                        plane_objective_or_gap=obj,**metrics))
                if w==10 and label in ('baseline','strict_ictal'):
                    spots.append(dict(start_idx=beg,end_idx=end,label=label,window_sec=w))
        phase.extend(phase_null_checks(x,spots,event_id,subj,edf.name))
    return manifest,models,ratios,phase




# STAGE 6: cohort inventory, independent patients, matched pseudo-onsets
# Direct continuation of Stage 5; revised complex-aware R_A and R_sigma used.
# Old Figure 8's R=K/Kc(Delta) is deliberately NOT used.
# 141 file CANDIDATES are transcribed from the user's original script.
# Only files with valid official annotations and compatible montage are analyzed.
# Group chb01 and chb21 as ONE patient.
# Relative file paths are referenced against the automatically detected dataset root.
# All results from this script require a local EDF run (not present here).

import hashlib
import traceback

# Fixed montage: use core16 to maximize cross-case coverage, separately from
# the original 21-channel pilot. Never pool results across montage dimensions.
MONTAGE_MODE = 'core16'     # 'core16' for cohort or 'pilot21' for V4 parity
CORE_16 = (
    'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2'
)
PILOT_21 = CANONICAL_21
CANONICAL_21 = CORE_16 if MONTAGE_MODE == 'core16' else PILOT_21
assert MONTAGE_MODE in ('core16','pilot21')
OUT = HERE / ('seizure_stage6_cohort_' + MONTAGE_MODE + '_output')

# Only 10 s windows are run by default: same 2 s hop, baseline endpoints
# [-180,-40] s, and fully ictal windows (non-overlap with baseline by design).
WINDOW_SECONDS = (10,)
MIN_STRICT_WINDOWS = 3
MIN_ADMISSIBLE_FRAC = .80
PLACEBOS_PER_EVENT = 1              # baseline-RMS-matched pseudo-onset
RUN_M3 = True                        # M2 is primary, M3 is sensitivity
RUN_MODEL_ORDER_SPOTCHECK = True     # ONE baseline and ONE ictal window / event
RUN_PHASE_NULL = False              # exploratory null is NOT an EEG gate
REUSE_CACHED_RECORDINGS = True       # safe incremental restart
MAX_FILES_TO_PROCESS = None          # e.g., 5 for quick debugging
MAX_EVENTS_PER_FILE = None           # None = all annotated events
PLACEBO_CANDIDATE_GRID_SEC = 30.0
REQUIRE_OFFICIAL_SUMMARIES = True

# No hard-coded user absolute path. Place script beside chb01/..chb24/
# or inside one case folder. Otherwise set DATA_ROOT_OVERRIDE manually.
DATA_ROOT_OVERRIDE = None            # Optional local EDF directory

OLD_SEIZURE_EDF_REL = (
    'chb01/chb01_03.edf',
    'chb01/chb01_04.edf',
    'chb01/chb01_15.edf',
    'chb01/chb01_16.edf',
    'chb01/chb01_18.edf',
    'chb01/chb01_21.edf',
    'chb01/chb01_26.edf',
    'chb02/chb02_16.edf',
    'chb02/chb02_16+.edf',
    'chb02/chb02_19.edf',
    'chb03/chb03_01.edf',
    'chb03/chb03_02.edf',
    'chb03/chb03_03.edf',
    'chb03/chb03_04.edf',
    'chb03/chb03_34.edf',
    'chb03/chb03_35.edf',
    'chb03/chb03_36.edf',
    'chb04/chb04_05.edf',
    'chb04/chb04_08.edf',
    'chb04/chb04_28.edf',
    'chb05/chb05_06.edf',
    'chb05/chb05_13.edf',
    'chb05/chb05_16.edf',
    'chb05/chb05_17.edf',
    'chb05/chb05_22.edf',
    'chb06/chb06_01.edf',
    'chb06/chb06_04.edf',
    'chb06/chb06_09.edf',
    'chb06/chb06_10.edf',
    'chb06/chb06_13.edf',
    'chb06/chb06_18.edf',
    'chb06/chb06_24.edf',
    'chb07/chb07_12.edf',
    'chb07/chb07_13.edf',
    'chb07/chb07_18.edf',
    'chb08/chb08_02.edf',
    'chb08/chb08_05.edf',
    'chb08/chb08_11.edf',
    'chb08/chb08_13.edf',
    'chb08/chb08_21.edf',
    'chb09/chb09_06.edf',
    'chb09/chb09_08.edf',
    'chb09/chb09_19.edf',
    'chb10/chb10_12.edf',
    'chb10/chb10_20.edf',
    'chb10/chb10_27.edf',
    'chb10/chb10_30.edf',
    'chb10/chb10_31.edf',
    'chb10/chb10_38.edf',
    'chb10/chb10_89.edf',
    'chb11/chb11_82.edf',
    'chb11/chb11_92.edf',
    'chb11/chb11_99.edf',
    'chb12/chb12_06.edf',
    'chb12/chb12_08.edf',
    'chb12/chb12_09.edf',
    'chb12/chb12_10.edf',
    'chb12/chb12_11.edf',
    'chb12/chb12_23.edf',
    'chb12/chb12_27.edf',
    'chb12/chb12_28.edf',
    'chb12/chb12_29.edf',
    'chb12/chb12_33.edf',
    'chb12/chb12_36.edf',
    'chb12/chb12_38.edf',
    'chb12/chb12_42.edf',
    'chb13/chb13_19.edf',
    'chb13/chb13_21.edf',
    'chb13/chb13_40.edf',
    'chb13/chb13_55.edf',
    'chb13/chb13_58.edf',
    'chb13/chb13_59.edf',
    'chb13/chb13_60.edf',
    'chb13/chb13_62.edf',
    'chb14/chb14_03.edf',
    'chb14/chb14_04.edf',
    'chb14/chb14_06.edf',
    'chb14/chb14_11.edf',
    'chb14/chb14_17.edf',
    'chb14/chb14_18.edf',
    'chb14/chb14_27.edf',
    'chb15/chb15_06.edf',
    'chb15/chb15_10.edf',
    'chb15/chb15_15.edf',
    'chb15/chb15_17.edf',
    'chb15/chb15_20.edf',
    'chb15/chb15_22.edf',
    'chb15/chb15_28.edf',
    'chb15/chb15_31.edf',
    'chb15/chb15_40.edf',
    'chb15/chb15_46.edf',
    'chb15/chb15_49.edf',
    'chb15/chb15_52.edf',
    'chb15/chb15_54.edf',
    'chb15/chb15_62.edf',
    'chb16/chb16_10.edf',
    'chb16/chb16_11.edf',
    'chb16/chb16_14.edf',
    'chb16/chb16_16.edf',
    'chb16/chb16_17.edf',
    'chb16/chb16_18.edf',
    'chb17/chb17a_03.edf',
    'chb17/chb17a_04.edf',
    'chb17/chb17b_63.edf',
    'chb18/chb18_29.edf',
    'chb18/chb18_30.edf',
    'chb18/chb18_31.edf',
    'chb18/chb18_32.edf',
    'chb18/chb18_35.edf',
    'chb18/chb18_36.edf',
    'chb19/chb19_28.edf',
    'chb19/chb19_29.edf',
    'chb19/chb19_30.edf',
    'chb20/chb20_12.edf',
    'chb20/chb20_13.edf',
    'chb20/chb20_14.edf',
    'chb20/chb20_15.edf',
    'chb20/chb20_16.edf',
    'chb20/chb20_68.edf',
    'chb21/chb21_19.edf',
    'chb21/chb21_20.edf',
    'chb21/chb21_21.edf',
    'chb21/chb21_22.edf',
    'chb22/chb22_20.edf',
    'chb22/chb22_25.edf',
    'chb22/chb22_38.edf',
    'chb23/chb23_06.edf',
    'chb23/chb23_08.edf',
    'chb23/chb23_09.edf',
    'chb24/chb24_01.edf',
    'chb24/chb24_03.edf',
    'chb24/chb24_04.edf',
    'chb24/chb24_06.edf',
    'chb24/chb24_07.edf',
    'chb24/chb24_09.edf',
    'chb24/chb24_11.edf',
    'chb24/chb24_13.edf',
    'chb24/chb24_14.edf',
    'chb24/chb24_15.edf',
    'chb24/chb24_17.edf',
    'chb24/chb24_21.edf',
)



def find_data_root():
    if DATA_ROOT_OVERRIDE is not None:
        root=Path(DATA_ROOT_OVERRIDE).expanduser().resolve()
        if not root.is_dir(): raise FileNotFoundError(f'DATA_ROOT_OVERRIDE missing: {root}')
        return root
    checks=[HERE,HERE.parent,HERE/'chbmit_seizure_data',HERE.parent/'chbmit_seizure_data',HERE.parent.parent/'chbmit_seizure_data']
    scored=[]
    for folder in checks:
        if folder.is_dir():
            count=sum((folder/r).is_file() for r in OLD_SEIZURE_EDF_REL)
            scored.append((count,folder))
    if not scored or max(z[0] for z in scored)==0:
        raise FileNotFoundError('Cannot find CHB-MIT subject folders. Put this script in dataset root or set DATA_ROOT_OVERRIDE.')
    return sorted(scored,key=lambda z:z[0],reverse=True)[0][1]


def official_events(edf):
    case=file_case(edf.name)
    source=edf.parent/(case+'-summary.txt')
    if not source.is_file(): raise ValueError('OFFICIAL_SUMMARY_MISSING: '+str(source))
    ann=parse_summary_text(source)
    if edf.name.lower() not in ann:
        raise ValueError('EDF_ABSENT_FROM_SUMMARY: '+edf.name)
    events=ann[edf.name.lower()]
    if not events: raise ValueError('NO_ANNOTATED_SEIZURES')
    for onset,offset in events:
        if not (np.isfinite(onset) and np.isfinite(offset) and offset>onset):
            raise ValueError('BAD_EVENT_ANNOTATION')
    return events,source


def prebaseline_rms(X,onset):
    # Selection uses PRE-ONSET activity ONLY, never seizure activity.
    i=int(round((onset+BASELINE_END_REL[0]-WINDOW_SECONDS[0])*FS))
    j=int(round((onset+BASELINE_END_REL[1])*FS))
    i=max(i,0)
    if j-i<FS*30 or j>len(X):return np.nan
    s=X[i:j,:]
    return float(np.sqrt(np.mean(np.square(s))))


def select_pseudo_onsets(X,onset,offset,events,already_used=()):
    if PLACEBOS_PER_EVENT==0: return []
    duration=offset-onset
    req_left=abs(BASELINE_END_REL[0])+WINDOW_SECONDS[0]  # >=190 s pre-onset
    req_right=duration+40.
    max_pseudo=len(X)/FS-req_right
    if max_pseudo<=req_left:return []
    real_rms=prebaseline_rms(X,onset)
    if not np.isfinite(real_rms) or real_rms<=0:return []
    candidates=[]
    # 30-s fixed grid; shift to a 2-s multiple to match window alignment.
    for t in np.arange(np.ceil(req_left/PLACEBO_CANDIDATE_GRID_SEC)*PLACEBO_CANDIDATE_GRID_SEC,
                       max_pseudo+1e-9,PLACEBO_CANDIDATE_GRID_SEC):
        t=round(t/STEP_SECONDS)*STEP_SECONDS
        # Every window in placebo baseline/event span must be free of true seizures,
        # including a 40s guard. This prevents control contamination.
        span_start=t-req_left
        span_end=t+duration+40.
        if any(span_start<b+40. and span_end>a-40. for a,b in events):continue
        if any(abs(t-v)<req_left+duration+40. for v in already_used):continue
        cand_rms=prebaseline_rms(X,t)
        if not np.isfinite(cand_rms) or cand_rms<=0:continue
        mismatch=abs(np.log(cand_rms/real_rms))
        candidates.append((float(mismatch),float(t),float(cand_rms)))
    candidates.sort(key=lambda z:(z[0],z[1]))
    return [dict(onset=c[1],offset=c[1]+duration,
                 baseline_log_rms_mismatch=c[0],
                 real_prebaseline_rms=real_rms,placebo_prebaseline_rms=c[2])
            for c in candidates[:PLACEBOS_PER_EVENT]]


def stage6_window_rows(X,event_onset,event_offset,all_events,recording,subject,event_id,
                       event_type,match_info=None,adequacy=False):
    # Returns per-window M2/M3 ratio rows and prespecified VAR-order spot checks.
    all_rows=[];selected_spots=[]
    for W in WINDOW_SECONDS:
        specs=[]
        for beg,en,start,end,label,occupancy in generate_window_specs(len(X),event_onset,event_offset,all_events,W):
            if label in ('baseline','strict_ictal'):
                specs.append((beg,en,start,end,label,occupancy))
        if not specs:continue
        for beg,en,start,end,label,occupancy in specs:
            data=X[beg:en]
            F=fit_v4_full_window(data)
            rhoF=float(max(abs(np.linalg.eigvals(F))))
            sigF=float(svdvals(F)[0])
            full_stable=bool(np.isfinite(rhoF) and rhoF<1.-TOL)
            for method in (('M2','M3') if RUN_M3 else ('M2',)):
                Q,objective=method2(F) if method=='M2' else method3(F)
                row=dict(recording=recording,subject=subject,event_id=event_id,
                         event_type=event_type,method=method,window_sec=W,
                         start_s=start,end_s=end,label=label,occupancy=occupancy,
                         n_channels=len(CANONICAL_21),montage=MONTAGE_MODE,
                         rho_F=rhoF,sigma_F=sigF,full_stable=full_stable,
                         plane_objective_or_gap=objective)
                if match_info:row.update(match_info)
                if Q is None:
                    row.update(dict(R_A=np.nan,R_sigma=np.nan,raw_R_A=np.nan,
                                    raw_R_sigma=np.nan,log_admissible=False,
                                    complex_pair=False,rho_G=np.nan,
                                    sigma_G=np.nan,K=np.nan,Kc_A=np.nan,
                                    capture=np.nan,reason='undefined_plane',
                                    admissible_RA=False,admissible_Rsigma=False))
                else:
                    m=ratio_metrics(Q.T@F@Q,1/FS)
                    if m['sigma_G']>sigF+1e-8:
                        raise AssertionError('Singular-value compression bound failed')
                    row.update(m)
                    row['raw_R_A']=m['R_A'];row['raw_R_sigma']=m['R_sigma']
                    row['R_A']=m['R_A'] if full_stable else np.nan
                    row['R_sigma']=m['R_sigma'] if full_stable else np.nan
                    row['admissible_RA']=bool(np.isfinite(row['R_A']))
                    row['admissible_Rsigma']=bool(np.isfinite(row['R_sigma']))
                    row['capture']=m['sigma_G']/(sigF+EPS)
                    if not full_stable:row['reason']='full_operator_unstable'
                all_rows.append(row)
        if adequacy and RUN_MODEL_ORDER_SPOTCHECK and W==10:
            for label in ('baseline','strict_ictal'):
                available=[s for s in specs if s[4]==label]
                if available:
                    s=available[len(available)//2]
                    selected_spots.append((s[0],s[1],label,W))
    order_rows=[]
    for beg,en,label,w in selected_spots:
        for result in fit_order_models(X[beg:en]):
            order_rows.append(dict(recording=recording,subject=subject,event_id=event_id,
                                   montage=MONTAGE_MODE,label=label,window_sec=w,
                                   window_end_s=en/FS,**result))
    return all_rows,order_rows


def summarize_stage6_events(ratios):
    from math import log2
    rows=[]
    if ratios.empty: return pd.DataFrame()
    for (rec,subj,ev,etype,w,method),g in ratios.groupby(
        ['recording','subject','event_id','event_type','window_sec','method'],sort=True):
        b=g[g.label=='baseline'];it=g[g.label=='strict_ictal']
        for measure in ('R_A','R_sigma'):
            xb=pd.to_numeric(b[measure],errors='coerce').dropna()
            xi=pd.to_numeric(it[measure],errors='coerce').dropna()
            bv=float(xb.median()) if len(xb) else np.nan
            iv=float(xi.median()) if len(xi) else np.nan
            enough_baseline=(len(b)==71)
            enough_ictal=(len(it)>=MIN_STRICT_WINDOWS)
            afb=len(xb)/len(b) if len(b) else 0.
            afi=len(xi)/len(it) if len(it) else 0.
            valid=(enough_baseline and enough_ictal and afb>=MIN_ADMISSIBLE_FRAC
                   and afi>=MIN_ADMISSIBLE_FRAC and bv>0 and iv>0)
            rows.append(dict(recording=rec,subject=subj,event_id=ev,
                event_type=etype,window_sec=w,method=method,measure=measure,
                n_baseline_total=len(b),n_ictal_total=len(it),
                n_baseline_admissible=len(xb),n_ictal_admissible=len(xi),
                baseline_admissibility_fraction=afb,ictal_admissibility_fraction=afi,
                complete_baseline=enough_baseline,
                enough_strict_windows=enough_ictal,
                baseline_median=bv,ictal_median=iv,
                log2_ictal_over_baseline=float(log2(iv/bv)) if bv>0 and iv>0 else np.nan,
                valid_primary_event=bool(valid),
                exclusion_reason=('' if valid else
                       ('incomplete_baseline' if not enough_baseline else
                        'too_few_strict_ictal' if not enough_ictal else
                        'insufficient_admissibility_or_nonpositive_median'))))
    return pd.DataFrame(rows)


def subject_hierarchy(events):
    # EVENT medians -> RECORDING median event and baseline -> log2 ratio
    # -> PATIENT median of recording ratios. Each patient contributes ONE unit.
    # Placebo 'ictal' means pseudo-event occupancy, not real seizure.
    good=events[events.valid_primary_event].copy()
    if good.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    recording=[]
    for (subj,rec,etype,w,method,measure),g in good.groupby(
        ['subject','recording','event_type','window_sec','method','measure'],sort=True):
        bv=float(g.baseline_median.median());iv=float(g.ictal_median.median())
        recording.append(dict(subject=subj,recording=rec,event_type=etype,
            window_sec=w,method=method,measure=measure,
            n_events=len(g),median_event_baseline=bv,median_event_ictal=iv,
            recording_log2_ratio=float(np.log2(iv/bv))))
    rec=pd.DataFrame(recording)
    subjects=[]
    for (subj,etype,w,method,measure),g in rec.groupby(
        ['subject','event_type','window_sec','method','measure'],sort=True):
        subjects.append(dict(subject=subj,event_type=etype,
           window_sec=w,method=method,measure=measure,
           n_recordings=len(g),n_events=int(g.n_events.sum()),
           patient_log2_ratio=float(g.recording_log2_ratio.median())))
    subjects=pd.DataFrame(subjects)
    real=subjects[subjects.event_type=='real']
    pla=subjects[subjects.event_type=='placebo']
    matched=real.merge(pla,on=['subject','window_sec','method','measure'],
         suffixes=('_real','_placebo'),validate='one_to_one')
    if not matched.empty:
        matched['real_minus_placebo_log2']=matched.patient_log2_ratio_real-matched.patient_log2_ratio_placebo
    return rec,subjects,matched


def cohort_summary(subjects,matched):
    rows=[]
    for (etype,w,method,measure),g in subjects.groupby(['event_type','window_sec','method','measure']):
        z=g.patient_log2_ratio.to_numpy(float);z=z[np.isfinite(z)]
        if len(z)==0:continue
        ci=[np.nan,np.nan]
        if len(z)>=5:
            rng=np.random.default_rng(52026)
            ci=np.quantile(np.median(rng.choice(z,size=(2000,len(z)),replace=True),axis=1),[.025,.975]).tolist()
        rows.append(dict(event_type=etype,window_sec=w,method=method,measure=measure,
             n_independent_patients=len(z),median_patient_log2_ratio=float(np.median(z)),
             n_positive=int(sum(z>0)),n_negative=int(sum(z<0)),n_zero=int(sum(z==0)),
             exploratory_bootstrap_ci_lower=float(ci[0]),
             exploratory_bootstrap_ci_upper=float(ci[1]),
             note='Descriptive patient-level; no confirmatory p-value or EEG-calibrated model gate'))
    if not matched.empty:
        for (w,method,measure),g in matched.groupby(['window_sec','method','measure']):
            z=g.real_minus_placebo_log2.to_numpy(float)
            rows.append(dict(event_type='real_minus_matched_placebo',window_sec=w,
              method=method,measure=measure,n_independent_patients=len(z),
              median_patient_log2_ratio=float(np.median(z)),
              n_positive=int(sum(z>0)),n_negative=int(sum(z<0)),n_zero=int(sum(z==0)),
              exploratory_bootstrap_ci_lower=np.nan,exploratory_bootstrap_ci_upper=np.nan,
              note='Paired descriptive contrast; one selected baseline-RMS placebo/event'))
    return pd.DataFrame(rows)


def save_json(path,body):
    path.write_text(json.dumps(body,indent=2,default=str))


def analyze_file(root,rel,all_events):
    edf=root/rel
    events=all_events
    X,channel_manifest=read_eeg(edf)
    if len(X)/FS<max(b for a,b in events):raise ValueError('ANNOTATION_OUTSIDE_EDF_DURATION')
    record=edf.name
    subject=subject_key(record)
    all_ratio=[];all_order=[];placebo_meta=[]
    used_placebo=[]
    usable_events=events if MAX_EVENTS_PER_FILE is None else events[:MAX_EVENTS_PER_FILE]
    for idx,(onset,offset) in enumerate(usable_events,1):
        print(f'    event {idx}/{len(usable_events)} {onset:g}-{offset:g}s',flush=True)
        ratio,order=stage6_window_rows(X,onset,offset,events,record,subject,idx,'real',adequacy=True)
        all_ratio+=ratio;all_order+=order
        if PLACEBOS_PER_EVENT:
            controls=select_pseudo_onsets(X,onset,offset,events,used_placebo)
            for j,ctrl in enumerate(controls,1):
                used_placebo.append(ctrl['onset'])
                fake=(ctrl['onset'],ctrl['offset'])
                # Real events were checked against the entire placebo span.
                p,_=stage6_window_rows(X,*fake,[fake],record,subject,idx,'placebo',
                    match_info=dict(placebo_rank=j,placebo_match_error=ctrl['baseline_log_rms_mismatch']),
                    adequacy=False)
                all_ratio+=p
                placebo_meta.append(dict(recording=record,subject=subject,
                   event_id=idx,real_onset=onset,real_offset=offset,**ctrl))
    channel_manifest['n_seizures']=len(usable_events)
    channel_manifest['subject']=subject
    channel_manifest['case']=file_case(record)
    channel_manifest['montage']=MONTAGE_MODE
    return pd.DataFrame(all_ratio),pd.DataFrame(all_order),placebo_meta,channel_manifest


def stage6_self_test():
    base=self_test()
    assert len(OLD_SEIZURE_EDF_REL)==141
    assert len(set(OLD_SEIZURE_EDF_REL))==141
    assert len({x.split('/')[0] for x in OLD_SEIZURE_EDF_REL})==24
    assert len(CANONICAL_21) in (16,21)
    assert subject_key('chb21_20.edf')==subject_key('chb01_03.edf')
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        p=root/'chb03';p.mkdir()
        s=p/'chb03-summary.txt'
        s.write_text('File Name: chb03_01.edf\nNumber of Seizures in File: 2\nSeizure 1 Start Time: 250 seconds\nSeizure 1 End Time: 275 seconds\nSeizure 2 Start Time: 600 seconds\nSeizure 2 End Time: 630 seconds\n')
        (p/'chb03_01.edf').touch()
        ev,src=official_events(p/'chb03_01.edf')
        assert ev==[(250.,275.),(600.,630.)]
        assert len(list(generate_window_specs(1000*32,250.,275.,ev,10)))>0
    # 32Hz, 1,000 s, reproducible stationary non-seizure synthetic signal.
    rng=np.random.default_rng(7)
    q=rng.normal(size=(1000*32,len(CANONICAL_21)))
    ctrls=select_pseudo_onsets(q,500.,540.,[(500.,540.)],[])
    assert ctrls and abs(ctrls[0]['onset']-500.)>=190
    return dict(status='passed',stage5_checks=base,
                old_inventory_count=len(OLD_SEIZURE_EDF_REL),
                candidate_cases=24,independent_patient_ids_max=23,
                parser_test='passed',placebo_guard_test='passed')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'stage6_self_test.json',stage6_self_test())
    root=find_data_root()
    run_cfg=dict(root=str(root),montage_mode=MONTAGE_MODE,
        canonical_channels=list(CANONICAL_21),windows=list(WINDOW_SECONDS),
        hop_seconds=STEP_SECONDS,baseline_endpoints=list(BASELINE_END_REL),
        min_ictal_windows=MIN_STRICT_WINDOWS,min_admissibility_fraction=MIN_ADMISSIBLE_FRAC,
        n_placebo=PLACEBOS_PER_EVENT,run_m3=RUN_M3,
        run_model_order_spotcheck=RUN_MODEL_ORDER_SPOTCHECK,
        m2_restarts=M2_RESTARTS,m2_iters=M2_ITERS,
        placebo_grid_seconds=PLACEBO_CANDIDATE_GRID_SEC,
        stage6_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        band=[FILTER_LOW,FILTER_HIGH],native_fs=FS_NATIVE,output_fs=FS,
        ridge_relative=V4_RIDGE_REL,shrink=V4_SHRINK_ALPHA,
        require_official_summaries=True,
        candidate_count=len(OLD_SEIZURE_EDF_REL))
    signature=hashlib.sha256(json.dumps(run_cfg,sort_keys=True).encode()).hexdigest()[:16]
    save_json(OUT/'run_settings.json',dict(run_cfg,run_signature=signature))
    print(f'STAGE 6 | data root: {root} | montage={MONTAGE_MODE}',flush=True)
    print(f'Candidate files: {len(OLD_SEIZURE_EDF_REL)} | run signature {signature}',flush=True)
    candidates=OLD_SEIZURE_EDF_REL if MAX_FILES_TO_PROCESS is None else OLD_SEIZURE_EDF_REL[:MAX_FILES_TO_PROCESS]
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    manifest=[];all_ratio=[];all_order=[];all_placebo=[];channel_manifest=[]
    for n,rel in enumerate(candidates,1):
        edf=root/rel
        subj=subject_key(edf.name)
        entry=dict(candidate=rel,subject=subj,edf_exists=edf.is_file(),
                   n_annotated_events=0,n_usable_event_contrasts=0,run_signature=signature)
        if not edf.is_file():
            entry['status']='EDF_NOT_AVAILABLE';manifest.append(entry);continue
        try:
            events,source=official_events(edf)
            entry.update(annotation_source=str(source),n_annotated_events=len(events))
        except Exception as exc:
            entry.update(status='ANNOTATION_EXCLUDED',reason=str(exc))
            manifest.append(entry);continue
        stem=edf.name.replace('+','plus').replace('.edf','')
        folder=cache/file_case(edf.name);folder.mkdir(exist_ok=True)
        prefix=folder/(stem+'_'+signature)
        rcsv=Path(str(prefix)+'_ratio.csv');ocsv=Path(str(prefix)+'_order.csv')
        jmeta=Path(str(prefix)+'_metadata.json')
        pcsv=Path(str(prefix)+'_placebo.csv')
        print(f'[{n:03d}/{len(candidates)}] {rel}: {len(events)} annotated',flush=True)
        try:
            if REUSE_CACHED_RECORDINGS and rcsv.is_file() and jmeta.is_file():
                meta=json.loads(jmeta.read_text())
                if meta.get('run_signature')!=signature:raise ValueError('CACHE_SIGNATURE_MISMATCH')
                ratio=pd.read_csv(rcsv)
                order=pd.read_csv(ocsv) if ocsv.exists() and ocsv.stat().st_size>1 else pd.DataFrame()
                placebo=pd.read_csv(pcsv).to_dict('records') if pcsv.exists() and pcsv.stat().st_size>1 else []
                channels=meta['channel_manifest']
                entry['status']='cached'
            else:
                ratio,order,placebo,channels=analyze_file(root,rel,events)
                # Save each complete EDF immediately: restart never recomputes successes.
                ratio.to_csv(rcsv,index=False)
                order.to_csv(ocsv,index=False)
                pd.DataFrame(placebo).to_csv(pcsv,index=False)
                save_json(jmeta,dict(run_signature=signature,channel_manifest=channels,
                                      candidate=rel,events=events))
                entry['status']='analyzed'
            entry.update(n_rows=len(ratio),n_usable_event_contrasts=int(
              summarize_stage6_events(ratio).query(
               "event_type == 'real' and method == 'M2' and measure == 'R_A' and valid_primary_event"
              ).shape[0]))
            all_ratio.append(ratio)
            if not order.empty:all_order.append(order)
            all_placebo.extend(placebo)
            channel_manifest.append(channels)
        except Exception as exc:
            entry.update(status='FAILED_OR_INCOMPATIBLE',reason=f'{type(exc).__name__}: {exc}')
            print(f'    {entry["status"]}: {entry["reason"]}',flush=True)
        manifest.append(entry)
        pd.DataFrame(manifest).to_csv(OUT/'candidate_manifest_progress.csv',index=False)
    # Save explicit manifest of skips and successes. Missing EDFs are not fabricated.
    mdf=pd.DataFrame(manifest)
    mdf.to_csv(OUT/'candidate_manifest.csv',index=False)
    if not all_ratio:
        save_json(OUT/'run_status.json',dict(status='NO_EDFS_ANALYZED',
               n_candidates=len(candidates),manifest=manifest,signature=signature))
        print('NO EDF analyzed. See candidate_manifest.csv',flush=True)
        return
    ratios=pd.concat(all_ratio,ignore_index=True)
    ratios.to_csv(OUT/'stage6_window_ratios.csv',index=False)
    if all_order:
        adeq=pd.concat(all_order,ignore_index=True)
        adeq.to_csv(OUT/'stage6_VAR_spotcheck.csv',index=False)
    else:adeq=pd.DataFrame()
    ev=summarize_stage6_events(ratios)
    ev.to_csv(OUT/'stage6_event_contrasts.csv',index=False)
    rec,subject,paired=subject_hierarchy(ev)
    rec.to_csv(OUT/'stage6_recording_contrasts.csv',index=False)
    subject.to_csv(OUT/'stage6_patient_contrasts.csv',index=False)
    paired.to_csv(OUT/'stage6_patient_matched_placebo.csv',index=False)
    cohort=cohort_summary(subject,paired) if not subject.empty else pd.DataFrame()
    cohort.to_csv(OUT/'stage6_cohort_summary.csv',index=False)
    pd.DataFrame(all_placebo).to_csv(OUT/'stage6_placebo_selection.csv',index=False)
    save_json(OUT/'stage6_channel_manifest.json',channel_manifest)
    status=dict(status='descriptive_cohort_complete_for_available_compatible_files',
       n_inventory_candidates=len(candidates),
       n_recordings_processed=len(channel_manifest),
       n_annotated_events_in_processed_recordings=int(sum(c['n_seizures'] for c in channel_manifest)),
       n_patient_identities_processed=len(set(c['subject'] for c in channel_manifest)),
       n_primary_R_A_patients=int(subject[(subject.event_type=='real') &
               (subject.method=='M2') &(subject.measure=='R_A')&(subject.window_sec==10)].subject.nunique()) if not subject.empty else 0,
       n_window_ratio_rows=len(ratios),
       n_order_model_spotcheck_rows=len(adeq),
       n_control_selection_rows=len(all_placebo),
       montage=MONTAGE_MODE,run_signature=signature,
       important_qualification=('Descriptive VAR(1) predictor geometry. Model adequacy not certified. '
       'Any p-values, cohort-wide physiological interpretations, and across-montage pooling are withheld. '
       'EEG files without annotations, compatible montage or complete clean baselines are excluded and recorded.'))
    save_json(OUT/'run_status.json',status)
    primary=cohort[(cohort.event_type=='real') & (cohort.window_sec==10) &
          (cohort.method=='M2') & (cohort.measure=='R_A')] if not cohort.empty else pd.DataFrame()
    print('\n=== Stage 6 PRIMARY descriptive outcome ===',flush=True)
    print(primary.to_string(index=False) if not primary.empty else 'No eligible patients',flush=True)
    print('\nStage 6 output:',OUT,flush=True)
    print('Please ZIP the entire output folder (including candidate_manifest.csv).',flush=True)

if __name__=='__main__':
    main()
