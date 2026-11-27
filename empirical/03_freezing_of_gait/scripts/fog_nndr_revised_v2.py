#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOG / Daphnet: manuscript-referenced NNDR pilot + complete dataset audit.

RUN IN SPYDER: put this file directly inside the folder shown in Finder,
next to S01R01.txt, S01R02.txt, ..., or edit DATA_DIR below.

MODE = 'representative': S01R01.txt, manuscript Fig. 14 event (label 03,
                        fourth contiguous annotation-2 interval), five-panel figure.
MODE = 'full': all S??R??.txt recordings, patient/recording/event tables,
               two *separate* rankings (quality vs observed contrast).
MODE = 'both': run both modes, but full scan may be lengthy.

The script implements the rate-based R_A and the exact one-step R_sigma
used in the draft rebuttal and the revised push-up script.  It does NOT
claim to reproduce the original manuscript's old numerical R, nor its
unprovided historical preprocessing, ridge or shrinkage choices.

Statistical status: screening/exploratory audit only. It does NOT compute
confirmatory placebo/switching-null p-values, C5 calibration, or Holm tests.
An onset window ending during the first 4 s of an episode can mix walking
and freezing, and must NOT be interpreted as proof of a physical mechanism.
"""
from __future__ import annotations
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ============================ USER SETTINGS ================================
DATA_DIR = Path(__file__).resolve().parent  # automatically your 'dataset' folder
MODE = 'representative'  # 'representative', 'full', or 'both'
REP_RECORDING = 'S01R01.txt'
REP_EVENT_NUMBER = 4        # manuscript Fig. 14: 'ankle event 03' is zero-based; onset ~1236.156 s

NOMINAL_FS = 64.0
WINDOW_SEC = 3.0           # original manuscript
STEP_SEC = 0.5            # original manuscript
ONSET_SEC = 4.0           # original manuscript
BASELINE_LOOKBACK_SEC = 30.0  # EXPLORATORY explicit baseline matching choice
BASELINE_GAP_SEC = 3.0       # avoid any window touching episode onset
MIN_EVENT_SEC_FOR_RANK = 4.0
MIN_BASELINE_WINDOWS = 4
MIN_ONSET_WINDOWS = 3

# These are REVISED PILOT choices, NOT verified historical FOG settings.
RIDGE_REL = 3e-3
SHRINK_ALPHA = 0.04
COND_MAX = 2e4
EFFECTIVE_RANK_MIN = 2.0
RESIDUAL_LAG = 4        # descriptive adequacy diagnostic, not C5 calibration
M2_RESTARTS = 5
M2_ITERS = 65
M2_TOL = 1e-9
SEED = 20261119

OUTDIR = DATA_DIR / 'FOG_NNDR_REVISED'
SAVE_FIG_PDF = True
EPS = 1e-12
STABLE_TOL = 1e-10


def load_recording(path):
    """The official raw Daphnet format: 11 whitespace-delimited columns."""
    a = np.loadtxt(path, dtype=float)
    if a.ndim != 2 or a.shape[1] != 11:
        raise ValueError(f'{path.name}: expected 11 columns, got {a.shape}')
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{path.name}: non-finite values in raw data')
    anno = a[:, 10].astype(np.int8)
    if not np.all(np.isin(anno, [0, 1, 2])):
        raise ValueError(f'{path.name}: annotations must be 0, 1, 2')
    time = (a[:, 0] - a[0, 0]) * 1e-3
    dt = np.diff(time)
    if not np.all(dt > 0):
        raise ValueError(f'{path.name}: non-monotonic timestamp')
    fs = 1.0 / np.median(dt)
    if abs(fs - NOMINAL_FS) / NOMINAL_FS > 0.05:
        warnings.warn(f'{path.name}: sampling frequency {fs:.3g} differs from 64 Hz')
    bad_gap = np.r_[False, dt > 2.0 / NOMINAL_FS]
    # Preserve physical sensor units (mg). Do NOT per-axis standardize.
    x = a[:, 1:4].copy()
    return dict(time=time, x=x, anno=anno, fs=fs, bad_gap=bad_gap,
                raw_magnitude=np.linalg.norm(x, axis=1), name=path.name)


def get_events(anno):
    f = np.asarray(anno) == 2
    starts = np.where(f & ~np.r_[False, f[:-1]])[0]
    ends = np.where(f & ~np.r_[f[1:], False])[0] + 1  # exclusive
    return [(i + 1, int(s), int(e)) for i, (s, e) in enumerate(zip(starts, ends))]


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else None


def m2_plane(f):
    """Constrained transverse action: max_{u perp v} u.T @ F @ v.
    Alternating projected power iteration, deterministic multiple restarts.
    """
    d = f.shape[0]
    rng = np.random.default_rng(SEED)
    initial = [np.eye(d)[:, j] for j in range(d)]
    initial.extend(rng.normal(size=d) for _ in range(M2_RESTARTS))
    best = -np.inf
    qbest = None
    for v0 in initial:
        v = unit(v0)
        if v is None:
            continue
        old = -np.inf
        for _ in range(M2_ITERS):
            u = f @ v
            u -= v * (v @ u)
            u = unit(u)
            if u is None:
                break
            vnew = f.T @ u
            vnew -= u * (u @ vnew)
            vnew = unit(vnew)
            if vnew is None:
                break
            v = vnew
            val = float(u @ f @ v)
            if abs(val - old) <= M2_TOL:
                break
            old = val
        # Reconstruct the response from the best final v to guarantee ordering.
        r = f @ v
        r -= v * (v @ r)
        score = float(np.linalg.norm(r))
        r = unit(r)
        if r is None:
            continue
        if score > best:
            best = score
            qbest = np.column_stack((r, v))
    return qbest, best


def m3_plane(f):
    """Largest-positive and most-negative commutator eigenvectors."""
    c = f @ f.T - f.T @ f
    evals, vecs = np.linalg.eigh(c)
    if np.max(np.abs(evals)) < EPS:
        return None
    q = vecs[:, [np.argmax(evals), np.argmin(evals)]]
    # Reorient within the same 2D plane only for response support.
    f2 = q.T @ f @ q
    q2, _ = m2_plane(f2)
    if q2 is not None:
        q = q @ q2
    return q


def reduced_indices(f, q, dt):
    """Both threshold conventions, including the real repeated-eigenvalue limit.

    R_A is calculated from the *departure of the principal-log generator*
    without dividing by the eigenvalue gap. For G=Gamma with eigenvalues
    lambda_1,2, L=Log(G)/dt = a I + b (G-m I)/dt, where
      b=(Log(lambda1)-Log(lambda2))/(lambda1-lambda2),
      b -> 1/lambda in the repeated-root limit.
    Hence d_F(L)=|b| d_F(G)/dt, which is robust near a Jordan block.
    """
    g = q.T @ f @ q
    eig = np.linalg.eigvals(g)
    gap = float(abs(eig[0] - eig[1]))
    rho = float(np.max(np.abs(eig)))
    sigma = float(np.linalg.svd(g, compute_uv=False)[0])
    fro2 = float(np.sum(g*g))
    efro2 = float(np.sum(np.abs(eig)**2))
    df = float(np.sqrt(max(0., fro2-efro2)))
    k = df / gap if gap > 1e-10 else (np.inf if df > 1e-10 else 0.)
    stable = rho < 1 - STABLE_TOL
    r_sigma = np.nan
    if stable:
        den2 = np.prod(1 - np.abs(eig)**2)
        if den2 > EPS:
            r_sigma = float(df / np.sqrt(den2))
    ra, kc, split, beta = np.nan, np.nan, np.nan, np.nan
    domain = 'inadmissible'
    spec = 'complex' if np.max(abs(np.imag(eig))) > 1e-9 else 'real'
    # A stable real 2x2 matrix with a negative real eigenvalue does not have
    # an admissible real *principal* logarithm. R_sigma may still exist.
    bad_real_log = bool(spec == 'real' and np.any(np.real(eig) <= 0))
    log_admissible = not bad_real_log
    if stable and rho > EPS and log_admissible:
        logs = np.log(eig.astype(complex))
        mu = logs / dt
        midpoint = (eig[0] + eig[1]) * .5
        dlam = (eig[0] - eig[1]) * .5
        if abs(midpoint) > EPS:
            if gap > 1e-6 * max(rho, 1e-4):
                divided_log = (logs[0]-logs[1]) / (eig[0]-eig[1])
            else:
                # Series of atanh(z)/z; accurate close to repeated roots.
                z = dlam / midpoint
                divided_log = (1 + z*z/3 + z**4/5 + z**6/7) / midpoint
            dfl = float(df * abs(divided_log) / dt)
            if spec == 'complex':
                gamma = -float(np.real(mu[0]))
                omega = abs(float(np.imag(mu[0])))
                if gamma > 0 and omega > EPS:
                    beta = omega / gamma
                    denom = 2*gamma*np.sqrt(np.hypot(1.,beta)+1.)
                    ra = float(dfl/denom)
                    kc = float(np.sqrt((np.hypot(1., beta)+1.)/beta**2))
                    split = beta
                    domain = 'complex_rates'
            else:
                rates_real = np.real(mu)
                if np.all(rates_real < 0):
                    delta = float(abs((rates_real[0]-rates_real[1]) /
                                      (rates_real[0]+rates_real[1])))
                    if 0 <= delta < 1:
                        s = float(np.sqrt(max(0., 1-delta**2)))
                        denom = abs(float(np.sum(rates_real))) * np.sqrt(s*(1+s))
                        if denom > EPS:
                            ra = float(dfl/denom)
                            kc = (float(np.sqrt(s*(1+s))/delta)
                                  if delta > 1e-13 else np.inf)
                            split = delta
                            domain = 'real_rates'
    support = float(np.linalg.norm(q[:, 0], 1) / np.sqrt(q.shape[0]))
    return dict(rhoGamma=rho, sigmaGamma=sigma, dF=df, K=k,
                RA=ra, KcA=kc, R_sigma=r_sigma, split_rate=split,
                beta=beta, spectrum=spec, A_domain=domain,
                bad_real_log=bad_real_log,
                support=support, stableGamma=bool(stable),
                stable_amplifying=bool(stable and sigma > 1 + STABLE_TOL),
                margin_sigma_minus1=sigma-1.0)


def effective_rank_and_condition(x):
    s = np.linalg.svd(x, compute_uv=False)
    s = s[s > 1e-12 * max(float(np.max(s)), EPS)]
    if not len(s):
        return 0., np.inf
    p = s / np.sum(s)
    return float(np.exp(-np.sum(p*np.log(p)))), float(s.max() / s.min())


def fit_one_window(x, dt):
    """No artificial joins; center X and Y separately (affine VAR fit).
    Relative ridge and trace-preserving isotropic shrinkage are labelled PILOT.
    """
    x0, y0 = x[:-1], x[1:]
    xx = x0 - x0.mean(axis=0)
    yy = y0 - y0.mean(axis=0)
    er, raw_cond = effective_rank_and_condition(xx)
    c00 = xx.T @ xx
    c10 = yy.T @ xx
    d = xx.shape[1]
    lam = RIDGE_REL * float(np.trace(c00)) / d
    c_reg = c00 + lam * np.eye(d)
    condreg = float(np.linalg.cond(c_reg)) if lam > 0 else np.inf
    # C1 is an explicit, tunable pilot cutoff, NOT a settled reviewer threshold.
    c1 = not np.isfinite(condreg) or condreg > COND_MAX
    rank_gate = bool(er < EFFECTIVE_RANK_MIN)
    if not np.isfinite(condreg) or lam <= 0:
        return None, dict(C1=True, rankGate=rank_gate, condReg=condreg, effRank=er)
    f0 = np.linalg.solve(c_reg.T, c10.T).T
    target = float(np.trace(f0)/d)
    f = (1 - SHRINK_ALPHA)*f0 + SHRINK_ALPHA*target*np.eye(d)
    rho = float(np.max(abs(np.linalg.eigvals(f))))
    sf = float(np.linalg.svd(f, compute_uv=False)[0])
    # Held-out skill, first 80% fit vs diagonal predictor, descriptive only.
    cut = max(12, min(len(xx)-10, int(round(.8*len(xx)))))
    xtr, ytr = xx[:cut], yy[:cut]
    xte, yte = xx[cut:], yy[cut:]
    den = xtr.T @ xtr
    rr = RIDGE_REL * np.trace(den)/d
    try:
        ft = np.linalg.solve((den + rr*np.eye(d)).T, (ytr.T@xtr).T).T
        diag = np.diag(np.sum(xtr*ytr,axis=0)/(np.sum(xtr*xtr,axis=0)+rr+EPS))
        mse_full = float(np.mean((yte - xte@ft.T)**2))
        mse_diag = float(np.mean((yte - xte@diag.T)**2))
        skill = float(1 - mse_full/(mse_diag+EPS))
    except np.linalg.LinAlgError:
        skill = np.nan
    # Short-lag memory and residual autocorrelation in full coordinate space.
    resid = yy - xx@f.T
    lag = min(RESIDUAL_LAG, len(resid)//5)
    rr1 = np.nan
    if lag:
        a, b = resid[:-lag], resid[lag:]
        va, vb = np.linalg.norm(a), np.linalg.norm(b)
        rr1 = float(np.sum(a*b) / (va*vb+EPS))
    info = dict(C1=bool(c1), rankGate=rank_gate, C2=bool(rho >= 1-STABLE_TOL),
                condReg=condreg, effRank=er, rhoF=rho, sigmaF=sf,
                ridgeLambda=lam, holdoutSkillVsDiagonal=skill,
                residualLagCorr=rr1, rawCond=raw_cond)
    return f, info


def windows_one_record(rec):
    x, labels = rec['x'], rec['anno']
    fs = NOMINAL_FS
    w = int(round(WINDOW_SEC * fs))
    step = int(round(STEP_SEC * fs))
    dt = 1.0/NOMINAL_FS  # nominal sampling step, recorded time for endpoints
    gap = rec['bad_gap']
    rows = []
    n = len(x)
    if n < w: return pd.DataFrame()
    # Start at the first w samples, use exact right-exclusive endpoint as paper.
    for stop in range(w, n+1, step):
        start = stop-w
        labels_w = labels[start:stop]
        # Nonexperiment label 0 and gaps are not contiguous walking/freezing data.
        if np.any(labels_w == 0) or np.any(gap[start+1:stop]):
            continue
        f, info = fit_one_window(x[start:stop], dt)
        if f is None:
            continue
        for method in ('M2','M3'):
            q = m2_plane(f)[0] if method == 'M2' else m3_plane(f)
            if q is None:
                continue
            vals = reduced_indices(f, q, dt)
            c3 = not vals['stableGamma']
            c4 = bool(vals['bad_real_log'])
            # C5 criteria not fixed or calibrated: skill and lag corr are saved,
            # but should NOT be silently used to discard event windows.
            rows.append(dict(recording=rec['name'],
                             subject=rec['name'][:3], method=method,
                             i_start=start, i_end=stop,
                             t_start=rec['time'][start],
                             t_end=rec['time'][stop-1] + dt,
                             nonfog_fraction=float(np.mean(labels_w==1)),
                             fog_fraction=float(np.mean(labels_w==2)),
                             label_transitions=int(np.count_nonzero(np.diff(labels_w))),
                             C3=c3, C4=c4,
                             good_RA=bool(not info['C1'] and not info['rankGate'] and
                                          not info['C2'] and not c3 and not c4 and
                                          np.isfinite(vals['RA'])),
                             good_sigma=bool(not info['C1'] and not info['rankGate'] and
                                             not info['C2'] and not c3 and
                                             np.isfinite(vals['R_sigma'])),
                             **info, **vals))
    return pd.DataFrame(rows)


def find_event_contrasts(rec, df):
    events = get_events(rec['anno'])
    dt = 1./NOMINAL_FS
    eventrows, matchrows = [], []
    for event_no, start, end in events:
        t0 = rec['time'][start]
        t1 = rec['time'][end-1] + dt
        duration = t1-t0
        for method in ('M2','M3'):
            z = df.loc[df.method.eq(method)]
            if z.empty: continue
            # Window endpoint post-onset (includes walking/freeze mixing),
            # exact 4s onset interval if duration permits.
            on = z[(z.t_end >= t0+1e-7) &
                   (z.t_end <= min(t0+ONSET_SEC,t1)+1e-7) &
                   (z.fog_fraction > 0)]
            # Entire baseline window must be walking (label==1). Use only
            # the nearest pre-event walking windows with 3s endpoint gap.
            baseline_pool = z[(z.t_end >= t0-BASELINE_LOOKBACK_SEC) &
                              (z.t_end <= t0-BASELINE_GAP_SEC+1e-7) &
                              (z.nonfog_fraction > 1-1e-12)]
            baseline_pool = baseline_pool.sort_values('t_end', ascending=False)
            # MATCH BY NUMBER of onset windows, not outcome value.
            base = baseline_pool.head(len(on))
            event_id = f'{rec["name"][:-4]}_E{event_no:02d}'
            for subset, subdf in [('onset',on),('annotated_nonFOG_baseline',base)]:
                for _,rr in subdf.iterrows():
                    matchrows.append(dict(event_id=event_id, recording=rec['name'],
                                          subject=rec['name'][:3],method=method,
                                          subset=subset,t_end=rr.t_end,
                                          RA=rr.RA, R_sigma=rr.R_sigma,
                                          good_RA=rr.good_RA,good_sigma=rr.good_sigma,
                                          fog_fraction=rr.fog_fraction,
                                          label_transitions=rr.label_transitions))
            row = dict(event_id=event_id, subject=rec['name'][:3],
                       recording=rec['name'],event_no=event_no,
                       t_on=t0,t_off=t1,duration_s=duration,
                       method=method, n_on=int(len(on)), n_baseline=int(len(base)),
                       n_on_goodRA=int(on.good_RA.sum()),
                       n_base_goodRA=int(base.good_RA.sum()),
                       n_on_goodSigma=int(on.good_sigma.sum()),
                       n_base_goodSigma=int(base.good_sigma.sum()),
                       onset_mixed_fraction=float((on.label_transitions>0).mean()) if len(on) else np.nan,
                       onset_fog_fraction=float(on.fog_fraction.mean()) if len(on) else np.nan)
            for key, good in [('RA','good_RA'),('R_sigma','good_sigma')]:
                onvals = on.loc[on[good],key].dropna().to_numpy()
                basevals = base.loc[base[good],key].dropna().to_numpy()
                a = float(np.median(onvals)) if len(onvals) else np.nan
                b = float(np.median(basevals)) if len(basevals) else np.nan
                row[f'on_{key}'] = a
                row[f'base_{key}'] = b
                row[f'delta_{key}'] = a-b
                row[f'log2contrast_{key}'] = (np.log2(a/b) if
                    np.isfinite(a) and np.isfinite(b) and a>0 and b>0 else np.nan)
            row['eligible'] = bool(duration >= MIN_EVENT_SEC_FOR_RANK and
                                   row['n_on_goodRA'] >= MIN_ONSET_WINDOWS and
                                   row['n_base_goodRA'] >= MIN_BASELINE_WINDOWS)
            # QUALITY-ONLY rank ignores size/direction of RA, sigma or contrasts.
            row['quality_score'] = float(
                .4*min(row['n_base_goodRA']/8,1)+
                .3*min(row['n_on_goodRA']/6,1)+
                .2*min(duration/10,1)+
                .1*min((row['n_on_goodRA']+row['n_base_goodRA']) /
                       max(row['n_on']+row['n_baseline'],1)/.90,1))
            eventrows.append(row)
    return pd.DataFrame(eventrows), pd.DataFrame(matchrows)


def plot_representative(rec, df, chosen, eventno, outfile):
    events = get_events(rec['anno'])
    match = next((e for e in events if e[0] == eventno),None)
    if match is None:
        raise ValueError(f'{rec["name"]}: annotated event {eventno} not present; '
                         f'available events: {[e[0] for e in events]}')
    _, a, b = match
    t0, t1 = rec['time'][a],rec['time'][b-1]+1./NOMINAL_FS
    left, right = t0-20, t1+15
    masksig = (rec['time'] >= left) & (rec['time'] <= right)
    fig, axes = plt.subplots(5,1,figsize=(9,9.5),sharex=True,
                             gridspec_kw={'height_ratios':[1.25,1,1,1,1]})
    axes[0].plot(rec['time'][masksig]-t0,rec['raw_magnitude'][masksig],
                 color='#333333',lw=.65)
    axes[0].set_ylabel('Ankle norm\n(mg)')
    styles = [('M2','#df7f25'),('M3','#206ca5')]
    for method,col in styles:
        sub = df[(df.method==method)&(df.t_end>=left)&(df.t_end<=right)].sort_values('t_end')
        t = sub.t_end.to_numpy()-t0
        for ax,colname in zip(axes[1:],['RA','R_sigma','support','split_rate']):
            ys = sub[colname].to_numpy(float)
            if colname == 'RA': ys[~sub.good_RA.to_numpy(bool)] = np.nan
            if colname == 'R_sigma': ys[~sub.good_sigma.to_numpy(bool)] = np.nan
            if colname == 'split_rate': ys[~sub.good_RA.to_numpy(bool)] = np.nan
            ax.plot(t,ys,label=method,color=col,lw=1.55,alpha=.9)
    axes[1].set_ylabel(r'$R_A$ (rates)')
    axes[2].set_ylabel(r'$R_\sigma$ (step)')
    axes[3].set_ylabel(r'$S(\hat r)$')
    axes[4].set_ylabel(r'$\Delta$ real / $\beta$ complex')
    axes[4].set_xlabel('Window end time from FOG onset (s)')
    for i,ax in enumerate(axes):
        ax.axvspan(0,min(ONSET_SEC,t1-t0),color='#f9ca65',alpha=.27,lw=0)
        if t1-t0 > ONSET_SEC:
            ax.axvspan(ONSET_SEC,t1-t0,color='#c9c9c9',alpha=.29,lw=0)
        ax.axvline(0,color='#666666',ls=':',lw=.8)
        ax.text(-.085,1.02,f'({chr(97+i)})',transform=ax.transAxes,
                ha='right',va='top',weight='bold')
        ax.grid(alpha=.12)
        ax.set_xlim(left-t0,right-t0)
        if i in (1,2): ax.axhline(1,color='#444444',ls='--',lw=.8)
    axes[1].legend(frameon=False,ncol=2,fontsize=9,loc='upper right')
    fig.suptitle(f'{rec["name"]}: annotated FOG episode {eventno:02d}\n'
                 '3 s windows; 0.5 s step; pilot ridge/shrinkage',fontsize=12)
    fig.subplots_adjust(left=.16,right=.98,top=.90,bottom=.075,hspace=.22)
    fig.savefig(outfile.with_suffix('.png'),dpi=250)
    if SAVE_FIG_PDF: fig.savefig(outfile.with_suffix('.pdf'))
    plt.close(fig)


def per_subject_summary(events):
    if events.empty: return pd.DataFrame(), pd.DataFrame()
    e = events[events.eligible & (events.method=='M2')].copy()
    # For each subject, median event contrast WITHIN recording, then median
    # across recordings. This avoids treating repeated events as independent.
    if e.empty: return pd.DataFrame(), pd.DataFrame()
    g = e.groupby(['subject','recording'],as_index=False).agg(
        n_events=('event_no','size'),
        contrast_RA=('log2contrast_RA','median'),
        contrast_sigma=('log2contrast_R_sigma','median'))
    subjects = g.groupby('subject',as_index=False).agg(
        n_recordings=('recording','size'),
        n_events=('n_events','sum'),
        contrast_RA=('contrast_RA','median'),
        contrast_sigma=('contrast_sigma','median'))
    loo=[]
    a = subjects[['subject','contrast_RA']].dropna()
    for s in a.subject:
        other = a[a.subject != s].contrast_RA.to_numpy()
        loo.append(dict(excluded_subject=s,n_remaining=len(other),
                        median_RA_without_subject=float(np.median(other)) if len(other) else np.nan))
    return subjects, pd.DataFrame(loo)


def write_run_manifest():
    OUTDIR.mkdir(parents=True,exist_ok=True)
    txt = f'''FOG NNDR revised pilot; analysis parameters (NOT prespecified confirmatory inference)
Data folder: {DATA_DIR}
Window / step / onset: {WINDOW_SEC} / {STEP_SEC} / {ONSET_SEC} s
Nominal fs: {NOMINAL_FS} Hz
Ankle columns: 2-4 (1-based); annotation: column 11
Valid classes: 1 experiment without FOG (NOT proof of walking), 2 FOG; 0 excluded
Fit: raw mg units; center x/y separately per window (affine VAR);
relative ridge = {RIDGE_REL}; isotropic shrinkage = {SHRINK_ALPHA};
no forced stability and no per-channel standardization.
M2: max transverse action, {M2_RESTARTS} random + axis starts, {M2_ITERS} iterations;
M3: extreme eigenvectors of FF^T-F^TF.
C1 regularized Gram conditioning: {COND_MAX};
Additional pilot effective-rank gate: {EFFECTIVE_RANK_MIN};
C2 full-map spectral stability; C3 reduced stability; C4 real principal log domain.
C5 not calibrated: held-out skill and residual lag correlation are descriptive.
Time labels: endpoint convention, [t_end - W, t_end) as in manuscript.
Onset: first {ONSET_SEC}s after annotated label 2 begins; may mix states.
Baseline: last matching number of *fully annotated class-1* windows within
{BASELINE_LOOKBACK_SEC}s before onset and at least {BASELINE_GAP_SEC}s before onset.
Window count matching, quality ranking, and thresholds are exploratory choices;
not verified as the originally submitted FOG code's choices.
The complete dataset is on the user's Mac, not embedded in this script.
Figures and tables are descriptive until placebo/regime-switching controls,
pre-specified patient-level tests and calibrated adequacy screens are added.
'''
    (OUTDIR/'ANALYSIS_MANIFEST.txt').write_text(txt,encoding='utf-8')


def process_one(path):
    rec = load_recording(path)
    wdf = windows_one_record(rec)
    events, matched = find_event_contrasts(rec,wdf) if not wdf.empty else (pd.DataFrame(),pd.DataFrame())
    return rec,wdf,events,matched


def representative():
    p = DATA_DIR/REP_RECORDING
    if not p.exists():
        candidates = list(DATA_DIR.rglob(REP_RECORDING))
        if candidates: p=candidates[0]
    if not p.exists():
        print(f'*** Missing {REP_RECORDING} under {DATA_DIR}')
        print('Move the script into the Finder dataset folder, or edit DATA_DIR.')
        return
    print('Representative:',p)
    rec,wdf,events,matched = process_one(p)
    out = OUTDIR/'representative'
    out.mkdir(parents=True,exist_ok=True)
    wdf.to_csv(out/'S01R01_window_diagnostics.csv',index=False)
    events.to_csv(out/'S01R01_events.csv',index=False)
    matched.to_csv(out/'S01R01_event_matched_windows.csv',index=False)
    plot_representative(rec,wdf,events,REP_EVENT_NUMBER,
                        out/f'S01R01_event{REP_EVENT_NUMBER:02d}_five_panel')
    print('Event count:',len(get_events(rec['anno'])))
    cols=['event_no','method','duration_s','n_on_goodRA','n_base_goodRA',
          'on_RA','base_RA','log2contrast_RA','on_R_sigma','base_R_sigma']
    print(events.loc[events.event_no==REP_EVENT_NUMBER,cols].to_string(index=False))
    print('Representative results:',out)


def full_scan():
    paths=sorted({p.resolve() for p in DATA_DIR.rglob('*.txt')
                  if re.fullmatch(r'S\d\dR\d\d\.txt',p.name,re.IGNORECASE)})
    if not paths:
        print('No SxxRyy.txt files found under',DATA_DIR)
        return
    out = OUTDIR/'full_scan'
    out.mkdir(parents=True,exist_ok=True)
    recording_rows=[]; wtables=[]; etables=[]; matched_tables=[]; errors=[]; event_ledger=[]
    print(f'Found {len(paths)} Daphnet recordings. Full audit may be lengthy.')
    for i,p in enumerate(paths,1):
        print(f'[{i}/{len(paths)}] {p.name}',flush=True)
        try:
            rec,wdf,events,matched=process_one(p)
            for event_no, i_start, i_end in get_events(rec['anno']):
                event_ledger.append(dict(recording=p.name,subject=p.name[:3],
                    event_no=event_no,t_on=rec['time'][i_start],
                    t_off=rec['time'][i_end-1]+1/NOMINAL_FS,
                    duration_s=rec['time'][i_end-1]+1/NOMINAL_FS-rec['time'][i_start]))
            if not wdf.empty:
                recording_rows.append(dict(recording=p.name,subject=p.name[:3],
                    n_samples=len(rec['anno']),n_events=len(get_events(rec['anno'])),
                    recording_duration_s=float(rec['time'][-1]-rec['time'][0]),
                    n_windows_per_method=int((wdf.method=='M2').sum()),
                    complex_frac_M2=float((wdf.loc[wdf.method=='M2','spectrum']=='complex').mean()),
                    full_unstable_frac_M2=float(wdf.loc[wdf.method=='M2','C2'].mean()),
                    RA_admissible_frac_M2=float(wdf.loc[wdf.method=='M2','good_RA'].mean()),
                    R_sigma_admissible_frac_M2=float(wdf.loc[wdf.method=='M2','good_sigma'].mean()),
                    C1_fraction=float(wdf.loc[wdf.method=='M2','C1'].mean()),
                    rank_gate_fraction=float(wdf.loc[wdf.method=='M2','rankGate'].mean()),
                    C3_fraction_M2=float(wdf.loc[wdf.method=='M2','C3'].mean()),
                    C4_fraction_M2=float(wdf.loc[wdf.method=='M2','C4'].mean()),
                    C3_fraction_M3=float(wdf.loc[wdf.method=='M3','C3'].mean()),
                    C4_fraction_M3=float(wdf.loc[wdf.method=='M3','C4'].mean()),
                    median_holdout_skill=float(wdf.loc[wdf.method=='M2','holdoutSkillVsDiagonal'].median())))
                wtables.append(wdf)
            if not events.empty: etables.append(events)
            if not matched.empty: matched_tables.append(matched)
        except Exception as exc:
            errors.append(dict(recording=p.name,error=str(exc)))
            print('  ERROR:',exc)
    n_recording_errors = len(errors)
    pd.DataFrame(event_ledger, columns=['recording','subject','event_no','t_on','t_off','duration_s']).to_csv(
        out/'annotated_event_ledger.csv',index=False)
    if recording_rows:
        pd.DataFrame(recording_rows).to_csv(out/'recording_audit.csv',index=False)
    if not etables:
        print('No annotated event contrasts could be constructed.')
        return
    all_events=pd.concat(etables,ignore_index=True)
    all_events.to_csv(out/'all_event_contrasts.csv',index=False)
    if matched_tables:
        pd.concat(matched_tables,ignore_index=True).to_csv(out/'all_matched_windows.csv',index=False)
    if wtables:
        pd.concat(wtables,ignore_index=True).to_csv(out/'all_window_diagnostics.csv',index=False)
    quality=(all_events[(all_events.method=='M2') & all_events.eligible]
             .sort_values(['quality_score','event_id'],ascending=[False,True]))
    quality.to_csv(out/'candidate_rank_by_quality_NOT_effect.csv',index=False)
    effects=(quality.sort_values('log2contrast_RA',ascending=False))
    effects.to_csv(out/'EXPLORATORY_rank_by_observed_RA_effect.csv',index=False)
    # Draw the top quality-selected and top effect-selected cases separately.
    # Their names and ranking rationale are preserved so the second cannot be
    # mistaken for an independently prespecified representative result.
    selected=[]
    if len(quality):
        selected.append(('TOP_QUALITY_CANDIDATE', quality.iloc[0]))
    finite_effect=effects[np.isfinite(effects.log2contrast_RA)]
    if len(finite_effect):
        selected.append(('EXPLORATORY_TOP_POSITIVE_RA',finite_effect.iloc[0]))
    for tag,item in selected:
        try:
            path=next(p for p in paths if p.name==item.recording)
            rec2,wdf2,_,_=process_one(path)
            plot_representative(rec2,wdf2,None,int(item.event_no),
                out/f'{tag}_{item.event_id}')
        except Exception as exc:
            errors.append(dict(recording=str(item.recording),error=f'candidate_plot: {exc}'))
            print(f'Candidate plot failed for {item.event_id}: {type(exc).__name__}: {exc}',flush=True)
    # Write the final error ledger AFTER candidate plotting (fixes earlier mismatch).
    pd.DataFrame(errors,columns=['recording','error']).to_csv(out/'scan_errors.csv',index=False)
    n_plot_errors = len(errors) - n_recording_errors
    subjects,loo=per_subject_summary(all_events)
    subjects.to_csv(out/'subject_level_M2.csv',index=False)
    loo.to_csv(out/'leave_one_subject_out_M2.csv',index=False)
    # Descriptive summary; avoid pooled episode p-values (pseudo-replication).
    lines=[f'Recordings scanned: {len(paths)}',
           f'Recordings with load/analysis errors: {n_recording_errors}',
           f'Candidate plotting errors: {n_plot_errors}',
           f'FOG episodes annotated: {all_events[all_events.method=="M2"].shape[0]}',
           f'Eligible episodes (M2): {len(quality)}',
           f'Subjects entering per-subject M2 summary: {len(subjects)}',
           '', 'TOP QUALITY CANDIDATES (selection blind to effect):',
           quality[['event_id','subject','quality_score','n_on_goodRA','n_base_goodRA',
                    'duration_s','log2contrast_RA']].head(12).to_string(index=False),
           '', 'TOP POSITIVE CONTRAST CANDIDATES (EXPLORATORY, selection-biased):',
           effects[['event_id','subject','quality_score','log2contrast_RA',
                    'log2contrast_R_sigma']].head(12).to_string(index=False)]
    vals=subjects.contrast_RA.to_numpy(float)
    vals=vals[np.isfinite(vals)]
    if len(vals)>=2 and np.any(vals!=0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stat,p=wilcoxon(vals,alternative='two-sided')
        lines += ['',f'Descriptive subject-median log2 R_A contrast: {np.median(vals):.4g}',
                  f'UNCORRECTED EXPLORATORY Wilcoxon p={p:.4g} (n={len(vals)});',
                  'NOT a confirmatory result: placebo/switching/adequacy controls outstanding.']
    text='\n'.join(lines)+'\n'
    (out/'FULL_SCAN_SUMMARY.txt').write_text(text,encoding='utf-8')
    print(text)
    print('Full scan results:',out)


def main():
    write_run_manifest()
    if MODE in ('representative','both'): representative()
    if MODE in ('full','both'): full_scan()
    if MODE not in ('representative','full','both'):
        raise ValueError('MODE must be representative, full or both')


if __name__=='__main__':
    main()
