#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover original Figure 11 panels from the corrected EHG v2 analysis.

Spyder F5. Put the enclosing package folder directly in the project's `files`
folder (beside ehg_final_outputs and ehg_final_v2_full_cohort.py).

STAGE="statistics" reads cached window CSVs only; no raw waveforms or NNDR fits.
STAGE="geometry" refits *only high-activity windows* to reconstruct response
vectors, verifies their G and support against the same cached rows, and writes
R2(r->meanfield) and |<r,1/sqrt(N)>|. Never overwrites existing outputs.
STAGE="all" does statistics and then geometry.

The calculations intentionally follow manuscript (67)-(69) and (65)-(66),
while using corrected R_A/R_sigma and saved v2 acceptance flags.
No outlier filtering, baseline normalization or inferred events.
"""
from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent
FILES_DIR = Path(os.environ.get('NNDR_EHG_FILES_DIR', str(PKG.parent))).expanduser()
CACHE = FILES_DIR / 'ehg_final_outputs' / 'records'
SOURCE_SCRIPT = FILES_DIR / 'ehg_final_v2_full_cohort.py'
OUT = PKG / 'output'

STAGE = 'geometry'  # 'statistics' first; then change to 'geometry'; 'all' also supported
METRICS = ('RA', 'R_sigma')
GATE = 'common'  # common M2/M3 valid ratio windows for Spearman and Cliff's delta
GEOMETRY_GATE = 'plane_valid'  # geometry is defined for valid planes, including inadmissible ratios
MIN_CLASS_N = 8
RATIO_COMPARE_TOL = 2.e-5  # 16-channel cached vs re-extracted 2x2 operator
SUPPORT_COMPARE_TOL = 2.e-5
PROGRESS_EVERY = 10
STRICT_COHORT_COUNTS = True
EXPECTED_RECORDS = 123
EXPECTED_SUBJECTS = 45
EXPECTED_COMMON_RA_SUBJECTS = 44
# On a source mismatch the script stops rather than silently accepting a new fit.
FAIL_ON_ANY_MISMATCH = True
# If geometry phase has been completed, retain verified CSV checkpoints.
USE_GEOMETRY_CHECKPOINTS = True


def _write(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    df.to_csv(p, index=False)
    return p


def _read_cached_record(stem: str) -> pd.DataFrame:
    p = CACHE / stem / 'windows.csv'
    d = pd.read_csv(p)
    req = {'method','time_start_sec','time_end_sec','activity_at_end',
           'RA','R_sigma','Delta','support','method_status','G11','G12','G21','G22'}
    missing = req - set(d)
    if missing:
        raise ValueError(f'{p} missing columns {sorted(missing)}')
    if set(d.method.unique()) != {'M2','M3'}:
        raise ValueError(f'{p}: both M2 and M3 are required')
    if d.duplicated(['method','time_end_sec']).any():
        raise ValueError(f'{p}: duplicate window-end times')
    if len(d[d.method=='M2']) != len(d[d.method=='M3']):
        raise ValueError(f'{p}: M2/M3 physical-window count differs')
    if not np.array_equal(d.loc[d.method=='M2','time_end_sec'].sort_values().to_numpy(),
                          d.loc[d.method=='M3','time_end_sec'].sort_values().to_numpy()):
        raise ValueError(f'{p}: M2/M3 time grids do not agree')
    return d


def records() -> list[str]:
    if not CACHE.is_dir():
        raise FileNotFoundError(f'Cached fitted windows not found: {CACHE}.\n'
                                'Place this package beside ehg_final_outputs in your Mac files folder.')
    stems = sorted(p.parent.name for p in CACHE.glob('*/windows.csv'))
    if not stems:
        raise FileNotFoundError(f'No records/*/windows.csv under {CACHE}')
    ns=len({s.split('_')[0] for s in stems})
    if STRICT_COHORT_COUNTS and (len(stems)!=EXPECTED_RECORDS or ns!=EXPECTED_SUBJECTS):
        raise RuntimeError(f'Incomplete cohort: {len(stems)} records/{ns} subjects; expected {EXPECTED_RECORDS}/{EXPECTED_SUBJECTS}. STOP rather than generate a misleading plot.')
    return stems


def record_join(df: pd.DataFrame) -> pd.DataFrame:
    """One row per *physical* window; extract untouched M2/M3 cache values."""
    m2 = df[df.method=='M2'].set_index('time_end_sec').sort_index()
    m3 = df[df.method=='M3'].set_index('time_end_sec').sort_index()
    j = pd.DataFrame(index=m2.index)
    j['time_start_sec'] = m2.time_start_sec.to_numpy()
    j['activity'] = m2.activity_at_end.to_numpy()
    if not np.allclose(m2.activity_at_end.to_numpy(),m3.activity_at_end.to_numpy(),atol=1e-8,rtol=0):
        raise ValueError('M2/M3 activity reference mismatch')
    for method, obj in [('M2',m2),('M3',m3)]:
        for field in ('RA','R_sigma','Delta','support','method_status','fit_status'):
            if field in obj:
                j[f'{method}_{field}'] = obj[field].to_numpy()
    j = j.reset_index()
    if j.activity.notna().sum()<10:
        raise ValueError('Insufficient activity reference')
    q50, q80 = np.quantile(j.activity.to_numpy(float),[.5,.8])
    j['baseline'] = j.activity <= q50
    j['high'] = j.activity >= q80
    return j


def cliffs_delta(high: np.ndarray, base: np.ndarray) -> float:
    """Manuscript delta=P(high>base)-P(high<base), ties contribute zero."""
    high=np.asarray(high,float); base=np.asarray(base,float)
    high=high[np.isfinite(high)]; base=base[np.isfinite(base)]
    if not (len(high) and len(base)):
        return np.nan
    # U counts wins + half of tied pairs; ties therefore cancel in 2U/(mn)-1.
    U=stats.mannwhitneyu(high,base,alternative='two-sided',method='asymptotic').statistic
    return float(2.*U/(len(high)*len(base))-1.)


def valid_mask(j: pd.DataFrame, metric: str, method: str) -> np.ndarray:
    good=np.isfinite(j[f'{method}_{metric}'].to_numpy(float))
    if GATE=='common':
        good &= np.isfinite(j[f'{"M3" if method=="M2" else "M2"}_{metric}'].to_numpy(float))
    return good


def _median(y):
    y=np.asarray(y,float);y=y[np.isfinite(y)]
    return float(np.median(y)) if len(y) else np.nan


def compute_stat_record(stem: str) -> list[dict]:
    j=record_join(_read_cached_record(stem))
    output=[]
    for metric in METRICS:
        for method in ('M2','M3'):
            v=j[f'{method}_{metric}'].to_numpy(float)
            ok=valid_mask(j,metric,method)
            b=v[ok & j.baseline.to_numpy()]
            h=v[ok & j.high.to_numpy()]
            b=b[np.isfinite(b)];h=h[np.isfinite(h)]
            sufficient=len(b)>=MIN_CLASS_N and len(h)>=MIN_CLASS_N
            bmed=_median(b) if sufficient else np.nan
            hmed=_median(h) if sufficient else np.nan
            xx=j.activity.to_numpy(float)[ok]
            yy=v[ok]
            corr=float(stats.spearmanr(xx,yy).statistic) if len(xx)>=8 and len(set(xx))>1 and len(set(yy))>1 else np.nan
            cliff=cliffs_delta(h,b) if sufficient else np.nan
            output.append(dict(record=stem,subject_id=stem.split('_')[0],method=method,
                               metric=metric,gate=GATE,n_physical_windows=len(j),
                               n_high_all=int(j.high.sum()),n_base_all=int(j.baseline.sum()),
                               n_valid_all=int(ok.sum()),n_high_valid=len(h),n_base_valid=len(b),
                               baseline_median=bmed,high_median=hmed,
                               difference=hmed-bmed if sufficient else np.nan,
                               log2_high_over_baseline=np.log2(hmed/bmed) if sufficient and min(hmed,bmed)>0 else np.nan,
                               spearman_rho=corr,cliffs_delta=cliff))
    return output


def compute_high_geom_from_cache(stem: str) -> list[dict]:
    j=record_join(_read_cached_record(stem))
    out=[]
    for method in ('M2','M3'):
        plane_ok=j[f'{method}_method_status'].eq('ok').to_numpy()
        high=j.high.to_numpy()
        if GEOMETRY_GATE=='common_RA':
            allowed=high & plane_ok & np.isfinite(j.M2_RA.to_numpy(float)) & np.isfinite(j.M3_RA.to_numpy(float))
        elif GEOMETRY_GATE=='plane_valid':
            allowed=high & plane_ok
        else:
            raise ValueError(GEOMETRY_GATE)
        q=j[allowed]
        for field in ('Delta','support'):
            v=q[f'{method}_{field}'].to_numpy(float)
            out.append(dict(record=stem,subject_id=stem.split('_')[0],method=method,
                            metric=field,gate=GEOMETRY_GATE,n_high_valid=len(v[np.isfinite(v)]),
                            high_median=_median(v)))
    return out


def aggregate_subjects(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out=[]
    for (sub,method,metric),g in rows.groupby(['subject_id','method','metric']):
        r=dict(subject_id=sub,method=method,metric=metric,n_records=len(g))
        for col in metrics:
            if col in g:
                r[col]=_median(g[col])
        out.append(r)
    return pd.DataFrame(out)


def statistics_phase() -> None:
    stems=records()
    ra=[];geom=[]
    for i,stem in enumerate(stems,1):
        ra.extend(compute_stat_record(stem))
        geom.extend(compute_high_geom_from_cache(stem))
        if i%PROGRESS_EVERY==0 or i==len(stems):
            print(f'Statistics {i}/{len(stems)}: {stem}',flush=True)
    record_stats=pd.DataFrame(ra);geom_record=pd.DataFrame(geom)
    subject_stats=aggregate_subjects(record_stats,
        ['baseline_median','high_median','difference','log2_high_over_baseline',
         'spearman_rho','cliffs_delta'])
    geom_subject=aggregate_subjects(geom_record,['high_median'])
    _write(record_stats,'Fig11_record_statistical_panels.csv')
    _write(subject_stats,'Fig11_subject_statistical_panels.csv')
    _write(geom_record,'Fig11_record_high_geometry_from_cache.csv')
    _write(geom_subject,'Fig11_subject_high_geometry_from_cache.csv')
    if STRICT_COHORT_COUNTS:
        eligible=subject_stats[(subject_stats.metric=='RA') & (subject_stats.method=='M2')].difference.notna().sum()
        if eligible!=EXPECTED_COMMON_RA_SUBJECTS:
            raise RuntimeError(f'Expected {EXPECTED_COMMON_RA_SUBJECTS} common-window subjects in RA contrast, recovered {eligible}; STOP and audit the cache gate.')
    print(f'Statistics complete: {len(stems)} records, {record_stats.subject_id.nunique()} subjects')
    _audit_reference(subject_stats)
    _assemble_if_ready()


def _audit_reference(subject_stats:pd.DataFrame):
    # Reuse the completed common-window audit for an *independent* numerical check.
    f=FILES_DIR/'NNDR_EHG_Codex_Final_Plotting_v3'/'Previous_Verified_Cohort'/'data'/'subject_effects_method_specific_vs_common.csv'
    if not f.exists():
        print('Reference common-window effects not found; check after copying package to Mac.')
        return
    reference=pd.read_csv(f)
    ref=reference[(reference['metric'].isin(METRICS)) & (reference['mode']=='common_M2_M3')]
    joined=subject_stats.merge(ref[['subject_id','method','metric','difference']],
                               on=['subject_id','method','metric'],how='inner',suffixes=('_recalc','_reference'))
    joined['abs_error']=(joined.difference_recalc-joined.difference_reference).abs()
    _write(joined,'Fig11_common_gate_reference_audit.csv')
    eligible=joined.dropna(subset=['difference_recalc','difference_reference'])
    worst=float(eligible.abs_error.max()) if len(eligible) else np.nan
    print(f'Common-gate comparison: {len(eligible)} valid entries; max absolute error={worst:.4g}')
    if len(eligible) and worst>1e-7:
        raise RuntimeError(f'Subject contrasts do not reproduce the verified common-gate results (max difference {worst:.3g}). STOP and audit quantile/gate semantics.')


def _load_v2():
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(f'Missing original full-cohort NNDR script: {SOURCE_SCRIPT}')
    spec=importlib.util.spec_from_file_location('ehg_v2_source_for_fig11',SOURCE_SCRIPT)
    if spec is None or spec.loader is None: raise ImportError(str(SOURCE_SCRIPT))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DATA_DIR=FILES_DIR
    if not getattr(mod,'DATASET_ALL_AVAILABLE_CHANNELS',False):
        raise RuntimeError('Source script not set to all available EHG channels')
    return mod


def _compare_config(v2,meta,stem):
    pp=meta.get('params',{})
    fields={'band_hz':list(v2.FILTER_BAND_HZ),'fs_analysis':v2.FS_ANALYSIS,
            'crop_first_sec':v2.CROP_FIRST_SEC,'crop_last_sec':v2.CROP_LAST_SEC,
            'win_sec':v2.WIN_SEC,'step_sec':v2.STEP_SEC,
            'ridge_relative':v2.RIDGE_REL,'shrink_alpha':v2.SHRINK_ALPHA,
            'min_effective_rank':v2.MIN_EFFECTIVE_RANK,'condition_max':v2.COND_MAX,
            'seed':v2.SEED,'m2_restarts':v2.M2_RESTARTS,'m2_iters':v2.M2_ITERS}
    for k,v in fields.items():
        old=pp.get(k)
        if isinstance(v,list): good=np.allclose(np.asarray(v,float),np.asarray(old,float)) if old is not None else False
        elif isinstance(v,(int,float)): good=old is not None and np.isclose(float(old),float(v),atol=1e-12,rtol=0)
        else: good=old==v
        if not good:raise RuntimeError(f'{stem}: cached settings {k}={old!r} differ from source script {v!r}. STOP; use exact original script.')


def _nndr_geometry_one(v2,X,wi,cached_rows):
    F, fit=v2.fit_map(X)
    if F is None:
        raise RuntimeError(f'Window {wi}: source fit rejected despite valid cached plane')
    result=[]
    for method in ('M2','M3'):
        old=cached_rows[method]
        if old['method_status']!='ok':continue
        Q, _ = (v2.m2_plane(F,v2.SEED+wi) if method=='M2'
                 else v2.m3_plane(F,v2.SEED+wi+100000))
        if Q is None:raise RuntimeError(f'Window {wi} {method}: plane missing')
        G=Q.T@F@Q
        Gold=np.array([[float(old.G11),float(old.G12)],[float(old.G21),float(old.G22)]])
        gerr=float(np.max(np.abs(G-Gold)))
        r=Q[:,0]; n=len(r)
        supp=float(np.linalg.norm(r,1)/(np.sqrt(n)*np.linalg.norm(r)))
        serr=abs(supp-float(old.support))
        if FAIL_ON_ANY_MISMATCH and (gerr>RATIO_COMPARE_TOL or serr>SUPPORT_COMPARE_TOL):
            raise RuntimeError(f'Window {wi} {method}: recovered direction/source mismatch: max|G-Gcached|={gerr:.3g}, |S-Scached|={serr:.3g}. STOP.')
        yr=X@r
        mf=X.mean(axis=1)
        r2=float(np.corrcoef(yr,mf)[0,1]**2) if np.std(yr)>1.e-12 and np.std(mf)>1.e-12 else np.nan
        uniform=float(abs(np.sum(r))/(math.sqrt(n)*np.linalg.norm(r)))
        if np.isfinite(r2) and (r2 < -1e-9 or r2>1.+1e-9):raise RuntimeError('Bad meanfield R²')
        if not (0<=uniform<=1+1e-9):raise RuntimeError('Bad uniform alignment')
        result.append(dict(method=method,n_channels=n,
                           meanfield_R2=r2,uniform_alignment=uniform,
                           reconstructed_support=supp,
                           max_G_error=gerr,support_error=serr,
                           fit_ridge_lambda=float(fit['ridge_lambda']),
                           fit_rhoF=float(fit['rhoF'])))
    return result


def geometry_record(stem: str,v2) -> pd.DataFrame:
    path=OUT/'geometry_checkpoints'/f'{stem}_high_geometry.csv'
    if USE_GEOMETRY_CHECKPOINTS and path.exists():
        return pd.read_csv(path)
    cache=_read_cached_record(stem)
    meta_file=CACHE/stem/'record_metadata.json'
    if not meta_file.exists():raise FileNotFoundError(meta_file)
    meta=json.loads(meta_file.read_text())
    if 'config_sha256' not in meta: raise RuntimeError(f'{stem}: no original config hash')
    _compare_config(v2,meta,stem)
    data=v2.read_ehg(stem,representative=False)
    X=data['X']; fs=v2.FS_ANALYSIS
    if len(data['metadata']['channels'])!=16:
        raise RuntimeError(f'{stem}: unexpected {len(data["metadata"]["channels"])} channels; cohort requires exact original channel state')
    M2=cache[cache.method=='M2'].set_index('time_end_sec')
    M3=cache[cache.method=='M3'].set_index('time_end_sec')
    w=record_join(cache)
    selected=w[w.high].copy()
    if GEOMETRY_GATE=='common_RA':
        selected=selected[np.isfinite(selected.M2_RA)&np.isfinite(selected.M3_RA)]
    results=[]
    for _,q in selected.iterrows():
        key=float(q.time_end_sec)
        r0=M2.loc[key];r1=M3.loc[key]
        if r0['method_status']!='ok' and r1['method_status']!='ok':continue
        a=int(round((float(q.time_start_sec)-v2.CROP_FIRST_SEC)*fs))
        win=int(round(v2.WIN_SEC*fs));step=int(round(v2.STEP_SEC*fs))
        if a<0 or a+win>len(X) or a%step:
            raise RuntimeError(f'{stem}: cache window is not on the original sampling grid: {a=}')
        wi=a//step
        xw=X[a:a+win]
        calc=_nndr_geometry_one(v2,xw,wi,{'M2':r0,'M3':r1})
        for rr in calc:
            rr.update(record=stem,subject_id=stem.split('_')[0],
                      time_start_sec=float(q.time_start_sec),time_end_sec=key,
                      activity_at_end=float(q.activity),high=True,
                      source_config_sha256=meta['config_sha256'])
            results.append(rr)
    df=pd.DataFrame(results)
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)
    return df


def geometry_phase():
    stems=records()
    v2=_load_v2()
    allrows=[];manifest=[]
    for i,stem in enumerate(stems,1):
        try:
            df=geometry_record(stem,v2)
        except Exception as exc:
            _write(pd.DataFrame(manifest+[dict(record=stem,success=False,reason=str(exc))]),'Fig11_geometry_progress.csv')
            raise
        allrows.append(df)
        manifest.append(dict(record=stem,success=True,n_recovered=len(df)))
        _write(pd.DataFrame(manifest),'Fig11_geometry_progress.csv')
        if i%PROGRESS_EVERY==0 or i==len(stems):print(f'Geometry {i}/{len(stems)} {stem}: {len(df)} plane windows',flush=True)
    g=pd.concat(allrows,ignore_index=True)
    _write(g,'Fig11_reaction_geometry_high_windows.csv')
    rec=g.groupby(['subject_id','record','method'],as_index=False).agg(
        meanfield_R2=('meanfield_R2','median'),
        uniform_alignment=('uniform_alignment','median'),
        n_geometry_windows=('meanfield_R2','count'),
        worst_G_error=('max_G_error','max'),
        worst_support_error=('support_error','max'))
    _write(rec,'Fig11_reaction_geometry_record.csv')
    sub=rec.groupby(['subject_id','method'],as_index=False).agg(
        meanfield_R2=('meanfield_R2','median'),
        uniform_alignment=('uniform_alignment','median'),
        n_records=('record','nunique'),
        max_G_error=('worst_G_error','max'),
        max_support_error=('worst_support_error','max'))
    _write(sub,'Fig11_reaction_geometry_subject.csv')
    print(f'Geometry complete: {len(g)} method-window rows across {sub.subject_id.nunique()} subjects.')
    _assemble_if_ready()


def _assemble_if_ready():
    a=OUT/'Fig11_subject_statistical_panels.csv'
    b=OUT/'Fig11_subject_high_geometry_from_cache.csv'
    c=OUT/'Fig11_reaction_geometry_subject.csv'
    if not (a.exists() and b.exists()):return
    stats_sub=pd.read_csv(a);geom_sub=pd.read_csv(b)
    for metric in METRICS:
        sub=stats_sub[stats_sub.metric==metric].copy()
        piv=geom_sub.pivot_table(index=['subject_id','method'],columns='metric',values='high_median').reset_index()
        sub=sub.merge(piv,on=['subject_id','method'],how='left').rename(columns={'Delta':'Delta_high_median','support':'support_high_median'})
        if c.exists():
            sub=sub.merge(pd.read_csv(c)[['subject_id','method','meanfield_R2','uniform_alignment']],
                          on=['subject_id','method'],how='left',validate='one_to_one')
        _write(sub,f'Fig11_original_panels_subject_{metric}.csv')
        print(metric,'subject output rows',len(sub),'complete R2',sub.get('meanfield_R2',pd.Series(dtype=float)).notna().sum())


def main():
    print('Original Fig. 11 equation recovery (source: manuscript Eqs. 65-69)')
    print('FILES_DIR:',FILES_DIR)
    print('CACHE:',CACHE)
    if STAGE in ('statistics','all'):statistics_phase()
    if STAGE in ('geometry','all'):geometry_phase()
    if STAGE not in ('statistics','geometry','all'):raise ValueError('Invalid STAGE')
    print('Outputs:',OUT)
    print('Do NOT call the geometry or original-nine-panel figure empirically verified until geometry_stage and the cached-G checks have completed on all 123 records.')


if __name__=='__main__':main()
