#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NNDR EHG revision v1: reproducible representative + dataset analysis.

Run with F5 in Spyder after editing DATA_DIR and RUN_MODE below.
Dependencies: pip install numpy scipy pandas matplotlib wfdb
Input: PhysioNet Icelandic 16-electrode EHG WFDB .hea/.dat/.atr files.

This is a transparent corrected re-analysis, not a byte-for-byte recreation
of the original submitted generator calculation. The project's public
REPRODUCIBILITY.md documents a local ridge VAR(1), and that documented
one-step convention is explicitly implemented here. NO fabricated results.

Numerical plane-extraction and threshold conventions adapted from
fig15_pushup_revised_final.py and NNDR_complex_threshold_note.tex.
All outputs are checkpoints and CSVs; old KKc columns are never read.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import platform
import re
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy import linalg, signal, stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ========================= USER SETTINGS (Spyder F5) =========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT  # screenshot: .hea/.dat/.atr in same folder as this script
OUTDIR = ROOT / "ehg_final_outputs"
RUN_MODE = "dataset"  # "dataset", "annotations_only", "representative", "both", "selftest"
REP_RECORD = "ice012_p_1of4"
REP_CHANNELS = ("EHG9", "EHG10", "EHG11", "EHG12")
DATASET_ALL_AVAILABLE_CHANNELS = True
CROP_FIRST_SEC = 200.0   # PROVISIONAL: taken from old exploratory script
CROP_LAST_SEC = 60.0    # PROVISIONAL: taken from old exploratory script
FS_ANALYSIS = 10.0
FILTER_BAND_HZ = (0.08, 2.0)
FILTER_ORDER = 4
WIN_SEC, STEP_SEC = 60.0, 5.0
RIDGE_REL = 3e-3         # PROVISIONAL relative ridge; sensitivity needed
SHRINK_ALPHA = 0.10     # original manuscript EHG level
COND_MAX = 2e4
MIN_EFFECTIVE_RANK = 1.8
M2_RESTARTS = 10
M2_ITERS = 160
SEED = 20261118
EPS = 1e-12
STABLE_TOL = 1e-9
# Activity is independent of NNDR; 120s centered smoothing as old display.
ACTIVITY_SMOOTH_SEC = 120.0
HIGH_Q, BASE_Q = 0.80, 0.50
MIN_CLASS_WINDOWS = 8
# Independent EHG-activity peaks; minimum separation avoids pseudo-replication.
PEAK_MIN_DISTANCE_SEC = 240.0
PEAK_PROMINENCE_STD = 0.20
PEAK_HEIGHT_QUANTILE = 0.60
ALIGN_HALF_SEC = 120.0
ALIGN_STEP_SEC = STEP_SEC
ANNOTATION_EXCLUSION_SEC = 150.0  # for optional C-marker sensitivity
N_SUBJECT_BOOT = 4000
RUN_DATASET_ANNOTATIONS = True  # primary C-marked sensitivity, strict/inclusive separately
# Restart cached recordings even in a large dataset run.
FORCE_RECOMPUTE = False
SAVE_DPI = 500
COLORS = {"M2": "#E69F00", "M3": "#009E73"}
METHOD_LABELS = {"M2": "Optimization (M2)", "M3": "Commutator (M3)"}
METRICS = ("RA", "R_sigma", "support", "Delta")

# v2 cohort and independent-annotation audit. These settings do not alter the
# window estimator or corrected amplification definitions in v1.
ANN_STRICT_SEPARATION_SEC = 240.0
ANN_STRICT_ARTIFACT_SEC = 150.0
ANN_POSSIBLE_EXCLUSION_SEC = 90.0
AUTO_REPRESENTATIVE = True   # after dataset, refit EHG9-EHG12 for top quality cases
N_AUTO_REP = 2
ANNOTATION_EVENT_MIN_VALID_PHASE_POINTS = 3



# ========================= MATH: FIT, PLANES, THRESHOLDS =====================
def effective_rank_condition(X: np.ndarray) -> tuple[float, float]:
    s = np.linalg.svd(X, compute_uv=False)
    s = s[np.isfinite(s) & (s > EPS * max(np.max(s, initial=0), 1.0))]
    if not len(s):
        return 0., np.inf
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p)))), float(s.max() / s.min())


def fit_map(xw: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Column-state ridge VAR(1): x[k+1] ~ F x[k], no forced stabilization."""
    n, d = xw.shape
    info = {"n_samples": int(n), "n_channels": int(d), "fit_status": "ok"}
    if n < max(40, 3*d + 1) or not np.all(np.isfinite(xw)):
        return None, {**info, "fit_status": "nonfinite_or_short"}
    X, Y = xw[:-1], xw[1:]
    er, cond = effective_rank_condition(X)
    info.update(eff_rank=er, condition=cond)
    if er < MIN_EFFECTIVE_RANK or cond > COND_MAX:
        return None, {**info, "fit_status": "rank_or_condition"}
    scale = float(np.trace(X.T @ X) / d)
    lam = float(RIDGE_REL * max(scale, EPS))
    try:
        F0 = linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y,
                          assume_a="pos", check_finite=True).T
    except linalg.LinAlgError:
        return None, {**info, "fit_status": "ridge_solve_failure"}
    mu = float(np.trace(F0) / d)
    F = (1. - SHRINK_ALPHA) * F0 + SHRINK_ALPHA * mu * np.eye(d)
    residual = Y - X @ F.T
    den = float(np.sum(Y**2))
    info.update(ridge_lambda=lam, shrink_mu=mu,
                rhoF=float(max(abs(linalg.eigvals(F)))),
                sigmaF=float(linalg.svdvals(F)[0]),
                pred_r2=1. - float(np.sum(residual**2))/den if den > EPS else np.nan)
    return F, info


def m2_plane(F: np.ndarray, seed: int) -> tuple[np.ndarray | None, float]:
    """M2 orthogonal directed-coupling alternating optimization."""
    rng = np.random.default_rng(seed)
    d = len(F)
    best, score = None, -np.inf
    for _ in range(M2_RESTARTS):
        v = rng.normal(size=d)
        v /= max(np.linalg.norm(v), EPS)
        u = rng.normal(size=d)
        u -= (v @ u) * v
        u /= max(np.linalg.norm(u), EPS)
        previous = -np.inf
        for _ in range(M2_ITERS):
            u = F @ v
            u -= (v @ u) * v
            nu = np.linalg.norm(u)
            if nu < EPS: break
            u /= nu
            v = F.T @ u
            v -= (u @ v) * u
            nv = np.linalg.norm(v)
            if nv < EPS: break
            v /= nv
            val = float(u @ F @ v)
            if abs(val - previous) < 1e-11: break
            previous = val
        else:
            val = float(u @ F @ v)
        if nu < EPS or nv < EPS: continue
        val = float(u @ F @ v)
        if val > score:
            score = val
            best = np.column_stack((u.copy(), v.copy()))
    if best is None: return None, np.nan
    Q, _ = np.linalg.qr(best)
    return Q[:, :2], float(score)


def m3_plane(F: np.ndarray, seed: int) -> tuple[np.ndarray | None, float]:
    """M3 commutator extremes; orient the same plane via 2D M2."""
    B = F @ F.T - F.T @ F
    vals, vecs = linalg.eigh(B)
    if max(abs(vals[0]), abs(vals[-1])) < EPS:
        return None, np.nan
    P = vecs[:, [len(vals)-1, 0]]
    G = P.T @ F @ P
    Q2, obj = m2_plane(G, seed)
    return (P @ Q2 if Q2 is not None else P), obj


def henrici(G: np.ndarray, eig: np.ndarray) -> float:
    return float(np.sqrt(max(0., np.sum(G*G) - np.sum(abs(eig)**2))))


def thresholds(G: np.ndarray, dt: float) -> dict:
    """R_sigma exact, R_A rate-based with real principal log admissibility.

    Continuous-time K_c formulas from note / validated push-up script,
    with algebraically stable d_F(L) forms near repeated eigenvalues.
    The real-rate stable expression is
      R_A=d_F(L)/[(sqrt(a)+sqrt(b))*sqrt(2 sqrt(a b))], a,b>0.
    The complex rate expression is
      R_A=d_F(L)/[2 gamma sqrt(sqrt(1+beta^2)+1)], beta=omega/gamma.
    Both equal K/Kc,A away from eigenvalue coalescence.
    """
    eig = linalg.eigvals(G)
    sigma = float(linalg.svdvals(G)[0])
    rho = float(np.max(abs(eig)))
    dF = henrici(G, eig)
    denom = float(np.prod(1. - abs(eig)**2))
    spectrum = "complex_pair" if max(abs(eig.imag)) > 1e-9 else "real_pair"
    if spectrum == "real_pair" and abs(eig[0]-eig[1]) < 1e-9:
        spectrum = "repeated_real"
    eigen_sum = abs(eig[0] + eig[1])
    Delta = float(abs(eig[0]-eig[1])/eigen_sum) if eigen_sum > EPS else np.nan
    good_B = (rho < 1.-STABLE_TOL and denom > EPS)
    RB = dF/np.sqrt(denom) if good_B else np.nan
    out = {"RA": np.nan, "R_sigma": float(RB), "Delta": Delta,
           "dF": dF, "rhoGamma": rho, "sigmaGamma": sigma,
           "spectrum": spectrum, "RA_status": "unavailable",
           "R_sigma_status": "ok" if good_B else "reduced_unstable_or_margin",
           "RA_domain": "", "log_residual": np.nan}
    if not good_B:
        out["RA_status"] = "reduced_unstable_or_margin"
        return out
    # Negative real eigenvalues admit no unambiguous real principal log.
    # This also excludes singular / near-zero maps in the rate diagnostic.
    if np.min(abs(eig)) <= 1e-12:
        out["RA_status"] = "singular_map"
        return out
    if spectrum != "complex_pair" and np.any(eig.real <= 0):
        out["RA_status"] = "negative_real_multiplier"
        return out
    try:
        Lc = linalg.logm(G) / float(dt)
        imag_err = float(np.linalg.norm(np.imag(Lc), ord='fro'))
        real_norm = max(1., float(np.linalg.norm(np.real(Lc), ord='fro')))
        if imag_err > 1e-7 * real_norm:
            out["RA_status"] = "nonreal_principal_log"
            return out
        L = np.real(Lc)
        lrel = float(np.linalg.norm(linalg.expm(L*dt)-G, ord='fro') /
                     max(np.linalg.norm(G, ord='fro'), EPS))
        out["log_residual"] = lrel
        if lrel > 1e-7:
            out["RA_status"] = "log_reconstruction_error"
            return out
        rates = linalg.eigvals(L)
        if np.max(rates.real) >= -STABLE_TOL:
            out["RA_status"] = "nonstable_rates"
            return out
        dFL = henrici(L, rates)
        if max(abs(rates.imag)) > 1e-8:
            if abs(rates[0]-np.conj(rates[1])) > 1e-7:
                out["RA_status"] = "nonconjugate_rates"
                return out
            gamma = float(-np.mean(rates.real))
            omega = float(max(abs(rates.imag)))
            beta = omega/gamma
            RA = dFL / (2.*gamma*np.sqrt(np.hypot(1., beta) + 1.))
            out["RA_domain"] = "complex_rates"
        else:
            a, b = -np.real(rates)
            if min(a,b) <= 0:
                out["RA_status"] = "nonnegative_real_rate"
                return out
            RA = dFL / ((np.sqrt(a)+np.sqrt(b)) * np.sqrt(2.*np.sqrt(a*b)))
            out["RA_domain"] = "real_rates"
        out.update(RA=float(RA), RA_status="ok")
    except (ValueError, linalg.LinAlgError, FloatingPointError) as exc:
        out["RA_status"] = f"log_error:{type(exc).__name__}"
    return out


def analyze_one(xw: np.ndarray, wi: int) -> list[dict]:
    F, fit = fit_map(xw)
    rows = []
    for method in ("M2", "M3"):
        row = {"method": method, **fit}
        if F is None:
            row.update(method_status="fit_rejected", RA=np.nan, R_sigma=np.nan,
                       support=np.nan, Delta=np.nan, RA_status="fit_rejected",
                       R_sigma_status="fit_rejected", spectrum="not_fitted",
                       full_stable=np.nan, reduced_stable=np.nan)
            rows.append(row)
            continue
        Q, score = (m2_plane(F, SEED + wi) if method == "M2"
                    else m3_plane(F, SEED + wi + 100000))
        if Q is None:
            row.update(method_status="plane_rejected", RA=np.nan, R_sigma=np.nan,
                       support=np.nan, Delta=np.nan, RA_status="plane_rejected",
                       R_sigma_status="plane_rejected", spectrum="not_fitted",
                       full_stable=int(fit["rhoF"] < 1.-STABLE_TOL),
                       reduced_stable=np.nan)
            rows.append(row)
            continue
        # Orient response direction = Q[:,0], retain exact L1 support.
        G = Q.T @ F @ Q
        d = Q.shape[0]
        r = Q[:, 0]
        support = float(np.sum(abs(r))/(np.sqrt(d)*np.linalg.norm(r)))
        res = thresholds(G, 1./FS_ANALYSIS)
        row.update(**res, support=support, method_status="ok", objective=score,
                   G11=float(G[0,0]), G12=float(G[0,1]),
                   G21=float(G[1,0]), G22=float(G[1,1]),
                   full_stable=int(fit["rhoF"] < 1.-STABLE_TOL),
                   reduced_stable=int(res["rhoGamma"] < 1.-STABLE_TOL))
        if np.isfinite(row["R_sigma"]) and abs(row["R_sigma"]-1) > 1e-7:
            assert (row["R_sigma"] > 1) == (row["sigmaGamma"] > 1), \
                "Exact R_sigma versus singular-value identity violated"
        rows.append(row)
    return rows

# ================================ DATA ======================================
def dataset_record_stems() -> list[str]:
    return sorted({p.stem for p in DATA_DIR.rglob("*.hea")
                   if p.with_suffix(".dat").exists()})


def wfdb_path(stem: str) -> Path:
    found = list(DATA_DIR.rglob(stem+".hea"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one .hea for {stem} under {DATA_DIR}, found {len(found)}")
    return found[0].with_suffix("")


def read_ehg(stem: str, representative: bool) -> dict:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install WFDB: pip install wfdb") from exc
    base = wfdb_path(stem)
    rec = wfdb.rdrecord(str(base), physical=True)
    fs = float(rec.fs)
    if rec.p_signal is None or fs <= 2 * FILTER_BAND_HZ[1]:
        raise ValueError("Missing physical signal or incompatible sampling rate")
    names = [str(n).upper().strip() for n in rec.sig_name]
    channel_indices = ([names.index(ch) for ch in REP_CHANNELS] if representative
                       else [i for i, n in enumerate(names) if re.fullmatch(r"EHG\d+", n)])
    if len(channel_indices) < 2:
        raise ValueError("Fewer than two EHG channels")
    units = [str(rec.units[i]).replace("µ", "u").strip() for i in channel_indices]
    if all(u.lower() == "mv" for u in units):
        scale = 1000.  # physical mV -> microvolts; NOT raw MAT 131.068.
    elif all(u.lower() in {"uv", "microv"} for u in units):
        scale = 1.
    else:
        raise ValueError(f"Unknown/mixed WFDB physical channel units: {units}")
    n0, n1 = int(round(CROP_FIRST_SEC*fs)), int(round(CROP_LAST_SEC*fs))
    end = len(rec.p_signal) - n1
    if end - n0 < int(2*WIN_SEC*fs):
        raise ValueError("Recording too short after requested crop")
    X = rec.p_signal[n0:end, channel_indices] * scale
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite waveform samples; no interpolation applied")
    sos = signal.butter(FILTER_ORDER, FILTER_BAND_HZ, btype="bandpass",
                        fs=fs, output="sos")
    xf = signal.sosfiltfilt(sos, X, axis=0)
    # Rational anti-aliased resampling, preserving correct physical times.
    from fractions import Fraction
    ratio = Fraction(FS_ANALYSIS/fs).limit_denominator(100000)
    xd = signal.resample_poly(xf, ratio.numerator, ratio.denominator, axis=0)
    xd -= xd.mean(axis=0, keepdims=True)  # no per-channel variance normalization
    t = CROP_FIRST_SEC + np.arange(len(xd))/FS_ANALYSIS
    env = np.mean(np.abs(signal.hilbert(xd, axis=0)), axis=1)
    annotations = []
    if base.with_suffix(".atr").exists():
        ann = wfdb.rdann(str(base), "atr")
        for idx, sample in enumerate(ann.sample):
            sym = str(ann.symbol[idx]) if idx < len(ann.symbol) else ""
            aux = (str(ann.aux_note[idx]).strip(" \x00") if ann.aux_note is not None
                   and idx < len(ann.aux_note) else "")
            # The auxiliary label usually carries C, (c), pm, pos, fm, em.
            label = aux if aux else sym
            if label.startswith("C") and label != "C":
                label = label.strip()  # do not merge possible (c) into C
            annotations.append({"record": stem, "time_sec": float(sample/fs),
                                "symbol": sym, "aux_note": aux, "label": label})
    meta = {"record": stem, "channels": [rec.sig_name[i] for i in channel_indices],
            "units_before": units, "conversion_to_uV": scale,
            "fs_raw": fs, "n_raw_samples": int(len(rec.p_signal)),
            "duration_sec": float(len(rec.p_signal)/fs),
            "n_after_crop_downsample": int(len(xd)), "n_annotations":len(annotations),
            "record_type": stem.split("_")[1] if "_" in stem else "unknown"}
    return {"X": xd, "env": env, "time": t,
            "annotations": pd.DataFrame(annotations), "metadata": meta}


def smooth_activity_on_windows(raw_env: np.ndarray, t: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    # centered 120-s mean of envelope, evaluated at endpoint times; descriptive
    # event reference, not the instantaneous physiological state at t_end.
    smooth = pd.Series(raw_env).rolling(
        max(1, int(round(ACTIVITY_SMOOTH_SEC*FS_ANALYSIS))),
        center=True, min_periods=1).mean().to_numpy()
    return np.interp(t_end, t, smooth)


def save_record(stem: str, representative: bool = False) -> tuple[pd.DataFrame,dict]:
    sub = OUTDIR / ("representative" if representative else "records") / stem
    sub.mkdir(parents=True, exist_ok=True)
    csv = sub / "windows.csv"
    meta_csv = sub / "record_metadata.json"
    config_hash = hashlib.sha256(json.dumps({"params":parameters(),
        "representative":representative,"rep_channels":REP_CHANNELS,
        "m2_restarts":M2_RESTARTS,"m2_iters":M2_ITERS},
        sort_keys=True,default=str).encode()).hexdigest()
    if not FORCE_RECOMPUTE and csv.exists() and meta_csv.exists():
        cached_meta = json.loads(meta_csv.read_text(encoding="utf-8"))
        if cached_meta.get("config_sha256")==config_hash:
            return pd.read_csv(csv), cached_meta
        print(f"Settings changed; recomputing {stem}")
    D = read_ehg(stem, representative)
    X, t, env = D["X"], D["time"], D["env"]
    win, step = int(round(WIN_SEC*FS_ANALYSIS)), int(round(STEP_SEC*FS_ANALYSIS))
    starts = list(range(0, len(X)-win+1, step))
    t_ends = t[np.array(starts) + win-1] + 1./FS_ANALYSIS
    reference = smooth_activity_on_windows(env, t, t_ends)
    rows = []
    for j, a in enumerate(starts):
        for row in analyze_one(X[a:a+win], j):
            rows.append({"record":stem, "subject_id":stem.split("_")[0],
                         "time_start_sec":float(t[a]), "time_end_sec":float(t_ends[j]),
                         "activity_at_end":float(reference[j]), **row})
    win_df = pd.DataFrame(rows)
    win_df.to_csv(csv,index=False)
    D["annotations"].reindex(columns=["record","time_sec","symbol","aux_note","label"]).to_csv(sub/"annotations.csv", index=False)
    if representative:
        pd.DataFrame({"time_sec":t,"ehg_envelope_uV":env}).to_csv(sub/"envelope.csv", index=False)
    metadata = {**D["metadata"], "config_sha256":config_hash,
                "fit_model":"ridge VAR(1), 1/10-second discrete map",
                "original_generator_unverified":True,
                "params": parameters(), "n_rolling_windows": len(starts),
                "n_method_rows":len(rows),
                "method_status_counts":win_df.groupby(["method","method_status"]).size().to_dict().__str__(),
                "RA_status_counts":win_df.groupby(["method","RA_status"]).size().to_dict().__str__()}
    meta_csv.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    return win_df, metadata


def parameters() -> dict:
    return {"band_hz": FILTER_BAND_HZ, "fs_analysis":FS_ANALYSIS,
            "crop_first_sec":CROP_FIRST_SEC,"crop_last_sec":CROP_LAST_SEC,
            "win_sec":WIN_SEC,"step_sec":STEP_SEC,"ridge_relative":RIDGE_REL,
            "shrink_alpha":SHRINK_ALPHA,"min_effective_rank":MIN_EFFECTIVE_RANK,
            "condition_max":COND_MAX, "envelope_smoothing_sec":ACTIVITY_SMOOTH_SEC,
            "activity_high_quantile":HIGH_Q,"activity_baseline_quantile":BASE_Q,
            "peak_min_distance_sec":PEAK_MIN_DISTANCE_SEC,
            "peak_prominence_std":PEAK_PROMINENCE_STD,
            "seed": SEED,"m2_restarts":M2_RESTARTS,"m2_iters":M2_ITERS,
            "run_dataset_annotations":RUN_DATASET_ANNOTATIONS,
            "numpy":np.__version__,"scipy":scipy_version,
            "python":platform.python_version()}

# =========================== STATS AND EVENTS ================================
def valid_pair_group(df: pd.DataFrame, metric: str) -> tuple[np.ndarray,np.ndarray]:
    g = df.sort_values("time_end_sec")
    activity = g.activity_at_end.to_numpy(dtype=float)
    if not len(g) or np.sum(np.isfinite(activity)) < 10:
        return np.array([]), np.array([])
    lo, hi = np.nanquantile(activity,[BASE_Q,HIGH_Q])
    values = g[metric].to_numpy(dtype=float)
    return values[(activity<=lo)&np.isfinite(values)], values[(activity>=hi)&np.isfinite(values)]


def record_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for method,g in df.groupby("method"):
        for metric in METRICS:
            baseline, high=valid_pair_group(g,metric)
            ok = len(baseline)>=MIN_CLASS_WINDOWS and len(high)>=MIN_CLASS_WINDOWS
            mb = float(np.median(baseline)) if ok else np.nan
            mh = float(np.median(high)) if ok else np.nan
            rows.append({"record":str(g.record.iloc[0]),
                         "subject_id":str(g.subject_id.iloc[0]),
                         "method":method,"metric":metric,
                         "n_baseline_valid":len(baseline),"n_high_valid":len(high),
                         "baseline_median":mb,"high_median":mh,
                         "difference":mh-mb if ok else np.nan,
                         "log2_high_over_baseline":np.log2(mh/mb)
                         if ok and min(mh,mb)>0 else np.nan})
    return pd.DataFrame(rows)


def bootstrap_subject_median(x: np.ndarray) -> tuple[float,float,float]:
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if not len(x): return np.nan,np.nan,np.nan
    rng=np.random.default_rng(SEED+81)
    idx=rng.integers(0,len(x),size=(N_SUBJECT_BOOT,len(x)))
    b=np.median(x[idx],axis=1)
    return float(np.median(x)),*map(float,np.quantile(b,[0.025,0.975]))


def subject_summary(rec: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    # Weight subjects equally by median of their recording-specific effects.
    sub=rec.groupby(["subject_id","method","metric"],as_index=False).agg(
        difference=("difference","median"),
        log2_high_over_baseline=("log2_high_over_baseline","median"),
        n_records=("record","nunique"))
    rows=[]
    for (method,metric),g in sub.groupby(["method","metric"]):
        x=g.difference.to_numpy(float)
        point,ci_lo,ci_hi=bootstrap_subject_median(x)
        rows.append({"method":method,"metric":metric,"n_subjects_valid":int(np.isfinite(x).sum()),
                     "median_subject_effect":point,"CI95_low":ci_lo,"CI95_high":ci_hi,
                     "n_positive":int(np.sum(x>0)),"n_negative":int(np.sum(x<0)),
                     "n_zero":int(np.sum(x==0))})
    return sub,pd.DataFrame(rows)


def method_agreement(sub:pd.DataFrame) -> pd.DataFrame:
    """Agreement across paired subject effects, not across overlapping windows."""
    rows=[]
    for metric,g in sub.groupby("metric"):
        pivot=g.pivot(index="subject_id",columns="method",values="difference")
        if not {"M2","M3"}.issubset(pivot.columns):continue
        pair=pivot[["M2","M3"]].dropna()
        n=len(pair)
        corr=float(stats.spearmanr(pair.M2,pair.M3).statistic) if n>=3 else np.nan
        rows.append({"metric":metric,"n_paired_subjects":n,
                     "spearman_M2_M3":corr,
                     "same_nonzero_sign_frac":float(np.mean(np.sign(pair.M2)==np.sign(pair.M3)))
                         if n else np.nan})
    return pd.DataFrame(rows)


def peak_times(df:pd.DataFrame) -> pd.DataFrame:
    g=(df[df.method=="M2"].sort_values("time_end_sec")
       .drop_duplicates("time_end_sec"))
    tt=g.time_end_sec.to_numpy(float)
    a=g.activity_at_end.to_numpy(float)
    if len(a)<10: return pd.DataFrame(columns=["peak_time_sec","peak_activity"])
    dist=max(1,int(round(PEAK_MIN_DISTANCE_SEC/STEP_SEC)))
    prom=max(float(PEAK_PROMINENCE_STD*np.std(a)),EPS)
    idx,_=signal.find_peaks(a,distance=dist,prominence=prom,
                             height=float(np.quantile(a,PEAK_HEIGHT_QUANTILE)))
    idx=[i for i in idx if tt[i]-tt[0] >= ALIGN_HALF_SEC
         and tt[-1]-tt[i] >= ALIGN_HALF_SEC]
    return pd.DataFrame({"peak_time_sec":tt[idx],"peak_activity":a[idx]})


def aligned_curves(df:pd.DataFrame, events:pd.DataFrame, event_col:str) -> tuple[pd.DataFrame,pd.DataFrame]:
    if events.empty or event_col not in events:
        return pd.DataFrame(),pd.DataFrame()
    taus=np.arange(-ALIGN_HALF_SEC, ALIGN_HALF_SEC+0.01,ALIGN_STEP_SEC)
    individual=[]
    for method,g in df.groupby("method"):
        g=g.sort_values("time_end_sec")
        time=g.time_end_sec.to_numpy(float)
        for j,event in events.iterrows():
            et=float(event[event_col])
            for tau in taus:
                ix=int(np.argmin(abs(time-(et+tau))))
                row=g.iloc[ix]
                if abs(time[ix]-(et+tau)) > STEP_SEC/2+.01: continue
                obs={"record":row.record,"subject_id":row.subject_id,
                     "event_id":int(j),"event_time_sec":et,"method":method,
                     "tau_end_sec":float(tau),"activity":row.activity_at_end}
                for m in METRICS: obs[m]=row.get(m,np.nan)
                individual.append(obs)
    ev=pd.DataFrame(individual)
    if ev.empty: return ev,pd.DataFrame()
    summary=[]
    for (method,tau),g in ev.groupby(["method","tau_end_sec"]):
        o={"method":method,"tau_end_sec":tau,"n_events":g.event_id.nunique()}
        for m in ("activity",)+METRICS:
            vals=g[m].to_numpy(float); vals=vals[np.isfinite(vals)]
            o[m+"_median"]=float(np.median(vals)) if len(vals) else np.nan
            o[m+"_q25"]=float(np.quantile(vals,.25)) if len(vals) else np.nan
            o[m+"_q75"]=float(np.quantile(vals,.75)) if len(vals) else np.nan
            o[m+"_nvalid"]=int(len(vals))
        summary.append(o)
    return ev,pd.DataFrame(summary)


def event_phase_contrasts(events:pd.DataFrame) -> pd.DataFrame:
    # Prespecified window-end comparisons. Reference is an EHG activity peak,
    # not true contraction onset, and 60-s windows overlap phases physically.
    if events.empty:return pd.DataFrame()
    masks={"pre":(-120.,-60.),"peak":(-20.,20.),"post":(60.,120.)}
    rows=[]
    for (method,eid),g in events.groupby(["method","event_id"]):
        o={"method":method,"event_id":eid,"record":g.record.iloc[0]}
        for metric in METRICS:
            for name,(lo,hi) in masks.items():
                sel=g.loc[(g.tau_end_sec>=lo)&(g.tau_end_sec<=hi),metric]
                o[f"{metric}_{name}"]=sel.median() if sel.notna().sum()>=3 else np.nan
            o[f"{metric}_peak_minus_pre"]=o[f"{metric}_peak"]-o[f"{metric}_pre"]
            o[f"{metric}_post_minus_peak"]=o[f"{metric}_post"]-o[f"{metric}_peak"]
        rows.append(o)
    return pd.DataFrame(rows)


def annotation_events(ann:pd.DataFrame, df:pd.DataFrame) -> pd.DataFrame:
    """Separate secondary C-marker test; no conflation with (c) possible events."""
    if ann.empty or "label" not in ann: return pd.DataFrame(columns=["time_sec"])
    subset=ann.loc[ann.label.astype(str).str.strip().eq("C")].sort_values("time_sec")
    if subset.empty:return subset
    other=ann.loc[ann.label.astype(str).str.strip().isin(["pm","pos","em"])]
    times=[]
    t0,t1=df.time_end_sec.min(),df.time_end_sec.max()
    for _,r in subset.iterrows():
        t=float(r.time_sec)
        if t<t0+ALIGN_HALF_SEC or t>t1-ALIGN_HALF_SEC: continue
        if times and t-times[-1]<2*ALIGN_HALF_SEC:continue
        if not other.empty and np.any(abs(other.time_sec.to_numpy(float)-t)<ANNOTATION_EXCLUSION_SEC):
            continue
        times.append(t)
    return pd.DataFrame({"time_sec":times})

# ================================ PLOTTING ==================================
def set_style():
    plt.rcParams.update({"font.family":"sans-serif",
        "font.sans-serif":["Arial","Helvetica","DejaVu Sans"], "font.size":8,
        "axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
        "axes.linewidth":0.7,"pdf.fonttype":42,"savefig.dpi":SAVE_DPI})


def panel_letters(axs):
    for i,ax in enumerate(axs):
        ax.text(-.075,1.06,f"({chr(97+i)})",transform=ax.transAxes,
                fontsize=10,fontweight="bold",va="bottom",clip_on=False)

def shared_legend(fig):
    handles=[Line2D([0],[0],color=COLORS[m],lw=1.6,label=METHOD_LABELS[m])
             for m in ("M2","M3")]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,
               bbox_to_anchor=(.5,.005),fontsize=8)


def plot_representative(df:pd.DataFrame, envelope:pd.DataFrame, peaks:pd.DataFrame, path:Path, C_marker_times=None):
    set_style()
    fig,ax=plt.subplots(5,1,figsize=(18/2.54,8.7),sharex=True)
    fig.subplots_adjust(left=.14,right=.98,top=.98,bottom=.085,hspace=.25)
    t=envelope.time_sec.to_numpy(float)
    ax[0].plot(t,envelope.ehg_envelope_uV,color="0.55",lw=.42,alpha=.35)
    g=df.loc[df.method=="M2"].sort_values("time_end_sec")
    ax[0].plot(g.time_end_sec,g.activity_at_end,color="0.18",lw=1.3)
    if not peaks.empty:
        ax[0].scatter(peaks.peak_time_sec,peaks.peak_activity,s=17,marker="v",color="0.15",zorder=4)
    ax[0].set_ylabel("EHG envelope\n($\\mu$V)")
    labels=[r"$R_A$",r"$R_\sigma$",r"$S(\widehat{r})$",r"$\Delta$"]
    for ia,(metric,label) in enumerate(zip(METRICS,labels),start=1):
        for method in ("M2","M3"):
            g=df.loc[df.method==method].sort_values("time_end_sec")
            tm=g.time_end_sec.to_numpy(float)
            y=g[metric].to_numpy(float)
            ax[ia].plot(tm,y,color=COLORS[method],lw=.45,alpha=.25)
            smooth=pd.Series(y).rolling(7,center=True,min_periods=3).median()
            ax[ia].plot(tm,smooth,color=COLORS[method],lw=1.3)
        if metric in ("RA","R_sigma"):
            ax[ia].axhline(1,color=".35",ls=(0,(3,3)),lw=.7)
        ax[ia].set_ylabel(label)
        if metric in ("RA","R_sigma"):
            yy=np.r_[df.loc[df.method=="M2",metric],df.loc[df.method=="M3",metric]]
            yy=yy[np.isfinite(yy)]
            if len(yy)>10:
                ymax=max(1.1,float(np.quantile(yy,.99))*1.1)
                ax[ia].set_ylim(0,ymax)
                if np.any(yy>ymax):
                    ax[ia].text(.995,.93,"Upper 1% outside display",ha="right",va="top",
                                fontsize=6,transform=ax[ia].transAxes)
    # Independent manually entered C markers in the activity panel only.
    # Timestamp uncertainty (approx. 30 s) is discussed in the caption.
    for mt in ([] if C_marker_times is None else C_marker_times):
        if np.isfinite(mt):
            ax[0].axvline(float(mt),color="#8D6E63",ls=":",lw=.8,alpha=.55)
    ax[-1].set_xlabel("Recording time (s)")
    for a in ax:a.grid(False);a.margins(x=0)
    panel_letters(ax);shared_legend(fig)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"),dpi=SAVE_DPI)
    plt.close(fig)


def plot_aligned(summary:pd.DataFrame, filename:Path, annotation:bool=False):
    if summary.empty:return
    set_style()
    fig,ax=plt.subplots(5,1,figsize=(18/2.54,8.2),sharex=True)
    fig.subplots_adjust(left=.14,right=.98,top=.97,bottom=.09,hspace=.22)
    for ia,metric in enumerate(("activity",)+METRICS):
        for method in (("M2",) if ia==0 else ("M2","M3")):
            g=summary.loc[summary.method==method].sort_values("tau_end_sec")
            x=g.tau_end_sec.to_numpy(float)
            med=g[f"{metric}_median"].to_numpy(float)
            q25=g[f"{metric}_q25"].to_numpy(float)
            q75=g[f"{metric}_q75"].to_numpy(float)
            color=".15" if ia==0 else COLORS[method]
            ax[ia].plot(x,med,color=color,lw=1.4)
            ax[ia].fill_between(x,q25,q75,color=color,alpha=.13,lw=0)
        ax[ia].axvline(0,color="0.35",lw=.9,ls=":")
        if metric in ("RA","R_sigma"):
            ax[ia].axhline(1,color=".45",lw=.7,ls="--")
        ax[ia].set_ylabel({"activity":"Envelope", "RA":r"$R_A$",
                           "R_sigma":r"$R_\sigma$","support":r"$S(\hat r)$",
                           "Delta":r"$\Delta$"}[metric])
        ax[ia].grid(False)
    ax[-1].set_xlabel("Window-end time relative to "+("C marker" if annotation else "EHG activity peak")+" (s)")
    panel_letters(ax);shared_legend(fig)
    fig.savefig(filename.with_suffix(".pdf"))
    fig.savefig(filename.with_suffix(".png"),dpi=SAVE_DPI)
    plt.close(fig)


def plot_subjects(sub:pd.DataFrame,filename:Path):
    set_style()
    fig,ax=plt.subplots(2,2,figsize=(18/2.54,5.3),sharex=False)
    fig.subplots_adjust(left=.14,right=.98,top=.91,bottom=.17,wspace=.42,hspace=.52)
    rng=np.random.default_rng(SEED+2)
    for a,metric in zip(ax.flat,METRICS):
        g=sub.loc[sub.metric==metric]
        pivot=g.pivot(index="subject_id",columns="method",values="difference")
        if {"M2","M3"}.issubset(pivot.columns):
            for _,r in pivot[["M2","M3"]].dropna().iterrows():
                a.plot([1,2],[r.M2,r.M3],color="0.6",lw=.5,alpha=.25,zorder=0)
        for ix,m in enumerate(("M2","M3")):
            y=g.loc[g.method==m,"difference"].to_numpy(float)
            y=y[np.isfinite(y)]
            x=ix+1+rng.normal(0,.045,len(y))
            a.scatter(x,y,s=12,color=COLORS[m],alpha=.65,edgecolor="none")
            if len(y):
                med,lo,hi=bootstrap_subject_median(y)
                a.plot([ix+.82,ix+1.18],[med,med],color="black",lw=1.25)
                a.plot([ix+1,ix+1],[lo,hi],color="black",lw=1.05)
        a.axhline(0,color=".35",ls="--",lw=.8)
        a.set_xticks([1,2],labels=["M2","M3"])
        a.set_xlim(.55,2.45)
        a.set_ylabel(f"High minus baseline: {metric}")
    panel_letters(ax.flat)
    fig.savefig(filename.with_suffix(".pdf"))
    fig.savefig(filename.with_suffix(".png"),dpi=SAVE_DPI)
    plt.close(fig)

# =========================== FINAL RUN / CHECKS ===============================
def selftest() -> None:
    """Independent SVD/matrix-log tests on synthetic matrices only."""
    from scipy.linalg import expm
    rng=np.random.default_rng(6)
    for j in range(200):
        # Real/complex; stable by explicitly constructing a known generator.
        if j%2:
            L=np.array([[-.3,2.],[ -1.,-.3]]) * rng.uniform(.25,2)
        else:
            L=np.array([[-.2,4.],[0.,-1.]]) * rng.uniform(.25,2)
        G=expm(L*.1)
        D=thresholds(G,.1)
        assert D["RA_status"]=="ok",D
        assert D["R_sigma_status"]=="ok",D
        if abs(D["R_sigma"]-1)>1e-8:
            assert (D["R_sigma"]>1)==(linalg.svdvals(G)[0]>1)
        # Independently compute K/Kc on nondegenerate rate eigenvalues.
        lam,V=linalg.eig(L)
        p1=V[:,0]/np.linalg.norm(V[:,0]);p2=V[:,1]/np.linalg.norm(V[:,1])
        c=min(1-1e-14,abs(np.vdot(p1,p2)))
        K=c/np.sqrt(1-c*c)
        if j%2:
            gamma=-lam[0].real; omega=abs(lam[0].imag)
            beta=omega/gamma
            expected=K*np.sqrt(np.hypot(1.,beta)-1.)
        else:
            Delta=abs((lam[0]-lam[1])/(lam[0]+lam[1]))
            s=np.sqrt(1-Delta**2)
            expected=K/np.sqrt(s/(1-s))
        assert np.isclose(D["RA"],expected,rtol=1e-6,atol=1e-7),(D,expected)
    # Full operator/plane support identical to manuscript L1 formula.
    F=np.array([[.2,1.,0],[0,.4,.1],[0,0,.5]])
    for fun in (m2_plane,m3_plane):
        Q,_=fun(F,42)
        assert np.allclose(Q.T@Q,np.eye(2),atol=1e-8)
        r=Q[:,0]
        assert 1/np.sqrt(3)-1e-8<=np.linalg.norm(r,1)/np.sqrt(3)<=1+1e-8
    # Negative real eigenvalues must reject RA but may retain R_sigma.
    G=np.array([[-.8,.03],[0.,-.5]])
    D=thresholds(G,.1)
    assert np.isnan(D["RA"]) and np.isfinite(D["R_sigma"])
    print("PASS: 200 random known-generator ratios; M2/M3; negative-real gate")


def run_representative():
    print("EHG representative:",REP_RECORD)
    df,meta=save_record(REP_RECORD, representative=True)
    sub=OUTDIR/"representative"/REP_RECORD
    peaks=peak_times(df)
    peaks.to_csv(sub/"detected_activity_peaks.csv",index=False)
    env=pd.read_csv(sub/"envelope.csv")
    plot_representative(df,env,peaks,sub/"Fig_E1_corrected_representative")
    individual,summary=aligned_curves(df,peaks,"peak_time_sec")
    individual.to_csv(sub/"activity_peak_aligned_events.csv",index=False)
    summary.to_csv(sub/"activity_peak_aligned_summary.csv",index=False)
    phase=event_phase_contrasts(individual)
    phase.to_csv(sub/"activity_peak_phase_effects.csv",index=False)
    plot_aligned(summary,sub/"Fig_E2_peak_aligned")
    ann=pd.read_csv(sub/"annotations.csv") if (sub/"annotations.csv").exists() else pd.DataFrame()
    C=annotation_events(ann,df)
    C.to_csv(sub/"C_annotation_screened.csv",index=False)
    cev,csummary=aligned_curves(df,C,"time_sec")
    cev.to_csv(sub/"C_annotation_aligned_events.csv",index=False)
    csummary.to_csv(sub/"C_annotation_aligned_summary.csv",index=False)
    if not csummary.empty:plot_aligned(csummary,sub/"Fig_E2b_C_annotation_sensitivity",annotation=True)
    summary_rows=[]
    for method,g in df.groupby("method"):
        summary_rows.append({"method":method,"n_windows":g.time_end_sec.nunique(),
                             "n_plane_valid":int((g.method_status=="ok").sum()),
                             "n_RA_valid":int(g.RA.notna().sum()),
                             "n_R_sigma_valid":int(g.R_sigma.notna().sum()),
                             "n_complex":int((g.spectrum=="complex_pair").sum()),
                             "n_raw_peaks":len(peaks),"n_screened_C":len(C),
                             "n_RA_gt1":int((g.RA>1).sum()),
                             "n_R_sigma_gt1":int((g.R_sigma>1).sum())})
    pd.DataFrame(summary_rows).to_csv(sub/"representative_screening_summary.csv",index=False)
    print("Representative outputs:",sub)
    print(pd.DataFrame(summary_rows).to_string(index=False))



# ================= V2: ALL-RECORD ANNOTATION AUDIT ===========================
# The v1 fit, M2, M3, threshold, and Hilbert-envelope code above are left
# unchanged. In particular, annotations NEVER enter operator estimation,
# choice of plane or amplification thresholds.

EVENT_PHASES = {"pre": (-120.0, -60.0),
                "marker": (-20.0, 20.0),
                "post": (60.0, 120.0)}
EVENT_EFFECTS = ("marker_minus_pre", "post_minus_marker")


def _clean_annotation_label(row) -> str:
    """Read the PhysioNet auxiliary annotation label, without relabeling (c)."""
    for key in ("aux_note", "label", "symbol"):
        val = row.get(key, "")
        if pd.notna(val):
            val = str(val).replace("\\x00", "").strip()
            if val and val.lower() != "nan":
                return val
    return ""


def audit_C_markers(ann: pd.DataFrame, record: str,
                    t_start: float, t_end: float) -> pd.DataFrame:
    """Return ONE ROW PER RAW C MARKER, not just the surviving events.

    Primary (strict): complete +/-120 s support, no other C within 240 s,
    no movement/equipment event within 150 s, no possible contraction within
    90 s. Inclusive sensitivity: complete support only; correlation among
    nearby events is NOT treated as statistical independence.
    """
    columns = ["record", "subject_id", "record_type", "marker_id", "time_sec",
               "raw_label", "strict_status", "strict_accepted", "inclusive_accepted",
               "nearest_other_C_sec", "nearest_artifact_sec", "nearest_possible_c_sec",
               "n_all_annotations", "raw_C_total", "raw_possible_c_total"]
    if ann is None or ann.empty:
        return pd.DataFrame(columns=columns)
    a = ann.copy()
    a["raw_label"] = a.apply(_clean_annotation_label, axis=1)
    a["time_sec"] = pd.to_numeric(a["time_sec"], errors="coerce")
    a = a[np.isfinite(a["time_sec"])].sort_values("time_sec").reset_index(drop=True)
    all_C = a.loc[a.raw_label.eq("C")].copy()
    if all_C.empty:
        return pd.DataFrame(columns=columns)
    C_t = all_C.time_sec.to_numpy(dtype=float)
    movement_t = a.loc[a.raw_label.isin(["pm", "pos", "em"])].time_sec.to_numpy(dtype=float)
    possible_t = a.loc[a.raw_label.eq("(c)")].time_sec.to_numpy(dtype=float)
    rows=[]
    for j,(index,r) in enumerate(all_C.iterrows()):
        t=float(r.time_sec)
        nearest_C = float(np.min(np.abs(np.delete(C_t,j)-t))) if len(C_t)>1 else np.inf
        nearest_art = float(np.min(np.abs(movement_t-t))) if len(movement_t) else np.inf
        nearest_pos = float(np.min(np.abs(possible_t-t))) if len(possible_t) else np.inf
        coverage = bool(t >= t_start+ALIGN_HALF_SEC and t <= t_end-ALIGN_HALF_SEC)
        if not coverage:
            status="outside_full_window_support"
        elif nearest_C < ANN_STRICT_SEPARATION_SEC:
            status="neighbor_C_within_240s"
        elif nearest_art < ANN_STRICT_ARTIFACT_SEC:
            status="motion_or_equipment_within_150s"
        elif nearest_pos < ANN_POSSIBLE_EXCLUSION_SEC:
            status="possible_contraction_within_90s"
        else:
            status="accepted"
        rows.append(dict(record=record, subject_id=record.split("_")[0],
                         record_type=record.split("_")[1] if "_" in record else "?",
                         marker_id=int(j), time_sec=t, raw_label="C",
                         strict_status=status, strict_accepted=status=="accepted",
                         inclusive_accepted=coverage,
                         nearest_other_C_sec=nearest_C,
                         nearest_artifact_sec=nearest_art,
                         nearest_possible_c_sec=nearest_pos,
                         n_all_annotations=int(len(a)),raw_C_total=int(len(all_C)),
                         raw_possible_c_total=int(len(possible_t))))
    return pd.DataFrame(rows,columns=columns)


def tier_events(audit: pd.DataFrame, tier: str) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(columns=["time_sec", "marker_id", "record"])
    if tier not in ("strict", "inclusive"):
        raise ValueError(tier)
    col="strict_accepted" if tier=="strict" else "inclusive_accepted"
    return audit.loc[audit[col].astype(bool),["time_sec", "marker_id", "record"]].reset_index(drop=True)


def annotation_file_audit() -> pd.DataFrame:
    """Fast, standalone scan of .atr labels: NO .dat waveform or NNDR required."""
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("For .atr parsing: pip install wfdb") from exc
    rows=[]
    annotation_dir=OUTDIR/"annotation_audit"
    annotation_dir.mkdir(exist_ok=True, parents=True)
    headers=sorted(DATA_DIR.rglob("ice*.hea"))
    if not headers:
        raise FileNotFoundError(f"No ice*.hea headers under {DATA_DIR}")
    labels=[]
    markers=[]
    for i,head in enumerate(headers,1):
        record=head.stem
        base=head.with_suffix("")
        row={"record":record,"subject_id":record.split("_")[0],
             "record_type":record.split("_")[1] if "_" in record else "?",
             "has_dat":int(base.with_suffix(".dat").exists()),
             "has_atr":int(base.with_suffix(".atr").exists()),
             "n_annotation_total":0,"n_raw_C":0,"n_raw_possible_c":0,
             "n_motion_pm":0,"n_position_pos":0,"n_equipment_em":0,
             "n_fetal_fm":0,"read_status":"ok"}
        try:
            h=wfdb.rdheader(str(base))
            row["duration_sec"]=float(h.sig_len/h.fs)
            if row["has_atr"]:
                a=wfdb.rdann(str(base),"atr")
                ann=[]
                for j,sample in enumerate(a.sample):
                    aux=(str(a.aux_note[j]) if a.aux_note and j<len(a.aux_note) else "")
                    symbol=str(a.symbol[j]) if j<len(a.symbol) else ""
                    obs={"record":record,"subject_id":row["subject_id"],
                         "record_type":row["record_type"],
                         "time_sec":float(sample/h.fs),"aux_note":aux,"symbol":symbol}
                    obs["raw_label"]=_clean_annotation_label(obs)
                    ann.append(obs)
                aa=pd.DataFrame(ann)
                labels.extend(ann)
                row["n_annotation_total"]=len(aa)
                for lab,col in [("C","n_raw_C"),("(c)","n_raw_possible_c"),
                                ("pm","n_motion_pm"),("pos","n_position_pos"),
                                ("em","n_equipment_em"),("fm","n_fetal_fm")]:
                    row[col]=int((aa.raw_label==lab).sum()) if len(aa) else 0
                if len(aa):
                    audit=audit_C_markers(aa.rename(columns={"raw_label":"label"}),
                        record,CROP_FIRST_SEC+WIN_SEC,row["duration_sec"]-CROP_LAST_SEC)
                    if not audit.empty: markers.append(audit)
        except Exception as exc:
            row["read_status"]=f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if i%20==0: print(f"Annotation scan: {i}/{len(headers)}")
    summary=pd.DataFrame(rows)
    if markers:
        all_m=pd.concat(markers,ignore_index=True)
    else:
        all_m=audit_C_markers(pd.DataFrame(),"",0,0)
    all_m.to_csv(annotation_dir/"raw_C_marker_screening.csv",index=False)
    pd.DataFrame(labels,columns=["record","subject_id","record_type", "time_sec", "aux_note","symbol","raw_label"]).to_csv(
        annotation_dir/"all_raw_annotation_labels.csv",index=False)
    count_status=all_m.groupby(["record","strict_status"]).size().unstack(fill_value=0) if len(all_m) else pd.DataFrame()
    for col in ["accepted","outside_full_window_support","neighbor_C_within_240s",
                "motion_or_equipment_within_150s","possible_contraction_within_90s"]:
        summary["n_C_"+col]=summary.record.map(count_status[col]).fillna(0).astype(int) if col in count_status else 0
    summary["n_C_inclusive_covered"]=summary.record.map(
        all_m.groupby("record").inclusive_accepted.sum()).fillna(0).astype(int) if len(all_m) else 0
    summary.to_csv(annotation_dir/"annotation_record_coverage.csv",index=False)
    print("Annotation scan:",len(summary),"headers,",int(summary.n_raw_C.sum()),
          "raw C markers,",int(summary.n_C_accepted.sum()),"strict eligible,",
          int(summary.n_C_inclusive_covered.sum()),"inclusive covered")
    return summary


def phase_effects_v2(individual: pd.DataFrame) -> pd.DataFrame:
    """Same pre/mid/post intervals as v1, but give empty cases explicit IDs."""
    if individual.empty:
        return pd.DataFrame()
    rows=[]
    for (record,method,event_id),g in individual.groupby(["record","method","event_id"],sort=True):
        o={"record":record,"subject_id":record.split("_")[0],
           "record_type":record.split("_")[1] if "_" in record else "?",
           "method":method,"event_id":event_id,
           "event_time_sec":float(g.event_time_sec.iloc[0])}
        for metric in METRICS:
            for phase,(lo,hi) in EVENT_PHASES.items():
                vals=g.loc[g.tau_end_sec.between(lo,hi),metric].dropna().to_numpy(float)
                o[f"{metric}_{phase}_nvalid"]=len(vals)
                o[f"{metric}_{phase}"]=float(np.median(vals)) if len(vals)>=ANNOTATION_EVENT_MIN_VALID_PHASE_POINTS else np.nan
            o[f"{metric}_marker_minus_pre"]=o[f"{metric}_marker"]-o[f"{metric}_pre"]
            o[f"{metric}_post_minus_marker"]=o[f"{metric}_post"]-o[f"{metric}_marker"]
        rows.append(o)
    return pd.DataFrame(rows)


def subject_weighted_annotation_effects(phase: pd.DataFrame, tier: str):
    """Event -> recording median -> subject median, then subject bootstrap."""
    if phase.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    valcols=[f"{m}_{eff}" for m in METRICS for eff in EVENT_EFFECTS]
    rec=(phase.groupby(["subject_id","record","record_type","method"],as_index=False)
         .agg({**{c:"median" for c in valcols},"event_id":"nunique"}))
    rec=rec.rename(columns={"event_id":"n_annotation_events"})
    sub=(rec.groupby(["subject_id","record_type","method"],as_index=False)
         .agg({**{c:"median" for c in valcols},"record":"nunique","n_annotation_events":"sum"}))
    sub=sub.rename(columns={"record":"n_records"})
    # Subject can have both pregnancy and labor recordings: do not let that
    # subject count twice in the ALL-strata cohort summary.
    sub_all=(rec.groupby(["subject_id","method"],as_index=False)
             .agg({**{c:"median" for c in valcols},"record":"nunique","n_annotation_events":"sum"}))
    sub_all=sub_all.rename(columns={"record":"n_records"})
    rows=[]
    for grouping,g in [("all",sub_all)]+[(typ,sub[sub.record_type==typ]) for typ in ("p","l")]:
        for metric in METRICS:
            for eff in EVENT_EFFECTS:
                col=f"{metric}_{eff}"
                for method in ("M2","M3"):
                    a=g.loc[g.method==method,col].to_numpy(float)
                    n=int(np.isfinite(a).sum())
                    med,lo,hi=bootstrap_subject_median(a)
                    rows.append({"tier":tier,"stratum":grouping,"metric":metric,"effect":eff,
                                 "method":method,"n_subjects_valid":n,
                                 "n_subjects_positive":int(np.sum(a>0)),
                                 "n_subjects_negative":int(np.sum(a<0)),
                                 "median_subject_effect":med,"CI95_low":lo,"CI95_high":hi})
    return rec,sub,pd.DataFrame(rows)


def annotation_method_agreement(rec:pd.DataFrame,tier:str) -> pd.DataFrame:
    """M2/M3 consistency for subject effects, not window-level pseudoreplication."""
    if rec.empty:return pd.DataFrame()
    fields=[f"{m}_{e}" for m in METRICS for e in EVENT_EFFECTS]
    subs=rec.groupby(["subject_id","method"],as_index=False)[fields].median()
    rows=[]
    for field in fields:
        piv=subs.pivot(index="subject_id",columns="method",values=field)
        if not {"M2","M3"}.issubset(piv.columns):continue
        both=piv[["M2","M3"]].dropna()
        n=len(both)
        rho=float(stats.spearmanr(both.M2,both.M3).statistic) if n>=3 and both.M2.nunique()>1 and both.M3.nunique()>1 else np.nan
        rows.append({"tier":tier,"effect":field,"n_paired_subjects":n,
                     "M2_M3_spearman":rho,
                     "M2_M3_agree_sign_frac":float((np.sign(both.M2)==np.sign(both.M3)).mean()) if n else np.nan})
    return pd.DataFrame(rows)


def subject_weighted_annotation_curves(events:pd.DataFrame,tier:str):
    """Events -> record -> subject at each relative time, then equal-subject IQR."""
    if events.empty:return pd.DataFrame(),pd.DataFrame()
    cols=["activity",*METRICS]
    record_time=events.groupby(["subject_id","record","method","tau_end_sec"],as_index=False)[cols].median()
    subj_time=record_time.groupby(["subject_id","method","tau_end_sec"],as_index=False)[cols].median()
    rows=[]
    for (method,tau),g in subj_time.groupby(["method","tau_end_sec"],sort=True):
        r={"tier":tier,"method":method,"tau_end_sec":float(tau),
           "n_subjects_total":int(g.subject_id.nunique())}
        for c in cols:
            v=g[c].to_numpy(float);v=v[np.isfinite(v)]
            r[c+"_median"]=float(np.median(v)) if len(v) else np.nan
            r[c+"_q25"]=float(np.quantile(v,.25)) if len(v) else np.nan
            r[c+"_q75"]=float(np.quantile(v,.75)) if len(v) else np.nan
            r[c+"_nvalid"]=len(v)
        rows.append(r)
    return subj_time,pd.DataFrame(rows)


def plot_C_subject_effects(summary:pd.DataFrame,filename:Path,tier:str):
    if summary.empty:return
    s=summary.loc[(summary.tier==tier)&(summary.stratum=="all")]
    if s.empty:return
    set_style()
    fig, axes=plt.subplots(2,2,figsize=(18/2.54,6.5),sharex=True)
    fig.subplots_adjust(left=.14,right=.98,top=.93,bottom=.17,wspace=.4,hspace=.4)
    for ax,(metric,effect) in zip(axes.flat,[("RA","marker_minus_pre"),
                                  ("RA","post_minus_marker"),
                                  ("R_sigma","marker_minus_pre"),
                                  ("R_sigma","post_minus_marker")]):
        for j,method in enumerate(("M2","M3")):
            z=s.loc[(s.method==method)&(s.metric==metric)&(s.effect==effect)]
            if len(z):
                r=z.iloc[0]; x=j+1
                if np.isfinite(r.median_subject_effect):
                    ax.scatter([x],[r.median_subject_effect],color=COLORS[method],s=45,zorder=3)
                    ax.plot([x,x],[r.CI95_low,r.CI95_high],color=COLORS[method],lw=1.5)
                ax.text(x,.96,f"n={int(r.n_subjects_valid)}",transform=ax.get_xaxis_transform(),
                        ha="center",va="top",fontsize=7)
        ax.axhline(0,color=".35",ls="--",lw=.8)
        ax.set_xlim(.5,2.5);ax.set_xticks([1,2],labels=["M2","M3"])
        ax.set_ylabel(r"$"+("R_A" if metric=="RA" else "R_\\sigma")+r"$ difference")
        ax.set_title("Marker minus pre" if effect=="marker_minus_pre" else "Post minus marker",fontsize=8)
    panel_letters(axes.flat)
    for ext in (".pdf",".png"):
        fig.savefig(filename.with_suffix(ext),dpi=SAVE_DPI if ext==".png" else None)
    plt.close(fig)


def generate_annotated_rep(stem:str):
    """Refit selected recording with original four-channel representative model."""
    print("Four-channel representative from QUALITY selection:",stem)
    df,meta=save_record(stem,representative=True)
    output=OUTDIR/"representative"/stem
    env=pd.read_csv(output/"envelope.csv")
    peaks=peak_times(df)
    peaks.to_csv(output/"detected_activity_peaks.csv",index=False)
    annfile=output/"annotations.csv"
    ann=pd.read_csv(annfile,keep_default_na=False) if annfile.exists() else pd.DataFrame()
    c_times=(ann.loc[ann.apply(_clean_annotation_label,axis=1).eq("C"),"time_sec"].to_numpy(float)
             if len(ann) else np.array([],dtype=float))
    plot_representative(df,env,peaks,output/"Fig_E1_corrected_representative",C_marker_times=c_times)
    peakev,peakmed=aligned_curves(df,peaks,"peak_time_sec")
    peakmed.to_csv(output/"EHG_activity_peak_summary.csv",index=False)
    phase_effects_v2(peakev).to_csv(output/"EHG_activity_peak_phase_effects.csv",index=False)
    if len(peakmed):plot_aligned(peakmed,output/"Fig_E2_activity_peak")
    audit=audit_C_markers(ann,stem,float(df.time_end_sec.min()),float(df.time_end_sec.max()))
    audit.to_csv(output/"C_annotation_all_screening_representative.csv",index=False)
    for tier in ("strict","inclusive"):
        chosen=tier_events(audit,tier)
        if chosen.empty:continue
        ev,summary=aligned_curves(df,chosen,"time_sec")
        ev.to_csv(output/f"C_aligned_events_{tier}.csv",index=False)
        summary.to_csv(output/f"C_aligned_summary_{tier}.csv",index=False)
        phase_effects_v2(ev).to_csv(output/f"C_phase_effects_{tier}.csv",index=False)
        if len(summary):plot_aligned(summary,output/f"Fig_C_aligned_{tier}",annotation=True)
    return meta


def select_quality_representatives(coverage:pd.DataFrame, raw_audit:pd.DataFrame,
                                   annot_summary:pd.DataFrame) -> pd.DataFrame:
    """Selection uses ONLY coverage/admissibility and marker availability.

    No RA magnitude, direction of effect, p-value or activity correlation is
    used. This avoids looking for a striking effect and then reporting it as
    typical of a prospectively selected recording.
    """
    if coverage.empty:return pd.DataFrame()
    coverage=coverage.copy()
    coverage["valid_RA_frac"]=coverage.n_RA_valid / coverage.n_windows.clip(lower=1)
    coverage["valid_Rsigma_frac"]=coverage.n_R_sigma_valid/coverage.n_windows.clip(lower=1)
    qq=coverage.groupby("record",as_index=False).agg(
        min_RA_frac=("valid_RA_frac","min"),
        min_Rsigma_frac=("valid_Rsigma_frac","min"),
        min_n_windows=("n_windows","min"),
        min_full_stable_frac=("n_full_stable",lambda x:float(np.min(x.to_numpy() / coverage.loc[x.index,"n_windows"].clip(lower=1).to_numpy()))))
    if len(raw_audit):
        qq=qq.merge(raw_audit[["record","record_type","n_raw_C","n_C_accepted","n_C_inclusive_covered"]],on="record",how="left")
    else:
        qq["record_type"]="?"
        for c in ("n_raw_C","n_C_accepted","n_C_inclusive_covered"):qq[c]=0
    if len(annot_summary):
        q=annot_summary.groupby("record",as_index=False).agg(
            n_C_strict_final=("n_strict_accepted","max"),
            n_C_inclusive_final=("n_inclusive_accepted","max"))
        qq=qq.merge(q,on="record",how="left")
    for c in ("n_raw_C","n_C_accepted","n_C_inclusive_covered", "n_C_strict_final","n_C_inclusive_final"):
        if c not in qq:qq[c]=0
        qq[c]=qq[c].fillna(0).astype(int)
    # Quality filter only: prioritize strict markers, then high completeness.
    qq["selection_score"]=(4*np.minimum(qq.n_C_strict_final,4)
        +1*np.minimum(qq.n_C_inclusive_final,6)
        +5*qq.min_RA_frac+3*qq.min_Rsigma_frac+4*qq.min_full_stable_frac
        +.4*np.minimum(qq.min_n_windows/100,6))
    qq["selection_criterion"]="Independent C marker coverage, RA/Rsigma admissibility, full stability, duration; no outcome magnitude"
    qq=qq.sort_values(["selection_score","record"],ascending=[False,True]).reset_index(drop=True)
    qq["rank"]=np.arange(1,len(qq)+1)
    return qq


def _coverage_rows(df:pd.DataFrame,stem:str) -> list[dict]:
    rows=[]
    for method,g in df.groupby("method"):
        rows.append({"record":stem,"subject_id":stem.split("_")[0],
            "record_type":stem.split("_")[1] if "_" in stem else "?",
            "method":method,"n_windows":g.time_end_sec.nunique(),
            "n_plane_valid":int((g.method_status=="ok").sum()),
            "n_RA_valid":int(g.RA.notna().sum()),
            "n_R_sigma_valid":int(g.R_sigma.notna().sum()),
            "n_complex":int((g.spectrum=="complex_pair").sum()),
            "n_full_stable":int((g.full_stable==1).sum()),
            "n_reduced_stable":int((g.reduced_stable==1).sum()),
            "n_RA_gt1":int((g.RA>1).sum()),
            "n_R_sigma_gt1":int((g.R_sigma>1).sum()),
            "n_negative_real_log_rejected":int((g.RA_status=="negative_real_multiplier").sum()),
            "n_reduced_unstable":int((g.R_sigma_status=="reduced_unstable_or_margin").sum()),
            "n_fit_rejected":int((g.fit_status!="ok").sum()),
            "n_plane_rejected":int((g.method_status=="plane_rejected").sum())})
    return rows


def run_dataset():
    """Full dataset, strict/inclusive C labels, subject-weighted inference."""
    stems=dataset_record_stems()
    if not stems:raise FileNotFoundError(f"No .hea AND .dat WFDB pairs inside {DATA_DIR}")
    print(f"DATASET: found {len(stems)} complete EHG waveform recordings, {len(set(s.split('_')[0] for s in stems))} subjects")
    OUTDIR.mkdir(parents=True,exist_ok=True)
    raw_all=annotation_file_audit() if RUN_DATASET_ANNOTATIONS else pd.DataFrame()
    manifest=[];record_contrast=[];coverage=[];markers=[];event_frames={"strict":[],"inclusive":[]}
    screen_rows=[]
    for i,stem in enumerate(stems,1):
        print(f"[{i}/{len(stems)}] {stem}",flush=True)
        try:
            df,metadata=save_record(stem,representative=False)
            subdir=OUTDIR/"records"/stem
            record_contrast.append(record_contrasts(df))
            coverage.extend(_coverage_rows(df,stem))
            if RUN_DATASET_ANNOTATIONS:
                afile=subdir/"annotations.csv"
                ann=pd.read_csv(afile,keep_default_na=False) if afile.exists() else pd.DataFrame()
                aud=audit_C_markers(ann,stem,float(df.time_end_sec.min()),float(df.time_end_sec.max()))
                aud.to_csv(subdir/"C_annotation_all_screening.csv",index=False)
                if len(aud):markers.append(aud)
                scr={"record":stem,"subject_id":stem.split("_")[0],
                     "record_type":stem.split("_")[1],
                     "n_raw_annotation_rows":len(ann),
                     "n_raw_C":int((ann.apply(_clean_annotation_label,axis=1)=="C").sum()) if len(ann) else 0,
                     "n_raw_possible_c":int((ann.apply(_clean_annotation_label,axis=1)=="(c)").sum()) if len(ann) else 0,
                     "n_strict_accepted":int(aud.strict_accepted.sum()) if len(aud) else 0,
                     "n_inclusive_accepted":int(aud.inclusive_accepted.sum()) if len(aud) else 0}
                for reason in ["outside_full_window_support", "neighbor_C_within_240s",
                               "motion_or_equipment_within_150s", "possible_contraction_within_90s"]:
                    scr["excluded_"+reason]=int((aud.strict_status==reason).sum()) if len(aud) else 0
                screen_rows.append(scr)
                for tier in ("strict","inclusive"):
                    chosen=tier_events(aud,tier)
                    if not len(chosen):continue
                    individual,summary=aligned_curves(df,chosen,"time_sec")
                    if len(individual):
                        individual["tier"]=tier
                        individual["record_type"]=scr["record_type"]
                        event_frames[tier].append(individual)
                        phase=phase_effects_v2(individual)
                        phase.to_csv(subdir/f"C_{tier}_phase_effects.csv",index=False)
                        summary.to_csv(subdir/f"C_{tier}_aligned_summary.csv",index=False)
            manifest.append({"record":stem,"success":True,"n_windows_per_method":int(len(df)/2),"error":""})
        except Exception as e:
            manifest.append({"record":stem,"success":False,"error":f"{type(e).__name__}: {e}"})
            warnings.warn(f"Skipped {stem}: {e}")
        pd.DataFrame(manifest).to_csv(OUTDIR/"dataset_manifest.csv",index=False)
        if coverage:pd.DataFrame(coverage).to_csv(OUTDIR/"dataset_coverage.csv",index=False)
        if record_contrast:pd.concat(record_contrast,ignore_index=True).to_csv(
            OUTDIR/"dataset_record_effects.csv",index=False)
        if screen_rows:pd.DataFrame(screen_rows).to_csv(
            OUTDIR/"dataset_C_marker_screening_by_record.csv",index=False)
    if not record_contrast:
        raise RuntimeError("No recordings processed; see dataset_manifest.csv")
    rec=pd.concat(record_contrast,ignore_index=True)
    rec.to_csv(OUTDIR/"dataset_record_effects.csv",index=False)
    sub, summary=subject_summary(rec)
    sub.to_csv(OUTDIR/"dataset_subject_effects.csv",index=False)
    summary.to_csv(OUTDIR/"dataset_subject_summary.csv",index=False)
    method_agreement(sub).to_csv(OUTDIR/"dataset_method_agreement.csv",index=False)
    plot_subjects(sub,OUTDIR/"Fig_E3_high_vs_baseline_subject_effects")
    cohort_cov=pd.DataFrame(coverage)
    cohort_cov.to_csv(OUTDIR/"dataset_coverage.csv",index=False)
    all_screen=pd.DataFrame(screen_rows)
    if len(all_screen):all_screen.to_csv(OUTDIR/"dataset_C_marker_screening_by_record.csv",index=False)
    allmarkers=pd.concat(markers,ignore_index=True) if markers else pd.DataFrame()
    if len(allmarkers):allmarkers.to_csv(OUTDIR/"dataset_C_marker_all_rows_screening.csv",index=False)
    for tier,frames in event_frames.items():
        if not frames:continue
        events=pd.concat(frames,ignore_index=True)
        # Retain per-event identifier within the record; no pooling ambiguity.
        phase=phase_effects_v2(events)
        phase.to_csv(OUTDIR/f"dataset_C_{tier}_event_phase_effects.csv",index=False)
        rr,ss,stat=subject_weighted_annotation_effects(phase,tier)
        rr.to_csv(OUTDIR/f"dataset_C_{tier}_record_effects.csv",index=False)
        ss.to_csv(OUTDIR/f"dataset_C_{tier}_subject_effects.csv",index=False)
        annotation_method_agreement(rr,tier).to_csv(
            OUTDIR/f"dataset_C_{tier}_method_agreement.csv",index=False)
        stat.to_csv(OUTDIR/f"dataset_C_{tier}_subject_summary.csv",index=False)
        stime,curves=subject_weighted_annotation_curves(events,tier)
        stime.to_csv(OUTDIR/f"dataset_C_{tier}_subject_time_curves.csv",index=False)
        curves.to_csv(OUTDIR/f"dataset_C_{tier}_subject_weighted_aligned.csv",index=False)
        plot_aligned(curves,OUTDIR/f"Fig_C_{tier}_subject_weighted",annotation=True)
        plot_C_subject_effects(stat,OUTDIR/f"Fig_C_{tier}_subject_effects",tier)
    ranks=select_quality_representatives(cohort_cov,raw_all,all_screen)
    if len(ranks):ranks.to_csv(OUTDIR/"quality_ranked_representatives.csv",index=False)
    print("\n-- FULL COHORT SUMMARY --")
    print("Unique complete records:",len(stems),"Successful:",sum(x["success"] for x in manifest))
    print("Unique subjects successfully included:",cohort_cov.subject_id.nunique())
    print(summary.to_string(index=False))
    if len(all_screen):
        print("\nC marker coverage: raw",int(all_screen.n_raw_C.sum()),
              "strict",int(all_screen.n_strict_accepted.sum()),
              "inclusive",int(all_screen.n_inclusive_accepted.sum()))
    if len(ranks):
        print("\nTop representative candidates selected by quality only:")
        print(ranks[["rank","record","selection_score","min_RA_frac","n_raw_C", "n_C_strict_final"]].head(8).to_string(index=False))
    # Refitting candidates four-channel costs extra; do it only after all
    # 16-channel cohort work, and preserve rank even when a refit fails.
    if AUTO_REPRESENTATIVE and len(ranks):
        used_subjects=set();chosen=[]
        for _,r in ranks.iterrows():
            stem=str(r.record); subj=stem.split("_")[0]
            if subj in used_subjects:continue
            used_subjects.add(subj);chosen.append(stem)
            if len(chosen)>=N_AUTO_REP:break
        reports=[]
        for stem in chosen:
            try:
                meta=generate_annotated_rep(stem)
                reports.append({"record":stem,"success":True,"error":""})
            except Exception as exc:
                reports.append({"record":stem,"success":False,"error":str(exc)})
                warnings.warn(f"Auto-representative {stem} failed: {exc}")
        pd.DataFrame(reports).to_csv(OUTDIR/"auto_representative_refit_manifest.csv",index=False)
    print("EHG cohort complete. Audit raw annotation labels, screening and subject tables BEFORE any claim.")


def selftest_v2():
    """Additional annotation status and independent-subject-weighting tests."""
    ann=pd.DataFrame({"record":["ice001_p_1of1"]*7,
         "time_sec":[10.,300.,600.,630.,900.,1200.,1550.],
         "aux_note":["C","C","C","(c)","pm","C","C"],
         "symbol":["~"]*7})
    audit=audit_C_markers(ann,"ice001_p_1of1",250,1700)
    assert len(audit)==5
    assert int(audit.strict_accepted.sum())==2, audit[["time_sec","strict_status"]]
    assert int(audit.inclusive_accepted.sum())==3
    assert len(tier_events(audit,"strict"))==2
    assert len(tier_events(audit,"inclusive"))==3
    phase=[]
    for sub,record,vals in [("ice001","ice001_p_1of2",[1,1,1]),
                            ("ice001","ice001_p_2of2",[1,1,1]),
                            ("ice002","ice002_p_1of1",[-1,-1,-1])]:
        for i,val in enumerate(vals):
            r={"subject_id":sub,"record":record,"record_type":"p",
               "method":"M2","event_id":i}
            for metric in METRICS:
                for eff in EVENT_EFFECTS:r[f"{metric}_{eff}"]=val
            phase.append(r)
    rr,ss,stat=subject_weighted_annotation_effects(pd.DataFrame(phase),"strict")
    v=stat.loc[(stat.stratum=="all")&(stat.method=="M2")&(stat.metric=="RA")&
               (stat.effect=="marker_minus_pre")].iloc[0]
    assert v.n_subjects_valid==2 and abs(v.median_subject_effect)<1e-12
    print("PASS: v2 raw C label separation, strict/inclusive screen, subject weighting")


def main():
    OUTDIR.mkdir(exist_ok=True,parents=True)
    (OUTDIR/"analysis_parameters.json").write_text(
        json.dumps({**parameters(), "version":"v2 cohort annotated", "data_dir":str(DATA_DIR),
                    "strict_separation_sec":ANN_STRICT_SEPARATION_SEC,
                    "strict_artifact_sec":ANN_STRICT_ARTIFACT_SEC,
                    "possible_contraction_exclusion_sec":ANN_POSSIBLE_EXCLUSION_SEC},indent=2),
        encoding="utf-8")
    selftest();selftest_v2()
    if RUN_MODE=="selftest":return
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
    if RUN_MODE=="annotations_only":
        annotation_file_audit();return
    if RUN_MODE in ("representative","both"):
        generate_annotated_rep(REP_RECORD)
    if RUN_MODE in ("dataset","both"):
        run_dataset()
    if RUN_MODE not in ("selftest","annotations_only","representative","dataset","both"):
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE}")


if __name__=="__main__":
    main()
