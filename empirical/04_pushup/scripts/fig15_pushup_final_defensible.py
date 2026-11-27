#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 16 / Push-up NNDR analysis -- final defensible revision
==============================================================

Scientific purpose
------------------
This script is the clean post-review push-up analysis.

The analysis follows the logic requested by Didier:

1. PRIMARY analysis: use raw demeaned Apple-Watch user acceleration. Nothing
   is removed from the primary signal.
2. PREPROCESSING SENSITIVITY: if the gross voluntary push-up cycle needs to be
   separated from faster corrections, remove it with a time-varying adaptive
   phase model with K=3 harmonics, then analyze the residual in:
       (i) 1--15 Hz (main adaptive sensitivity);
       (ii) 1--24 Hz (wide-band sensitivity, safely below Nyquist for ~50 Hz).
3. The old fixed-f0, K=2, 1--8 Hz analysis is deliberately NOT included.
4. One common high/low activity definition is used for every signal variant.
   It is constructed only from the RAW demeaned acceleration magnitude using a
   causal trailing 0.5-s envelope. The upper 15% is "high activity".
5. One-step pair (x_t, x_{t+1}) is assigned by the activity class at t only.
   No future class label is used.
6. Fitted operators are NEVER rescaled to force stability. rho(F) is reported
   as estimated.
7. Physical acceleration coordinates are primary; no per-axis z-scoring is
   applied before the operator fit.
8. For every reduced 2x2 operator Gamma, report spectrum-independent
   diagnostics:
       rho(Gamma)
       sigma_max(Gamma)
       d_F(Gamma)  (Henrici departure)
       R1 = d_F / sqrt[(1-|lambda1|^2)(1-|lambda2|^2)]
   when rho(Gamma)<1. For a stable 2x2 map,
       R1 > 1  <=>  sigma_max(Gamma) > 1.
   R1 applies to real and complex eigenvalue pairs.
9. Uncertainty:
       moving-block bootstrap of the observed one-step pairs.
10. Null controls for the high-minus-low contrast:
       (a) stationary fitted VAR(1) surrogate with residual-vector resampling;
       (b) multivariate common-phase randomized surrogate preserving the
           full second-order cross-spectrum of the analyzed signal.
    These nulls keep the observed RAW causal activity labels fixed and ask
    whether a stationary/spectrum-preserving signal can generate an equally
    large operator contrast at those same times.
11. A circular-shift test of reaction-coordinate alignment is retained only as
    a secondary timing-specific control.

Input
-----
raw_samples_Pushup_duo_27Dec2025..csv
columns required:
    timestamp, accX, accY, accZ

Outputs
-------
pushup_final_defensible/
    pushup_final_observed.csv
    pushup_final_bootstrap_replicates.csv
    pushup_final_bootstrap_summary.csv
    pushup_final_null_replicates.csv
    pushup_final_null_summary.csv
    pushup_final_timing_null.csv
    pushup_final_report.txt
    pushup_final_reproducibility_snippet.txt
    fig16_pushup_final_summary.png
    fig16_pushup_final_summary.pdf

Notes
-----
* The adaptive phase fit is an OFFLINE sensitivity analysis. It uses
  zero-phase filtering and local weighted regression. The activity labels,
  however, are causal and are constructed from raw acceleration only.
* This script does not choose a favorable preprocessing after seeing the
  answer. It reports raw, adaptive 1--15 Hz, and adaptive 1--24 Hz together.
* No legacy K/Kc(Delta) value is used in the push-up conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import numpy.linalg as LA
import matplotlib.pyplot as plt

try:
    from scipy.signal import butter, sosfiltfilt, welch, hilbert
except Exception as exc:
    raise ImportError(
        "This script requires scipy.signal (butter, sosfiltfilt, welch, hilbert)."
    ) from exc


# USER SETTINGS
INPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "raw_samples_Pushup_duo_27Dec2025..csv"
OUTDIR = Path(__file__).resolve().parents[1] / "results" / "pushup_final_defensible"

# Raw activity class: common to ALL variants.
ENVELOPE_SEC = 0.50
HIGH_ACTIVITY_QUANTILE = 0.85  # upper 15%

# Adaptive phase-cycle removal requested by Didier.
LC_FREQ_BAND_HZ = (0.20, 1.20)
ADAPTIVE_N_HARMONICS = 3
ADAPTIVE_LOCAL_WINDOW_SEC = 6.0
ADAPTIVE_GAUSSIAN_SIGMA_FRAC = 0.35
PHASE_SOURCE = "pca"            # "pca" or "magnitude"
FILTER_ORDER = 4
ADAPTIVE_MAIN_BAND_HZ = (1.0, 15.0)
ADAPTIVE_WIDE_BAND_HZ = (1.0, 24.0)

# Operator estimator.
RIDGE_REL = 3e-3
SHRINK_ALPHA = 0.04
MIN_EFF_RANK = 2.20
COND_MAX = 2e4

# Plane extraction.
METHOD2_RESTARTS = 10
METHOD2_ITERS = 180
SEED = 0

# Uncertainty / null controls.
N_BLOCK_BOOT = 300
BLOCK_BOOT_SEC = 1.0
N_VAR_NULL = 300
N_PHASE_NULL = 300
N_SHIFT_NULL = 1000
MIN_SHIFT_SEC = 3.0

# Simulation.
VAR_BURNIN_SEC = 10.0

# Plotting.
DPI = 500
SAVE_PDF = True
SAVE_PNG = True

# Numerical tolerances.
EPS = 1e-12
STABLE_TOL = 1e-10


# DATA STRUCTURES
@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    mode: str
    band_hz: tuple[float, float] | None = None


VARIANTS = [
    Variant(
        name="raw_demeaned",
        label="Raw demeaned acceleration",
        mode="raw",
        band_hz=None,
    ),
    Variant(
        name="adaptive_phase_residual_1_15",
        label="Adaptive phase residual, 1–15 Hz",
        mode="adaptive",
        band_hz=ADAPTIVE_MAIN_BAND_HZ,
    ),
    Variant(
        name="adaptive_phase_residual_1_24",
        label="Adaptive phase residual, 1–24 Hz",
        mode="adaptive",
        band_hz=ADAPTIVE_WIDE_BAND_HZ,
    ),
]


# BASIC UTILITIES
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def robust_fs_from_time(t_abs: np.ndarray) -> tuple[float, float]:
    dt = np.diff(np.asarray(t_abs, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Could not infer a positive sampling interval.")
    dt_med = float(np.median(dt))
    return 1.0 / dt_med, dt_med


def causal_trailing_mean(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Trailing mean using samples [t-w+1, ..., t] only."""
    x = np.asarray(x, float).ravel()
    w = max(1, int(window_samples))
    cs = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(x.size)
    lo = np.maximum(0, i - w + 1)
    return (cs[i + 1] - cs[lo]) / (i - lo + 1)


def centered_mean_for_display(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Display only; never used to create classes."""
    x = np.asarray(x, float).ravel()
    w = max(1, int(window_samples))
    return np.convolve(x, np.ones(w) / float(w), mode="same")


def corr1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(LA.eigvals(A))))


def sigma_max(A: np.ndarray) -> float:
    return float(LA.svd(A, compute_uv=False)[0])


def effective_rank(X: np.ndarray, rel_tol: float = 1e-12) -> tuple[float, float]:
    X = np.asarray(X, float)
    s = LA.svd(X, compute_uv=False)
    if s.size == 0:
        return 0.0, np.inf
    cutoff = rel_tol * max(float(np.max(s)), EPS)
    s = s[np.isfinite(s) & (s > cutoff)]
    if s.size == 0:
        return 0.0, np.inf
    p = s / np.sum(s)
    er = float(np.exp(-np.sum(p * np.log(p + EPS))))
    cond = float(np.max(s) / max(np.min(s), EPS))
    return er, cond


def orthonormalize_cols(A: np.ndarray) -> np.ndarray:
    Q, _ = LA.qr(np.asarray(A, float))
    return Q[:, :A.shape[1]]


def percentile_ci(x: np.ndarray, q=(2.5, 97.5)) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    return tuple(float(v) for v in np.percentile(x, q))


# FILTERING AND ADAPTIVE PHASE-CYCLE REMOVAL
def butter_filter(
    X: np.ndarray,
    fs: float,
    btype: str,
    cutoff,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    X = np.asarray(X, float)
    nyq = 0.5 * float(fs)

    if btype == "bandpass":
        lo, hi = map(float, cutoff)
        lo = max(lo, 1e-6)
        # Keep a numerical margin from Nyquist without silently turning 24 Hz
        # into the old 1--8 Hz choice.
        hi = min(hi, 0.985 * nyq)
        if hi <= lo:
            raise ValueError(
                f"Invalid band {cutoff} for fs={fs:.6g} Hz (Nyquist={nyq:.6g} Hz)."
            )
        wn = [lo / nyq, hi / nyq]
    elif btype in ("lowpass", "highpass"):
        c = min(float(cutoff), 0.985 * nyq)
        wn = c / nyq
    else:
        raise ValueError(f"Unknown filter type {btype!r}")

    sos = butter(order, wn, btype=btype, output="sos")
    return sosfiltfilt(sos, X, axis=0)


def estimate_pushup_fundamental(
    X: np.ndarray,
    fs: float,
    band: tuple[float, float] = LC_FREQ_BAND_HZ,
) -> float:
    mag = LA.norm(np.asarray(X, float), axis=1)
    mag = mag - np.mean(mag)
    nperseg = min(len(mag), max(64, int(round(8.0 * fs))))
    f, pxx = welch(mag, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return np.nan
    j = int(np.argmax(pxx[m]))
    return float(f[m][j])


def principal_component_1(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    _, _, Vt = LA.svd(Xc, full_matrices=False)
    return np.asarray(Xc @ Vt[0], float)


def estimate_instantaneous_phase(
    t: np.ndarray,
    X: np.ndarray,
    fs: float,
    source: str = PHASE_SOURCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline phase estimate for the adaptive-cycle sensitivity analysis."""
    X = np.asarray(X, float)

    if source == "magnitude":
        ref = LA.norm(X, axis=1)
        ref = ref - np.mean(ref)
    elif source == "pca":
        ref = principal_component_1(X)
    else:
        raise ValueError("PHASE_SOURCE must be 'pca' or 'magnitude'.")

    ref_band = butter_filter(
        ref[:, None], fs, "bandpass", LC_FREQ_BAND_HZ, order=FILTER_ORDER
    ).ravel()

    analytic = hilbert(ref_band)
    phi = np.unwrap(np.angle(analytic))
    if np.nanmedian(np.diff(phi)) < 0:
        phi = -phi

    dt = float(np.median(np.diff(t)))
    f_inst = np.gradient(phi, dt) / (2.0 * np.pi)
    f_inst = causal_trailing_mean(f_inst, max(3, int(round(0.75 * fs))))
    return phi, f_inst, ref_band


def harmonic_design(phi: np.ndarray, n_harmonics: int) -> np.ndarray:
    cols = [np.ones_like(phi)]
    for k in range(1, int(n_harmonics) + 1):
        cols.extend([np.cos(k * phi), np.sin(k * phi)])
    return np.column_stack(cols)


def adaptive_phase_cycle_fit(
    t: np.ndarray,
    X: np.ndarray,
    phi: np.ndarray,
    n_harmonics: int = ADAPTIVE_N_HARMONICS,
    local_window_sec: float = ADAPTIVE_LOCAL_WINDOW_SEC,
    sigma_frac: float = ADAPTIVE_GAUSSIAN_SIGMA_FRAC,
) -> np.ndarray:
    """
    Offline local weighted harmonic regression:

      x_i(t) = sum_{k=1}^K [
                  a_ik(t) cos(k phi(t)) + b_ik(t) sin(k phi(t))
               ] + epsilon_i(t).

    The adaptive residual is a sensitivity analysis, not the causal event
    definition. The event labels are always taken from raw acceleration.
    """
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    Phi_all = harmonic_design(phi, n_harmonics)

    n, p = Phi_all.shape
    half = 0.5 * float(local_window_sec)
    sigma = max(1e-6, sigma_frac * float(local_window_sec))
    X_cycle = np.full_like(X, np.nan, dtype=float)
    ridge = 1e-8

    for i in range(n):
        mask = np.abs(t - t[i]) <= half
        if int(np.sum(mask)) < p + 2:
            nearest = np.argsort(np.abs(t - t[i]))[: p + 2]
            mask = np.zeros(n, dtype=bool)
            mask[nearest] = True

        Phi = Phi_all[mask]
        Y = X[mask]
        w = np.exp(-0.5 * ((t[mask] - t[i]) / sigma) ** 2)
        sw = np.sqrt(w)[:, None]
        A = Phi * sw
        B = Y * sw

        lhs = A.T @ A + ridge * np.eye(p)
        rhs = A.T @ B
        try:
            coef = LA.solve(lhs, rhs)
        except LA.LinAlgError:
            coef, *_ = LA.lstsq(A, B, rcond=None)

        X_cycle[i] = Phi_all[i] @ coef

    return X_cycle


# SIGNAL VARIANTS AND COMMON CAUSAL ACTIVITY LABELS
def build_common_activity_labels(
    X_raw_demeaned: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Define one common causal activity label from raw acceleration only.

    high[t] is determined using samples no later than t.
    Pair (x_t, x_{t+1}) uses high[t].
    """
    raw_mag = LA.norm(np.asarray(X_raw_demeaned, float), axis=1)
    w = max(1, int(round(ENVELOPE_SEC * fs)))
    env = causal_trailing_mean(raw_mag, w)
    thr = float(np.quantile(env, HIGH_ACTIVITY_QUANTILE))
    high = env >= thr
    return env, high, thr


def build_variants(
    t: np.ndarray,
    X_raw: np.ndarray,
    fs: float,
) -> tuple[dict[str, dict], dict]:
    X = np.asarray(X_raw, float)
    X = X - np.mean(X, axis=0, keepdims=True)

    phi, f_inst, phase_ref = estimate_instantaneous_phase(t, X, fs)
    X_cycle = adaptive_phase_cycle_fit(
        t=t,
        X=X,
        phi=phi,
        n_harmonics=ADAPTIVE_N_HARMONICS,
        local_window_sec=ADAPTIVE_LOCAL_WINDOW_SEC,
        sigma_frac=ADAPTIVE_GAUSSIAN_SIGMA_FRAC,
    )
    X_residual = X - X_cycle

    result: dict[str, dict] = {}

    for v in VARIANTS:
        if v.mode == "raw":
            X_fit = X.copy()
        elif v.mode == "adaptive":
            X_fit = butter_filter(
                X_residual, fs, "bandpass", v.band_hz, order=FILTER_ORDER
            )
        else:
            raise ValueError(v.mode)

        result[v.name] = {
            "name": v.name,
            "label": v.label,
            "mode": v.mode,
            "band_hz": v.band_hz,
            # Physical coordinates: no per-axis standardization.
            "X": np.asarray(X_fit, float),
        }

    aux = {
        "X_raw_demeaned": X,
        "X_cycle": X_cycle,
        "X_residual": X_residual,
        "phi": phi,
        "f_inst_hz": f_inst,
        "phase_ref": phase_ref,
        "f0_global_hz": estimate_pushup_fundamental(X, fs),
    }
    return result, aux


# LOCAL OPERATOR ESTIMATION
def ridge_fit_pairs(
    X0: np.ndarray,
    X1: np.ndarray,
    ridge_rel: float = RIDGE_REL,
) -> tuple[np.ndarray, float]:
    X0 = np.asarray(X0, float)
    X1 = np.asarray(X1, float)

    d = X0.shape[1]
    scale = float(np.trace(X0.T @ X0) / max(d, 1))
    lam = ridge_rel * max(scale, EPS)

    lhs = X0.T @ X0 + lam * np.eye(d)
    rhs = X0.T @ X1
    Fhat = LA.solve(lhs, rhs).T
    return Fhat, lam


def isotropic_shrinkage(
    Fhat: np.ndarray,
    alpha: float = SHRINK_ALPHA,
) -> tuple[np.ndarray, float]:
    d = Fhat.shape[0]
    mu = float(np.trace(Fhat) / d)
    F = (1.0 - alpha) * Fhat + alpha * mu * np.eye(d)
    return F, mu


def fit_operator_from_pairs(
    X0: np.ndarray,
    X1: np.ndarray,
) -> dict | None:
    X0 = np.asarray(X0, float)
    X1 = np.asarray(X1, float)

    if X0.shape != X1.shape or X0.ndim != 2 or X0.shape[0] < 20:
        return None

    er, cond = effective_rank(X0)
    if er < MIN_EFF_RANK or cond > COND_MAX:
        return None

    Fhat, lam = ridge_fit_pairs(X0, X1)
    F, mu = isotropic_shrinkage(Fhat, SHRINK_ALPHA)

    # IMPORTANT: no forced stability rescaling.
    return {
        "F": F,
        "rhoF": spectral_radius(F),
        "sigmaF": sigma_max(F),
        "ridge_lambda": float(lam),
        "shrink_mu": float(mu),
        "eff_rank": float(er),
        "cond": float(cond),
        "n_pairs": int(X0.shape[0]),
    }


# M2 / M3 RESPONSE-PLANE EXTRACTION
def method2_optimization(
    F: np.ndarray,
    n_restarts: int = METHOD2_RESTARTS,
    max_iter: int = METHOD2_ITERS,
    tol: float = 1e-10,
    seed: int = SEED,
) -> dict | None:
    rng = np.random.default_rng(seed)
    d = F.shape[0]

    best_uv = None
    best_val = -np.inf

    for _ in range(n_restarts):
        v = rng.standard_normal(d)
        v /= LA.norm(v) + EPS

        u = rng.standard_normal(d)
        u -= (v @ u) * v
        u /= LA.norm(u) + EPS

        prev = 0.0
        for _it in range(max_iter):
            u = F @ v
            u -= (v @ u) * v
            nu = LA.norm(u)
            if nu < EPS:
                break
            u /= nu

            v = F.T @ u
            v -= (u @ v) * u
            nv = LA.norm(v)
            if nv < EPS:
                break
            v /= nv

            val = float(u.T @ (F @ v))
            if abs(val - prev) < tol:
                prev = val
                break
            prev = val

        if prev > best_val:
            best_val = prev
            best_uv = (u.copy(), v.copy())

    if best_uv is None:
        return None

    u, v = best_uv
    Q = orthonormalize_cols(np.column_stack([u, v]))
    return {
        "method": "M2",
        "Q": Q,
        "r": Q[:, 0],
        "n": Q[:, 1],
        "objective": float(best_val),
    }


def method3_commutator(F: np.ndarray) -> dict | None:
    B = F @ F.T - F.T @ F
    evals, E = LA.eigh(B)

    i_min = int(np.argmin(evals))
    i_max = int(np.argmax(evals))

    if max(abs(float(evals[i_min])), abs(float(evals[i_max]))) < EPS:
        return None

    w_plus = E[:, i_max] / (LA.norm(E[:, i_max]) + EPS)
    w_minus = E[:, i_min] / (LA.norm(E[:, i_min]) + EPS)
    Qplane = orthonormalize_cols(np.column_stack([w_plus, w_minus]))

    # Reorientation inside the same plane changes neither eigenvalues nor
    # singular values of the reduced operator; it only makes r/n comparable.
    G0 = Qplane.T @ F @ Qplane
    out2 = method2_optimization(G0, n_restarts=6, max_iter=120, seed=SEED)

    if out2 is None:
        Q = Qplane
    else:
        r = Qplane @ out2["r"]
        n = Qplane @ out2["n"]
        Q = orthonormalize_cols(np.column_stack([r, n]))

    return {
        "method": "M3",
        "Q": Q,
        "r": Q[:, 0],
        "n": Q[:, 1],
        "lamB_min": float(evals[i_min]),
        "lamB_max": float(evals[i_max]),
    }


def extract_method(F: np.ndarray, method: str) -> dict | None:
    if method == "M2":
        return method2_optimization(F)
    if method == "M3":
        return method3_commutator(F)
    raise ValueError(method)


# EXACT 2D REDUCED DIAGNOSTICS
def reduced_diagnostics(F: np.ndarray, Q: np.ndarray) -> dict:
    G = np.asarray(Q, float).T @ np.asarray(F, float) @ np.asarray(Q, float)
    eig, V = LA.eig(G)
    sv = LA.svd(G, compute_uv=False)

    rhoG = float(np.max(np.abs(eig)))
    sigmaG = float(sv[0])

    # Spectrum type.
    scale = max(1.0, float(np.max(np.abs(eig))))
    complex_pair = bool(np.max(np.abs(np.imag(eig))) > 1e-9 * scale)
    if complex_pair:
        spectrum_type = "complex_pair"
    elif abs(eig[0] - eig[1]) <= 1e-9 * scale:
        spectrum_type = "repeated_real"
    else:
        spectrum_type = "real_pair"

    # Eigenvector-overlap K, retained for connection to the manuscript.
    p1 = V[:, 0] / (LA.norm(V[:, 0]) + EPS)
    p2 = V[:, 1] / (LA.norm(V[:, 1]) + EPS)
    c = float(np.clip(abs(np.vdot(p1, p2)), 0.0, 1.0 - 1e-12))
    kappa2D = float(np.sqrt((1.0 + c) / (1.0 - c)))
    K = float(0.5 * (kappa2D - 1.0 / kappa2D))

    # Henrici departure. For a real 2x2 matrix this equals
    # |lambda1-lambda2| K (away from the defective limit), but this expression
    # is numerically safer and does not need eigenvectors.
    fro2 = float(np.sum(np.abs(G) ** 2))
    eig2 = float(np.sum(np.abs(eig) ** 2))
    dF = float(np.sqrt(max(fro2 - eig2, 0.0)))

    stable_reduced = bool(rhoG < 1.0 - STABLE_TOL)
    denom = float(np.prod(1.0 - np.abs(eig) ** 2))

    if stable_reduced and denom > EPS:
        R1 = float(dF / np.sqrt(denom))
    else:
        R1 = np.nan

    amplifying_stable = bool(stable_reduced and sigmaG > 1.0 + STABLE_TOL)

    criterion_agreement = True
    if np.isfinite(R1):
        criterion_agreement = bool((R1 > 1.0) == (sigmaG > 1.0))

    return {
        "Gamma": G,
        "eig1_real": float(np.real(eig[0])),
        "eig1_imag": float(np.imag(eig[0])),
        "eig2_real": float(np.real(eig[1])),
        "eig2_imag": float(np.imag(eig[1])),
        "spectrum_type": spectrum_type,
        "rhoGamma": rhoG,
        "sigmaGamma": sigmaG,
        "sigmaGamma_min": float(sv[-1]),
        "K": K,
        "kappa2D": kappa2D,
        "dF": dF,
        "R1": R1,
        "stable_reduced": int(stable_reduced),
        "stable_amplifying": int(amplifying_stable),
        "criterion_agreement": int(criterion_agreement),
    }


def analyze_fit(fit: dict | None, method: str) -> dict | None:
    if fit is None:
        return None

    plane = extract_method(fit["F"], method)
    if plane is None:
        return None

    red = reduced_diagnostics(fit["F"], plane["Q"])
    return {**fit, **plane, **red}


# OBSERVED REGIME ANALYSIS
def pair_labels_from_high(high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Causal rule:
        pair (x_t, x_{t+1}) is HIGH iff high[t] is True.
    """
    high_t = np.asarray(high[:-1], bool)
    return ~high_t, high_t  # low, high


def analyze_variant_observed(
    variant: dict,
    high: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    X = np.asarray(variant["X"], float)
    low_pair, high_pair = pair_labels_from_high(high)

    X0_all = X[:-1]
    X1_all = X[1:]

    pair_sets = {
        "low": (X0_all[low_pair], X1_all[low_pair]),
        "high": (X0_all[high_pair], X1_all[high_pair]),
    }

    rows = []
    fits: dict[tuple[str, str], dict] = {}

    for regime, (X0, X1) in pair_sets.items():
        fit = fit_operator_from_pairs(X0, X1)

        for method in ("M2", "M3"):
            res = analyze_fit(fit, method)
            if res is None:
                continue

            fits[(regime, method)] = res
            rows.append({
                "variant": variant["name"],
                "variant_label": variant["label"],
                "regime": regime,
                "method": method,
                "n_pairs": res["n_pairs"],
                "rhoF": res["rhoF"],
                "sigmaF": res["sigmaF"],
                "rhoGamma": res["rhoGamma"],
                "sigmaGamma": res["sigmaGamma"],
                "K": res["K"],
                "dF": res["dF"],
                "R1": res["R1"],
                "spectrum_type": res["spectrum_type"],
                "stable_reduced": res["stable_reduced"],
                "stable_amplifying": res["stable_amplifying"],
                "eig1_real": res["eig1_real"],
                "eig1_imag": res["eig1_imag"],
                "eig2_real": res["eig2_real"],
                "eig2_imag": res["eig2_imag"],
                "eff_rank": res["eff_rank"],
                "cond": res["cond"],
                "ridge_lambda": res["ridge_lambda"],
            })

    return pd.DataFrame(rows), fits


# MOVING-BLOCK BOOTSTRAP
def circular_block_indices(
    n: int,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block bootstrap indices of length n."""
    block_len = max(1, min(int(block_len), n))
    out = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        block = (start + np.arange(block_len)) % n
        out.extend(block.tolist())
    return np.asarray(out[:n], dtype=int)


def block_bootstrap_variant(
    variant: dict,
    high: np.ndarray,
    fs: float,
    n_boot: int = N_BLOCK_BOOT,
) -> pd.DataFrame:
    X = np.asarray(variant["X"], float)
    X0_all = X[:-1]
    X1_all = X[1:]
    high_t = np.asarray(high[:-1], bool)

    n = len(high_t)
    block_len = max(2, int(round(BLOCK_BOOT_SEC * fs)))
    rng = np.random.default_rng(SEED + 5001)

    rows = []

    for b in range(n_boot):
        idx = circular_block_indices(n, block_len, rng)
        labels = high_t[idx]

        for regime, mask in (("low", ~labels), ("high", labels)):
            if int(np.sum(mask)) < 20:
                continue

            fit = fit_operator_from_pairs(X0_all[idx][mask], X1_all[idx][mask])

            for method in ("M2", "M3"):
                res = analyze_fit(fit, method)
                if res is None:
                    continue

                rows.append({
                    "variant": variant["name"],
                    "boot": b,
                    "regime": regime,
                    "method": method,
                    "rhoF": res["rhoF"],
                    "rhoGamma": res["rhoGamma"],
                    "sigmaGamma": res["sigmaGamma"],
                    "R1": res["R1"],
                    "stable_amplifying": res["stable_amplifying"],
                })

    return pd.DataFrame(rows)


def summarize_bootstrap(
    boot_df: pd.DataFrame,
) -> pd.DataFrame:
    if boot_df.empty:
        return pd.DataFrame()

    rows = []

    for (variant, method), g in boot_df.groupby(["variant", "method"]):
        low = g[g["regime"] == "low"].set_index("boot")
        high = g[g["regime"] == "high"].set_index("boot")
        common = low.index.intersection(high.index)

        for regime, h in (("low", low), ("high", high)):
            for metric in ("rhoF", "rhoGamma", "sigmaGamma", "R1"):
                vals = h[metric].to_numpy(float)
                lo, hi = percentile_ci(vals)
                rows.append({
                    "variant": variant,
                    "method": method,
                    "summary_type": "regime",
                    "regime": regime,
                    "metric": metric,
                    "median": float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })

        if common.size:
            for metric in ("sigmaGamma", "R1"):
                delta = (
                    high.loc[common, metric].to_numpy(float)
                    - low.loc[common, metric].to_numpy(float)
                )
                lo, hi = percentile_ci(delta)
                rows.append({
                    "variant": variant,
                    "method": method,
                    "summary_type": "contrast_high_minus_low",
                    "regime": "",
                    "metric": metric,
                    "median": float(np.nanmedian(delta)) if np.any(np.isfinite(delta)) else np.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })

    return pd.DataFrame(rows)


# NULL 1: STATIONARY FITTED VAR(1)
def fit_global_var(X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    X = np.asarray(X, float)
    fit = fit_operator_from_pairs(X[:-1], X[1:])
    if fit is None:
        return None

    F = fit["F"]
    if spectral_radius(F) >= 1.0:
        warnings.warn(
            "Global fitted VAR is not stable; stationary VAR null is skipped "
            f"(rho={spectral_radius(F):.4f})."
        )
        return None

    residuals = X[1:] - X[:-1] @ F.T
    residuals = residuals - np.mean(residuals, axis=0, keepdims=True)
    return F, residuals


def simulate_var_residual_bootstrap(
    F: np.ndarray,
    residuals: np.ndarray,
    n: int,
    fs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    d = F.shape[0]
    burn = max(100, int(round(VAR_BURNIN_SEC * fs)))
    total = n + burn

    X = np.zeros((total, d), dtype=float)
    X[0] = 0.0

    for t in range(1, total):
        eps = residuals[int(rng.integers(0, len(residuals)))]
        X[t] = F @ X[t - 1] + eps

    return X[burn:]


# NULL 2: MULTIVARIATE COMMON-PHASE RANDOMIZATION
def common_phase_randomized_surrogate(
    X: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the same random Fourier phase at each frequency to all channels.

    This preserves each channel spectrum and the complete cross-spectrum
    between channels, hence the multivariate second-order covariance structure.
    """
    X = np.asarray(X, float)
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu

    n = Xc.shape[0]
    Z = np.fft.rfft(Xc, axis=0)
    nfreq = Z.shape[0]

    factors = np.ones(nfreq, dtype=complex)

    if n % 2 == 0:
        random_idx = np.arange(1, nfreq - 1)  # leave DC and Nyquist real
    else:
        random_idx = np.arange(1, nfreq)      # leave DC only

    theta = rng.uniform(0.0, 2.0 * np.pi, size=random_idx.size)
    factors[random_idx] = np.exp(1j * theta)

    Zs = Z * factors[:, None]
    Xs = np.fft.irfft(Zs, n=n, axis=0)
    return np.asarray(Xs + mu, float)


# NULL CONTRAST CALCULATION
def contrast_from_signal(
    X: np.ndarray,
    high: np.ndarray,
    method: str,
) -> tuple[float, float]:
    """
    Return:
        delta_sigma = sigmaGamma_high - sigmaGamma_low
        delta_R1    = R1_high - R1_low
    """
    low_pair, high_pair = pair_labels_from_high(high)
    X0 = X[:-1]
    X1 = X[1:]

    fit_low = fit_operator_from_pairs(X0[low_pair], X1[low_pair])
    fit_high = fit_operator_from_pairs(X0[high_pair], X1[high_pair])

    low = analyze_fit(fit_low, method)
    high_res = analyze_fit(fit_high, method)

    if low is None or high_res is None:
        return np.nan, np.nan

    d_sigma = float(high_res["sigmaGamma"] - low["sigmaGamma"])
    if np.isfinite(low["R1"]) and np.isfinite(high_res["R1"]):
        d_R1 = float(high_res["R1"] - low["R1"])
    else:
        d_R1 = np.nan
    return d_sigma, d_R1


def null_controls_variant(
    variant: dict,
    high: np.ndarray,
    fs: float,
    observed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = np.asarray(variant["X"], float)
    rng_var = np.random.default_rng(SEED + 7001)
    rng_phase = np.random.default_rng(SEED + 8001)

    replicate_rows = []
    summary_rows = []

    var_fit = fit_global_var(X)

    for method in ("M2", "M3"):
        o_low = observed_df[
            (observed_df["method"] == method) & (observed_df["regime"] == "low")
        ]
        o_high = observed_df[
            (observed_df["method"] == method) & (observed_df["regime"] == "high")
        ]

        if o_low.empty or o_high.empty:
            continue

        obs_d_sigma = float(o_high["sigmaGamma"].iloc[0] - o_low["sigmaGamma"].iloc[0])
        low_R1 = float(o_low["R1"].iloc[0])
        high_R1 = float(o_high["R1"].iloc[0])
        obs_d_R1 = high_R1 - low_R1 if np.isfinite(low_R1) and np.isfinite(high_R1) else np.nan

        # VAR null.
        var_sigmas = []
        var_r1 = []
        if var_fit is not None:
            Fnull, residuals = var_fit
            for b in range(N_VAR_NULL):
                Xs = simulate_var_residual_bootstrap(
                    Fnull, residuals, len(X), fs, rng_var
                )
                ds, dr = contrast_from_signal(Xs, high, method)
                var_sigmas.append(ds)
                var_r1.append(dr)
                replicate_rows.append({
                    "variant": variant["name"],
                    "method": method,
                    "null_type": "stationary_VAR1",
                    "replicate": b,
                    "delta_sigmaGamma": ds,
                    "delta_R1": dr,
                })

        # Cross-spectrum-preserving phase null.
        phase_sigmas = []
        phase_r1 = []
        for b in range(N_PHASE_NULL):
            Xs = common_phase_randomized_surrogate(X, rng_phase)
            ds, dr = contrast_from_signal(Xs, high, method)
            phase_sigmas.append(ds)
            phase_r1.append(dr)
            replicate_rows.append({
                "variant": variant["name"],
                "method": method,
                "null_type": "common_phase_randomized",
                "replicate": b,
                "delta_sigmaGamma": ds,
                "delta_R1": dr,
            })

        for null_name, vals_sigma, vals_r1 in (
            ("stationary_VAR1", var_sigmas, var_r1),
            ("common_phase_randomized", phase_sigmas, phase_r1),
        ):
            vs = np.asarray(vals_sigma, float)
            vs = vs[np.isfinite(vs)]

            if vs.size:
                p_sigma = float(
                    (1 + np.sum(vs >= obs_d_sigma)) / (1 + vs.size)
                )
                qlo, qhi = percentile_ci(vs)
            else:
                p_sigma = np.nan
                qlo, qhi = np.nan, np.nan

            vr = np.asarray(vals_r1, float)
            vr = vr[np.isfinite(vr)]
            if np.isfinite(obs_d_R1) and vr.size:
                p_r1 = float(
                    (1 + np.sum(vr >= obs_d_R1)) / (1 + vr.size)
                )
            else:
                p_r1 = np.nan

            summary_rows.append({
                "variant": variant["name"],
                "method": method,
                "null_type": null_name,
                "observed_delta_sigmaGamma": obs_d_sigma,
                "null_delta_sigma_median": float(np.median(vs)) if vs.size else np.nan,
                "null_delta_sigma_ci95_low": qlo,
                "null_delta_sigma_ci95_high": qhi,
                "p_delta_sigma_upper": p_sigma,
                "observed_delta_R1": obs_d_R1,
                "p_delta_R1_upper": p_r1,
                "n_valid_sigma": int(vs.size),
                "n_valid_R1": int(vr.size),
            })

    return pd.DataFrame(replicate_rows), pd.DataFrame(summary_rows)


# SECONDARY TIMING-SPECIFIC CIRCULAR-SHIFT TEST
def infer_global_plane(X: np.ndarray, method: str) -> dict | None:
    fit = fit_operator_from_pairs(X[:-1], X[1:])
    return analyze_fit(fit, method)


def circular_shift_timing_test(
    signal: np.ndarray,
    envelope: np.ndarray,
    fs: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    signal = np.asarray(signal, float).ravel()
    envelope = np.asarray(envelope, float).ravel()

    n = min(len(signal), len(envelope))
    signal = signal[:n]
    envelope = envelope[:n]

    obs = corr1d(envelope, signal)

    min_shift = max(1, int(round(MIN_SHIFT_SEC * fs)))
    allowed = np.arange(min_shift, n - min_shift + 1, dtype=int)

    if allowed.size == 0:
        return obs, np.nan, np.nan, 0

    vals = []
    for _ in range(N_SHIFT_NULL):
        sh = int(rng.choice(allowed))
        vals.append(corr1d(envelope, np.roll(signal, sh)))

    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 2:
        return obs, np.nan, np.nan, int(vals.size)

    z = float((obs - np.mean(vals)) / (np.std(vals, ddof=1) + EPS))
    p = float((1 + np.sum(vals >= obs)) / (1 + vals.size))
    return obs, z, p, int(vals.size)


def timing_nulls_all(
    variants: dict[str, dict],
    raw_activity_envelope: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + 9001)

    for v in VARIANTS:
        X = variants[v.name]["X"]

        for method in ("M2", "M3"):
            res = infer_global_plane(X, method)
            if res is None:
                continue

            reaction = np.abs(X @ res["r"])
            obs, z, p, nvalid = circular_shift_timing_test(
                reaction, raw_activity_envelope, fs, rng
            )

            rows.append({
                "variant": v.name,
                "method": method,
                "observed_corr_raw_activity_vs_reaction": obs,
                "z_shift": z,
                "p_shift_upper": p,
                "n_valid_shifts": nvalid,
            })

    return pd.DataFrame(rows)


# FINAL SUMMARY TABLE
def attach_intervals_and_nulls(
    observed: pd.DataFrame,
    boot_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = observed.copy()

    # Regime-specific bootstrap CIs.
    for metric in ("rhoF", "rhoGamma", "sigmaGamma", "R1"):
        sub = boot_summary[
            (boot_summary["summary_type"] == "regime")
            & (boot_summary["metric"] == metric)
        ][["variant", "method", "regime", "ci95_low", "ci95_high"]].copy()

        sub = sub.rename(columns={
            "ci95_low": f"{metric}_ci95_low",
            "ci95_high": f"{metric}_ci95_high",
        })
        out = out.merge(sub, on=["variant", "method", "regime"], how="left")

    # Contrast CIs and null p-values belong to each method/variant pair.
    contrast = boot_summary[
        (boot_summary["summary_type"] == "contrast_high_minus_low")
        & (boot_summary["metric"] == "sigmaGamma")
    ][["variant", "method", "median", "ci95_low", "ci95_high"]].rename(columns={
        "median": "bootstrap_delta_sigmaGamma_median",
        "ci95_low": "bootstrap_delta_sigmaGamma_ci95_low",
        "ci95_high": "bootstrap_delta_sigmaGamma_ci95_high",
    })

    out = out.merge(contrast, on=["variant", "method"], how="left")

    # Put both null p-values into columns.
    if not null_summary.empty:
        ptab = null_summary.pivot_table(
            index=["variant", "method"],
            columns="null_type",
            values="p_delta_sigma_upper",
            aggfunc="first",
        ).reset_index()

        ptab.columns = [
            "variant" if c == "variant" else
            "method" if c == "method" else
            f"p_delta_sigma_{c}"
            for c in ptab.columns
        ]
        out = out.merge(ptab, on=["variant", "method"], how="left")

    return out


# FIGURE
def make_summary_figure(
    t: np.ndarray,
    raw_env: np.ndarray,
    high: np.ndarray,
    threshold: float,
    final_table: pd.DataFrame,
    null_summary: pd.DataFrame,
    outbase: Path,
) -> None:
    """
    Compact four-panel diagnostic figure. The purpose is transparency rather
    than choosing a favorable variant.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (a) causal raw envelope and high-activity class.
    ax1.plot(t, raw_env, lw=1.2)
    ax1.axhline(threshold, ls="--", lw=1.0)
    ax1.fill_between(t, 0.0, raw_env, where=high, alpha=0.18)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Raw causal envelope")
    ax1.set_title("(a) Common causal activity definition")

    # Helper x positions.
    groups = []
    for v in VARIANTS:
        for m in ("M2", "M3"):
            groups.append((v.name, m, f"{v.label}\n{m}"))
    x = np.arange(len(groups), dtype=float)

    # (b) sigma_max(Gamma), low/high with CIs.
    for offset, regime, marker in [(-0.12, "low", "o"), (0.12, "high", "s")]:
        vals, elo, ehi = [], [], []
        for vname, method, _ in groups:
            r = final_table[
                (final_table["variant"] == vname)
                & (final_table["method"] == method)
                & (final_table["regime"] == regime)
            ]
            if r.empty:
                vals.append(np.nan); elo.append(np.nan); ehi.append(np.nan)
                continue
            y = float(r["sigmaGamma"].iloc[0])
            lo = float(r["sigmaGamma_ci95_low"].iloc[0])
            hi = float(r["sigmaGamma_ci95_high"].iloc[0])
            vals.append(y)
            elo.append(y - lo if np.isfinite(lo) else np.nan)
            ehi.append(hi - y if np.isfinite(hi) else np.nan)
        ax2.errorbar(
            x + offset, vals, yerr=[elo, ehi],
            fmt=marker, capsize=3, label=regime
        )
    ax2.axhline(1.0, ls="--", lw=1.0)
    ax2.set_ylabel(r"$\sigma_{\max}(\Gamma)$")
    ax2.set_title("(b) Reduced one-step gain")
    ax2.set_xticks(x)
    ax2.set_xticklabels([g[2] for g in groups], rotation=30, ha="right", fontsize=7)
    ax2.legend(frameon=False)

    # (c) R1, low/high with CIs.
    for offset, regime, marker in [(-0.12, "low", "o"), (0.12, "high", "s")]:
        vals, elo, ehi = [], [], []
        for vname, method, _ in groups:
            r = final_table[
                (final_table["variant"] == vname)
                & (final_table["method"] == method)
                & (final_table["regime"] == regime)
            ]
            if r.empty:
                vals.append(np.nan); elo.append(np.nan); ehi.append(np.nan)
                continue
            y = float(r["R1"].iloc[0])
            lo = float(r["R1_ci95_low"].iloc[0])
            hi = float(r["R1_ci95_high"].iloc[0])
            vals.append(y)
            elo.append(y - lo if np.isfinite(y) and np.isfinite(lo) else np.nan)
            ehi.append(hi - y if np.isfinite(y) and np.isfinite(hi) else np.nan)
        ax3.errorbar(
            x + offset, vals, yerr=[elo, ehi],
            fmt=marker, capsize=3, label=regime
        )
    ax3.axhline(1.0, ls="--", lw=1.0)
    ax3.set_ylabel(r"$R_1$")
    ax3.set_title("(c) Exact stable-map criterion")
    ax3.set_xticks(x)
    ax3.set_xticklabels([g[2] for g in groups], rotation=30, ha="right", fontsize=7)

    # (d) observed high-low sigma contrast against null 95% ranges.
    ypos = []
    labels = []
    for i, (vname, method, label) in enumerate(groups):
        obs_row = null_summary[
            (null_summary["variant"] == vname)
            & (null_summary["method"] == method)
        ]
        if obs_row.empty:
            continue

        obs = float(obs_row["observed_delta_sigmaGamma"].iloc[0])
        y0 = float(len(ypos))
        ypos.append(y0)
        labels.append(label)

        # Two null intervals.
        for j, null_name in enumerate(("stationary_VAR1", "common_phase_randomized")):
            r = obs_row[obs_row["null_type"] == null_name]
            if r.empty:
                continue
            lo = float(r["null_delta_sigma_ci95_low"].iloc[0])
            hi = float(r["null_delta_sigma_ci95_high"].iloc[0])
            yy = y0 + (-0.12 if j == 0 else 0.12)
            ax4.plot([lo, hi], [yy, yy], lw=3, alpha=0.55)

        ax4.plot(obs, y0, "ko", ms=4)

    ax4.axvline(0.0, ls="--", lw=1.0)
    ax4.set_yticks(ypos)
    ax4.set_yticklabels(labels, fontsize=7)
    ax4.set_xlabel(r"$\Delta\sigma_{\max}=\sigma_{\rm high}-\sigma_{\rm low}$")
    ax4.set_title("(d) Observed contrast vs null 95% ranges")

    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(outbase.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# REPORT / REPRODUCIBILITY TEXT
def write_report(
    path: Path,
    fs: float,
    dt: float,
    activity_threshold: float,
    final_table: pd.DataFrame,
    null_summary: pd.DataFrame,
    timing_df: pd.DataFrame,
    aux: dict,
) -> None:
    lines = []
    lines.append("PUSH-UP FINAL DEFENSIBLE ANALYSIS")
    lines.append("=" * 72)
    lines.append(f"n/a sampling: fs={fs:.6f} Hz, dt={dt:.8f} s")
    lines.append(f"Raw causal envelope: {ENVELOPE_SEC:.3f} s")
    lines.append(f"High-activity threshold: Q_{HIGH_ACTIVITY_QUANTILE:.2f} = {activity_threshold:.8g}")
    lines.append(f"Global push-up spectral peak: {aux['f0_global_hz']:.6f} Hz")
    lines.append(f"Adaptive K={ADAPTIVE_N_HARMONICS}; main band={ADAPTIVE_MAIN_BAND_HZ}; wide band={ADAPTIVE_WIDE_BAND_HZ}")
    lines.append("No per-axis standardization. No forced stability rescaling.")
    lines.append("Pairs are classified from raw causal activity at t only.")
    lines.append("")

    cols = [
        "variant", "method", "regime", "n_pairs",
        "rhoF", "rhoGamma", "sigmaGamma", "R1",
        "spectrum_type", "stable_amplifying",
        "sigmaGamma_ci95_low", "sigmaGamma_ci95_high",
        "R1_ci95_low", "R1_ci95_high",
    ]
    lines.append("OBSERVED RESULTS")
    lines.append("-" * 72)
    lines.append(final_table[cols].to_string(index=False))
    lines.append("")

    lines.append("NULL TESTS FOR HIGH-MINUS-LOW sigma_max(Gamma)")
    lines.append("-" * 72)
    if null_summary.empty:
        lines.append("No valid null results.")
    else:
        lines.append(null_summary.to_string(index=False))
    lines.append("")

    lines.append("SECONDARY CIRCULAR-SHIFT TIMING TEST")
    lines.append("-" * 72)
    lines.append(timing_df.to_string(index=False))
    lines.append("")

    lines.append("INTERPRETATION RULE")
    lines.append("-" * 72)
    lines.append(
        "For a reduced map with rho(Gamma)<1, stable one-step amplification is "
        "present iff sigma_max(Gamma)>1, equivalently R1>1. "
        "A manuscript claim should be based on agreement of M2 and M3, "
        "bootstrap uncertainty, and null controls, not on a single point estimate."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_reproducibility_snippet(
    path: Path,
    fs: float,
) -> None:
    nyq = 0.5 * fs
    txt = f"""## Rhythmic push-up motion -- final revised analysis

**State.** Apple Watch Core Motion user acceleration, three axes, in units of g.
The per-channel mean is removed. The primary analysis uses the raw demeaned
three-axis acceleration without per-axis standardization and without forced
stability rescaling of the fitted operator.

**Common activity definition.** One activity label is constructed from the raw
demeaned acceleration magnitude only. A causal trailing {ENVELOPE_SEC:.2f}-s
mean envelope is formed, and samples above the within-recording
Q_{HIGH_ACTIVITY_QUANTILE:.2f} threshold are designated high activity. The
one-step pair (x_t, x_(t+1)) is assigned using the class at t only. The same raw
activity labels are reused for every signal variant.

**Primary signal.** Raw demeaned acceleration.

**Adaptive-cycle sensitivity.** The instantaneous push-up phase is estimated
from the first principal component after {LC_FREQ_BAND_HZ[0]:.2f}--{LC_FREQ_BAND_HZ[1]:.2f} Hz
bandpass filtering. A locally weighted K={ADAPTIVE_N_HARMONICS} harmonic model

x_i(t) = sum_(k=1)^K [a_ik(t) cos(k phi(t)) + b_ik(t) sin(k phi(t))] + epsilon_i(t)

is fitted with a {ADAPTIVE_LOCAL_WINDOW_SEC:.1f}-s local window. The residual is
analyzed in {ADAPTIVE_MAIN_BAND_HZ[0]:.0f}--{ADAPTIVE_MAIN_BAND_HZ[1]:.0f} Hz,
with {ADAPTIVE_WIDE_BAND_HZ[0]:.0f}--{ADAPTIVE_WIDE_BAND_HZ[1]:.0f} Hz as a
wide-band sensitivity analysis. The sampling rate inferred from this recording
is {fs:.6f} Hz (Nyquist {nyq:.6f} Hz).

**Operator fit.** Separate low- and high-activity one-step operators are fitted
by ridge regression with relative ridge level {RIDGE_REL:.3g}, followed by
trace-preserving isotropic shrinkage alpha={SHRINK_ALPHA:.3g}. Operators are not
rescaled to force stability.

**Reduced diagnostics.** M2 and M3 independently extract the response plane.
For Gamma=Q^T F Q, report rho(Gamma), sigma_max(Gamma), Henrici departure d_F,
and

R1 = d_F / sqrt[(1-|lambda_1|^2)(1-|lambda_2|^2)]

when rho(Gamma)<1. For a stable 2x2 map, R1>1 iff sigma_max(Gamma)>1. This
criterion applies to real and complex eigenvalue pairs.

**Uncertainty and controls.** Moving-block bootstrap uses
{BLOCK_BOOT_SEC:.1f}-s blocks. The high-minus-low sigma_max(Gamma) contrast is
tested against (i) a stationary fitted VAR(1) surrogate with innovation-vector
resampling and (ii) a multivariate common-phase randomized surrogate preserving
the analyzed signal's cross-spectrum. A circular-shift test of the reaction
coordinate against the raw causal activity envelope is reported only as a
secondary timing-specific control.
"""
    path.write_text(txt, encoding="utf-8")


# IO
def read_pushup_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    df = pd.read_csv(path)
    required = ["timestamp", "accX", "accY", "accZ"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; found {list(df.columns)}")

    t_abs = df["timestamp"].to_numpy(float)
    t = t_abs - t_abs[0]
    fs, dt = robust_fs_from_time(t_abs)
    X = df[["accX", "accY", "accZ"]].to_numpy(float)
    return t, X, fs, dt


# MAIN
def main() -> None:
    ensure_dir(OUTDIR)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}. Put the CSV beside this script "
            "or update INPUT_CSV."
        )

    t, X_raw, fs, dt = read_pushup_csv(INPUT_CSV)
    print(f"Loaded {INPUT_CSV}: n={len(t)}, fs={fs:.6f} Hz, dt={dt:.8f} s")

    variants, aux = build_variants(t, X_raw, fs)

    raw_env, high, activity_threshold = build_common_activity_labels(
        aux["X_raw_demeaned"], fs
    )
    print(
        f"Common causal high-activity class: upper "
        f"{100*(1-HIGH_ACTIVITY_QUANTILE):.1f}% of raw envelope "
        f"(threshold={activity_threshold:.6g})"
    )

    # Observed results.
    observed_parts = []
    observed_by_variant = {}

    for v in VARIANTS:
        obs, _fits = analyze_variant_observed(variants[v.name], high)
        observed_parts.append(obs)
        observed_by_variant[v.name] = obs

    observed = pd.concat(observed_parts, ignore_index=True)
    observed.to_csv(OUTDIR / "pushup_final_observed.csv", index=False)

    # Moving-block bootstrap.
    boot_parts = []
    for v in VARIANTS:
        print(f"Block bootstrap: {v.name}")
        bd = block_bootstrap_variant(variants[v.name], high, fs)
        boot_parts.append(bd)

    boot_df = pd.concat(boot_parts, ignore_index=True)
    boot_df.to_csv(
        OUTDIR / "pushup_final_bootstrap_replicates.csv", index=False
    )
    boot_summary = summarize_bootstrap(boot_df)
    boot_summary.to_csv(
        OUTDIR / "pushup_final_bootstrap_summary.csv", index=False
    )

    # VAR(1) and cross-spectrum-preserving phase nulls.
    null_rep_parts = []
    null_sum_parts = []

    for v in VARIANTS:
        print(f"Null controls: {v.name}")
        reps, summ = null_controls_variant(
            variants[v.name], high, fs, observed_by_variant[v.name]
        )
        null_rep_parts.append(reps)
        null_sum_parts.append(summ)

    null_reps = pd.concat(null_rep_parts, ignore_index=True)
    null_summary = pd.concat(null_sum_parts, ignore_index=True)

    null_reps.to_csv(
        OUTDIR / "pushup_final_null_replicates.csv", index=False
    )
    null_summary.to_csv(
        OUTDIR / "pushup_final_null_summary.csv", index=False
    )

    # Secondary timing control.
    timing_df = timing_nulls_all(variants, raw_env, fs)
    timing_df.to_csv(
        OUTDIR / "pushup_final_timing_null.csv", index=False
    )

    # Final compact table.
    final_table = attach_intervals_and_nulls(
        observed, boot_summary, null_summary
    )
    final_table.to_csv(
        OUTDIR / "pushup_final_summary.csv", index=False
    )

    # Human-readable outputs.
    write_report(
        OUTDIR / "pushup_final_report.txt",
        fs=fs,
        dt=dt,
        activity_threshold=activity_threshold,
        final_table=final_table,
        null_summary=null_summary,
        timing_df=timing_df,
        aux=aux,
    )

    write_reproducibility_snippet(
        OUTDIR / "pushup_final_reproducibility_snippet.txt",
        fs=fs,
    )

    make_summary_figure(
        t=t,
        raw_env=raw_env,
        high=high,
        threshold=activity_threshold,
        final_table=final_table,
        null_summary=null_summary,
        outbase=OUTDIR / "fig16_pushup_final_summary",
    )

    # Console table: the one to inspect before manuscript/email changes.
    show_cols = [
        "variant", "method", "regime", "n_pairs",
        "rhoF", "rhoGamma", "sigmaGamma", "R1",
        "spectrum_type", "stable_amplifying",
        "sigmaGamma_ci95_low", "sigmaGamma_ci95_high",
        "R1_ci95_low", "R1_ci95_high",
        "p_delta_sigma_stationary_VAR1",
        "p_delta_sigma_common_phase_randomized",
    ]
    show_cols = [c for c in show_cols if c in final_table.columns]

    print("\nFINAL OBSERVED + UNCERTAINTY TABLE")
    print("=" * 100)
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 40,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.6g}",
    ):
        print(final_table[show_cols].to_string(index=False))

    print("\nSaved outputs in:", OUTDIR.resolve())
    print(
        "\nDo not update the manuscript or email numerical conclusions until "
        "pushup_final_summary.csv and pushup_final_null_summary.csv are inspected."
    )


if __name__ == "__main__":
    main()
