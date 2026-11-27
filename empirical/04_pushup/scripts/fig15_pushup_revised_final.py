#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Push-up NNDR analysis -- revised final reproducibility script
=============================================================

This is the single public-source script for the revised push-up example.

It combines the final analyses retained after review:

1. Primary raw demeaned three-axis acceleration, with no per-axis
   standardization and no forced stabilization.
2. A common causal trailing 0.5-s activity envelope from the raw signal;
   one-step pairs are classified by the activity state at t only.
3. Adaptive K=3 cycle-removal sensitivity checks in 1--15 Hz and 1--24 Hz.
   The obsolete fixed-frequency K=2, 1--8 Hz pipeline is not included.
4. M2 and M3 response-plane extraction.
5. Both threshold conventions carried while the manuscript convention is
   being finalized:
       R_A: continuous-time rates-based variance-response normalization;
       R_B = R_sigma: exact discrete one-step normalization.
6. Definitive class-conditioned uncertainty from a 5-s circular block
   bootstrap with 2000 replicates.
7. Stationary fitted VAR(1), multivariate common-phase randomized, and
   circular-shift timing controls.
8. The pre-registered M3 moving-window statistic (1-s windows, 0.1-s step)
   compared with a time-varying spectrum- and variance-matched normal null.

Input
-----
raw_samples_Pushup_duo_27Dec2025..csv
with columns: timestamp, accX, accY, accZ

Run
---
python fig15_pushup_revised_final.py

No command-line arguments are required. Results are written to
pushup_revised_final/.
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
OUTDIR = Path(__file__).resolve().parents[1] / "results" / "pushup_revised_final"

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


# FINAL CONSOLIDATED REVISION ANALYSIS
#
# This section supersedes the earlier script-level orchestration above.  It
# keeps the tested preprocessing/operator/plane routines and combines the
# three analyses used in the revised manuscript:
#
#   A. causal low/high subset analysis with 5-s circular-block bootstrap;
#   B. stationary VAR(1), common-phase and timing controls;
#   C. registered 1-s M3 moving-window statistic against a time-varying
#      spectrum- and variance-matched normal null.
#
# Both threshold conventions under coauthor discussion are carried:
#   RA = continuous-time variance-response normalization (rates);
#   RB = R_sigma = exact discrete one-step normalization.
#
# No obsolete fixed-frequency 1--8 Hz pipeline is included.

from scipy.linalg import schur, solve_discrete_lyapunov


# Final reproducibility settings
FINAL_ROOT = Path(__file__).resolve().parent
FINAL_INPUT_CSV = FINAL_ROOT / "raw_samples_Pushup_duo_27Dec2025..csv"
FINAL_OUTDIR = FINAL_ROOT / "pushup_revised_final"

# Definitive class-conditioned bootstrap requested in the rebuttal.
FINAL_BOOT_BLOCK_SEC = 5.0
FINAL_N_BOOT = 2000
FINAL_BOOT_SEED = 20261011

# Registered moving-window statistic.
REGISTERED_WINDOW_SEC = 1.0
REGISTERED_STEP_SEC = 0.1
REGISTERED_METHOD = "M3"
REGISTERED_N_NULL = 500
REGISTERED_NULL_SEED = 20261009
REGISTERED_BURNIN_SEC = 10.0
REGISTERED_BOOT_BLOCK_SEC = 5.0
REGISTERED_N_BOOT = 2000
REGISTERED_BOOT_SEED = 20261010
REGISTERED_GAIN_TOL = 1e-10

# Option-A numerical tolerance.
RATE_TOL = 1e-9


# OPTION A: CONTINUOUS-TIME VARIANCE-RESPONSE NORMALIZATION

def option_a_variance_ratio(
    Gamma: np.ndarray,
    K: float,
    dt: float,
) -> dict:
    """
    Evaluate the rates-based variance-response normalization R_A.

    The discrete reduced multipliers are converted with the principal
    logarithm mu_i = Log(lambda_i)/dt.

    Complex pair:
        mu_± = -gamma ± i omega,
        beta = |omega|/gamma,
        Kc_A = sqrt[1/(sqrt(1+beta^2)-1)].

    Stable real-rate pair:
        Delta = |(mu1-mu2)/(mu1+mu2)|,
        s = sqrt(1-Delta^2),
        Kc_A = sqrt[s/(1-s)].

    R_A is returned as NaN rather than extrapolated when the principal-log
    rates are not in an admissible real-pair or complex-conjugate domain.
    """
    eig = LA.eigvals(np.asarray(Gamma, complex))
    rho = float(np.max(np.abs(eig)))

    out = {
        "RA": np.nan,
        "KcA": np.nan,
        "A_domain": "inadmissible",
        "gamma_A": np.nan,
        "omega_A": np.nan,
        "beta_A": np.nan,
        "Delta_rate_A": np.nan,
    }

    if (not np.isfinite(rho)) or rho <= 0.0 or rho >= 1.0:
        return out

    rates = np.log(eig) / float(dt)
    scale = max(1.0, float(np.max(np.abs(rates))))
    is_complex = np.max(np.abs(np.imag(rates))) > RATE_TOL * scale

    if is_complex:
        if abs(rates[0] - np.conj(rates[1])) > 1e-7 * scale:
            return out

        mu = rates[0] if np.imag(rates[0]) >= 0 else rates[1]
        gamma = -float(np.real(mu))
        omega = abs(float(np.imag(mu)))

        if gamma <= 0.0 or omega <= EPS:
            return out

        beta = omega / gamma
        denom = np.sqrt(1.0 + beta**2) - 1.0
        if denom <= EPS:
            return out

        KcA = float(np.sqrt(1.0 / denom))
        RA = float(K / KcA)

        out.update({
            "RA": RA,
            "KcA": KcA,
            "A_domain": "complex_rate_pair",
            "gamma_A": gamma,
            "omega_A": omega,
            "beta_A": beta,
        })
        return out

    rates_real = np.real(rates)
    if not np.all(rates_real < -RATE_TOL):
        return out

    denom_rate = float(rates_real[0] + rates_real[1])
    if abs(denom_rate) <= EPS:
        return out

    Delta = float(abs((rates_real[0] - rates_real[1]) / denom_rate))
    if Delta < 0.0 or Delta >= 1.0:
        return out

    s = float(np.sqrt(max(0.0, 1.0 - Delta**2)))
    denom = 1.0 - s

    if denom <= EPS:
        KcA = np.inf
        RA = 0.0
    else:
        KcA = float(np.sqrt(s / denom))
        RA = float(K / KcA)

    out.update({
        "RA": RA,
        "KcA": KcA,
        "A_domain": "real_rate_pair",
        "Delta_rate_A": Delta,
    })
    return out


def option_b_one_step_threshold(Gamma: np.ndarray) -> dict:
    """
    Exact discrete one-step normalization.

    For a stable two-dimensional map:
        Kc_B = sqrt[(1-|l1|^2)(1-|l2|^2)] / |l1-l2|,
        RB = d_F / sqrt[(1-|l1|^2)(1-|l2|^2)],

    with RB > 1 iff sigma_max(Gamma) > 1.

    The d_F form is used for RB because it remains finite at the defective
    limit and needs no eigenvectors.
    """
    G = np.asarray(Gamma, float)
    eig = LA.eigvals(G)
    rho = float(np.max(np.abs(eig)))
    sigma = sigma_max(G)

    fro2 = float(np.sum(np.abs(G) ** 2))
    eig2 = float(np.sum(np.abs(eig) ** 2))
    dF = float(np.sqrt(max(fro2 - eig2, 0.0)))

    margin2 = float(np.prod(1.0 - np.abs(eig) ** 2))
    gap = float(abs(eig[0] - eig[1]))

    RB = np.nan
    KcB = np.nan
    if rho < 1.0 - STABLE_TOL and margin2 > EPS:
        margin = float(np.sqrt(margin2))
        RB = float(dF / margin)
        if gap > EPS:
            KcB = float(margin / gap)
        else:
            KcB = np.inf

    return {
        "RB": RB,
        "R_sigma": RB,
        "KcB": KcB,
        "dF": dF,
        "rhoGamma": rho,
        "sigmaGamma": sigma,
    }


# FINAL LOW/HIGH SUBSET ANALYSIS

def analyze_pair_set_final(
    X0: np.ndarray,
    X1: np.ndarray,
    method: str,
    dt: float,
) -> dict | None:
    fit = fit_operator_from_pairs(X0, X1)
    res = analyze_fit(fit, method)

    if res is None:
        return None

    A = option_a_variance_ratio(res["Gamma"], res["K"], dt)
    B = option_b_one_step_threshold(res["Gamma"])

    rhoG = float(res["rhoGamma"])
    sigmaG = float(res["sigmaGamma"])

    return {
        "n_pairs": int(len(X0)),
        "rhoF": float(res["rhoF"]),
        "sigmaF": float(res["sigmaF"]),
        "rhoGamma": rhoG,
        "sigmaGamma": sigmaG,
        "gain_over_rho": sigmaG / rhoG if rhoG > EPS else np.nan,
        "K": float(res["K"]),
        "dF": float(B["dF"]),
        "RA": float(A["RA"]) if np.isfinite(A["RA"]) else np.nan,
        "KcA": float(A["KcA"]) if np.isfinite(A["KcA"]) else A["KcA"],
        "A_domain": A["A_domain"],
        "gamma_A": A["gamma_A"],
        "omega_A": A["omega_A"],
        "beta_A": A["beta_A"],
        "Delta_rate_A": A["Delta_rate_A"],
        "RB": float(B["RB"]) if np.isfinite(B["RB"]) else np.nan,
        "R_sigma": float(B["R_sigma"]) if np.isfinite(B["R_sigma"]) else np.nan,
        # Alias retained so older plotting/null helpers use the same exact
        # one-step quantity without maintaining a second implementation.
        "R1": float(B["RB"]) if np.isfinite(B["RB"]) else np.nan,
        "KcB": B["KcB"],
        "spectrum_type": res["spectrum_type"],
        "stable_reduced": int(res["stable_reduced"]),
        "stable_amplifying": int(res["stable_amplifying"]),
        "eig1_real": float(res["eig1_real"]),
        "eig1_imag": float(res["eig1_imag"]),
        "eig2_real": float(res["eig2_real"]),
        "eig2_imag": float(res["eig2_imag"]),
        "eff_rank": float(res["eff_rank"]),
        "cond": float(res["cond"]),
        "ridge_lambda": float(res["ridge_lambda"]),
    }


def circular_block_pair_indices_final(
    n_pairs: int,
    block_len_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample indices of ORIGINAL one-step pairs in circular contiguous blocks.

    The pair (x_t, x_{t+1}) remains intact. Therefore no artificial transition
    is created at a bootstrap block boundary.
    """
    block_len_pairs = max(1, min(int(block_len_pairs), int(n_pairs)))
    out = []
    while len(out) < n_pairs:
        start = int(rng.integers(0, n_pairs))
        block = (start + np.arange(block_len_pairs)) % n_pairs
        out.extend(block.tolist())
    return np.asarray(out[:n_pairs], dtype=int)


def bootstrap_variant_final(
    variant_name: str,
    X: np.ndarray,
    high: np.ndarray,
    fs: float,
    dt: float,
) -> pd.DataFrame:
    X = np.asarray(X, float)
    X0_all = X[:-1]
    X1_all = X[1:]
    labels_all = np.asarray(high[:-1], bool)

    n_pairs = len(X0_all)
    block_len = max(1, int(round(FINAL_BOOT_BLOCK_SEC * fs)))
    variant_index = [v.name for v in VARIANTS].index(variant_name)
    rng = np.random.default_rng(FINAL_BOOT_SEED + 1000 * variant_index)

    rows = []

    for b in range(FINAL_N_BOOT):
        idx = circular_block_pair_indices_final(n_pairs, block_len, rng)
        labels = labels_all[idx]
        X0b = X0_all[idx]
        X1b = X1_all[idx]

        for regime, mask in (("low", ~labels), ("high", labels)):
            if int(np.sum(mask)) < 20:
                continue

            for method in ("M2", "M3"):
                res = analyze_pair_set_final(
                    X0b[mask], X1b[mask], method, dt
                )
                if res is None:
                    continue

                rows.append({
                    "variant": variant_name,
                    "bootstrap": b,
                    "regime": regime,
                    "method": method,
                    **res,
                })

        if (b + 1) % 100 == 0 or b == 0:
            print(f"{variant_name}: 5-s bootstrap {b+1}/{FINAL_N_BOOT}")

    return pd.DataFrame(rows)


FINAL_POINT_METRICS = [
    "rhoF",
    "sigmaF",
    "rhoGamma",
    "sigmaGamma",
    "gain_over_rho",
    "K",
    "RA",
    "RB",
]


def _ci95_final(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    return (
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    )


def summarize_bootstrap_final(
    boot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    contrast_rows = []

    for (variant, method, regime), g in boot.groupby(
        ["variant", "method", "regime"]
    ):
        row = {
            "variant": variant,
            "method": method,
            "regime": regime,
            "n_boot_valid": int(g["bootstrap"].nunique()),
        }

        for metric in FINAL_POINT_METRICS:
            vals = g[metric].to_numpy(float)
            lo, hi = _ci95_final(vals)
            row[f"{metric}_boot_median"] = (
                float(np.nanmedian(vals))
                if np.any(np.isfinite(vals))
                else np.nan
            )
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        row["frac_sigmaGamma_gt1"] = float(
            np.mean(g["sigmaGamma"].to_numpy(float) > 1.0)
        )
        row["frac_rhoGamma_lt1_sigmaGamma_gt1"] = float(
            np.mean(
                (g["rhoGamma"].to_numpy(float) < 1.0)
                & (g["sigmaGamma"].to_numpy(float) > 1.0)
            )
        )
        row["frac_rhoF_ge1"] = float(
            np.mean(g["rhoF"].to_numpy(float) >= 1.0)
        )
        row["frac_RA_valid"] = float(
            np.mean(np.isfinite(g["RA"].to_numpy(float)))
        )

        summary_rows.append(row)

    for (variant, method), g in boot.groupby(["variant", "method"]):
        low = g[g["regime"] == "low"].set_index("bootstrap")
        high_g = g[g["regime"] == "high"].set_index("bootstrap")
        common = low.index.intersection(high_g.index)

        for metric in (
            "rhoF",
            "sigmaF",
            "rhoGamma",
            "sigmaGamma",
            "gain_over_rho",
            "RA",
            "RB",
        ):
            if common.size == 0:
                continue

            delta = (
                high_g.loc[common, metric].to_numpy(float)
                - low.loc[common, metric].to_numpy(float)
            )
            delta = delta[np.isfinite(delta)]

            if delta.size:
                med = float(np.median(delta))
                lo, hi = _ci95_final(delta)
                frac_le_zero = float(np.mean(delta <= 0.0))
            else:
                med = lo = hi = frac_le_zero = np.nan

            contrast_rows.append({
                "variant": variant,
                "method": method,
                "metric": metric,
                "n_boot_paired": int(delta.size),
                "delta_high_minus_low_median": med,
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "bootstrap_fraction_delta_le_zero": frac_le_zero,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def build_point_table_final(
    variants: dict[str, dict],
    high: np.ndarray,
    dt: float,
) -> pd.DataFrame:
    low_pair, high_pair = pair_labels_from_high(high)
    rows = []

    for v in VARIANTS:
        X = np.asarray(variants[v.name]["X"], float)
        X0_all = X[:-1]
        X1_all = X[1:]

        for regime, mask in (("low", low_pair), ("high", high_pair)):
            for method in ("M2", "M3"):
                res = analyze_pair_set_final(
                    X0_all[mask],
                    X1_all[mask],
                    method,
                    dt,
                )
                if res is None:
                    raise RuntimeError(
                        f"Point fit failed for {v.name}, {method}, {regime}"
                    )

                rows.append({
                    "variant": v.name,
                    "variant_label": v.label,
                    "method": method,
                    "regime": regime,
                    **res,
                })

    return pd.DataFrame(rows)


def merge_point_bootstrap_final(
    point: pd.DataFrame,
    boot_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = point.merge(
        boot_summary,
        on=["variant", "method", "regime"],
        how="left",
        suffixes=("", "_boot"),
    )

    # Aliases used by the legacy plotting routine, all pointing to R_B/R_sigma.
    if "RB_ci95_low" in out.columns:
        out["R1_ci95_low"] = out["RB_ci95_low"]
        out["R1_ci95_high"] = out["RB_ci95_high"]

    return out


# REGISTERED 1-S MOVING-WINDOW / TIME-VARYING NORMAL NULL

def moving_window_registered(
    X: np.ndarray,
    fs: float,
    causal_env: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Registered statistic: M3, W=1 s, step=0.1 s, no amplitude selection.
    """
    X = np.asarray(X, float)
    n = len(X)
    W = int(round(REGISTERED_WINDOW_SEC * fs))
    step = max(1, int(round(REGISTERED_STEP_SEC * fs)))
    starts = np.arange(0, n - W + 1, step, dtype=int)

    rows = []

    for j, s in enumerate(starts):
        e = s + W
        Xw = X[s:e]

        fit = fit_operator_from_pairs(Xw[:-1], Xw[1:])
        res = analyze_fit(fit, REGISTERED_METHOD)
        if res is None:
            continue

        Xwc = Xw - np.mean(Xw, axis=0, keepdims=True)
        local_total_variance = float(
            np.mean(np.sum(Xwc**2, axis=1))
        )

        registered_event = bool(
            res["rhoGamma"] < 1.0 - STABLE_TOL
            and res["sigmaGamma"] > 1.0 + REGISTERED_GAIN_TOL
        )

        row = {
            "window_id": j,
            "start_sample": int(s),
            "end_sample": int(e - 1),
            "center_sample": int(round((s + e - 1) / 2)),
            "start_sec": s / fs,
            "end_sec": (e - 1) / fs,
            "center_sec": 0.5 * (s + e - 1) / fs,
            "n_samples": int(W),
            "rhoF": float(res["rhoF"]),
            "sigmaF": float(res["sigmaF"]),
            "rhoGamma": float(res["rhoGamma"]),
            "sigmaGamma": float(res["sigmaGamma"]),
            "R_sigma": float(res["R1"]) if np.isfinite(res["R1"]) else np.nan,
            "complex_pair": int(res["spectrum_type"] == "complex_pair"),
            "registered_event": int(registered_event),
            "eig1_real": float(res["eig1_real"]),
            "eig1_imag": float(res["eig1_imag"]),
            "eig2_real": float(res["eig2_real"]),
            "eig2_imag": float(res["eig2_imag"]),
            "local_total_variance": local_total_variance,
            "eff_rank": float(res["eff_rank"]),
            "cond": float(res["cond"]),
        }

        if causal_env is not None:
            row["causal_envelope_mean"] = float(np.mean(causal_env[s:e]))

        rows.append(row)

    return pd.DataFrame(rows)


def spectrum_matched_normal_operator(F: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Construct a real normal matrix with the same real-Schur eigenvalue blocks.

    Strictly upper Schur couplings are removed. Each complex 2x2 block is
    replaced by a normal damped-rotation block [[a,w],[-w,a]].
    """
    T, U = schur(np.asarray(F, float), output="real")
    n = T.shape[0]
    Tn = np.zeros_like(T)

    i = 0
    while i < n:
        if i < n - 1 and abs(T[i + 1, i]) > 1e-10:
            block = T[i:i + 2, i:i + 2]
            ev = LA.eigvals(block)
            a = float(np.real(ev[0]))
            omega = float(abs(np.imag(ev[0])))
            Tn[i:i + 2, i:i + 2] = np.array(
                [[a, omega], [-omega, a]]
            )
            i += 2
        else:
            Tn[i, i] = T[i, i]
            i += 1

    Fn = U @ Tn @ U.T
    Fn = np.real_if_close(Fn).astype(float)

    normality_error = float(
        LA.norm(Fn @ Fn.T - Fn.T @ Fn, "fro")
    )
    return Fn, normality_error


def isotropic_noise_for_target_variance(
    F_normal: np.ndarray,
    target_total_variance: float,
) -> float:
    """
    Isotropic innovation strength q such that the frozen-time stationary
    covariance has trace equal to the observed local total variance.
    """
    if spectral_radius(F_normal) >= 1.0 - STABLE_TOL:
        return np.nan

    try:
        P_unit = solve_discrete_lyapunov(
            F_normal,
            np.eye(F_normal.shape[0]),
        )
    except Exception:
        return np.nan

    tr = float(np.trace(P_unit))
    if (not np.isfinite(tr)) or tr <= EPS:
        return np.nan

    q = float(target_total_variance / tr)
    if (not np.isfinite(q)) or q <= 0.0:
        return np.nan
    return q


def build_tv_normal_schedule(
    X: np.ndarray,
    observed_windows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    model_centers = []
    F_models = []
    q_models = []
    normality_errors = []

    for _, row in observed_windows.iterrows():
        s = int(row["start_sample"])
        e = int(row["end_sample"]) + 1
        Xw = X[s:e]

        fit = fit_operator_from_pairs(Xw[:-1], Xw[1:])
        if fit is None:
            continue

        Fn, normality_error = spectrum_matched_normal_operator(fit["F"])
        if spectral_radius(Fn) >= 1.0 - STABLE_TOL:
            continue

        q = isotropic_noise_for_target_variance(
            Fn,
            float(row["local_total_variance"]),
        )
        if not np.isfinite(q):
            continue

        model_centers.append(int(row["center_sample"]))
        F_models.append(Fn)
        q_models.append(q)
        normality_errors.append(normality_error)

    if not F_models:
        raise RuntimeError("No valid stable local normal models.")

    model_centers = np.asarray(model_centers, dtype=int)
    F_models = np.asarray(F_models)
    q_models = np.asarray(q_models)

    transition_samples = np.arange(len(X) - 1)
    schedule_idx = np.empty(len(transition_samples), dtype=int)

    for ii, t in enumerate(transition_samples):
        k = np.searchsorted(model_centers, t)
        if k == 0:
            schedule_idx[ii] = 0
        elif k >= len(model_centers):
            schedule_idx[ii] = len(model_centers) - 1
        else:
            left, right = k - 1, k
            schedule_idx[ii] = (
                left
                if abs(t - model_centers[left]) <= abs(t - model_centers[right])
                else right
            )

    max_normality_error = float(np.max(normality_errors))
    return (
        F_models,
        q_models,
        schedule_idx,
        model_centers,
        max_normality_error,
    )


def simulate_tv_normal(
    n: int,
    fs: float,
    F_models: np.ndarray,
    q_models: np.ndarray,
    schedule_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    d = F_models.shape[1]
    X = np.zeros((n, d), dtype=float)

    burn = max(100, int(round(REGISTERED_BURNIN_SEC * fs)))
    x = np.zeros(d)
    F0 = F_models[0]
    q0 = q_models[0]

    for _ in range(burn):
        x = F0 @ x + rng.normal(0.0, np.sqrt(q0), size=d)

    X[0] = x

    for t in range(n - 1):
        k = schedule_idx[t]
        F = F_models[k]
        q = q_models[k]
        X[t + 1] = F @ X[t] + rng.normal(0.0, np.sqrt(q), size=d)

    return X - X.mean(axis=0, keepdims=True)


def registered_fraction(window_df: pd.DataFrame) -> float:
    if len(window_df) == 0:
        return np.nan
    return float(np.mean(window_df["registered_event"].to_numpy(float)))


def circular_block_bootstrap_fraction_registered(
    indicator: np.ndarray,
    block_windows: int,
) -> tuple[np.ndarray, float, float]:
    indicator = np.asarray(indicator, float)
    n = len(indicator)
    rng = np.random.default_rng(REGISTERED_BOOT_SEED)

    vals = []
    for _ in range(REGISTERED_N_BOOT):
        out = []
        while len(out) < n:
            start = int(rng.integers(0, n))
            block = (start + np.arange(block_windows)) % n
            out.extend(indicator[block].tolist())
        vals.append(float(np.mean(np.asarray(out[:n], float))))

    vals = np.asarray(vals, float)
    return (
        vals,
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def run_registered_tvnormal(
    X_raw_demeaned: np.ndarray,
    raw_env: np.ndarray,
    fs: float,
    outdir: Path,
) -> dict:
    observed = moving_window_registered(
        X_raw_demeaned,
        fs,
        causal_env=raw_env,
    )
    if observed.empty:
        raise RuntimeError("No valid registered moving windows.")

    observed.to_csv(
        outdir / "pushup_registered_observed_windows.csv",
        index=False,
    )

    S_obs = registered_fraction(observed)
    n_complex = int(observed["complex_pair"].sum())
    n_full_unstable = int(np.sum(observed["rhoF"] >= 1.0))

    block_windows = max(
        1,
        int(round(REGISTERED_BOOT_BLOCK_SEC / REGISTERED_STEP_SEC)),
    )
    boot_vals, boot_lo, boot_hi = (
        circular_block_bootstrap_fraction_registered(
            observed["registered_event"].to_numpy(),
            block_windows,
        )
    )

    (
        F_models,
        q_models,
        schedule_idx,
        model_centers,
        max_normality_error,
    ) = build_tv_normal_schedule(
        X_raw_demeaned,
        observed,
    )

    rng = np.random.default_rng(REGISTERED_NULL_SEED)
    null_rows = []

    for b in range(REGISTERED_N_NULL):
        Xnull = simulate_tv_normal(
            n=len(X_raw_demeaned),
            fs=fs,
            F_models=F_models,
            q_models=q_models,
            schedule_idx=schedule_idx,
            rng=rng,
        )

        nulldf = moving_window_registered(Xnull, fs, causal_env=None)
        S_null = registered_fraction(nulldf)

        null_rows.append({
            "replicate": b,
            "registered_fraction": S_null,
            "n_valid_windows": len(nulldf),
            "fraction_complex": (
                float(nulldf["complex_pair"].mean())
                if len(nulldf)
                else np.nan
            ),
            "fraction_full_rho_ge_1": (
                float(np.mean(nulldf["rhoF"] >= 1.0))
                if len(nulldf)
                else np.nan
            ),
        })

        if (b + 1) % 25 == 0 or b == 0:
            print(
                f"time-varying normal null {b+1:4d}/{REGISTERED_N_NULL}: "
                f"S={S_null:.4f}"
            )

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(
        outdir / "pushup_registered_tvnormal_null.csv",
        index=False,
    )

    valid_null = (
        null_df["registered_fraction"]
        .dropna()
        .to_numpy(float)
    )

    p = float(
        (1 + np.sum(valid_null >= S_obs))
        / (1 + len(valid_null))
    )

    result = {
        "S_obs": S_obs,
        "n_valid_windows": int(len(observed)),
        "n_complex": n_complex,
        "fraction_complex": n_complex / len(observed),
        "n_full_unstable": n_full_unstable,
        "bootstrap_ci95_low": boot_lo,
        "bootstrap_ci95_high": boot_hi,
        "bootstrap_replicates": REGISTERED_N_BOOT,
        "bootstrap_block_sec": REGISTERED_BOOT_BLOCK_SEC,
        "null_replicates": int(len(valid_null)),
        "null_median": float(np.median(valid_null)),
        "null_ci95_low": float(np.percentile(valid_null, 2.5)),
        "null_ci95_high": float(np.percentile(valid_null, 97.5)),
        "null_max": float(np.max(valid_null)),
        "p_one_sided": p,
        "n_local_normal_models": int(len(F_models)),
        "max_normality_error": max_normality_error,
    }

    # Registered-null figure.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(valid_null, bins=30, density=True, alpha=0.65)
    ax.axvline(
        S_obs,
        linestyle="--",
        linewidth=2.0,
        label=f"Observed = {S_obs:.3f}",
    )
    ax.set_xlabel(
        r"Fraction of 1-s windows with "
        r"$\rho(\Gamma)<1<\sigma_{\max}(\Gamma)$"
    )
    ax.set_ylabel("Null density")
    ax.set_title("Time-varying normal null")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        outdir / "fig15_pushup_registered_tvnormal.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "fig15_pushup_registered_tvnormal.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    return result


# FINAL OUTPUT TABLES / REPORT

def write_final_latex_table(
    path: Path,
    final: pd.DataFrame,
) -> None:
    label_map = {v.name: v.label for v in VARIANTS}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\begin{tabular}{lllcccccccc}",
        r"\toprule",
        (
            r"Signal & Method & Class & $n$ & "
            r"$\rho(\widehat F)$ & $\sigma_{\max}(\widehat F)$ & "
            r"$\rho(\Gamma)$ & $\sigma_{\max}(\Gamma)$ & "
            r"$R_A$ & $R_B$ & "
            r"$P_{\rm boot}[\rho(\Gamma)<1<\sigma_{\max}(\Gamma)]$ \\"
        ),
        r"\midrule",
    ]

    for v in VARIANTS:
        g = final[final["variant"] == v.name]
        first = True
        for method in ("M2", "M3"):
            for regime in ("low", "high"):
                rr = g[
                    (g["method"] == method)
                    & (g["regime"] == regime)
                ]
                if rr.empty:
                    continue
                rr = rr.iloc[0]
                sig_label = label_map[v.name] if first else ""
                first = False

                def f3(x):
                    return "--" if not np.isfinite(x) else f"{x:.3f}"

                lines.append(
                    f"{sig_label} & {method} & {regime} & "
                    f"{int(rr['n_pairs'])} & "
                    f"{f3(rr['rhoF'])} & "
                    f"{f3(rr['sigmaF'])} & "
                    f"{f3(rr['rhoGamma'])} & "
                    f"{f3(rr['sigmaGamma'])} & "
                    f"{f3(rr['RA'])} & "
                    f"{f3(rr['RB'])} & "
                    f"{f3(rr['frac_rhoGamma_lt1_sigmaGamma_gt1'])} \\\\"
                )
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Final class-conditioned push-up diagnostics. "
            r"The activity class is causal and derived from the raw acceleration "
            r"envelope. $R_A$ is the continuous-time variance-response "
            r"normalization and $R_B=R_\sigma$ the exact discrete one-step "
            r"normalization. The final column gives the fraction of 5-s "
            r"circular-block bootstrap replicates satisfying "
            r"$\rho(\Gamma)<1<\sigma_{\max}(\Gamma)$.}"
        ),
        r"\label{tab:pushup_final_subsets}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    path: Path,
    fs: float,
    dt: float,
    threshold: float,
    point: pd.DataFrame,
    final: pd.DataFrame,
    contrasts: pd.DataFrame,
    null_summary: pd.DataFrame,
    timing_df: pd.DataFrame,
    registered: dict,
    aux: dict,
) -> None:
    raw_m3_high = final[
        (final["variant"] == "raw_demeaned")
        & (final["method"] == "M3")
        & (final["regime"] == "high")
    ]

    if raw_m3_high.empty:
        high_point_crossing = False
    else:
        r = raw_m3_high.iloc[0]
        high_point_crossing = bool(
            r["rhoF"] < 1.0
            and r["rhoGamma"] < 1.0
            and r["sigmaGamma"] > 1.0
        )

    if registered["p_one_sided"] < 0.05 and high_point_crossing:
        placement = "Option (a)"
        placement_text = (
            "registered moving-window statistic significant and raw high-activity "
            "M3 point estimate is stable-amplifying."
        )
    elif registered["p_one_sided"] < 0.05:
        placement = "Option (b)"
        placement_text = (
            "registered moving-window statistic significant, while the "
            "class-conditioned high-activity point estimate remains below "
            "the one-step threshold."
        )
    else:
        placement = "Option (c)"
        placement_text = (
            "registered moving-window statistic is not significant against "
            "the time-varying normal null."
        )

    cols = [
        "variant",
        "method",
        "regime",
        "n_pairs",
        "rhoF",
        "sigmaF",
        "rhoGamma",
        "sigmaGamma",
        "gain_over_rho",
        "K",
        "RA",
        "RB",
        "spectrum_type",
    ]

    lines = [
        "PUSH-UP REVISED FINAL ANALYSIS",
        "=" * 78,
        "",
        f"Sampling rate              : {fs:.8f} Hz",
        f"Sampling step              : {dt:.8f} s",
        f"Samples                    : 2252",
        f"Causal envelope            : {ENVELOPE_SEC:.3f} s trailing raw magnitude",
        f"High-activity threshold    : Q_{HIGH_ACTIVITY_QUANTILE:.2f} = {threshold:.8g}",
        f"Global spectral peak       : {aux['f0_global_hz']:.8f} Hz",
        f"Relative ridge             : {RIDGE_REL}",
        f"Shrinkage alpha            : {SHRINK_ALPHA}",
        "Per-axis standardization   : none",
        "Forced stabilization       : none",
        f"Subset bootstrap           : {FINAL_N_BOOT} circular 5-s pair blocks",
        "",
        "POINT ESTIMATES",
        "-" * 78,
        point[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        ),
        "",
        "REGISTERED MOVING-WINDOW TEST",
        "-" * 78,
        (
            "Observed fraction rho(Gamma)<1<sigma_max(Gamma): "
            f"{registered['S_obs']:.8f}"
        ),
        (
            "5-s block-bootstrap 95% CI: "
            f"[{registered['bootstrap_ci95_low']:.8f}, "
            f"{registered['bootstrap_ci95_high']:.8f}]"
        ),
        (
            "Complex reduced windows: "
            f"{registered['n_complex']}/{registered['n_valid_windows']} "
            f"= {registered['fraction_complex']:.8f}"
        ),
        (
            "Time-varying normal null median [95% range]: "
            f"{registered['null_median']:.8f} "
            f"[{registered['null_ci95_low']:.8f}, "
            f"{registered['null_ci95_high']:.8f}]"
        ),
        f"Maximum null fraction      : {registered['null_max']:.8f}",
        f"One-sided Monte Carlo p    : {registered['p_one_sided']:.8f}",
        f"Normality numerical error  : {registered['max_normality_error']:.3e}",
        "",
        f"Rebuttal placement         : {placement}",
        f"Reason                     : {placement_text}",
        "",
        "HIGH-MINUS-LOW 5-S BOOTSTRAP CONTRASTS",
        "-" * 78,
        contrasts.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        ),
        "",
        "STATIONARY / PHASE NULLS FOR CLASS-CONDITIONED CONTRAST",
        "-" * 78,
        null_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        ),
        "",
        "SECONDARY CIRCULAR-SHIFT TIMING CONTROL",
        "-" * 78,
        timing_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        ),
        "",
        "INTERPRETATION",
        "-" * 78,
        (
            "R_A and R_B are both reported because the manuscript-level threshold "
            "convention remains an open coauthor decision. R_A is the rates-based "
            "continuous-time variance-response normalization. R_B=R_sigma is the "
            "exact discrete one-step normalization and, for stable Gamma, "
            "R_B>1 iff sigma_max(Gamma)>1."
        ),
        (
            "The obsolete fixed-frequency K=2, 1--8 Hz pipeline is not part of "
            "this script and is not used for any revised conclusion."
        ),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# CONSOLIDATED MAIN

def main() -> None:
    FINAL_OUTDIR.mkdir(parents=True, exist_ok=True)

    if not FINAL_INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {FINAL_INPUT_CSV.name}. Put the CSV beside this "
            "script without renaming it, or update FINAL_INPUT_CSV."
        )

    # Common data, variants and causal labels.
    t, X_raw, fs, dt = read_pushup_csv(FINAL_INPUT_CSV)
    variants, aux = build_variants(t, X_raw, fs)
    raw_env, high, activity_threshold = build_common_activity_labels(
        aux["X_raw_demeaned"],
        fs,
    )
    low_pair, high_pair = pair_labels_from_high(high)

    print(
        f"Loaded {FINAL_INPUT_CSV.name}: n={len(t)}, fs={fs:.6f} Hz, "
        f"dt={dt:.8f} s"
    )
    print(
        f"Causal raw-envelope Q85={activity_threshold:.8g}; "
        f"pair counts low={int(np.sum(low_pair))}, "
        f"high={int(np.sum(high_pair))}"
    )

    # A. Definitive class-conditioned point estimates.
    point = build_point_table_final(variants, high, dt)
    point.to_csv(
        FINAL_OUTDIR / "pushup_final_point_estimates.csv",
        index=False,
    )

    # B. 5-s circular-block bootstrap of original pair sequence.
    boot_parts = []
    for v in VARIANTS:
        print(f"\n5-s subset bootstrap: {v.name}")
        bd = bootstrap_variant_final(
            v.name,
            variants[v.name]["X"],
            high,
            fs,
            dt,
        )
        boot_parts.append(bd)

    boot = pd.concat(boot_parts, ignore_index=True)
    boot.to_csv(
        FINAL_OUTDIR / "pushup_final_subset_bootstrap_replicates.csv",
        index=False,
    )

    boot_summary, contrasts = summarize_bootstrap_final(boot)
    boot_summary.to_csv(
        FINAL_OUTDIR / "pushup_final_subset_bootstrap_summary.csv",
        index=False,
    )
    contrasts.to_csv(
        FINAL_OUTDIR / "pushup_final_subset_contrasts.csv",
        index=False,
    )

    final = merge_point_bootstrap_final(point, boot_summary)
    final.to_csv(
        FINAL_OUTDIR / "pushup_final_subset_table.csv",
        index=False,
    )
    write_final_latex_table(
        FINAL_OUTDIR / "pushup_final_subset_table.tex",
        final,
    )

    # C. Stationary VAR(1) / common-phase nulls.
    #    These are intentionally the same corrected controls as the
    #    defensible pipeline and test sigma_Gamma and R_B/R_sigma, not R_A.
    null_rep_parts = []
    null_sum_parts = []

    # Base observed table supplies the R1 column expected by the tested null
    # helper; R1 is exactly R_B/R_sigma.
    observed_by_variant = {}

    for v in VARIANTS:
        obs_base, _ = analyze_variant_observed(variants[v.name], high)
        observed_by_variant[v.name] = obs_base

        print(f"Stationary/phase nulls: {v.name}")
        reps, summ = null_controls_variant(
            variants[v.name],
            high,
            fs,
            obs_base,
        )
        null_rep_parts.append(reps)
        null_sum_parts.append(summ)

    null_reps = pd.concat(null_rep_parts, ignore_index=True)
    null_summary = pd.concat(null_sum_parts, ignore_index=True)

    null_reps.to_csv(
        FINAL_OUTDIR / "pushup_final_stationary_null_replicates.csv",
        index=False,
    )
    null_summary.to_csv(
        FINAL_OUTDIR / "pushup_final_stationary_null_summary.csv",
        index=False,
    )

    # D. Secondary timing control.
    timing_df = timing_nulls_all(variants, raw_env, fs)
    timing_df.to_csv(
        FINAL_OUTDIR / "pushup_final_timing_null.csv",
        index=False,
    )

    # E. Registered M3 moving-window statistic + time-varying normal null.
    registered = run_registered_tvnormal(
        aux["X_raw_demeaned"],
        raw_env,
        fs,
        FINAL_OUTDIR,
    )

    pd.DataFrame([registered]).to_csv(
        FINAL_OUTDIR / "pushup_registered_summary.csv",
        index=False,
    )

    # F. Figures and report.
    make_summary_figure(
        t=t,
        raw_env=raw_env,
        high=high,
        threshold=activity_threshold,
        final_table=final,
        null_summary=null_summary,
        outbase=FINAL_OUTDIR / "fig15_pushup_revised_summary",
    )

    write_final_report(
        FINAL_OUTDIR / "pushup_revised_final_report.txt",
        fs=fs,
        dt=dt,
        threshold=activity_threshold,
        point=point,
        final=final,
        contrasts=contrasts,
        null_summary=null_summary,
        timing_df=timing_df,
        registered=registered,
        aux=aux,
    )

    # Console audit.
    show = [
        "variant",
        "method",
        "regime",
        "n_pairs",
        "rhoF",
        "sigmaF",
        "rhoGamma",
        "sigmaGamma",
        "gain_over_rho",
        "K",
        "RA",
        "RB",
        "sigmaGamma_ci95_low",
        "sigmaGamma_ci95_high",
        "RA_ci95_low",
        "RA_ci95_high",
        "RB_ci95_low",
        "RB_ci95_high",
        "frac_rhoGamma_lt1_sigmaGamma_gt1",
        "frac_rhoF_ge1",
    ]

    print("\nFINAL PUSH-UP LOW/HIGH TABLE")
    print("=" * 120)
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 50,
        "display.width", 280,
        "display.float_format", lambda x: f"{x:.6g}",
    ):
        print(final[show].to_string(index=False))

    print("\nREGISTERED PLACEMENT STATISTIC")
    print("=" * 120)
    for key, value in registered.items():
        print(f"{key}: {value}")

    print("\nSaved outputs to:", FINAL_OUTDIR.resolve())


if __name__ == "__main__":
    main()
