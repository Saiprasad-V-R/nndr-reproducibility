#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Push-up registered placement statistic
=======================================

Purpose
-------
Compute the single pre-registered statistic proposed for deciding the
placement of the push-up example:

    fraction of 1-s moving windows (M3) satisfying

        rho(Gamma) < 1 < sigma_max(Gamma)

and compare that fraction with a time-varying NORMAL null whose
local variance follows the recording.

IMPORTANT
---------
This script uses:

    raw demeaned 3-axis acceleration
    no per-axis standardization
    no forced stabilization of fitted operators
    relative ridge = 3e-3
    isotropic shrinkage alpha = 0.04
    M3 only
    window = 1.0 s
    step = 0.1 s

The causal 0.5-s raw-acceleration envelope is computed and reported,
but it is NOT used to select moving windows. Thus the registered
moving-window statistic does not suffer from amplitude-selection bias.

Null construction
-----------------
For every observed moving window:

1. fit the same local one-step operator F;
2. construct a NORMAL operator with the SAME local eigenvalues:
   - real Schur decomposition F = U T U^T;
   - remove strictly upper Schur couplings;
   - replace each complex 2x2 block by the corresponding
     normal rotation-damping block;
3. choose isotropic innovation variance so that the frozen-time
   stationary covariance has the same TOTAL local variance as the
   observed window.

The sequence of these locally spectrum-matched normal operators is then
used as a piecewise time-varying VAR process. Every surrogate is run
through exactly the same 1-s / 0.1-s M3 moving-window analysis.

This null therefore preserves:
    - local spectral damping / oscillatory frequency;
    - local scalar variance profile;
    - sampling rate and window geometry;

but deliberately removes:
    - non-normal Schur coupling.

Primary test
------------
One-sided Monte Carlo:

    p = (1 + #{S_null >= S_obs}) / (B + 1)

Default B = 500.

Outputs
-------
pushup_tvnormal_registered/
    observed_windows.csv
    null_registered_fractions.csv
    registered_summary.txt
    registered_null_test.png
    registered_null_test.pdf

"""

from pathlib import Path

import numpy as np
import pandas as pd
import numpy.linalg as LA
import matplotlib.pyplot as plt

from scipy.linalg import schur, solve_discrete_lyapunov


# SETTINGS

INPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "raw_samples_Pushup_duo_27Dec2025..csv"
OUTDIR = Path(__file__).resolve().parents[1] / "results" / "pushup_tvnormal_registered"

# Registered moving-window statistic
WINDOW_SEC = 1.0
STEP_SEC = 0.1
METHOD = "M3"

# Causal activity envelope: descriptive only here
ENVELOPE_SEC = 0.5
HIGH_QUANTILE = 0.85

# Same operator estimator as corrected push-up analysis
RIDGE_REL = 3e-3
SHRINK_ALPHA = 0.04

# Numerical screening
MIN_PAIRS = 20
MIN_EFF_RANK = 2.20
COND_MAX = 2e4

# Null
N_NULL = 500
NULL_SEED = 20261009
BURNIN_SEC = 10.0

# Dependence-preserving interval for the observed fraction
BOOT_BLOCK_SEC = 5.0
N_BOOT = 2000
BOOT_SEED = 20261010

# Threshold tolerances
STABLE_TOL = 1e-10
GAIN_TOL = 1e-10

EPS = 1e-12


# BASIC UTILITIES

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def spectral_radius(A):
    return float(np.max(np.abs(LA.eigvals(A))))


def sigma_max(A):
    return float(LA.svd(A, compute_uv=False)[0])


def robust_fs(t):
    dt = np.diff(np.asarray(t, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Cannot infer sampling interval.")
    dt_med = float(np.median(dt))
    return 1.0 / dt_med, dt_med


def causal_trailing_mean(x, w):
    """Mean over samples [t-w+1, ..., t], never future samples."""
    x = np.asarray(x, float).ravel()
    w = max(1, int(w))

    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.maximum(0, i - w + 1)

    return (cs[i + 1] - cs[lo]) / (i - lo + 1)


def effective_rank_and_cond(X):
    s = LA.svd(X, compute_uv=False)

    if len(s) == 0 or np.max(s) < EPS:
        return 0.0, np.inf

    keep = s > EPS * np.max(s)
    s = s[keep]

    p = s / np.sum(s)
    erank = float(np.exp(-np.sum(p * np.log(p + EPS))))
    cond = float(np.max(s) / max(np.min(s), EPS))

    return erank, cond


# OPERATOR FIT

def fit_local_operator(X):
    """
    Fit x_{t+1} = F x_t in one window.

    No channel standardization.
    No forced stabilization.
    """
    X = np.asarray(X, float)

    X0 = X[:-1]
    X1 = X[1:]

    if len(X0) < MIN_PAIRS:
        return None

    erank, cond = effective_rank_and_cond(X0)

    if erank < MIN_EFF_RANK or cond > COND_MAX:
        return None

    d = X.shape[1]

    C = X0.T @ X0

    lam = RIDGE_REL * float(np.trace(C) / d)

    try:
        Fhat = LA.solve(
            C + lam * np.eye(d),
            X0.T @ X1
        ).T
    except LA.LinAlgError:
        return None

    # Trace-preserving isotropic shrinkage
    mu = float(np.trace(Fhat) / d)

    F = (
        (1.0 - SHRINK_ALPHA) * Fhat
        + SHRINK_ALPHA * mu * np.eye(d)
    )

    return {
        "F": F,
        "rhoF": spectral_radius(F),
        "sigmaF": sigma_max(F),
        "ridge_lambda": lam,
        "eff_rank": erank,
        "cond": cond,
    }


# M3 REDUCED PLANE

def m3_plane(F):
    """
    M3: extreme eigenvectors of
        C = F F^T - F^T F.

    Only the plane is needed here; its internal orientation
    does not change rho(Gamma), sigma_max(Gamma), or R_sigma.
    """

    C = F @ F.T - F.T @ F

    evals, evecs = LA.eigh(C)

    i_min = int(np.argmin(evals))
    i_max = int(np.argmax(evals))

    if max(abs(evals[i_min]), abs(evals[i_max])) < EPS:
        return None

    Q = np.column_stack([
        evecs[:, i_max],
        evecs[:, i_min]
    ])

    Q, _ = LA.qr(Q)

    return Q[:, :2]


def reduced_metrics(F, Q):
    Gamma = Q.T @ F @ Q

    eig = LA.eigvals(Gamma)

    rhoG = float(np.max(np.abs(eig)))
    sigmaG = sigma_max(Gamma)

    # Henrici departure
    dF2 = (
        float(LA.norm(Gamma, "fro") ** 2)
        - float(np.sum(np.abs(eig) ** 2))
    )

    dF = float(np.sqrt(max(dF2, 0.0)))

    denom = float(
        np.prod(1.0 - np.abs(eig) ** 2)
    )

    if rhoG < 1.0 - STABLE_TOL and denom > EPS:
        R_sigma = dF / np.sqrt(denom)
    else:
        R_sigma = np.nan

    complex_pair = bool(
        np.max(np.abs(np.imag(eig)))
        >
        1e-9 * max(1.0, np.max(np.abs(eig)))
    )

    registered_event = bool(
        rhoG < 1.0 - STABLE_TOL
        and sigmaG > 1.0 + GAIN_TOL
    )

    return {
        "Gamma": Gamma,
        "rhoGamma": rhoG,
        "sigmaGamma": sigmaG,
        "R_sigma": R_sigma,
        "complex_pair": complex_pair,
        "registered_event": registered_event,
        "eig1_real": float(np.real(eig[0])),
        "eig1_imag": float(np.imag(eig[0])),
        "eig2_real": float(np.real(eig[1])),
        "eig2_imag": float(np.imag(eig[1])),
    }


# MOVING-WINDOW ANALYSIS

def moving_window_analysis(X, fs, causal_env=None):
    """
    Registered analysis:
        W = 1 s
        step = 0.1 s
        M3 only

    No amplitude-based window selection.
    """

    n = len(X)

    W = int(round(WINDOW_SEC * fs))
    step = max(1, int(round(STEP_SEC * fs)))

    rows = []

    starts = np.arange(
        0,
        n - W + 1,
        step,
        dtype=int
    )

    for j, s in enumerate(starts):

        e = s + W

        Xw = X[s:e]

        fit = fit_local_operator(Xw)

        if fit is None:
            continue

        Q = m3_plane(fit["F"])

        if Q is None:
            continue

        red = reduced_metrics(
            fit["F"],
            Q
        )

        # local total variance
        Xwc = Xw - np.mean(Xw, axis=0, keepdims=True)

        local_total_variance = float(
            np.mean(np.sum(Xwc ** 2, axis=1))
        )

        row = {
            "window_id": j,
            "start_sample": int(s),
            "end_sample": int(e - 1),
            "center_sample": int(round((s + e - 1) / 2)),
            "start_sec": s / fs,
            "end_sec": (e - 1) / fs,
            "center_sec": 0.5 * (s + e - 1) / fs,

            "n_samples": len(Xw),

            "rhoF": fit["rhoF"],
            "sigmaF": fit["sigmaF"],

            "rhoGamma": red["rhoGamma"],
            "sigmaGamma": red["sigmaGamma"],
            "R_sigma": red["R_sigma"],

            "complex_pair": int(red["complex_pair"]),
            "registered_event": int(red["registered_event"]),

            "eig1_real": red["eig1_real"],
            "eig1_imag": red["eig1_imag"],
            "eig2_real": red["eig2_real"],
            "eig2_imag": red["eig2_imag"],

            "local_total_variance": local_total_variance,

            "eff_rank": fit["eff_rank"],
            "cond": fit["cond"],
        }

        if causal_env is not None:
            envw = causal_env[s:e]

            row["causal_envelope_mean"] = float(
                np.mean(envw)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# NORMAL OPERATOR WITH SAME SPECTRUM

def spectrum_matched_normal_operator(F):
    """
    Construct a REAL NORMAL matrix with the same eigenvalues as F.

    Real Schur:
        F = U T U^T

    Then:
        * discard strictly upper Schur couplings;
        * retain real 1x1 eigenvalue blocks;
        * replace each complex 2x2 block with
              [[a, +omega],
               [-omega, a]]
          having eigenvalues a ± i omega.

    The resulting operator is normal by construction.
    """

    T, U = schur(F, output="real")

    n = T.shape[0]

    Tn = np.zeros_like(T)

    i = 0

    while i < n:

        # Complex 2x2 Schur block
        if (
            i < n - 1
            and abs(T[i + 1, i]) > 1e-10
        ):

            block = T[i:i + 2, i:i + 2]

            ev = LA.eigvals(block)

            # conjugate pair
            a = float(np.real(ev[0]))
            omega = float(abs(np.imag(ev[0])))

            Tn[i:i + 2, i:i + 2] = np.array([
                [a, omega],
                [-omega, a],
            ])

            i += 2

        else:
            # real eigenvalue
            Tn[i, i] = T[i, i]
            i += 1

    F_normal = U @ Tn @ U.T

    # numerical cleanup
    F_normal = np.real_if_close(F_normal).astype(float)

    normality_error = LA.norm(
        F_normal @ F_normal.T
        -
        F_normal.T @ F_normal,
        "fro"
    )

    return F_normal, float(normality_error)


# LOCAL VARIANCE MATCHING

def isotropic_noise_for_target_variance(
    F_normal,
    target_total_variance,
):
    """
    For

        x_{t+1} = F_normal x_t + eps_t,
        Cov(eps_t) = q I,

    solve P_unit from

        P_unit = F P_unit F^T + I.

    Because covariance scales linearly with q,

        P(q) = q P_unit.

    Choose q so that

        trace(P(q)) = target_total_variance.

    This preserves the local SCALAR variance while retaining
    isotropic forcing and therefore does not import a
    non-normal covariance geometry from the data.
    """

    rho = spectral_radius(F_normal)

    if rho >= 1.0 - STABLE_TOL:
        return np.nan

    try:
        P_unit = solve_discrete_lyapunov(
            F_normal,
            np.eye(F_normal.shape[0])
        )
    except Exception:
        return np.nan

    tr = float(np.trace(P_unit))

    if not np.isfinite(tr) or tr <= EPS:
        return np.nan

    q = float(target_total_variance / tr)

    if not np.isfinite(q) or q <= 0:
        return np.nan

    return q


# BUILD TIME-VARYING NORMAL NULL MODEL

def build_tv_normal_schedule(
    X,
    fs,
    observed_windows,
):
    """
    Construct one locally normal spectrum-matched operator
    and isotropic innovation level per observed window.

    Invalid/unstable local windows are not used to define the
    null schedule; the nearest valid local model is used.
    """

    model_centers = []
    F_models = []
    q_models = []
    normality_errors = []

    for _, row in observed_windows.iterrows():

        s = int(row["start_sample"])
        e = int(row["end_sample"]) + 1

        fit = fit_local_operator(X[s:e])

        if fit is None:
            continue

        F_normal, normality_error = (
            spectrum_matched_normal_operator(
                fit["F"]
            )
        )

        # Null is explicitly stable normal dynamics
        if spectral_radius(F_normal) >= 1.0 - STABLE_TOL:
            continue

        q = isotropic_noise_for_target_variance(
            F_normal,
            float(row["local_total_variance"])
        )

        if not np.isfinite(q):
            continue

        model_centers.append(
            int(row["center_sample"])
        )

        F_models.append(F_normal)
        q_models.append(q)
        normality_errors.append(normality_error)

    if len(F_models) == 0:
        raise RuntimeError(
            "No valid stable local normal models."
        )

    model_centers = np.asarray(
        model_centers,
        dtype=int
    )

    F_models = np.asarray(F_models)
    q_models = np.asarray(q_models)

    print(
        "Time-varying normal null:"
        f" {len(F_models)} local models"
    )

    print(
        "maximum normality error =",
        np.max(normality_errors)
    )

    # Assign every sample transition t -> t+1
    # to the nearest local model center.
    transition_samples = np.arange(
        len(X) - 1
    )

    schedule_idx = np.empty(
        len(transition_samples),
        dtype=int
    )

    for i, t in enumerate(transition_samples):

        k = np.searchsorted(
            model_centers,
            t
        )

        if k == 0:
            schedule_idx[i] = 0

        elif k >= len(model_centers):
            schedule_idx[i] = len(model_centers) - 1

        else:
            left = k - 1
            right = k

            if (
                abs(t - model_centers[left])
                <=
                abs(t - model_centers[right])
            ):
                schedule_idx[i] = left
            else:
                schedule_idx[i] = right

    return (
        F_models,
        q_models,
        schedule_idx,
        model_centers,
    )


# SIMULATE TIME-VARYING NORMAL NULL

def simulate_tv_normal(
    n,
    fs,
    F_models,
    q_models,
    schedule_idx,
    rng,
):
    d = F_models.shape[1]

    X = np.zeros(
        (n, d),
        dtype=float
    )

    # Burn in with the first local model.
    F0 = F_models[0]
    q0 = q_models[0]

    burn = max(
        100,
        int(round(BURNIN_SEC * fs))
    )

    x = np.zeros(d)

    for _ in range(burn):
        eps = rng.normal(
            0.0,
            np.sqrt(q0),
            size=d
        )

        x = F0 @ x + eps

    X[0] = x

    for t in range(n - 1):

        k = schedule_idx[t]

        F = F_models[k]
        q = q_models[k]

        eps = rng.normal(
            0.0,
            np.sqrt(q),
            size=d
        )

        X[t + 1] = (
            F @ X[t] + eps
        )

    # Remove residual global mean exactly as for observed data
    X = X - X.mean(
        axis=0,
        keepdims=True
    )

    return X


# REGISTERED STATISTIC

def registered_fraction(window_df):
    if len(window_df) == 0:
        return np.nan

    return float(
        np.mean(
            window_df[
                "registered_event"
            ].to_numpy(float)
        )
    )


# 5-S CIRCULAR BLOCK BOOTSTRAP OF WINDOW INDICATOR

def circular_block_bootstrap_fraction(
    indicator,
    block_windows,
    n_boot=N_BOOT,
):
    """
    Dependence-preserving CI for the registered fraction.

    The binary moving-window indicator is itself the statistic
    being averaged. We therefore block-bootstrap this sequence
    using 5-s blocks in window time.

    With a 0.1-s window step, 5 s corresponds to ~50 windows.
    """

    indicator = np.asarray(
        indicator,
        dtype=float
    )

    n = len(indicator)

    rng = np.random.default_rng(
        BOOT_SEED
    )

    values = []

    for _ in range(n_boot):

        out = []

        while len(out) < n:

            start = int(
                rng.integers(0, n)
            )

            block = (
                start
                +
                np.arange(block_windows)
            ) % n

            out.extend(
                indicator[block].tolist()
            )

        arr = np.asarray(
            out[:n],
            float
        )

        values.append(
            float(np.mean(arr))
        )

    values = np.asarray(values)

    return (
        values,
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


# FIGURE

def make_figure(
    null_values,
    observed_fraction,
    outdir,
):
    fig, ax = plt.subplots(
        figsize=(6.0, 4.0)
    )

    ax.hist(
        null_values,
        bins=30,
        density=True,
        alpha=0.65
    )

    ax.axvline(
        observed_fraction,
        linestyle="--",
        linewidth=2.0,
        label=(
            "Observed = "
            f"{observed_fraction:.3f}"
        )
    )

    ax.set_xlabel(
        "Fraction of 1-s windows with "
        r"$\rho(\Gamma)<1<\sigma_{\max}(\Gamma)$"
    )

    ax.set_ylabel(
        "Null density"
    )

    ax.set_title(
        "Time-varying normal null"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        outdir /
        "registered_null_test.png",
        dpi=500,
        bbox_inches="tight"
    )

    fig.savefig(
        outdir /
        "registered_null_test.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


# MAIN

def main():

    ensure_dir(OUTDIR)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}"
        )

    # Load

    df = pd.read_csv(
        INPUT_CSV
    )

    required = [
        "timestamp",
        "accX",
        "accY",
        "accZ",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    t_abs = df[
        "timestamp"
    ].to_numpy(float)

    fs, dt = robust_fs(
        t_abs
    )

    X = df[
        ["accX", "accY", "accZ"]
    ].to_numpy(float)

    # Physical raw coordinates:
    # demean only, no scaling.
    X = X - X.mean(
        axis=0,
        keepdims=True
    )

    n = len(X)

    print(
        f"n={n}, fs={fs:.6f} Hz, "
        f"dt={dt:.8f} s"
    )

    # Causal activity envelope
    # descriptive only for this registered statistic

    magnitude = LA.norm(
        X,
        axis=1
    )

    env_w = max(
        1,
        int(round(
            ENVELOPE_SEC * fs
        ))
    )

    causal_env = causal_trailing_mean(
        magnitude,
        env_w
    )

    env_threshold = float(
        np.quantile(
            causal_env,
            HIGH_QUANTILE
        )
    )

    print(
        "causal raw-envelope Q85 =",
        env_threshold
    )

    # Observed moving-window statistic

    observed = moving_window_analysis(
        X,
        fs,
        causal_env=causal_env
    )

    if observed.empty:
        raise RuntimeError(
            "No valid observed windows."
        )

    observed.to_csv(
        OUTDIR /
        "observed_windows.csv",
        index=False
    )

    S_obs = registered_fraction(
        observed
    )

    n_complex = int(
        observed["complex_pair"].sum()
    )

    n_full_unstable = int(
        np.sum(
            observed["rhoF"] >= 1.0
        )
    )

    print(
        "\nREGISTERED OBSERVED STATISTIC"
    )

    print(
        "valid windows:",
        len(observed)
    )

    print(
        "complex reduced spectra:",
        n_complex,
        "/",
        len(observed),
        f"= {n_complex/len(observed):.3f}"
    )

    print(
        "full fitted rho(F)>=1:",
        n_full_unstable
    )

    print(
        "fraction rho(Gamma)<1<"
        "sigma_max(Gamma):",
        f"{S_obs:.6f}"
    )

    # 5-s block-bootstrap CI

    block_windows = max(
        1,
        int(
            round(
                BOOT_BLOCK_SEC /
                STEP_SEC
            )
        )
    )

    boot_values, boot_lo, boot_hi = (
        circular_block_bootstrap_fraction(
            observed[
                "registered_event"
            ].to_numpy(),
            block_windows,
            N_BOOT
        )
    )

    print(
        "5-s block-bootstrap 95% CI:",
        f"[{boot_lo:.6f}, "
        f"{boot_hi:.6f}]"
    )

    # Build locally spectrum-matched
    # time-varying NORMAL null

    (
        F_models,
        q_models,
        schedule_idx,
        model_centers,
    ) = build_tv_normal_schedule(
        X,
        fs,
        observed
    )

    # Null replicates

    rng = np.random.default_rng(
        NULL_SEED
    )

    null_rows = []

    for b in range(N_NULL):

        Xnull = simulate_tv_normal(
            n=n,
            fs=fs,
            F_models=F_models,
            q_models=q_models,
            schedule_idx=schedule_idx,
            rng=rng,
        )

        nulldf = moving_window_analysis(
            Xnull,
            fs,
            causal_env=None
        )

        S_null = registered_fraction(
            nulldf
        )

        null_rows.append({
            "replicate": b,
            "registered_fraction": S_null,
            "n_valid_windows": len(nulldf),
            "fraction_complex": (
                float(
                    nulldf[
                        "complex_pair"
                    ].mean()
                )
                if len(nulldf)
                else np.nan
            ),
            "fraction_full_rho_ge_1": (
                float(
                    np.mean(
                        nulldf[
                            "rhoF"
                        ] >= 1.0
                    )
                )
                if len(nulldf)
                else np.nan
            ),
        })

        if (
            (b + 1) % 25 == 0
            or b == 0
        ):
            print(
                f"null {b+1:4d}/"
                f"{N_NULL}: "
                f"S={S_null:.4f}"
            )

    null_df = pd.DataFrame(
        null_rows
    )

    null_df.to_csv(
        OUTDIR /
        "null_registered_fractions.csv",
        index=False
    )

    valid_null = (
        null_df[
            "registered_fraction"
        ]
        .dropna()
        .to_numpy(float)
    )

    # One-sided preregistered test
    p = float(
        (
            1
            +
            np.sum(
                valid_null >= S_obs
            )
        )
        /
        (
            1
            +
            len(valid_null)
        )
    )

    null_median = float(
        np.median(valid_null)
    )

    null_lo = float(
        np.percentile(
            valid_null,
            2.5
        )
    )

    null_hi = float(
        np.percentile(
            valid_null,
            97.5
        )
    )

    print(
        "\nTIME-VARYING NORMAL NULL"
    )

    print(
        "valid null replicates:",
        len(valid_null)
    )

    print(
        "null median fraction:",
        f"{null_median:.6f}"
    )

    print(
        "null 95% range:",
        f"[{null_lo:.6f}, "
        f"{null_hi:.6f}]"
    )

    print(
        "one-sided Monte Carlo p:",
        f"{p:.8f}"
    )

    # Placement rule from rebuttal

    if p < 0.05:
        decision = (
            "REGISTERED STATISTIC SIGNIFICANT: "
            "the moving-window excess survives "
            "the time-varying normal null."
        )
    else:
        decision = (
            "REGISTERED STATISTIC NOT SIGNIFICANT: "
            "under the pre-specified rule the "
            "push-up example should move to the "
            "Supplementary Material."
        )

    print(
        "\n",
        decision
    )

    # Text report

    report = f"""
PUSH-UP REGISTERED PLACEMENT TEST
=================================

Input
-----
n samples                  : {n}
sampling rate              : {fs:.8f} Hz
sampling step              : {dt:.8f} s
signal                     : raw demeaned acceleration
channel scaling            : none
forced stabilization       : none

Moving-window analysis
----------------------
method                     : M3
window                     : {WINDOW_SEC:.3f} s
step                       : {STEP_SEC:.3f} s
relative ridge             : {RIDGE_REL}
shrinkage alpha            : {SHRINK_ALPHA}

Causal envelope
---------------
window                     : {ENVELOPE_SEC:.3f} s
high threshold             : Q{HIGH_QUANTILE:.2f}
threshold value            : {env_threshold:.8g}

NOTE: the envelope is descriptive only for the registered
moving-window statistic; moving windows are not selected by amplitude.

Observed
--------
valid windows               : {len(observed)}
complex reduced windows     : {n_complex}
fraction complex            : {n_complex/len(observed):.8f}
full fits with rho(F)>=1    : {n_full_unstable}

registered statistic:
fraction rho(Gamma)<1<sigma_max(Gamma)
                            : {S_obs:.8f}

5-s circular block-bootstrap
95% interval                : [{boot_lo:.8f}, {boot_hi:.8f}]
bootstrap replicates        : {N_BOOT}
block length                : {BOOT_BLOCK_SEC:.3f} s
                              ({block_windows} moving-window samples)

Time-varying normal null
------------------------
null replicates             : {len(valid_null)}
local normal models         : {len(F_models)}
null median fraction        : {null_median:.8f}
null 95% range              : [{null_lo:.8f}, {null_hi:.8f}]

one-sided Monte Carlo test:
p = (1 + #{{S_null >= S_obs}})/(1+B)
                            : {p:.8f}

Decision under registered rule
------------------------------
{decision}

Null definition
---------------
Each local observed operator is replaced by a normal operator
with the same real-Schur eigenvalue blocks. Strictly upper
Schur couplings are removed. Complex blocks are replaced by
normal damped-rotation blocks. Isotropic innovations are
scaled so that the frozen-time stationary covariance has the
same total variance as the corresponding observed window.

Thus the null retains local spectral damping/rotation and the
local scalar variance profile but removes non-normal operator
geometry.
"""

    (
        OUTDIR /
        "registered_summary.txt"
    ).write_text(
        report,
        encoding="utf-8"
    )

    print(
        "\n",
        report
    )

    # Plot

    make_figure(
        valid_null,
        S_obs,
        OUTDIR
    )

    print(
        "\nSaved outputs to:",
        OUTDIR.resolve()
    )


if __name__ == "__main__":
    main()
