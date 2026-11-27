#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Push-up definitive low/high subset table
========================================

Purpose
-------
Complete the second push-up calculation requested by Virgile.

For the SAME corrected causal low/high activity definition used in
fig15_pushup_final_defensible.py, report for each signal variant and M2/M3:

    rho(F)
    sigma_max(F)
    rho(Gamma)
    sigma_max(Gamma)
    K
    Option A: R_A  (continuous-time variance-kernel criterion)
    Option B: R_B  (exact discrete-time one-step criterion = R_sigma)
    sigma_max(Gamma) / rho(Gamma)

with:
    * exact low/high pair counts;
    * 5-s circular moving-block bootstrap confidence intervals;
    * fraction of bootstrap replicates with sigma_max(Gamma) > 1;
    * fraction with rho(Gamma) < 1 < sigma_max(Gamma);
    * low-to-high bootstrap contrasts.

This script does NOT change:
    * preprocessing;
    * causal class definition;
    * fitted operator;
    * M2/M3 plane extraction;
    * ridge/shrinkage;
    * coordinate scaling.

It imports those definitions directly from:
    fig15_pushup_final_defensible.py

Expected files in the same directory
------------------------------------
    fig15_pushup_final_defensible.py
    raw_samples_Pushup_duo_27Dec2025..csv

Outputs
-------
pushup_subset_final/
    pushup_subset_point_estimates.csv
    pushup_subset_bootstrap_replicates.csv
    pushup_subset_bootstrap_summary.csv
    pushup_subset_contrasts.csv
    pushup_subset_final_table.csv
    pushup_subset_final_table.tex
    pushup_subset_report.txt

Bootstrap definition
--------------------
The bootstrap is over the chronological sequence of ORIGINAL one-step pairs
(x_t, x_{t+1}) together with their causal activity label at t.

Circular contiguous blocks of length 5 s are sampled until the original
number of pairs is reached.  This preserves within-block serial dependence
and, importantly, never creates artificial x_t -> x_{t+1} transitions across
bootstrap block boundaries.

The activity labels are therefore treated as part of the observed design.
This interval quantifies uncertainty of the class-conditioned operator
contrast conditional on the recorded causal activity segmentation.

Option A
--------
For a stable complex pair of continuous-time rates
    mu_± = -gamma ± i omega,
obtained from the principal logarithm of the discrete reduced multipliers,
    Kc_A = sqrt(1 / (sqrt(1 + beta^2) - 1)),
    beta = |omega| / gamma,
    R_A = K / Kc_A.

For a stable real-rate pair mu1, mu2 < 0,
    Delta_rate = |(mu1-mu2)/(mu1+mu2)|,
    s = sqrt(1-Delta_rate^2),
    Kc_A = sqrt(s/(1-s)),
    R_A = K / Kc_A.

If the principal logarithm does not produce an admissible real-rate or
complex-conjugate continuous-time pair, R_A is reported as NaN rather than
silently extrapolated.

Option B
--------
For stable reduced multipliers lambda1, lambda2,
    Kc_B = sqrt((1-|lambda1|^2)(1-|lambda2|^2)) / |lambda1-lambda2|,
    R_B = K / Kc_B.

The imported final pipeline evaluates the equivalent invariant form
    R_B = d_F(Gamma) /
          sqrt((1-|lambda1|^2)(1-|lambda2|^2)).

For rho(Gamma)<1:
    R_B > 1  <=>  sigma_max(Gamma) > 1.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd


# PATHS / SETTINGS

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "fig15_pushup_final_defensible.py"
INPUT_CSV = ROOT.parent / "data" / "raw_samples_Pushup_duo_27Dec2025..csv"
OUTDIR = ROOT.parent / "results" / "pushup_subset_final"

# Requested dependence-preserving bootstrap
BOOT_BLOCK_SEC = 5.0
N_BOOT = 2000
BOOT_SEED = 20261011

EPS = 1e-12
RATE_TOL = 1e-9

SIGNAL_ORDER = [
    "raw_demeaned",
    "adaptive_phase_residual_1_15",
    "adaptive_phase_residual_1_24",
]

SIGNAL_LABEL = {
    "raw_demeaned": "Raw",
    "adaptive_phase_residual_1_15": "Adaptive 1--15 Hz",
    "adaptive_phase_residual_1_24": "Adaptive 1--24 Hz",
}


# IMPORT THE AUTHORITATIVE FINAL PUSH-UP PIPELINE

def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find {BASE_SCRIPT.name}. "
            "Place this script beside fig15_pushup_final_defensible.py."
        )

    spec = importlib.util.spec_from_file_location(
        "pushup_final_base",
        str(BASE_SCRIPT),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# OPTION A

def option_a_variance_ratio(
    Gamma: np.ndarray,
    K: float,
    dt: float,
) -> dict:
    """
    Continuous-time variance-kernel normalization.

    Uses the principal logarithm of the two discrete reduced multipliers.

    Returns:
        RA, KcA, spectrum_domain_A, gamma, omega, beta, Delta_rate
    """
    eig = np.linalg.eigvals(np.asarray(Gamma, complex))
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

    if not np.isfinite(rho) or rho >= 1.0 or rho <= 0.0:
        return out

    # Principal continuous-time rates.
    rates = np.log(eig) / float(dt)

    # Complex-conjugate rate pair
    imag_scale = max(1.0, float(np.max(np.abs(rates))))
    is_complex = np.max(np.abs(np.imag(rates))) > RATE_TOL * imag_scale

    if is_complex:
        # Require conjugate pair to numerical tolerance.
        if abs(rates[0] - np.conj(rates[1])) > 1e-7 * imag_scale:
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
        RA = float(K / KcA) if KcA > 0 else np.nan

        out.update({
            "RA": RA,
            "KcA": KcA,
            "A_domain": "complex_rate_pair",
            "gamma_A": gamma,
            "omega_A": omega,
            "beta_A": beta,
        })
        return out

    # Real continuous-time rate pair
    rates_real = np.real(rates)

    # Require stable continuous-time rates.
    if not np.all(rates_real < -RATE_TOL):
        return out

    denom_rate = float(rates_real[0] + rates_real[1])
    if abs(denom_rate) <= EPS:
        return out

    Delta = float(
        abs((rates_real[0] - rates_real[1]) / denom_rate)
    )

    # Real-pair Eq. (16b) domain.
    if Delta < 0.0 or Delta >= 1.0:
        return out

    s = float(np.sqrt(max(0.0, 1.0 - Delta**2)))
    denom = 1.0 - s

    if denom <= EPS:
        # Equal-rate limit -> infinite threshold -> R_A=0.
        KcA = np.inf
        RA = 0.0
    else:
        KcA = float(np.sqrt(s / denom))
        RA = float(K / KcA) if KcA > 0 else np.nan

    out.update({
        "RA": RA,
        "KcA": KcA,
        "A_domain": "real_rate_pair",
        "Delta_rate_A": Delta,
    })
    return out


# POINT ANALYSIS

def analyze_pair_set(
    base,
    X0: np.ndarray,
    X1: np.ndarray,
    method: str,
    dt: float,
) -> dict | None:
    fit = base.fit_operator_from_pairs(X0, X1)
    res = base.analyze_fit(fit, method)

    if res is None:
        return None

    A = option_a_variance_ratio(
        res["Gamma"],
        res["K"],
        dt,
    )

    rhoG = float(res["rhoGamma"])
    sigmaG = float(res["sigmaGamma"])

    gain_over_rho = (
        sigmaG / rhoG
        if rhoG > EPS
        else np.nan
    )

    return {
        "n_pairs": int(len(X0)),

        "rhoF": float(res["rhoF"]),
        "sigmaF": float(res["sigmaF"]),

        "rhoGamma": rhoG,
        "sigmaGamma": sigmaG,
        "gain_over_rho": gain_over_rho,

        "K": float(res["K"]),

        "RA": float(A["RA"]) if np.isfinite(A["RA"]) else np.nan,
        "KcA": float(A["KcA"]) if np.isfinite(A["KcA"]) else A["KcA"],
        "A_domain": A["A_domain"],
        "gamma_A": A["gamma_A"],
        "omega_A": A["omega_A"],
        "beta_A": A["beta_A"],
        "Delta_rate_A": A["Delta_rate_A"],

        "RB": float(res["R1"]) if np.isfinite(res["R1"]) else np.nan,

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


# CIRCULAR MOVING-BLOCK BOOTSTRAP OF ORIGINAL PAIRS

def circular_block_pair_indices(
    n_pairs: int,
    block_len_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block_len_pairs = max(
        1,
        min(int(block_len_pairs), int(n_pairs)),
    )

    out = []
    while len(out) < n_pairs:
        start = int(rng.integers(0, n_pairs))
        block = (
            start + np.arange(block_len_pairs)
        ) % n_pairs
        out.extend(block.tolist())

    return np.asarray(out[:n_pairs], dtype=int)


def bootstrap_variant(
    base,
    variant_name: str,
    X: np.ndarray,
    high: np.ndarray,
    fs: float,
    dt: float,
) -> pd.DataFrame:
    """
    Resample the chronological original pair sequence in 5-s circular blocks.

    Each resampled item carries:
        X_t, X_{t+1}, causal label at t.

    No artificial transition is created at a bootstrap block boundary.
    """
    X = np.asarray(X, float)

    X0_all = X[:-1]
    X1_all = X[1:]
    labels_all = np.asarray(high[:-1], bool)

    n_pairs = len(X0_all)
    block_len = max(
        1,
        int(round(BOOT_BLOCK_SEC * fs)),
    )

    rng = np.random.default_rng(
        BOOT_SEED
        + 1000 * SIGNAL_ORDER.index(variant_name)
    )

    rows = []

    for b in range(N_BOOT):
        idx = circular_block_pair_indices(
            n_pairs,
            block_len,
            rng,
        )

        labels = labels_all[idx]
        X0b = X0_all[idx]
        X1b = X1_all[idx]

        for regime, mask in (
            ("low", ~labels),
            ("high", labels),
        ):
            if int(np.sum(mask)) < 20:
                continue

            X0 = X0b[mask]
            X1 = X1b[mask]

            for method in ("M2", "M3"):
                res = analyze_pair_set(
                    base,
                    X0,
                    X1,
                    method,
                    dt,
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
            print(
                f"{variant_name}: bootstrap "
                f"{b+1}/{N_BOOT}"
            )

    return pd.DataFrame(rows)


# SUMMARIES

POINT_METRICS = [
    "rhoF",
    "sigmaF",
    "rhoGamma",
    "sigmaGamma",
    "gain_over_rho",
    "K",
    "RA",
    "RB",
]


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return (
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    )


def summarize_bootstrap(
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

        for metric in POINT_METRICS:
            vals = g[metric].to_numpy(float)
            lo, hi = ci95(vals)

            row[f"{metric}_boot_median"] = (
                float(np.nanmedian(vals))
                if np.any(np.isfinite(vals))
                else np.nan
            )
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        # Exact requested fractions.
        row["frac_sigmaGamma_gt1"] = float(
            np.mean(
                g["sigmaGamma"].to_numpy(float) > 1.0
            )
        )

        row["frac_rhoGamma_lt1_sigmaGamma_gt1"] = float(
            np.mean(
                (g["rhoGamma"].to_numpy(float) < 1.0)
                &
                (g["sigmaGamma"].to_numpy(float) > 1.0)
            )
        )

        row["frac_rhoF_ge1"] = float(
            np.mean(
                g["rhoF"].to_numpy(float) >= 1.0
            )
        )

        row["frac_RA_valid"] = float(
            np.mean(
                np.isfinite(g["RA"].to_numpy(float))
            )
        )

        summary_rows.append(row)

    # Paired low-high contrasts by bootstrap replicate.
    for (variant, method), g in boot.groupby(
        ["variant", "method"]
    ):
        low = g[g["regime"] == "low"].set_index("bootstrap")
        high = g[g["regime"] == "high"].set_index("bootstrap")

        common = low.index.intersection(high.index)

        for metric in [
            "rhoF",
            "sigmaF",
            "rhoGamma",
            "sigmaGamma",
            "gain_over_rho",
            "RA",
            "RB",
        ]:
            if len(common) == 0:
                continue

            delta = (
                high.loc[common, metric].to_numpy(float)
                -
                low.loc[common, metric].to_numpy(float)
            )

            delta = delta[np.isfinite(delta)]

            if len(delta) == 0:
                med = lo = hi = p_le_zero = np.nan
            else:
                med = float(np.median(delta))
                lo, hi = ci95(delta)

                # Descriptive bootstrap sign probability.
                p_le_zero = float(
                    np.mean(delta <= 0.0)
                )

            contrast_rows.append({
                "variant": variant,
                "method": method,
                "metric": metric,
                "n_boot_paired": int(len(delta)),
                "delta_high_minus_low_median": med,
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "bootstrap_fraction_delta_le_zero": p_le_zero,
            })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(contrast_rows),
    )


def build_final_table(
    point: pd.DataFrame,
    boot_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = point.merge(
        boot_summary,
        on=["variant", "method", "regime"],
        how="left",
        suffixes=("", "_boot"),
    )
    return out


# LATEX TABLE

def write_latex_table(
    path: Path,
    final: pd.DataFrame,
) -> None:
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

    for sig in SIGNAL_ORDER:
        g = final[final["variant"] == sig]
        first = True

        for method in ("M2", "M3"):
            for regime in ("low", "high"):
                r = g[
                    (g["method"] == method)
                    & (g["regime"] == regime)
                ]

                if r.empty:
                    continue

                rr = r.iloc[0]
                sig_label = (
                    SIGNAL_LABEL[sig]
                    if first
                    else ""
                )
                first = False

                def f3(v):
                    return (
                        "--"
                        if not np.isfinite(v)
                        else f"{v:.3f}"
                    )

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
            r"The activity class is defined causally from the raw acceleration "
            r"envelope and the same labels are used for all signal variants. "
            r"$R_A$ is the continuous-time variance-kernel normalization and "
            r"$R_B$ is the exact discrete-time one-step amplification "
            r"normalization. The final column is the fraction of 5-s "
            r"circular-block bootstrap replicates satisfying "
            r"$\rho(\Gamma)<1<\sigma_{\max}(\Gamma)$.}"
        ),
        r"\label{tab:pushup_final_subsets}",
        r"\end{table*}",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# TEXT REPORT

def write_report(
    path: Path,
    fs: float,
    dt: float,
    threshold: float,
    point: pd.DataFrame,
    final: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    lines = []

    lines.append("PUSH-UP DEFINITIVE LOW/HIGH SUBSET ANALYSIS")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Sampling rate              : {fs:.8f} Hz")
    lines.append(f"Sampling step              : {dt:.8f} s")
    lines.append(f"Causal envelope threshold  : {threshold:.8g}")
    lines.append(f"Bootstrap block length     : {BOOT_BLOCK_SEC:.3f} s")
    lines.append(f"Bootstrap replicates       : {N_BOOT}")
    lines.append("")
    lines.append(
        "Bootstrap unit: chronological original one-step pairs "
        "(x_t,x_{t+1},label_t) in circular 5-s blocks."
    )
    lines.append("")

    lines.append("POINT ESTIMATES")
    lines.append("-" * 78)

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

    lines.append(
        point[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )
    lines.append("")

    lines.append("5-S CIRCULAR BLOCK BOOTSTRAP")
    lines.append("-" * 78)

    cols2 = [
        "variant",
        "method",
        "regime",
        "n_boot_valid",
        "rhoF_ci95_low",
        "rhoF_ci95_high",
        "sigmaF_ci95_low",
        "sigmaF_ci95_high",
        "rhoGamma_ci95_low",
        "rhoGamma_ci95_high",
        "sigmaGamma_ci95_low",
        "sigmaGamma_ci95_high",
        "RA_ci95_low",
        "RA_ci95_high",
        "RB_ci95_low",
        "RB_ci95_high",
        "frac_sigmaGamma_gt1",
        "frac_rhoGamma_lt1_sigmaGamma_gt1",
        "frac_rhoF_ge1",
        "frac_RA_valid",
    ]

    cols2 = [
        c for c in cols2
        if c in final.columns
    ]

    lines.append(
        final[cols2].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )
    lines.append("")

    lines.append("BOOTSTRAP HIGH-MINUS-LOW CONTRASTS")
    lines.append("-" * 78)

    lines.append(
        contrasts.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )
    lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 78)
    lines.append(
        "For a stable reduced map, R_B>1 is exactly equivalent to "
        "sigma_max(Gamma)>1. The reported bootstrap fractions therefore "
        "show how often the class-conditioned reduced fit crosses the exact "
        "one-step amplification threshold under dependence-preserving "
        "resampling. R_A is retained as the complementary continuous-time "
        "variance-kernel normalization."
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# MAIN

def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV.name}. "
            "Place it beside this script."
        )

    base = load_base_module()

    # Read data and rebuild the exact corrected final signal variants.
    t, X_raw, fs, dt = base.read_pushup_csv(
        INPUT_CSV
    )

    variants, aux = base.build_variants(
        t,
        X_raw,
        fs,
    )

    raw_env, high, threshold = (
        base.build_common_activity_labels(
            aux["X_raw_demeaned"],
            fs,
        )
    )

    low_pair, high_pair = (
        base.pair_labels_from_high(high)
    )

    print(
        f"n={len(t)}, fs={fs:.6f} Hz, dt={dt:.8f} s"
    )
    print(
        f"Causal raw-envelope Q85={threshold:.8g}"
    )
    print(
        f"Exact pair counts: "
        f"low={int(np.sum(low_pair))}, "
        f"high={int(np.sum(high_pair))}"
    )
    print(
        f"5-s block length: "
        f"{int(round(BOOT_BLOCK_SEC*fs))} original pair indices"
    )

    # Point estimates.
    point_rows = []

    for sig in SIGNAL_ORDER:
        X = np.asarray(
            variants[sig]["X"],
            float,
        )

        X0_all = X[:-1]
        X1_all = X[1:]

        for regime, mask in (
            ("low", low_pair),
            ("high", high_pair),
        ):
            X0 = X0_all[mask]
            X1 = X1_all[mask]

            for method in ("M2", "M3"):
                res = analyze_pair_set(
                    base,
                    X0,
                    X1,
                    method,
                    dt,
                )

                if res is None:
                    raise RuntimeError(
                        f"Point fit failed: "
                        f"{sig}, {method}, {regime}"
                    )

                point_rows.append({
                    "variant": sig,
                    "variant_label": SIGNAL_LABEL[sig],
                    "method": method,
                    "regime": regime,
                    **res,
                })

    point = pd.DataFrame(
        point_rows
    )

    point.to_csv(
        OUTDIR /
        "pushup_subset_point_estimates.csv",
        index=False,
    )

    # Bootstrap.
    boot_parts = []

    for sig in SIGNAL_ORDER:
        print(
            "\nBOOTSTRAP:",
            sig
        )

        bd = bootstrap_variant(
            base=base,
            variant_name=sig,
            X=variants[sig]["X"],
            high=high,
            fs=fs,
            dt=dt,
        )

        boot_parts.append(bd)

    boot = pd.concat(
        boot_parts,
        ignore_index=True,
    )

    boot.to_csv(
        OUTDIR /
        "pushup_subset_bootstrap_replicates.csv",
        index=False,
    )

    boot_summary, contrasts = (
        summarize_bootstrap(boot)
    )

    boot_summary.to_csv(
        OUTDIR /
        "pushup_subset_bootstrap_summary.csv",
        index=False,
    )

    contrasts.to_csv(
        OUTDIR /
        "pushup_subset_contrasts.csv",
        index=False,
    )

    # Final merged table.
    final = build_final_table(
        point,
        boot_summary,
    )

    final.to_csv(
        OUTDIR /
        "pushup_subset_final_table.csv",
        index=False,
    )

    write_latex_table(
        OUTDIR /
        "pushup_subset_final_table.tex",
        final,
    )

    write_report(
        OUTDIR /
        "pushup_subset_report.txt",
        fs=fs,
        dt=dt,
        threshold=threshold,
        point=point,
        final=final,
        contrasts=contrasts,
    )

    # Console output.
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
        "RA",
        "RB",
        "sigmaGamma_ci95_low",
        "sigmaGamma_ci95_high",
        "RB_ci95_low",
        "RB_ci95_high",
        "frac_sigmaGamma_gt1",
        "frac_rhoGamma_lt1_sigmaGamma_gt1",
        "frac_rhoF_ge1",
    ]

    print(
        "\nFINAL LOW/HIGH TABLE"
    )
    print(
        "=" * 120
    )

    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 40,
        "display.width", 260,
        "display.float_format",
        lambda x: f"{x:.6g}",
    ):
        print(
            final[show].to_string(
                index=False
            )
        )

    print(
        "\nHIGH-MINUS-LOW BOOTSTRAP CONTRASTS"
    )
    print(
        "=" * 120
    )

    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 20,
        "display.width", 220,
        "display.float_format",
        lambda x: f"{x:.6g}",
    ):
        print(
            contrasts.to_string(
                index=False
            )
        )

    print(
        "\nSaved outputs to:",
        OUTDIR.resolve(),
    )


if __name__ == "__main__":
    main()
