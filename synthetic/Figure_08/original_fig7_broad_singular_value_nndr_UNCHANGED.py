# -*- coding: utf-8 -*-
"""
Broad-singular-value NNDR validation figure.

Purpose
-------
Tests whether the 2D NNDR reduction survives when the full fitted operator has
an intentionally broad singular-value hierarchy. A controlled 2D non-normal
block is embedded in a high-dimensional stable background whose singular values
follow a power law. The reference plane is known by construction.

Outputs
-------
outputs_broad_singular_value/fig_broad_singular_value_nndr.pdf
outputs_broad_singular_value/fig_broad_singular_value_nndr.png
outputs_broad_singular_value/broad_singular_value_metrics.csv

Spyder usage
------------
%runfile /path/to/fig_broad_singular_value_nndr.py --wdir /path/to/folder
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# 0) Configuration

OUTDIR = "outputs_broad_singular_value"

N = 60
M = 10
T_TRAIN = 160
SIGMA_NOISE = 0.035
RIDGE_LAM = 1e-4


N = 80
M = 18
T_TRAIN = 220
SEEDS = np.arange(8)


# Eigenvalue splitting in the embedded 2D block.
DELTA_TARGET = 0.35
A_MEAN = 0.70
# If Delta=(a-b)/(a+b) and a+b=2*A_MEAN, then a=A_MEAN*(1+Delta), b=A_MEAN*(1-Delta).
A_EIG = A_MEAN * (1.0 + DELTA_TARGET)
B_EIG = A_MEAN * (1.0 - DELTA_TARGET)

# Test multiple amplification levels and multiple background power-law spectra.
R_TARGETS = np.array([1.0, 3.0, 10.0, 30.0], dtype=float)
POWER_EXPONENTS = np.array([0.25, 0.75, 1.25, 1.75, 2.25], dtype=float)
#SEEDS = np.arange(4, dtype=int)

# Background amplitude below the embedded block spectral scale.
BACKGROUND_SCALE = 0.74


# 1) Plot style

def set_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.25,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 500,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.86,
        "legend.edgecolor": "none",
    })


def format_axis(ax, nbx=5, nby=5, logx=False, logy=False) -> None:
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    if not logx:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbx))
    if not logy:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nby))


def add_panel_label(ax, label: str, x=-0.16, y=1.05) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", ha="left", va="bottom")


# 2) Linear algebra and NNDR diagnostics

def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q


def Kc_from_Delta(Delta: float) -> float:
    Delta = float(np.clip(Delta, 0.0, 1.0 - 1e-12))
    s = np.sqrt(1.0 - Delta**2)
    return float(np.sqrt(s / (1.0 - s + 1e-15)))


def twoD_metrics_from_Gamma(Gamma: np.ndarray) -> Tuple[float, float, float, float]:
    eigvals, eigvecs = np.linalg.eig(Gamma)
    lam1, lam2 = eigvals[0], eigvals[1]
    denom = lam1 + lam2
    Delta = 1.0 if abs(denom) < 1e-15 else abs((lam1 - lam2) / denom)
    Delta = float(np.clip(np.real_if_close(Delta), 0.0, 1.0 - 1e-12))

    p1 = eigvecs[:, 0] / (np.linalg.norm(eigvecs[:, 0]) + 1e-15)
    p2 = eigvecs[:, 1] / (np.linalg.norm(eigvecs[:, 1]) + 1e-15)
    c = abs(np.vdot(p1, p2))
    c = float(np.clip(np.real_if_close(c), 0.0, 1.0 - 1e-12))
    kappa2D = float(np.sqrt((1.0 + c) / (1.0 - c)))
    K = float(0.5 * (kappa2D - 1.0 / kappa2D))
    R = float(K / (Kc_from_Delta(Delta) + 1e-15))
    return Delta, kappa2D, K, R


def principal_angle_deg(Q_hat: np.ndarray, Q_star: np.ndarray) -> float:
    s = np.linalg.svd(Q_hat.T @ Q_star, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.degrees(np.arccos(s[-1])))


def orthonormalize_pair(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(np.column_stack([r, n]))
    return Q[:, :2]


def extract_nr_by_optimization(A: np.ndarray, iters: int = 350, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_dim = A.shape[0]
    v = rng.standard_normal(n_dim)
    v /= np.linalg.norm(v) + 1e-15
    for _ in range(iters):
        Av = A @ v
        Av = Av - v * (v @ Av)
        nu = np.linalg.norm(Av)
        if nu < 1e-12:
            v = rng.standard_normal(n_dim)
            v /= np.linalg.norm(v) + 1e-15
            continue
        u = Av / nu
        Atu = A.T @ u
        Atu = Atu - u * (u @ Atu)
        nv = np.linalg.norm(Atu)
        if nv < 1e-12:
            v = rng.standard_normal(n_dim)
            v /= np.linalg.norm(v) + 1e-15
            continue
        v = Atu / nv
    n = v
    r_raw = (np.eye(n_dim) - np.outer(n, n)) @ (A @ n)
    r = r_raw / (np.linalg.norm(r_raw) + 1e-15)
    return n, r


def method2_optimization(A: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    n, r = extract_nr_by_optimization(A, seed=seed)
    Q = orthonormalize_pair(r, n)
    Gamma = Q.T @ A @ Q
    Delta, kappa2D, K, R = twoD_metrics_from_Gamma(Gamma)
    return {"method": "M2", "Q": Q, "Gamma": Gamma, "Delta": Delta, "kappa2D": kappa2D, "K": K, "R": R}


def method3_commutator(A: np.ndarray) -> Dict[str, Any]:
    B = A @ A.T - A.T @ A
    evals, evecs = np.linalg.eigh(B)
    Q0 = np.column_stack([evecs[:, -1], evecs[:, 0]])
    # Order the plane internally using the same directed-coupling problem in 2D.
    G0 = Q0.T @ A @ Q0
    n2, r2 = extract_nr_by_optimization(G0, iters=250, seed=0)
    Q = orthonormalize_pair(Q0 @ r2, Q0 @ n2)
    Gamma = Q.T @ A @ Q
    Delta, kappa2D, K, R = twoD_metrics_from_Gamma(Gamma)
    return {"method": "M3", "Q": Q, "Gamma": Gamma, "Delta": Delta, "kappa2D": kappa2D, "K": K, "R": R,
            "comm_pos": float(evals[-1]), "comm_neg": float(evals[0])}


# 3) Broad-spectrum operator construction and data simulation

def make_2d_block(delta: float, R_target: float) -> Tuple[np.ndarray, float, float, float]:
    a = A_MEAN * (1.0 + delta)
    b = A_MEAN * (1.0 - delta)
    Kc = Kc_from_Delta(delta)
    q = float(R_target * Kc)
    B = np.array([[a, q * (a - b)],
                  [0.0, b]], dtype=float)
    return B, a, b, q


def background_diag(n_bg: int, exponent: float, scale: float = BACKGROUND_SCALE) -> np.ndarray:
    j = np.arange(1, n_bg + 1, dtype=float)
    vals = scale * j ** (-float(exponent))
    return vals


def build_operator(n: int, exponent: float, R_target: float, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    O = random_orthogonal(n, rng)
    B2, a, b, q = make_2d_block(DELTA_TARGET, R_target)
    bg = background_diag(n - 2, exponent)
    C = np.zeros((n, n), dtype=float)
    C[:2, :2] = B2
    C[2:, 2:] = np.diag(bg)
    F = O @ C @ O.T
    Q_star = O[:, :2]
    Gamma_star = Q_star.T @ F @ Q_star
    Delta_star, kappa_star, K_star, R_star = twoD_metrics_from_Gamma(Gamma_star)
    svals = np.linalg.svd(F, compute_uv=False)
    return {"F": F, "Q_star": Q_star, "Gamma_star": Gamma_star, "svals": svals,
            "a": a, "b": b, "q": q, "Delta_star": Delta_star,
            "kappa_star": kappa_star, "K_star": K_star, "R_star": R_star,
            "bg": bg, "O": O}


def simulate_var(F: np.ndarray, M: int, T: int, sigma_noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = F.shape[0]
    X_list, Y_list = [], []
    for _ in range(M):
        x = np.zeros(n)
        Xin = np.zeros((n, T))
        Yout = np.zeros((n, T))
        for t in range(T):
            Xin[:, t] = x
            x = F @ x + sigma_noise * rng.standard_normal(n)
            Yout[:, t] = x
        X_list.append(Xin)
        Y_list.append(Yout)
    return np.concatenate(X_list, axis=1), np.concatenate(Y_list, axis=1)


def ridge_fit_F(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    n = X.shape[0]
    return (np.linalg.solve((X @ X.T + lam * np.eye(n)).T, (Y @ X.T).T)).T


# 4) Benchmark and plotting

def run_benchmark() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exponent in POWER_EXPONENTS:
        for R_target in R_TARGETS:
            for seed in SEEDS:
                op = build_operator(N, exponent, R_target, seed=10_000 + 100 * seed + int(100 * exponent))
                F = op["F"]
                Q_star = op["Q_star"]
                X, Y = simulate_var(F, M=M, T=T_TRAIN, sigma_noise=SIGMA_NOISE,
                                    seed=200_000 + 1000 * seed + int(100 * exponent) + int(R_target))
                Fhat = ridge_fit_F(X, Y, RIDGE_LAM)
                rel_err = float(np.linalg.norm(Fhat - F, ord="fro") / (np.linalg.norm(F, ord="fro") + 1e-15))
                cond_F = float(np.linalg.cond(F))
                sv_ratio_10 = float(op["svals"][0] / (op["svals"][9] + 1e-15)) if len(op["svals"]) > 9 else np.nan

                for extractor in (method2_optimization, method3_commutator):
                    out = extractor(Fhat) if extractor is method3_commutator else extractor(Fhat, seed=seed)
                    theta = principal_angle_deg(out["Q"], Q_star)
                    E_R = abs(np.log10((out["R"] + 1e-15) / (op["R_star"] + 1e-15)))
                    rows.append({
                        "method": out["method"],
                        "power_exponent": float(exponent),
                        "R_target": float(R_target),
                        "R_star": float(op["R_star"]),
                        "R_hat": float(out["R"]),
                        "Delta_star": float(op["Delta_star"]),
                        "Delta_hat": float(out["Delta"]),
                        "K_star": float(op["K_star"]),
                        "K_hat": float(out["K"]),
                        "theta_deg": float(theta),
                        "E_R": float(E_R),
                        "relErr_F": rel_err,
                        "cond_F": cond_F,
                        "sv_ratio_10": sv_ratio_10,
                        "seed": int(seed),
                    })
    return pd.DataFrame(rows)


def mean_sem(df: pd.DataFrame, value: str) -> pd.DataFrame:
    g = df.groupby(["method", "power_exponent"])[value]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_figure(df: pd.DataFrame, outdir: str) -> None:
    set_publication_style()
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.6))
    axa, axb, axc, axd = axes.ravel()

    # Panel a: singular spectra for one seed and R=3, different exponents.
    for exponent in POWER_EXPONENTS:
        op = build_operator(N, exponent, R_target=3.0, seed=10_000 + int(100 * exponent))
        s = op["svals"] / op["svals"][0]
        axa.loglog(np.arange(1, N + 1), s, marker="o", markersize=2.3, linewidth=1.0,
                   label=fr"$\beta={exponent:.2g}$")
    axa.set_xlabel("Singular-value index $j$")
    axa.set_ylabel(r"$\sigma_j(F)/\sigma_1(F)$")
    # axa.set_title("Embedded broad singular spectra")
    axa.legend(loc="lower left", ncol=1)
    format_axis(axa, logx=True, logy=True)
    add_panel_label(axa, "a")

    # Panel b: plane angle.
    styles = {"M2": {"marker": "o", "linestyle": "-", "label": "Optimization (M2)"},
              "M3": {"marker": "^", "linestyle": "--", "label": "Commutator (M3)"}}
    stat_theta = mean_sem(df, "theta_deg")
    for method in ["M2", "M3"]:
        d = stat_theta[stat_theta["method"] == method]
        axb.errorbar(d["power_exponent"], d["mean"], yerr=d["sem"], capsize=2.5,
                     marker=styles[method]["marker"], linestyle=styles[method]["linestyle"],
                     label=styles[method]["label"])
    axb.set_xlabel(r"Power-law exponent $\beta$")
    axb.set_ylabel(r"Plane error $\theta$ (deg)")
    # axb.set_title("Recovery of embedded response plane")
    format_axis(axb)
    add_panel_label(axb, "b")

    # Panel c: R recovery error.
    stat_ER = mean_sem(df, "E_R")
    for method in ["M2", "M3"]:
        d = stat_ER[stat_ER["method"] == method]
        axc.errorbar(d["power_exponent"], d["mean"], yerr=d["sem"], capsize=2.5,
                     marker=styles[method]["marker"], linestyle=styles[method]["linestyle"],
                     label=styles[method]["label"])
    axc.set_xlabel(r"Power-law exponent $\beta$")
    axc.set_ylabel(r"$R$-recovery error $\mathcal{E}_R$")
    # axc.set_title(r"Recovery of $R=K/K_c(\Delta)$")
    format_axis(axc)
    add_panel_label(axc, "c")

    # Panel d: Rhat vs Rstar scatter.
    for method in ["M2", "M3"]:
        d = df[df["method"] == method]
        axd.scatter(d["R_star"], d["R_hat"], s=15, alpha=0.72,
                    marker=styles[method]["marker"], label=styles[method]["label"])
    lo = min(df["R_star"].min(), df["R_hat"].min()) * 0.75
    hi = max(df["R_star"].max(), df["R_hat"].max()) * 1.35
    axd.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, label="identity")
    axd.set_xscale("log")
    axd.set_yscale("log")
    axd.set_xlim(lo, hi)
    axd.set_ylim(lo, hi)
    axd.set_xlabel(r"Reference $R_\star$")
    axd.set_ylabel(r"Inferred $\widehat R$")
    # axd.set_title("Reference versus inferred ratio")
    format_axis(axd, logx=True, logy=True)
    add_panel_label(axd, "d")

    handles, labels = axb.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0), pad=1.1, w_pad=1.2, h_pad=1.2)
    fig.savefig(os.path.join(outdir, "fig_broad_singular_value_nndr.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "fig_broad_singular_value_nndr.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    df = run_benchmark()
    csv_path = os.path.join(OUTDIR, "broad_singular_value_metrics.csv")
    df.to_csv(csv_path, index=False)
    plot_figure(df, OUTDIR)
    print("Saved:")
    print(os.path.join(OUTDIR, "fig_broad_singular_value_nndr.pdf"))
    print(os.path.join(OUTDIR, "fig_broad_singular_value_nndr.png"))
    print(csv_path)
    print("Summary by method:")
    print(df.groupby("method")[["theta_deg", "E_R", "relErr_F"]].mean())
