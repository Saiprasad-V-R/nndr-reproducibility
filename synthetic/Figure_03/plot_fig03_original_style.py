# -*- coding: utf-8 -*-
"""
Fig. 3 synthetic transition -- Phase 3 revised version.

Outputs
-------
1. fig3_synthetic_transition_abc.png/pdf
   Remaining synthetic-transition figure with panels (a)--(c) only.
   The old phrase "subcritical to threshold" has been removed.

2. fig3_observable_alignment_standalone.png/pdf
   Former panel (d), promoted to a standalone figure.
   The x-axis is fixed to 10^{-2}--10^{2}; many additional R samples below R=1
   and repeated noise realizations per point stabilize the alignment curve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.linalg import logm, schur

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_synthetic_figures")
RUN_STANDALONE_ALIGNMENT = False

# Reproducibility parameters
N = 200
RHO = 0.95
DELTA_SIGMA = 0.05
CFG_SEED = 1

# Time series used for panels (a)--(c)
T_TRAJ = 600
SIGMA_NOISE = 0.05
NOISE_SEED_TRAJ = 123

# Alignment curve: longer trajectories and repeated noise realizations.
# This is the key revision requested for the standalone observable-alignment figure.
T_ALIGN = 1000
N_ALIGN_REPEATS = 16
NOISE_SEED_ALIGN = 2468

# Dense sweep.  IMPORTANT: keep kappa as a floating-point parameter.
# In the previous revision kappa was rounded to integers; for this particular
# configuration R<1 only occurred for kappa=1,...,6, so the left side of the
# S-curve had too few points.  The grid below intentionally oversamples
# 1 <= kappa <= 7 without rounding, producing many more R values below R=1,
# while still covering the full displayed range 1e-2 <= R <= 1e2.
KAPPA_GRID = np.unique(np.r_[
    np.linspace(1.00, 7.00, 46),      # many subcritical R<1 points
    np.linspace(7.25, 20.00, 28),     # transition shoulder
    np.geomspace(20.5, 450.0, 42)     # supercritical plateau
].astype(float))

# Representative cases shown in panels (a)--(c)
TRAJECTORY_TARGET_R = [0.1, 1.0, 10.0, 30.0]
COLUMN_GROUPS = [(0, 1), (2, 3)]
COLUMN_TITLES = ["Subcritical", "Supercritical"]

# Style

def set_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.75,
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


def despine(ax) -> None:
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)


def format_axis(ax, nbx=5, nby=5, logx=False, logy=False) -> None:
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    if not logx:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbx))
    if not logy:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nby))


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.86)
    frame.set_edgecolor("none")


def panel_label(ax, label: str, x=-0.23, y=1.04) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", ha="left", va="bottom")


def symmetric_ylim(curves, pad: float = 0.12, min_span: float = 1e-6):
    vals = np.concatenate([np.asarray(c).ravel() for c in curves])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)
    amp = max(float(np.max(np.abs(vals))), min_span)
    return (-(1.0 + pad) * amp, (1.0 + pad) * amp)

# Linear algebra and diagnostics

def random_orthogonal(N_: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((N_, N_)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def corrected_reduced_ratios(Gamma: np.ndarray, delta_t: float = 1.0):
    """Rate-based R_A and exact discrete-time R_sigma on the original reduced map.

    Formulae agree with the V19 suite's shared/nndr_thresholds.py.
    No simulation, fitted plane, coordinate, or plotting conventions are changed.
    """
    G = np.asarray(Gamma, dtype=float)
    eigenvalues = np.linalg.eigvals(G)
    if max(abs(eigenvalues)) >= 1:
        raise ValueError("R_A and R_sigma require a stable reduced map")
    if np.any(abs(eigenvalues) <= 1e-15) or np.any((abs(eigenvalues.imag) <= 1e-12) & (eigenvalues.real < 0)):
        raise ValueError("Invalid real principal-logarithm domain")
    Lc = logm(G)
    if np.max(abs(Lc.imag)) > 1e-8 * max(1.0, np.linalg.norm(Lc.real)):
        raise ValueError("Principal matrix logarithm is not real")
    L = Lc.real / delta_t
    rates = np.linalg.eigvals(L)

    def departure(A):
        T, _ = schur(A, output="real")
        if abs(T[1, 0]) <= 1e-14 * max(1.0, np.linalg.norm(A)):
            return abs(float(T[0, 1]))
        if T[0, 1]*T[1, 0] < 0:
            return abs(abs(float(T[0, 1])) - abs(float(T[1, 0])))
        eig = np.linalg.eigvals(A)
        return float(np.sqrt(max(0., np.linalg.norm(A, 'fro')**2 - np.sum(abs(eig)**2))))

    if np.max(abs(rates.imag)) <= 1e-12:
        g1, g2 = sorted([-float(z.real) for z in rates])
        if g1 <= 0:
            raise ValueError("Unstable continuous-time generator")
        denominator = (np.sqrt(g1)+np.sqrt(g2))*np.sqrt(2*np.sqrt(g1*g2))
    else:
        gamma = -float(np.mean(rates.real))
        omega = float(np.max(abs(rates.imag)))
        if gamma <= 0:
            raise ValueError("Unstable continuous-time generator")
        denominator = 2*np.sqrt(gamma*(np.hypot(gamma,omega)+gamma))
    RA = float(departure(L)/denominator)
    margin = float(np.sqrt(np.prod([1-abs(z)**2 for z in eigenvalues])))
    Rsigma = float(departure(G)/margin)
    if abs(Rsigma-1)>1e-8 and abs(np.linalg.svd(G,compute_uv=False)[0]-1)>1e-8:
        assert (Rsigma>1) == (np.linalg.svd(G,compute_uv=False)[0]>1)
    return RA, Rsigma


def twoD_metrics_from_Gamma(Gamma: np.ndarray):
    eigvals, eigvecs = np.linalg.eig(Gamma)
    lam1, lam2 = eigvals[0], eigvals[1]
    denom = lam1 + lam2
    Delta = 1.0 if abs(denom) < 1e-15 else abs((lam1 - lam2) / denom)
    Delta = float(np.real_if_close(Delta))
    p1 = eigvecs[:, 0] / (np.linalg.norm(eigvecs[:, 0]) + 1e-15)
    p2 = eigvecs[:, 1] / (np.linalg.norm(eigvecs[:, 1]) + 1e-15)
    c = abs(np.dot(p1.conj(), p2))
    c = float(np.clip(np.real_if_close(c), 0.0, 1.0 - 1e-12))
    kappa2D = float(np.sqrt((1.0 + c) / (1.0 - c)))
    K = float(0.5 * (kappa2D - 1.0 / kappa2D))
    return Delta, kappa2D, K


def make_fixed_config(N_: int, rho: float = 0.95, delta_sigma: float = 0.05, seed: int = 1):
    rng = np.random.default_rng(seed)
    U = random_orthogonal(N_, rng)
    V = random_orthogonal(N_, rng)
    sigma_base = 1.0 + delta_sigma * rng.standard_normal(N_ - 1)
    sigma_base = np.clip(sigma_base, 0.1, 3.0)
    lam = rng.uniform(0.2 * rho, 0.95 * rho, size=N_)
    lam[0] = rho
    Lambda = np.diag(lam)
    return {"N": N_, "rho": rho, "delta_sigma": delta_sigma, "seed": seed,
            "U": U, "V": V, "Lambda": Lambda, "sigma_base": sigma_base}


def build_F_from_config(cfg: dict, kappa: float) -> np.ndarray:
    U, V, Lambda, N_ = cfg["U"], cfg["V"], cfg["Lambda"], cfg["N"]
    sigma = np.ones(N_)
    sigma[:-1] = cfg["sigma_base"]
    sigma[-1] = 1.0 / float(kappa)
    Sigma = np.diag(sigma)
    Sigma_inv = np.diag(1.0 / sigma)
    return U @ Sigma @ V.T @ Lambda @ V @ Sigma_inv @ U.T


def extract_nr_by_optimization(A: np.ndarray, iters: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    N_ = A.shape[0]
    v = rng.standard_normal(N_)
    v /= np.linalg.norm(v) + 1e-15
    for _ in range(iters):
        Av = A @ v
        Av = Av - v * (v @ Av)
        nu = np.linalg.norm(Av)
        if nu < 1e-12:
            v = rng.standard_normal(N_)
            v /= np.linalg.norm(v) + 1e-15
            continue
        u = Av / nu
        Atu = A.T @ u
        Atu = Atu - u * (u @ Atu)
        nv = np.linalg.norm(Atu)
        if nv < 1e-12:
            v = rng.standard_normal(N_)
            v /= np.linalg.norm(v) + 1e-15
            continue
        v = Atu / nv
    n = v
    r_raw = (np.eye(N_) - np.outer(n, n)) @ (A @ n)
    r = r_raw / (np.linalg.norm(r_raw) + 1e-15)
    return n, r


def simulate_var_with_fixed_noise(F: np.ndarray, noise: np.ndarray, x0=None) -> np.ndarray:
    T_, N_ = noise.shape
    x = np.zeros(N_) if x0 is None else np.asarray(x0, dtype=float).copy()
    X = np.zeros((N_, T_))
    for k in range(T_):
        x = F @ x + noise[k]
        X[:, k] = x
    return X


@dataclass
class GeometryResult:
    kappa_internal: float
    Delta: float
    kappa2D: float
    K: float
    Kc: float
    R: float
    R_sigma: float
    F: np.ndarray
    n: np.ndarray
    r: np.ndarray


@dataclass
class CaseResult:
    kappa_internal: float
    Delta: float
    kappa2D: float
    K: float
    Kc: float
    R: float
    R_sigma: float
    corr_m_r: float
    abs_corr_m_r: float
    mean_field: np.ndarray
    n_coord: np.ndarray
    r_coord: np.ndarray


@dataclass
class AlignmentPoint:
    kappa_internal: float
    R: float
    corr_mean: float
    corr_median: float
    corr_q25: float
    corr_q75: float
    corr_std: float
    n_repeats: int


def compute_geometry(cfg: dict, kappa_internal: float) -> GeometryResult:
    F = build_F_from_config(cfg, kappa=kappa_internal)
    n, r = extract_nr_by_optimization(F, iters=250, seed=0)
    Q = np.column_stack([r, n])
    Gamma = Q.T @ F @ Q
    Delta, kappa2D, K = twoD_metrics_from_Gamma(Gamma)
    RA, Rsigma = corrected_reduced_ratios(Gamma)
    Kc_A = K/RA if RA > 0 else np.inf
    return GeometryResult(float(kappa_internal), float(Delta), float(kappa2D),
                          float(K), float(Kc_A), float(RA), float(Rsigma), F, n, r)


def compute_case_from_geometry(geo: GeometryResult, noise: np.ndarray) -> CaseResult:
    X = simulate_var_with_fixed_noise(geo.F, noise)
    mean_field = X.mean(axis=0)
    n_coord = geo.n @ X
    r_coord = geo.r @ X
    c_mr = corr(mean_field, r_coord)
    return CaseResult(
        kappa_internal=geo.kappa_internal, Delta=geo.Delta, kappa2D=geo.kappa2D,
        K=geo.K, Kc=geo.Kc, R=geo.R, R_sigma=geo.R_sigma, corr_m_r=float(c_mr),
        abs_corr_m_r=float(abs(c_mr)), mean_field=mean_field,
        n_coord=n_coord, r_coord=r_coord,
    )


def compute_alignment_point(geo: GeometryResult, rng: np.random.Generator) -> AlignmentPoint:
    vals = []
    for _ in range(N_ALIGN_REPEATS):
        noise = SIGMA_NOISE * rng.standard_normal((T_ALIGN, N))
        X = simulate_var_with_fixed_noise(geo.F, noise)
        vals.append(abs(corr(X.mean(axis=0), geo.r @ X)))
    vals = np.asarray(vals, dtype=float)
    return AlignmentPoint(
        kappa_internal=geo.kappa_internal,
        R=geo.R,
        corr_mean=float(vals.mean()),
        corr_median=float(np.median(vals)),
        corr_q25=float(np.quantile(vals, 0.25)),
        corr_q75=float(np.quantile(vals, 0.75)),
        corr_std=float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        n_repeats=int(vals.size),
    )


def select_geometries_by_R(geometries: list[GeometryResult], target_R: list[float]) -> list[GeometryResult]:
    Rvals = np.array([res.R for res in geometries], dtype=float)
    chosen = []
    used = set()
    for target in target_R:
        dist = np.abs(np.log10(Rvals + 1e-12) - np.log10(target + 1e-12))
        order = np.argsort(dist)
        idx = int(order[0])
        for alt in order:
            if int(alt) not in used:
                idx = int(alt)
                break
        used.add(idx)
        chosen.append(geometries[idx])
    return chosen


def label_for_case(res) -> str:
    R = float(res.R)
    if R < 0.1:
        text = f"{R:.3f}"
    elif R < 10.0:
        text = f"{R:.1f}"
    else:
        text = f"{int(np.rint(R))}"
    return rf"$R_A={text}$"

# Plotting

COLOR_BLUE   = "#2563EB"
COLOR_ORANGE = "#EA580C"
COLOR_THRESH = "#94A3B8"
COLOR_CURVE  = "#334155"
COLOR_DOT    = "#64748B"
COLOR_BG_STRIP = "#F8FAFC"


def plot_transition_abc(chosen_cases: list[CaseResult], outdir: str, include_rsigma=False) -> None:
    set_publication_style()
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure(figsize=(6.15, 5.55), facecolor="white")
    gs = GridSpec(nrows=3, ncols=2, figure=fig, wspace=0.28, hspace=0.42)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)]
    t = np.arange(chosen_cases[0].mean_field.size)

    row_data = [
        ("a", r" $m_k$", "mean_field"),
        ("b", r" $\widehat{\mathbf{n}}^{\top}\mathbf{x}_k$", "n_coord"),
        ("c", r" $\widehat{\mathbf{r}}^{\top}\mathbf{x}_k$", "r_coord"),
    ]

    for col, (idx_low, idx_high) in enumerate(COLUMN_GROUPS):
        res_low = chosen_cases[idx_low]
        res_high = chosen_cases[idx_high]
        for row, (_, ylabel, attr) in enumerate(row_data):
            ax = axes[row][col]
            y_low = getattr(res_low, attr)
            y_high = getattr(res_high, attr)

            ax.set_facecolor(COLOR_BG_STRIP)
            despine(ax)
            ax.axhline(0.0, color=COLOR_THRESH, lw=0.7, linestyle="--", zorder=1, alpha=0.8)
            ax.plot(t, y_high, color=COLOR_ORANGE, lw=0.9, alpha=0.80, zorder=2)
            ax.plot(t, y_low, color=COLOR_BLUE, lw=0.9, alpha=0.90, zorder=3)
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(symmetric_ylim([y_low, y_high], pad=0.12))
            format_axis(ax, nbx=6)

            if col == 0:
                ax.set_ylabel(ylabel, labelpad=6)
            if row < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"Time step $k$")

            if row == 0:
                ax.set_title(COLUMN_TITLES[col], fontweight="bold", pad=23, fontsize=9.75)
                handles = [
                    Line2D([0], [0], color=COLOR_BLUE, lw=2.0, solid_capstyle="round"),
                    Line2D([0], [0], color=COLOR_ORANGE, lw=2.0, solid_capstyle="round"),
                ]
                labels = [label_for_case(res_low), label_for_case(res_high)]
                leg = ax.legend(handles, labels, loc="lower center",
                                bbox_to_anchor=(0.50, 1.03), ncol=2,
                                handlelength=1.6, borderpad=0.0, columnspacing=1.0)
                style_legend(leg)
                if include_rsigma:
                    # Secondary values placed in faint grey within the top panel,
                    # leaving all original axes/legend/font/style parameters intact.
                    ax.text(0.5, 0.930,
                            rf"$R_\sigma={res_low.R_sigma:.2f}\qquad R_\sigma={res_high.R_sigma:.2f}$",
                            color="#94A3B8", fontsize=6.5, alpha=0.92,
                            ha="center", va="top", transform=ax.transAxes,
                            bbox={"facecolor": COLOR_BG_STRIP, "edgecolor": "none", "pad": 0.8, "alpha": 0.75})

    for row, (letter, _, _) in enumerate(row_data):
        panel_label(axes[row][0], letter)

    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.12, top=0.88)
    tag = "RA_Rsigma" if include_rsigma else "RA_only"
    out_png = os.path.join(outdir, f"Fig03_{tag}.png")
    out_pdf = os.path.join(outdir, f"Fig03_{tag}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, transparent=True, bbox_inches="tight")
    print("Saved:", out_png)
    print("Saved:", out_pdf)
    plt.close(fig)


def plot_alignment_standalone(points: list[AlignmentPoint], chosen_cases: list[CaseResult], outdir: str) -> None:
    """Standalone observable-alignment curve.

    Phase-3 clean version: no point markers and no highlighted example cases.
    The representative time-series cases are already shown in Fig. 3(a)--(c),
    so this figure should communicate only the smooth transition of the
    observable alignment versus R.
    """
    set_publication_style()
    os.makedirs(outdir, exist_ok=True)

    Rvals = np.array([p.R for p in points], dtype=float)
    med = np.array([p.corr_median for p in points], dtype=float)
    q25 = np.array([p.corr_q25 for p in points], dtype=float)
    q75 = np.array([p.corr_q75 for p in points], dtype=float)

    order = np.argsort(Rvals)
    Rvals, med, q25, q75 = Rvals[order], med[order], q25[order], q75[order]
    keep = (Rvals >= 1e-2) & (Rvals <= 1e2)

    fig, ax = plt.subplots(figsize=(4.95, 3.45), facecolor="white")
    ax.set_facecolor(COLOR_BG_STRIP)
    despine(ax)

    # Clean curve only: central 50% band + median line. No sampled-point markers and
    # no highlighted time-series cases, because these made the standalone panel
    # visually cluttered.
    ax.fill_between(Rvals[keep], q25[keep], q75[keep],
                    color=COLOR_DOT, alpha=0.16, linewidth=0,
                    label="Central 50% over noise trials")
    ax.plot(Rvals[keep], med[keep], color=COLOR_CURVE, lw=2.15,
            alpha=0.98, zorder=3, label="Median alignment")

    ax.axvline(1.0, color=COLOR_THRESH, lw=1.05, linestyle="--", zorder=1, alpha=0.85)
    ax.text(1.08, 0.065, r"$R_A=1$", transform=ax.get_xaxis_transform(),
            fontsize=8.2, color=COLOR_THRESH, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0.0, min(1.0, max(0.25, 1.08 * float(np.max(q75[keep])))))
    ax.set_xlabel(r"Reduced non-normality ratio $R_A=K/K_{c,A}$")
    ax.set_ylabel(r"$|\mathrm{corr}(m_k,\widehat{\mathbf{r}}^{\top}\mathbf{x}_k)|$")
    ax.set_title("Observable alignment across the synthetic transition", fontweight="bold")

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    format_axis(ax, logx=True)
    leg = ax.legend(loc="upper left", frameon=True, fontsize=7.5)
    style_legend(leg)

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.18, top=0.90)
    out_png = os.path.join(outdir, "fig3_observable_alignment_standalone.png")
    out_pdf = os.path.join(outdir, "fig3_observable_alignment_standalone.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, transparent=True, bbox_inches="tight")
    print("Saved:", out_png)
    print("Saved:", out_pdf)
    plt.close(fig)

# Main

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    cfg = make_fixed_config(N_=N, rho=RHO, delta_sigma=DELTA_SIGMA, seed=CFG_SEED)
    geometries = [compute_geometry(cfg, float(k)) for k in KAPPA_GRID]

    chosen_geometries = select_geometries_by_R(geometries, TRAJECTORY_TARGET_R)
    rng_traj = np.random.default_rng(NOISE_SEED_TRAJ)
    noise_traj = SIGMA_NOISE * rng_traj.standard_normal((T_TRAJ, N))
    chosen_cases = [compute_case_from_geometry(geo, noise_traj) for geo in chosen_geometries]

    plot_transition_abc(chosen_cases, OUTDIR, include_rsigma=False)
    plot_transition_abc(chosen_cases, OUTDIR, include_rsigma=True)

    import csv
    with open(os.path.join(OUTDIR, "Fig03_selected_corrected_ratios.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "kappa", "R_A", "R_sigma", "K", "Kc_A"])
        for j, res in enumerate(chosen_cases, 1):
            writer.writerow([j, repr(res.kappa_internal), repr(res.R),
                             repr(res.R_sigma), repr(res.K), repr(res.Kc)])
            print(f"Case {j}: kappa={res.kappa_internal:.8g}, R_A={res.R:.6g}, R_sigma={res.R_sigma:.6g}")

    if RUN_STANDALONE_ALIGNMENT:
        rng_align = np.random.default_rng(NOISE_SEED_ALIGN)
        alignment_points = [compute_alignment_point(geo, rng_align) for geo in geometries]
        plot_alignment_standalone(alignment_points, chosen_cases, OUTDIR)
    print("Done.")
