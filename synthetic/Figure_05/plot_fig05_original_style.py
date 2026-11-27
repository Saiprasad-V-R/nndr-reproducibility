# -*- coding: utf-8 -*-
"""
Figure 4/5: Method-comparison benchmark with M=1 and M=25
================================================================

Purpose of this revision
------------------------
This version adds the single-trajectory synthetic check requested by Didier.

The empirical analyses have M=1 by construction.  Therefore this synthetic
benchmark now runs the same method-comparison pipeline for

    M = 1   : one training trajectory, matching the empirical regime
    M = 25  : ensemble baseline used for the synthetic validation

Representative setting:
    N = 100
    M in {1, 25}
    T_train = 300

Panels:
    a. inferred R_hat versus true reduced non-normality ratio R_true
    b. signed deviation from reference, R_hat - R_true, versus R_true
    c. plane error theta(Q_hat, Q_true) versus R_true
    d. plane-invariant observable alignment c_plane versus R_true

Plot convention:
    solid colored lines  : M = 25 ensemble baseline
    faint open markers   : M = 1 single-trajectory check
    black reference line : Method 2 applied directly to the true operator F;
                           this reference is independent of M and is shown once

Visual-balance revision:
    The M=1 markers and the eigenbasis-SVD method are intentionally drawn with
    lower visual weight, because their role is to document finite-realization
    sensitivity rather than to dominate the main M2/M3 comparison.  Panel (b)
    uses a clipped main deviation axis with boundary markers for outliers; the
    full numerical values are saved in the CSV.

Notation:
    Q_true : reference plane obtained from the true operator F
    Q_hat  : estimated plane obtained from Fhat
    R      : reduced non-normality ratio K/Kc(Delta)

Figure legend:
    Reference    : Method 2 applied directly to the true operator F
    Eigenbasis   : eigenbasis-SVD extraction
    Optimization : optimization-based extraction
    Commutator   : commutator-plane extraction

Spyder usage:
    %runfile /path/to/fig5_method_comparison_M1_M25.py --wdir /path/to/output_folder
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from scipy.linalg import logm, schur
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List


# 1) Global plotting style

def set_publication_style():
    """
    Shared visual identity for manuscript figures.
    Matches Figure S1 and Figure S2.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.25,
        "axes.labelsize": 9.25,
        "axes.titlesize": 9.75,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
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


def despine(ax):
    """Apply the boxed axis treatment used in the reference figure."""
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)


def format_axis(ax, nbx=5, nby=5, logx=False, logy=False):
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    if not logx:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbx))
    if not logy:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nby))


def style_legend(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.86)
    frame.set_edgecolor("none")


def add_panel_label(ax, label: str, x=-0.13, y=1.08):
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# 2) Figure labels and colors

METHOD_STYLE = {
    "TRUE": {
        "label": "Reference",
        "color": "#000000",
        "linestyle": "-",
        "marker": "o",
        "lw": 1.45,
        "alpha": 1.0,
    },
    "M1_BF": {
        "label": "Eigenbasis (M1)",
        "color": "#0072B2",
        "linestyle": ":",
        "marker": "s",
        # Drawn slightly lighter because it is the sensitivity baseline.
        "lw": 0.95,
        "alpha": 0.62,
    },
    "M2_OPT": {
        "label": "Optimization (M2)",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "o",
        "lw": 1.35,
        "alpha": 1.0,
    },
    "M3_COMM": {
        "label": "Commutator (M3)",
        "color": "#009E73",
        "linestyle": "--",
        "marker": "^",
        "lw": 1.35,
        "alpha": 1.0,
    },
}


# M-comparison requested for synthetic-to-empirical consistency.
# M=1 matches the empirical regime by construction; M=25 is the ensemble baseline.
M_SINGLE = 1
M_BASELINE = 25
M_VALUES = [M_SINGLE, M_BASELINE]

# Visual balance constants for the M=1 overlay and deviation panel.
# These keep the single-trajectory check visible but secondary.
SINGLE_MARKER_SIZE = 2.35
SINGLE_MARKER_ALPHA = 0.55
SINGLE_MARKER_EDGEWIDTH = 0.55
EIGENBASIS_MARKER_SCALE = 0.78
DEVIATION_YCLIP_MAX = 35.0
DEVIATION_YCLIP_MIN = 6.0


# 3) Utilities

def random_orthogonal(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an orthogonal matrix via QR decomposition."""
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def corr(a, b) -> float:
    """Pearson correlation for two one-dimensional arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def spectral_radius(A: np.ndarray) -> float:
    """Spectral radius."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def principal_angle_deg(Q_hat: np.ndarray, Q_star: np.ndarray) -> float:
    """
    Largest principal angle between span(Q_hat) and span(Q_star).

    Q_hat, Q_star have shape (N, 2) with orthonormal columns.
    """
    M = Q_hat.T @ Q_star
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = float(np.arccos(s[-1]))
    return float(np.degrees(theta))


def best_plane_corr(Q: np.ndarray, X: np.ndarray, m: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Plane-invariant correlation:
        c_plane = max_{||a||=1} |corr(m(t), a^T Q^T x(t))|.

    Returns
    -------
    c_best : float
        Best absolute correlation.
    a_best : ndarray, shape (2,)
        Best direction in plane coordinates.
    r_best : ndarray, shape (N,)
        Corresponding lifted direction.
    """
    Z = Q.T @ X
    m = np.asarray(m).ravel()

    Zc = Z - Z.mean(axis=1, keepdims=True)
    mc = m - m.mean()

    g = Zc @ mc
    ng = np.linalg.norm(g)

    if ng < 1e-14:
        return 0.0, np.array([1.0, 0.0]), Q[:, 0]

    a = g / ng
    y = a @ Z
    c = abs(corr(m, y))
    r = Q @ a

    return float(c), a, r


# 4) Reduced 2D diagnostics

def Kc_from_Delta(Delta: float) -> float:
    """
    Critical threshold for the reduced 2D real-eigenvalue setting.

    Kc(Delta) = sqrt( sqrt(1 - Delta^2) / (1 - sqrt(1 - Delta^2)) )
    for 0 <= Delta < 1.
    """
    Delta = float(np.clip(Delta, 0.0, 1.0 - 1e-12))
    s = np.sqrt(1.0 - Delta**2)
    return float(np.sqrt(s / (1.0 - s + 1e-15)))


def twoD_metrics_from_Gamma(Gamma: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute reduced two-dimensional diagnostics.

    Returns
    -------
    Delta : float
        Normalized eigenvalue splitting.
    kappa2D : float
        Eigenvector non-orthogonality in the reduced plane.
    K : float
        Reduced non-normality index.
    """
    eigvals, eigvecs = np.linalg.eig(Gamma)
    lam1, lam2 = eigvals[0], eigvals[1]
    denom = lam1 + lam2

    Delta = 1.0 if abs(denom) < 1e-15 else abs((lam1 - lam2) / denom)

    p1 = eigvecs[:, 0] / (np.linalg.norm(eigvecs[:, 0]) + 1e-15)
    p2 = eigvecs[:, 1] / (np.linalg.norm(eigvecs[:, 1]) + 1e-15)

    c = abs(np.dot(p1.conj(), p2))
    c = float(np.clip(c, 0.0, 1.0 - 1e-12))

    kappa2D = float(np.sqrt((1.0 + c) / (1.0 - c)))
    K = float(0.5 * (kappa2D - 1.0 / kappa2D))

    return float(Delta), float(kappa2D), float(K)


def gamma_diagnostics(Gamma: np.ndarray, tol=1e-12) -> Tuple[complex, np.ndarray, bool]:
    """
    Basic safety diagnostics for the reduced 2 x 2 operator.
    """
    G = np.real_if_close(Gamma, tol=1000)
    tr = complex(np.trace(G))
    ev = np.linalg.eigvals(G)

    tr_ok = abs(tr) > tol

    ev_r = np.real_if_close(ev, tol=1000)
    if np.any(np.iscomplex(ev_r)):
        same_sign = False
    else:
        ev_r = np.real(ev_r)
        same_sign = (ev_r[0] * ev_r[1] > 0)

    ok = bool(tr_ok and same_sign)
    return tr, ev, ok


def _corrected_reduced_ratios(Gamma: np.ndarray, delta_t: float = 1.0):
    """Exact V19 diagnostics on the reduced map, without coercing invalid windows."""
    G = np.asarray(Gamma, dtype=float)
    w = np.linalg.eigvals(G)
    if np.max(np.abs(w)) >= 1 - 1e-12:
        return np.nan, np.nan, "unstable_reduced_map"
    if np.min(np.abs(w)) < 1e-14 or np.any((np.abs(w.imag)<1e-11) & (w.real<=0)):
        margin=np.sqrt(np.prod(1-np.abs(w)**2))
        dep=np.sqrt(max(0.,np.linalg.norm(G,'fro')**2-np.sum(np.abs(w)**2)))
        return np.nan, float(dep/margin), "principal_log_not_real"
    Lc=logm(G)
    if np.linalg.norm(Lc.imag)>1e-8*max(1.0,np.linalg.norm(Lc.real)):
        return np.nan,np.nan,"principal_log_not_real"
    L=Lc.real/delta_t
    rates=np.linalg.eigvals(L)
    def departure(A):
        eig=np.linalg.eigvals(A)
        return float(np.sqrt(max(0.,np.linalg.norm(A,'fro')**2-np.sum(np.abs(eig)**2))))
    if np.max(abs(rates.imag))<1e-10:
        g1,g2=sorted([-float(v.real) for v in rates])
        if g1<=0: return np.nan,np.nan,"unstable_rates"
        d=(np.sqrt(g1)+np.sqrt(g2))*np.sqrt(2*np.sqrt(g1*g2))
    else:
        gamma=-float(np.mean(rates.real)); omega=float(np.max(abs(rates.imag)))
        if gamma<=0: return np.nan,np.nan,"unstable_rates"
        d=2*np.sqrt(gamma*(np.hypot(gamma,omega)+gamma))
    margin=np.sqrt(np.prod(1-np.abs(w)**2))
    return departure(L)/d,departure(G)/margin,"ok"


def compute_indices_from_Q(A: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
    """Corrected R_A and companion R_sigma on the selected real plane."""
    Gamma=Q.T @ A @ Q
    RA, Rsigma, status = _corrected_reduced_ratios(Gamma)
    Delta,_,K=twoD_metrics_from_Gamma(Gamma)
    trG,evG,okG=gamma_diagnostics(Gamma)
    return dict(Gamma=Gamma,Delta=Delta,K=K,R=RA,R_sigma=Rsigma,
                status=status,trG=trG,evG=evG,okG=okG)


# 5) Fixed non-normal matrix family

def make_fixed_config(N: int, rho=0.95, delta_sigma=0.05, seed=1) -> Dict[str, Any]:
    """
    Fix U, V, Lambda and sigma_i for i < N.
    Vary only sigma_N = 1/kappa to control non-normality.
    """
    rng = np.random.default_rng(seed)

    U = random_orthogonal(N, rng)
    V = random_orthogonal(N, rng)

    sigma_base = 1.0 + delta_sigma * rng.standard_normal(N - 1)
    sigma_base = np.clip(sigma_base, 0.1, 3.0)

    lam = rng.uniform(0.2 * rho, 0.95 * rho, size=N)
    lam[0] = rho
    Lambda = np.diag(lam)

    return {
        "N": N,
        "rho": rho,
        "delta_sigma": delta_sigma,
        "seed": seed,
        "U": U,
        "V": V,
        "Lambda": Lambda,
        "sigma_base": sigma_base,
    }


def build_F_from_config(cfg: Dict[str, Any], kappa: float) -> np.ndarray:
    """
    Construct F(kappa) from the fixed configuration.

    F(kappa) = U Sigma(kappa) V^T Lambda V Sigma(kappa)^(-1) U^T.
    """
    U = cfg["U"]
    V = cfg["V"]
    Lambda = cfg["Lambda"]
    N = cfg["N"]

    sigma = np.ones(N)
    sigma[:-1] = cfg["sigma_base"]
    sigma[-1] = 1.0 / float(kappa)

    Sigma = np.diag(sigma)
    Sigma_inv = np.diag(1.0 / sigma)

    F = U @ Sigma @ V.T @ Lambda @ V @ Sigma_inv @ U.T
    return F


# 6) VAR simulation and stacking

def simulate_var_with_fixed_noise(F: np.ndarray, noise: np.ndarray, x0=None) -> np.ndarray:
    """
    Simulate x_{k+1} = F x_k + noise[k].

    noise shape:
        (T, N)

    returns:
        X shape (N, T), storing x_1, ..., x_T.
    """
    T, N = noise.shape
    x = np.zeros(N) if x0 is None else np.asarray(x0, dtype=float).copy()

    X = np.zeros((N, T))
    for k in range(T):
        x = F @ x + noise[k]
        X[:, k] = x

    return X


def stack_VAR_pairs_from_trajectories(X_list: List[np.ndarray], x0=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack one-step VAR regression pairs from independent trajectories.

    Each X is shape (N, T), storing x_1, ..., x_T.
    Predictors:
        x_0, x_1, ..., x_{T-1}
    Responses:
        x_1, x_2, ..., x_T
    """
    Xcols = []
    Ycols = []

    for X in X_list:
        N, T = X.shape
        x0_vec = np.zeros(N) if x0 is None else np.asarray(x0).reshape(N)
        Xin = np.column_stack([x0_vec, X[:, :-1]])
        Yout = X.copy()

        Xcols.append(Xin)
        Ycols.append(Yout)

    Xstack = np.concatenate(Xcols, axis=1)
    Ystack = np.concatenate(Ycols, axis=1)

    return Xstack, Ystack


# 7) Ridge estimation of Fhat

def ridge_fit_F(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve:
        min_F ||Y - F X||_F^2 + lam ||F||_F^2

    Closed form:
        Fhat = (Y X^T) (X X^T + lam I)^(-1).
    """
    N = X.shape[0]
    XXT = X @ X.T
    YXT = Y @ X.T
    A = XXT + float(lam) * np.eye(N)

    Fhat = (np.linalg.solve(A.T, YXT.T)).T
    return Fhat


def select_lambda_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: np.ndarray,
    frac_train=0.8,
    seed=0,
) -> Tuple[float, np.ndarray]:
    """
    Select ridge lambda using a train-validation split over stacked columns.
    """
    rng = np.random.default_rng(seed)
    T_eff = X.shape[1]

    idx = np.arange(T_eff)
    rng.shuffle(idx)

    ntr = int(frac_train * T_eff)
    itr = idx[:ntr]
    iva = idx[ntr:]

    Xtr = X[:, itr]
    Ytr = Y[:, itr]
    Xva = X[:, iva]
    Yva = Y[:, iva]

    mse = []

    for lam in lambdas:
        Fhat = ridge_fit_F(Xtr, Ytr, lam)
        R = Yva - Fhat @ Xva
        mse.append(float(np.mean(R * R)))

    mse = np.array(mse)
    j = int(np.argmin(mse))

    return float(lambdas[j]), mse


# 8) Extraction methods

def extract_nr_by_optimization(A: np.ndarray, iters=250, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate:
        max <u, A v>
        subject to ||u|| = ||v|| = 1 and u^T v = 0.

    The returned reaction direction is computed from projected A n.
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]

    v = rng.standard_normal(N)
    v /= np.linalg.norm(v) + 1e-15

    for _ in range(iters):
        Av = A @ v
        Av = Av - v * (v @ Av)
        nu = np.linalg.norm(Av)

        if nu < 1e-12:
            v = rng.standard_normal(N)
            v /= np.linalg.norm(v) + 1e-15
            continue

        u = Av / nu

        Atu = A.T @ u
        Atu = Atu - u * (u @ Atu)
        nv = np.linalg.norm(Atu)

        if nv < 1e-12:
            v = rng.standard_normal(N)
            v /= np.linalg.norm(v) + 1e-15
            continue

        v = Atu / nv

    n = v
    r_raw = (np.eye(N) - np.outer(n, n)) @ (A @ n)
    r = r_raw / (np.linalg.norm(r_raw) + 1e-15)

    return n, r


def method2_optimization(A: np.ndarray, iters=250, seed=0) -> Dict[str, Any]:
    """Optimization-based extraction."""
    n, r = extract_nr_by_optimization(A, iters=iters, seed=seed)

    Q, _ = np.linalg.qr(np.column_stack([r, n]))
    r = Q[:, 0]
    n = Q[:, 1]

    Gamma = Q.T @ A @ Q
    trG, evG, okG = gamma_diagnostics(Gamma)

    return {
        "name": "M2_OPT",
        "n": n,
        "r": r,
        "Q": Q,
        "Gamma": Gamma,
        "trG": trG,
        "evG": evG,
        "okG": okG,
    }


def _make_real_direction(z, tol=1e-12):
    """
    Convert a possibly complex vector into a normalized real direction.
    """
    if np.iscomplexobj(z):
        zr = np.real(z)
        zi = np.imag(z)

        nr = np.linalg.norm(zr)
        ni = np.linalg.norm(zi)

        if nr > tol:
            return zr / (nr + 1e-15)
        if ni > tol:
            return zi / (ni + 1e-15)

        v = np.random.randn(z.size)
        return v / (np.linalg.norm(v) + 1e-15)

    return z / (np.linalg.norm(z) + 1e-15)


def method1_bruteforce(A: np.ndarray, eps=1e-15) -> Dict[str, Any]:
    """
    Eigenbasis-SVD extraction.
    """
    _, P = np.linalg.eig(A)
    U, svals, _ = np.linalg.svd(P, full_matrices=False)

    kappa_P = float(svals[0] / (svals[-1] + eps))

    n = _make_real_direction(U[:, -1])
    r_raw = (np.eye(A.shape[0]) - np.outer(n, n)) @ (A @ n)
    r = r_raw / (np.linalg.norm(r_raw) + eps)

    Q, _ = np.linalg.qr(np.column_stack([r, n]))
    r = Q[:, 0]
    n = Q[:, 1]

    Gamma = Q.T @ A @ Q
    trG, evG, okG = gamma_diagnostics(Gamma)

    return {
        "name": "M1_BF",
        "n": n,
        "r": r,
        "Q": Q,
        "Gamma": Gamma,
        "kappa_P": kappa_P,
        "sigmas_P": svals,
        "trG": trG,
        "evG": evG,
        "okG": okG,
    }


def method3_commutator(A: np.ndarray, tol=1e-12) -> Dict[str, Any]:
    """
    Commutator-plane extraction.

        B = A A^T - A^T A.
    """
    B = A @ A.T - A.T @ A

    evals, evecs = np.linalg.eigh(B)

    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    okB = bool(max(abs(lam_max), abs(lam_min)) > tol)

    w_plus = evecs[:, -1]
    w_minus = evecs[:, 0]

    Q = np.column_stack([w_plus, w_minus])

    Gamma = Q.T @ A @ Q
    trG, evG, okG = gamma_diagnostics(Gamma)

    return {
        "name": "M3_COMM",
        "n": w_plus,
        "r": w_minus,
        "Q": Q,
        "Gamma": Gamma,
        "lamB_max": lam_max,
        "lamB_min": lam_min,
        "okB": okB,
        "trG": trG,
        "evG": evG,
        "okG": okG,
    }


# 9) Benchmark configuration

@dataclass
class BenchConfig:
    rho: float = 0.95
    delta_sigma: float = 0.05
    sigma_noise: float = 0.05

    lambdas: np.ndarray = field(default_factory=lambda: np.logspace(-8, 2, 21))
    frac_train: float = 0.8

    cfg_seed_base: int = 1
    seed_train_bank: int = 111
    seed_test_noise: int = 222
    seed_cv: int = 333
    seed_opt_reference: int = 0
    seed_opt_est: int = 0

    iters_opt: int = 250


# 10) Core benchmark

def run_representative_benchmark(
    N: int,
    M: int,
    T_train: int,
    kappas: List[float],
    bc: BenchConfig,
    T_test: int,
    outdir: str,
) -> pd.DataFrame:
    """
    Run the representative benchmark for Figure S3.
    """
    rows = []

    cfg = make_fixed_config(
        N=N,
        rho=bc.rho,
        delta_sigma=bc.delta_sigma,
        seed=bc.cfg_seed_base,
    )

    rng_test = np.random.default_rng(bc.seed_test_noise + 10_000 * N)
    noise_test = bc.sigma_noise * rng_test.standard_normal((T_test, N))

    rng_bank = np.random.default_rng(
        bc.seed_train_bank + 1_000_000 * N + 10_000 * M + 100 * T_train
    )
    noise_bank = bc.sigma_noise * rng_bank.standard_normal((M, T_train, N))

    for kappa in kappas:
        F = build_F_from_config(cfg, kappa=kappa)

        # Reference: optimization-based extraction on true F
        out_ref = method2_optimization(
            F,
            iters=bc.iters_opt,
            seed=bc.seed_opt_reference,
        )

        Q_star = out_ref["Q"]
        idx_star = compute_indices_from_Q(F, Q_star)

        R_star = float(idx_star["R"])
        Delta_star = float(idx_star["Delta"])

        # Training trajectories
        X_list = []
        for m in range(M):
            Xm = simulate_var_with_fixed_noise(
                F,
                noise_bank[m],
                x0=np.zeros(N),
            )
            X_list.append(Xm)

        Xstack, Ystack = stack_VAR_pairs_from_trajectories(
            X_list,
            x0=np.zeros(N),
        )

        # Estimate Fhat
        lam_star, _ = select_lambda_ridge(
            Xstack,
            Ystack,
            bc.lambdas,
            frac_train=bc.frac_train,
            seed=bc.seed_cv,
        )

        Fhat = ridge_fit_F(Xstack, Ystack, lam_star)

        relErr = float(
            np.linalg.norm(Fhat - F, ord="fro")
            / (np.linalg.norm(F, ord="fro") + 1e-15)
        )
        rho_hat = spectral_radius(Fhat)

        # Test trajectory for c_plane
        Xte = simulate_var_with_fixed_noise(
            F,
            noise_test,
            x0=np.zeros(N),
        )
        mte = Xte.mean(axis=0)

        c_ref, _, _ = best_plane_corr(Q_star, Xte, mte)

        rows.append({
            "N": int(N),
            "M": int(M),
            "T_train": int(T_train),
            "T_test": int(T_test),
            "kappa": float(kappa),
            "method": "TRUE",
            "method_label": METHOD_STYLE["TRUE"]["label"],
            "lambda_star": float(lam_star),
            "relErr_Fhat": float(relErr),
            "rho_Fhat": float(rho_hat),
            "R_star": R_star,
            "Delta_star": Delta_star,
            "R_hat": R_star,
            "Delta_hat": Delta_star,
            "R_signed_err": 0.0,
            "R_abs_err": 0.0,
            "R_scaled_abs_err": 0.0,
            "theta_deg": 0.0,
            "c_plane": float(c_ref),
        })

        methods = [
            method1_bruteforce(Fhat),
            method2_optimization(Fhat, iters=bc.iters_opt, seed=bc.seed_opt_est),
            method3_commutator(Fhat),
        ]

        for outm in methods:
            method = outm["name"]
            Q_hat = outm["Q"]
            idx_hat = compute_indices_from_Q(Fhat, Q_hat)

            R_hat = float(idx_hat["R"])
            Delta_hat = float(idx_hat["Delta"])

            signed_err = R_hat - R_star
            abs_err = abs(signed_err)
            scaled_abs_err = abs_err / (1.0 + abs(R_star))

            theta = principal_angle_deg(Q_hat, Q_star)
            c_plane, _, _ = best_plane_corr(Q_hat, Xte, mte)

            row = {
                "N": int(N),
                "M": int(M),
                "T_train": int(T_train),
                "T_test": int(T_test),
                "kappa": float(kappa),
                "method": method,
                "method_label": METHOD_STYLE[method]["label"],
                "lambda_star": float(lam_star),
                "relErr_Fhat": float(relErr),
                "rho_Fhat": float(rho_hat),
                "R_star": R_star,
                "Delta_star": Delta_star,
                "R_hat": R_hat,
                "Delta_hat": Delta_hat,
                "R_signed_err": float(signed_err),
                "R_abs_err": float(abs_err),
                "R_scaled_abs_err": float(scaled_abs_err),
                "theta_deg": float(theta),
                "c_plane": float(c_plane),
            }

            if method == "M1_BF":
                row["kappa_P"] = float(outm["kappa_P"])

            if method == "M3_COMM":
                row["lamB_max"] = float(outm["lamB_max"])
                row["lamB_min"] = float(outm["lamB_min"])
                row["okB"] = bool(outm["okB"])

            rows.append(row)

        print(f"[done] M = {M}, kappa = {kappa}")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(outdir, f"Figure_4_method_benchmark_M{M}_data.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved benchmark data:")
    print("  CSV:", csv_path)

    return df


# 11) Figure S3 plotting

def _format_R_tick_value(v):
    """Clean labels for log-scale reduced non-normality ratio ticks."""
    if v < 0.1:
        return f"{v:.2f}"
    if v < 1:
        return f"{v:.1f}"
    if abs(v - round(v)) < 1e-10:
        return f"{int(round(v))}"
    return f"{v:.1f}"


def _nice_R_ticks(R_min, R_max):
    """Compact major ticks for the log-scale true reduced ratio axis."""
    # Keep only a few labelled ticks.  Dense minor ticks remain unlabeled.
    preferred = np.array([0.02, 0.1, 1.0, 10.0, 100.0])
    lo = 0.80 * R_min
    hi = 1.20 * R_max
    ticks = preferred[(preferred >= lo) & (preferred <= hi)]
    if R_min < 1.0 < R_max and not np.any(np.isclose(ticks, 1.0)):
        ticks = np.sort(np.append(ticks, 1.0))
    return ticks


def _sparse_mark_indices(x):
    """
    Plot all computed samples as a line, but draw markers only at a sparse
    representative subset.  This preserves the dense subcritical sweep without
    visually overcrowding the interval below R=1.
    """
    x = np.asarray(x, dtype=float)
    if x.size <= 10:
        return list(range(x.size))

    targets = np.array([0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0,
                        10.0, 20.0, 50.0, 100.0])
    idx = []
    for target in targets:
        if np.nanmin(x) <= target <= np.nanmax(x):
            idx.append(int(np.nanargmin(np.abs(np.log10(x) - np.log10(target)))))

    # Also show the first and last simulated values.
    idx.extend([0, x.size - 1])
    idx = sorted(set(idx))
    return idx


def _apply_R_axis(ax, R_min, R_max, xlabel=True):
    """Consistent x-axis formatting for the true reduced non-normality ratio."""
    ax.set_xscale("log")
    ax.set_xlim(0.75 * R_min, 1.25 * R_max)

    ticks = _nice_R_ticks(R_min, R_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_R_tick_value(v) for v in ticks])
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if xlabel:
        ax.set_xlabel(r"True reduced ratio $R_{A,\mathrm{ref}}$")


def _mark_threshold(ax, yfrac=0.92):
    """Draw and label the R=1 threshold."""
    ax.axvline(1.0, color="0.72", lw=0.8, linestyle=":", zorder=0, alpha=0.85)
    trans = ax.get_xaxis_transform()
    ax.text(
        1.0, yfrac, r"$R_A=1$",
        transform=trans, ha="left", va="top",
        fontsize=6.7, color="0.45", rotation=90,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.7),
    )


def _available_M_values(df: pd.DataFrame) -> Tuple[int, int]:
    """Return (single_M, baseline_M) robustly."""
    ms = sorted(int(v) for v in pd.unique(df["M"]) if np.isfinite(v))
    if not ms:
        raise ValueError("Input dataframe must contain at least one finite M value.")

    single_M = M_SINGLE if M_SINGLE in ms else min(ms)
    baseline_M = M_BASELINE if M_BASELINE in ms else max(ms)
    return single_M, baseline_M


def _robust_symmetric_clip_value(values, min_clip=DEVIATION_YCLIP_MIN, max_clip=DEVIATION_YCLIP_MAX):
    """
    Robust symmetric clipping value for the signed-deviation panel.

    The full errors are retained in the CSV.  Clipping is only a visual device
    so the central M2/M3 deviations remain legible when a few single-trajectory
    outliers are very large.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(min_clip)
    clip = 1.25 * float(np.nanpercentile(np.abs(vals), 85))
    clip = max(float(min_clip), clip)
    clip = min(float(max_clip), clip)
    return clip


def _plot_clip_boundary_markers(ax, x, y, color, marker, yclip, alpha=0.65):
    """Show out-of-range signed deviations as small boundary triangles."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return 0

    hi = y > yclip
    lo = y < -yclip
    nclip = int(np.sum(hi) + np.sum(lo))

    if np.any(hi):
        ax.plot(
            x[hi], np.full(np.sum(hi), 0.985 * yclip),
            linestyle="None", marker="^", markersize=3.0,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=0.65, alpha=alpha, zorder=6,
        )
    if np.any(lo):
        ax.plot(
            x[lo], np.full(np.sum(lo), -0.985 * yclip),
            linestyle="None", marker="v", markersize=3.0,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=0.65, alpha=alpha, zorder=6,
        )
    return nclip


def _plot_method_with_M_overlay(
    ax,
    df: pd.DataFrame,
    method: str,
    ycol: str,
    single_M: int,
    baseline_M: int,
    label_baseline: bool = False,
    show_single: bool = True,
    marker_scale: float = 1.0,
    yclip: float | None = None,
):
    """
    Plot one method using the paper convention:
        M=25 baseline: solid/standard line with method styling
        M=1 check    : faint open markers with the same method color

    The true-operator reference is independent of M and is shown once.
    If yclip is supplied, values are clipped for display and out-of-range
    samples are indicated by boundary triangles.
    """
    st = METHOD_STYLE[method]

    method_marker_scale = EIGENBASIS_MARKER_SCALE if method == "M1_BF" else 1.0
    base_markersize = 3.15 * marker_scale * method_marker_scale
    base_alpha = st["alpha"]
    clipped_count = 0

    # Baseline line.
    dm_base = df[(df["method"] == method) & (df["M"] == baseline_M)].sort_values("R_star")
    if not dm_base.empty:
        x = dm_base["R_star"].to_numpy(dtype=float)
        y = dm_base[ycol].to_numpy(dtype=float)
        y_plot = np.clip(y, -yclip, yclip) if yclip is not None else y
        ax.plot(
            x,
            y_plot,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markevery=_sparse_mark_indices(x),
            markersize=base_markersize,
            lw=st["lw"],
            alpha=base_alpha,
            label=st["label"] if label_baseline else None,
            zorder=3,
        )
        if yclip is not None:
            clipped_count += _plot_clip_boundary_markers(
                ax, x, y, st["color"], st["marker"], yclip, alpha=max(0.45, base_alpha)
            )

    # Single-trajectory markers.  Do not add them for TRUE because the true
    # operator reference does not depend on the number of training trajectories.
    if show_single and method != "TRUE":
        dm_single = df[(df["method"] == method) & (df["M"] == single_M)].sort_values("R_star")
        if not dm_single.empty:
            x = dm_single["R_star"].to_numpy(dtype=float)
            y = dm_single[ycol].to_numpy(dtype=float)
            y_plot = np.clip(y, -yclip, yclip) if yclip is not None else y
            ax.plot(
                x,
                y_plot,
                color=st["color"],
                linestyle="None",
                marker=st["marker"],
                markevery=_sparse_mark_indices(x),
                markersize=SINGLE_MARKER_SIZE * marker_scale * method_marker_scale,
                markerfacecolor="white",
                markeredgecolor=st["color"],
                markeredgewidth=SINGLE_MARKER_EDGEWIDTH,
                alpha=SINGLE_MARKER_ALPHA,
                label=None,
                zorder=4,
            )
            if yclip is not None:
                clipped_count += _plot_clip_boundary_markers(
                    ax, x, y, st["color"], st["marker"], yclip, alpha=SINGLE_MARKER_ALPHA
                )

    return clipped_count


def plot_figure_S3(df: pd.DataFrame, outdir: str, include_rsigma=False):
    """
    Create the 2 x 2 method-comparison benchmark figure.

    Revision:
    The same benchmark is plotted for M=25 and M=1.  The M=25 ensemble baseline
    is shown as the standard colored line.  The M=1 single-trajectory result is
    overlaid as open markers.  This directly tests whether the synthetic method
    comparison remains meaningful in the empirical regime where M=1.
    """
    set_publication_style()
    os.makedirs(outdir, exist_ok=True)

    single_M, baseline_M = _available_M_values(df)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(7.1, 5.35),
        constrained_layout=False,
    )


    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    methods_all = ["TRUE", "M1_BF", "M2_OPT", "M3_COMM"]
    methods_est = ["M1_BF", "M2_OPT", "M3_COMM"]

    # True reduced non-normality ratio values used as the x-axis.
    dref = df[(df["method"] == "TRUE") & (df["M"] == baseline_M)].sort_values("R_star")
    if dref.empty:
        dref = df[df["method"] == "TRUE"].sort_values("R_star")

    R_values = np.asarray(dref["R_star"].values, dtype=float)
    R_min = float(np.nanmin(R_values))
    R_max = float(np.nanmax(R_values))
    xlim = (0.75 * R_min, 1.25 * R_max)

    # Panel a: inferred/observed R versus true R
    for method in methods_all:
        _plot_method_with_M_overlay(
            ax_a,
            df,
            method,
            "R_hat",
            single_M=single_M,
            baseline_M=baseline_M,
            label_baseline=True,
            show_single=True,
        )

    # y=x reference and R=1 threshold.
    xx = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
    ax_a.plot(xx, xx, color="0.55", lw=0.8, linestyle="--", zorder=0)
    _mark_threshold(ax_a, yfrac=0.86)
    ax_a.axhline(1.0, color="0.70", lw=0.8, linestyle=":", zorder=0)

    _apply_R_axis(ax_a, R_min, R_max)
    ax_a.set_yscale("log")

    positive_Rhat = df["R_hat"].replace([np.inf, -np.inf], np.nan).dropna()
    positive_Rhat = positive_Rhat[positive_Rhat > 0]
    ymin = max(1e-3, 0.75 * float(positive_Rhat.min()))
    ymax = 1.25 * float(positive_Rhat.max())
    ax_a.set_ylim(ymin, ymax)

    ax_a.set_ylabel(r"Inferred $\widehat R_A$")


    # Two-part legend: method identity and M encoding.
    leg_methods = ax_a.legend(loc="upper left", handlelength=2.4)
    style_legend(leg_methods)
    ax_a.add_artist(leg_methods)

    despine(ax_a)
    format_axis(ax_a, logx=True, logy=True)

    # Panel b: signed deviation from true R
    # Eigenbasis is shown only in an inset.
    # Main panel shows Optimization + Commutator only.
    methods_b_main = ["M2_OPT", "M3_COMM"]

    err_vals = df[df["method"].isin(methods_b_main)]["R_signed_err"].values
    err_lim = _robust_symmetric_clip_value(
        err_vals,
        min_clip=DEVIATION_YCLIP_MIN,
        max_clip=15.0,
    )

    for method in methods_b_main:
        _plot_method_with_M_overlay(
            ax_b,
            df,
            method,
            "R_signed_err",
            single_M=single_M,
            baseline_M=baseline_M,
            label_baseline=True,
            show_single=True,
            yclip=err_lim,
        )

    ax_b.axhline(0.0, color="0.45", lw=0.8, linestyle="--", zorder=0)
    _mark_threshold(ax_b, yfrac=0.90)
    _apply_R_axis(ax_b, R_min, R_max)

    ax_b.set_ylabel(r"$\widehat R_A-R_{A,\mathrm{ref}}$")
    ax_b.set_ylim(-1.05 * err_lim, 1.05 * err_lim)

    leg = ax_b.legend(loc="lower right", handlelength=2.4)
    style_legend(leg)

    despine(ax_b)
    format_axis(ax_b, logx=True)

    # Eigenbasis inset in panel b
    ax_b_in = ax_b.inset_axes([0.11, 0.13, 0.35, 0.30])

    _plot_method_with_M_overlay(
        ax_b_in,
        df,
        "M1_BF",
        "R_signed_err",
        single_M=single_M,
        baseline_M=baseline_M,
        label_baseline=False,
        show_single=True,
        marker_scale=0.75,
    )

    ax_b_in.axhline(0.0, color="0.55", lw=0.6, linestyle="--", zorder=0)
    ax_b_in.set_xscale("log")
    ax_b_in.set_xlim(0.75 * R_min, 1.25 * R_max)

    dm_eig_all = df[df["method"] == "M1_BF"].sort_values("R_star")
    eig_err = np.asarray(dm_eig_all["R_signed_err"].values, dtype=float)
    eig_finite = eig_err[np.isfinite(eig_err)]
    if eig_finite.size:
        eig_lim = max(1.0, 1.15 * np.nanpercentile(np.abs(eig_finite), 90))
        eig_lim = min(eig_lim, 10.0)
        ax_b_in.set_ylim(-eig_lim, eig_lim)

    ticks = _nice_R_ticks(R_min, R_max)
    if ticks.size > 4:
        ticks = ticks[[0, len(ticks)//2, -1]]

    ax_b_in.set_xticks(ticks)
    ax_b_in.set_xticklabels([_format_R_tick_value(v) for v in ticks])
    ax_b_in.tick_params(axis="both", labelsize=5.8, pad=1.2)
    ax_b_in.xaxis.set_minor_formatter(mticker.NullFormatter())

    format_axis(ax_b_in, nbx=3, nby=3, logx=True)

    for spine in ax_b_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.45")

    ax_b_in.grid(False)

    # Panel c: plane error versus true R
    for method in methods_est:
        _plot_method_with_M_overlay(
            ax_c,
            df,
            method,
            "theta_deg",
            single_M=single_M,
            baseline_M=baseline_M,
            label_baseline=False,
            show_single=True,
        )

    _mark_threshold(ax_c, yfrac=0.90)
    _apply_R_axis(ax_c, R_min, R_max)
    ax_c.set_ylabel(r"$\theta(\widehat Q,Q_{\rm true})$ (deg)")
    ax_c.set_ylim(0.0, 95.0)
    despine(ax_c)
    format_axis(ax_c, logx=True)

    # Panel d: plane-invariant observable alignment versus true R
    for method in methods_all:
        _plot_method_with_M_overlay(
            ax_d,
            df,
            method,
            "c_plane",
            single_M=single_M,
            baseline_M=baseline_M,
            label_baseline=False,
            show_single=True,
        )

    _mark_threshold(ax_d, yfrac=0.90)
    _apply_R_axis(ax_d, R_min, R_max)
    ax_d.set_ylabel(r"$c_{\mathrm{plane}}$")
    ax_d.set_ylim(0.0, min(1.0, 1.10 * df["c_plane"].max() + 0.02))
    despine(ax_d)
    format_axis(ax_d, logx=True)
    # Companion diagnostic: light-weight R_sigma overlays in panels (a) and (b).
    # The physical systems and planes are identical; only the reduced criterion changes.
    # Keep the user's line/marker palettes and 2x2 panel layout completely unchanged.
    if include_rsigma:
        for method in methods_all:
            dmr = df[(df["method"] == method) & (df["M"] == baseline_M)].sort_values("R_star")
            x = dmr["R_star"].to_numpy(dtype=float)
            y = dmr["R_sigma_hat"].to_numpy(dtype=float)
            ax_a.plot(x, y, color=METHOD_STYLE[method]["color"], ls=":", lw=1.0,
                      alpha=0.28, zorder=2, label=None)
            if method in methods_b_main:
                eb = dmr["R_sigma_signed_err"].to_numpy(dtype=float)
                eb_plot = np.clip(eb, -err_lim, err_lim)
                ax_b.plot(x, eb_plot, color=METHOD_STYLE[method]["color"], ls=":",
                          lw=0.9, alpha=0.28, zorder=2, label=None)
        # The faint trace indicates R_sigma on identical maps, not an uncertainty band.
        ys=df["R_sigma_hat"].to_numpy(dtype=float)
        ys=ys[np.isfinite(ys)&(ys>0)]
        ymax=max(ax_a.get_ylim()[1], 1.15*float(np.max(ys)))
        ax_a.set_ylim(ax_a.get_ylim()[0], ymax)
        ax_a.text(0.97,0.07,r"faint dotted: $R_\sigma$ ($M=25$)",
                  transform=ax_a.transAxes,ha="right",va="bottom",
                  fontsize=6.6,color="0.50",alpha=0.90)

    # Common axis formatting
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.tick_params(axis="x", rotation=0)

    panel_labels = ["a", "b", "c", "d"]
    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], panel_labels):
        add_panel_label(ax, lab)

    # Shared bottom legend (ONLY M-encoding)

    m_handles = [
        Line2D([0], [0],
               color="0.25",
               lw=1.4,
               marker="o",
               markerfacecolor="0.25",
               markeredgecolor="0.25",
               markersize=4.5,
               label=rf"$M={baseline_M}$ (ensemble)"),

        Line2D([0], [0],
               color="0.25",
               lw=0.0,
               marker="o",
               markerfacecolor="white",
               markeredgecolor="0.25",
               markeredgewidth=0.8,
               markersize=4.5,
               label=rf"$M={single_M}$ (single trajectory)")
    ]

    fig.legend(
        handles=m_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        fontsize=7.5,
        handlelength=1.8,
        columnspacing=2.0
    )


    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.90,
        wspace=0.34,
        hspace=0.42,
    )

    tag="RA_Rsigma" if include_rsigma else "RA_only"
    png_path = os.path.join(outdir, f"Fig05_{tag}.png")
    pdf_path = os.path.join(outdir, f"Fig05_{tag}.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved Figure 5 M-comparison with balanced M=1 overlay:")
    print("  PNG:", png_path)
    print("  PDF:", pdf_path)


# 12) Summary table

def print_summary_table(df: pd.DataFrame):
    """
    Print compact summary metrics for the representative benchmark.

    The M column is retained so that the M=1 empirical-regime check can be
    compared directly with the M=25 ensemble baseline.
    """
    d = df[df["method"] != "TRUE"].copy()

    summary = (
        d.groupby(["M", "method", "method_label"])
        .agg(
            median_R_abs_err=("R_abs_err", "median"),
            median_R_scaled_abs_err=("R_scaled_abs_err", "median"),
            median_theta_deg=("theta_deg", "median"),
            median_c_plane=("c_plane", "median"),
            median_relErr_Fhat=("relErr_Fhat", "median"),
        )
        .reset_index()
    )

    method_order = {"M1_BF": 0, "M2_OPT": 1, "M3_COMM": 2}
    summary["order"] = summary["method"].map(method_order)
    summary = summary.sort_values(["M", "order"]).drop(columns=["order"])

    print("\nRepresentative benchmark summary by M")
    print("------------------------------------------------------------")
    print(summary.to_string(index=False))
    print("------------------------------------------------------------")


# 13) Main script

def _load_v19_frozen_results(outdir: str) -> pd.DataFrame:
    """Load corrected V19 numerical evidence; never rerun the original simulation."""
    source_csv=os.path.join(outdir, "T01_fig05_full_results_V19.csv")
    d=pd.read_csv(source_csv)
    required=["method","M","kappa","R_true","R_sigma_true","R_hat","R_sigma_hat",
              "signed_error","principal_angle","c_plane","status"]
    assert all(c in d.columns for c in required), "Missing source columns"
    assert d.shape[0]==232, "Expected 29 x 2 x 4 results"
    assert set(d["method"].unique()) == {"TRUE","M1","M2","M3"}
    assert set(d["M"].unique()) == {1,25}
    assert d["kappa"].nunique()==29
    assert np.isclose(d["R_true"],d["R_true"].iloc[0]).sum() < len(d)
    # NA values are scientifically meaningful: rates cannot be computed for
    # inadmissible reduced maps and must not be replaced or interpolated.
    d=d.rename(columns={
        "R_true":"R_star", "principal_angle":"theta_deg",
        "signed_error":"R_signed_err"})
    d["method"]=d["method"].map({"TRUE":"TRUE","M1":"M1_BF","M2":"M2_OPT","M3":"M3_COMM"})
    d["R_sigma_signed_err"]=d["R_sigma_hat"]-d["R_sigma_true"]
    d["R_abs_err"]=d["R_signed_err"].abs()
    d["R_scaled_abs_err"]=d["R_abs_err"]/(1+d["R_star"].abs())
    d["relErr_Fhat"]=d["F_rel_error"]
    d["method_label"]=d["method"].map({k:v["label"] for k,v in METHOD_STYLE.items()})
    for meth in ["M1_BF","M2_OPT","M3_COMM"]:
        assert d.loc[d.method==meth].shape[0]==58
    d.to_csv(os.path.join(outdir, "Fig05_plotting_table_corrected.csv"),index=False)
    return d


def main():
    from pathlib import Path
    root=Path(__file__).resolve().parent
    outdir=root/"outputs_synthetic_figures"
    outdir.mkdir(exist_ok=True)
    df=_load_v19_frozen_results(str(outdir))
    print("Figure 5: using corrected saved V19 benchmark, NOT simulating")
    print("Missing R_A values:",int(df["R_hat"].isna().sum()))
    print_summary_table(df)
    plot_figure_S3(df,str(outdir),include_rsigma=False)
    plot_figure_S3(df,str(outdir),include_rsigma=True)
    print("Files in",outdir)


if __name__ == "__main__":
    main()
