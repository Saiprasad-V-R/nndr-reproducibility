# -*- coding: utf-8 -*-
"""
Figure S3: Scaling of mechanism recovery with data availability and dimension

Standalone script. No nndr package required.

Phase-3 revision responding to Fig. 5 feedback:
    * Scaling is evaluated at three true reduced non-normality regimes: R_true≈0.2, 1, and 10.
    * The actual median R_true reached in each regime is stored in the CSV and printed for captioning.
    * Dense/low-data tests are retained, including small T_train values.
    * The main text should state the data-efficiency point: NNDR estimates a reduced 2-D
      non-normality ratio from O(N) rather than O(N^2) samples.

Split-script revision: this file computes and writes CSV data only.

Final M-consistency revision:
    * Main scaling panels remain unchanged and use the original M=25 baseline.
    * The N-scaling and T_train-scaling rows include compact insets that compare
      M=1 against M=25 at the sensitive regime R_true≈1.
    * Insets are deliberately small, title-free, and use lighter markers to avoid
      cluttering the main scaling laws.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv

# 1) Publication style

def set_publication_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.75,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.35,
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
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)


def format_axis(ax, nbx=5, nby=5, set_x_locator=True, set_y_locator=True):
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    if set_x_locator:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbx))
    if set_y_locator:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nby))


def style_legend(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.86)
    frame.set_edgecolor("none")

def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

# 2) Utilities

def random_orthogonal(N, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))

def principal_angle_deg(Q1, Q2):
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = float(np.arccos(s[-1]))
    return float(np.degrees(theta))

def Kc_from_Delta(Delta):
    Delta = float(np.clip(Delta, 0.0, 1.0 - 1e-12))
    s = np.sqrt(1.0 - Delta**2)
    return float(np.sqrt(s / (1.0 - s + 1e-15)))

def twoD_metrics_from_Gamma(Gamma):
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

def compute_indices_from_Q(A, Q):
    Gamma = Q.T @ A @ Q
    Delta, kappa2D, K = twoD_metrics_from_Gamma(Gamma)
    Kc = Kc_from_Delta(Delta)
    KKc = K / (Kc + 1e-15)
    return Delta, kappa2D, K, Kc, KKc

def log_ratio_error(est, true, eps=1e-12):
    est = max(float(est), eps)
    true = max(float(true), eps)
    return abs(np.log10(est / true))


def find_kappa_for_target_R(N, target_R, cfg_seed_base=1, rho=0.95, delta_sigma=0.05,
                            iters_opt=250, kappa_min=1.0, kappa_max=2000.0):
    """Choose kappa so the true reduced ratio K/Kc(Delta) is close to target_R.

    The mapping kappa -> R_true depends mildly on N and on the fixed random
    operator configuration.  Therefore the scaling experiments should not use
    the same kappa for all N when the goal is to compare fixed R regimes.
    """
    cfg = make_fixed_config(N=N, rho=rho, delta_sigma=delta_sigma, seed=cfg_seed_base + N)
    grid = np.geomspace(kappa_min, kappa_max, 240)
    Rvals = []
    for kap in grid:
        F = build_F_from_config(cfg, kappa=kap)
        Qtrue = method2_optimization(F, iters=iters_opt, seed=0)
        _, _, _, _, Rtrue = compute_indices_from_Q(F, Qtrue)
        Rvals.append(Rtrue)
    Rvals = np.asarray(Rvals, dtype=float)
    j = int(np.nanargmin(np.abs(np.log10(np.maximum(Rvals, 1e-12)) - np.log10(target_R))))
    return float(grid[j]), float(Rvals[j])

# 3) Controlled non-normal VAR generator

def make_fixed_config(N, rho=0.95, delta_sigma=0.05, seed=1):
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

def build_F_from_config(cfg, kappa):
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

def simulate_var_with_fixed_noise(F, noise, x0=None):
    T, N = noise.shape
    x = np.zeros(N) if x0 is None else np.asarray(x0, dtype=float).copy()

    X = np.zeros((N, T))
    for k in range(T):
        x = F @ x + noise[k]
        X[:, k] = x

    return X

# 4) Ridge regression

def ridge_fit_F(X, Y, lam):
    N = X.shape[0]
    XXT = X @ X.T
    YXT = Y @ X.T
    A = XXT + float(lam) * np.eye(N)
    Fhat = np.linalg.solve(A.T, YXT.T).T
    return Fhat

def select_lambda_ridge(X, Y, lambdas, frac_train=0.8, seed=0):
    rng = np.random.default_rng(seed)
    T_eff = X.shape[1]

    idx = np.arange(T_eff)
    rng.shuffle(idx)

    ntr = max(2, int(frac_train * T_eff))
    itr = idx[:ntr]
    iva = idx[ntr:]

    if iva.size == 0:
        iva = itr

    Xtr, Ytr = X[:, itr], Y[:, itr]
    Xva, Yva = X[:, iva], Y[:, iva]

    mse = []
    for lam in lambdas:
        Fhat = ridge_fit_F(Xtr, Ytr, lam)
        R = Yva - Fhat @ Xva
        mse.append(float(np.mean(R * R)))

    mse = np.array(mse)
    j = int(np.argmin(mse))
    return float(lambdas[j]), mse

def stack_VAR_pairs_from_trajectories(X_list, x0):
    Xcols = []
    Ycols = []

    for X in X_list:
        N, T = X.shape
        x0_vec = np.asarray(x0).reshape(N)
        Xin = np.column_stack([x0_vec, X[:, :-1]])
        Yout = X.copy()

        Xcols.append(Xin)
        Ycols.append(Yout)

    Xstack = np.concatenate(Xcols, axis=1)
    Ystack = np.concatenate(Ycols, axis=1)
    return Xstack, Ystack

# 5) Extraction methods

def extract_nr_by_optimization(A, iters=250, seed=0):
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

    Q, _ = np.linalg.qr(np.column_stack([r, n]))
    return Q

def method1_eigenbasis_svd(A):
    eigvals, P = np.linalg.eig(A)
    U, svals, _ = np.linalg.svd(P, full_matrices=False)

    z = U[:, -1]
    if np.iscomplexobj(z):
        zr = np.real(z)
        zi = np.imag(z)
        if np.linalg.norm(zr) >= np.linalg.norm(zi):
            n = zr
        else:
            n = zi
    else:
        n = z

    n = n / (np.linalg.norm(n) + 1e-15)

    r_raw = (np.eye(A.shape[0]) - np.outer(n, n)) @ (A @ n)
    r = r_raw / (np.linalg.norm(r_raw) + 1e-15)

    Q, _ = np.linalg.qr(np.column_stack([r, n]))
    return Q

def method2_optimization(A, iters=250, seed=0):
    return extract_nr_by_optimization(A, iters=iters, seed=seed)

def method3_commutator(A):
    B = A @ A.T - A.T @ A
    evals, evecs = np.linalg.eigh(B)

    w_plus = evecs[:, -1]
    w_minus = evecs[:, 0]

    Q, _ = np.linalg.qr(np.column_stack([w_plus, w_minus]))
    return Q

# 6) One benchmark condition

def run_one_condition(
    N, M, T_train, kappas, cfg_seed_base=1, rho=0.95, delta_sigma=0.05,
    sigma_noise=0.05, lambdas=None, frac_train=0.8, seed_train_bank=111,
    seed_cv=333, iters_opt=250,
):
    if lambdas is None:
        lambdas = np.logspace(-8, 2, 21)

    cfg = make_fixed_config(N=N, rho=rho, delta_sigma=delta_sigma, seed=cfg_seed_base + N)
    rng_bank = np.random.default_rng(seed_train_bank + 1_000_000 * N + 10_000 * M + 100 * T_train)
    noise_bank = sigma_noise * rng_bank.standard_normal((M, T_train, N))

    rows = []
    for kappa in kappas:
        F = build_F_from_config(cfg, kappa=kappa)

        Qtrue = method2_optimization(F, iters=iters_opt, seed=0)
        _, _, _, _, KKc_true = compute_indices_from_Q(F, Qtrue)

        X_list = []
        for m in range(M):
            Xm = simulate_var_with_fixed_noise(F, noise_bank[m], x0=np.zeros(N))
            X_list.append(Xm)

        Xstack, Ystack = stack_VAR_pairs_from_trajectories(X_list, x0=np.zeros(N))

        lam_star, _ = select_lambda_ridge(Xstack, Ystack, lambdas=lambdas, frac_train=frac_train, seed=seed_cv)
        Fhat = ridge_fit_F(Xstack, Ystack, lam_star)

        relErr = float(np.linalg.norm(Fhat - F, ord="fro") / (np.linalg.norm(F, ord="fro") + 1e-15))

        method_outputs = [
            ("M1_BF", method1_eigenbasis_svd(Fhat)),
            ("M2_OPT", method2_optimization(Fhat, iters=iters_opt, seed=0)),
            ("M3_COMM", method3_commutator(Fhat)),
        ]

        for method_name, Q in method_outputs:
            _, _, _, _, KKc_est = compute_indices_from_Q(Fhat, Q)
            angle = principal_angle_deg(Qtrue, Q)

            rows.append({
                "N": int(N),
                "M": int(M),
                "T_train": int(T_train),
                "kappa": float(kappa),
                "method": method_name,
                "lam_star": float(lam_star),
                "relErr_Fhat": float(relErr),
                "KKc_true": float(KKc_true),
                "target_R_regime": np.nan,
                "regime_label": "",
                "KKc_est": float(KKc_est),
                "angle_deg": float(angle),
                "log_KKc_error": float(log_ratio_error(KKc_est, KKc_true)),
                "rho_Fhat": spectral_radius(Fhat),
            })

    return rows

# 7) Scaling experiments

def median_by_method(rows, x_key):
    methods = ["M1_BF", "M2_OPT", "M3_COMM"]
    x_values = sorted(set(row[x_key] for row in rows))
    out = []

    for x in x_values:
        rx = [r for r in rows if r[x_key] == x]
        rel_err_vals = [r["relErr_Fhat"] for r in rx]
        rel_err_med = float(np.median(rel_err_vals))

        for method in methods:
            rm = [r for r in rx if r["method"] == method]
            out.append({
                x_key: x,
                "method": method,
                "median_angle_deg": float(np.median([r["angle_deg"] for r in rm])),
                "median_log_KKc_error": float(np.median([r["log_KKc_error"] for r in rm])),
                "median_relErr_Fhat": rel_err_med,
            })
    return out

def run_all_scaling_experiments():
    # Target true reduced non-normality regimes.
    # Labels use the reduced non-normality wording.
    regimes = [
        (0.20, r"$R_{\rm true}\approx0.2$"),
        (1.00, r"$R_{\rm true}\approx1$"),
        (10.0, r"$R_{\rm true}\approx10$"),
    ]

    # Main scaling axes.
    M_list = [1, 2, 5, 10, 15, 25, 50, 100]
    N_list = [50, 100, 150, 200, 250, 300, 350, 400]
    T_list = [3, 4, 10, 25, 50, 100, 200, 400, 800]

    rows_all = []

    def run_and_tag(N, M, T_train, target_R, regime_label):
        kappa_star, R_reached = find_kappa_for_target_R(N=N, target_R=target_R)
        rows = run_one_condition(N=N, M=M, T_train=T_train, kappas=[kappa_star])
        for r in rows:
            r["target_R_regime"] = float(target_R)
            r["regime_label"] = regime_label
            r["R_true_reached"] = float(R_reached)
        return rows

    print("\nRunning scaling with M at three true R regimes...")
    rows_M = []
    for target_R, regime_label in regimes:
        for M in M_list:
            rows = run_and_tag(N=100, M=M, T_train=200, target_R=target_R, regime_label=regime_label)
            for r in rows:
                r["scaling_axis"] = "M"
            rows_M.extend(rows)

    print("\nRunning scaling with N at three true R regimes using the main M=25 baseline...")
    rows_N = []
    for target_R, regime_label in regimes:
        for N in N_list:
            rows = run_and_tag(N=N, M=25, T_train=200, target_R=target_R, regime_label=regime_label)
            for r in rows:
                r["scaling_axis"] = "N"
            rows_N.extend(rows)

    print("\nRunning scaling with T_train at three true R regimes using the main M=25 baseline...")
    rows_T = []
    for target_R, regime_label in regimes:
        for T_train in T_list:
            rows = run_and_tag(N=100, M=25, T_train=T_train, target_R=target_R, regime_label=regime_label)
            for r in rows:
                r["scaling_axis"] = "T_train"
            rows_T.extend(rows)

    # Didier M=1 consistency checks.
    # These rows are NOT included in the main summaries. They are used only
    # to draw compact insets in panels (c)--(f), at the sensitive regime R≈1.
    rows_N_M1_inset = []
    target_R, regime_label = regimes[1]
    print("\nRunning M=1 inset check for the N-scaling row at R_true≈1...")
    for N in N_list:
        rows = run_and_tag(N=N, M=1, T_train=200, target_R=target_R, regime_label=regime_label)
        for r in rows:
            r["scaling_axis"] = "N_M1_inset"
        rows_N_M1_inset.extend(rows)

    rows_T_M1_inset = []
    print("\nRunning M=1 inset check for the T_train-scaling row at R_true≈1...")
    for T_train in T_list:
        rows = run_and_tag(N=100, M=1, T_train=T_train, target_R=target_R, regime_label=regime_label)
        for r in rows:
            r["scaling_axis"] = "T_train_M1_inset"
        rows_T_M1_inset.extend(rows)

    rows_all.extend(rows_M)
    rows_all.extend(rows_N)
    rows_all.extend(rows_T)
    rows_all.extend(rows_N_M1_inset)
    rows_all.extend(rows_T_M1_inset)

    # Print actual R values for the caption.
    print("\nActual median R_true reached by target regime:")
    for target_R, regime_label in regimes:
        vals = [r["R_true_reached"] for r in rows_all if abs(r["target_R_regime"] - target_R) < 1e-12]
        print(f"  target {target_R:g}: median={np.median(vals):.3g}, min={np.min(vals):.3g}, max={np.max(vals):.3g}")

    return (
        rows_all,
        median_by_method_regime(rows_M, "M"),
        median_by_method_regime(rows_N, "N"),
        median_by_method_regime(rows_T, "T_train"),
    )

def median_by_method_regime(rows, x_key):
    methods = ["M1_BF", "M2_OPT", "M3_COMM"]
    x_values = sorted(set(row[x_key] for row in rows))
    regimes = sorted(set(row["target_R_regime"] for row in rows))
    out = []

    for reg in regimes:
        for x in x_values:
            rx = [r for r in rows if r[x_key] == x and r["target_R_regime"] == reg]
            if not rx:
                continue
            rel_err_vals = [r["relErr_Fhat"] for r in rx]
            rel_err_med = float(np.median(rel_err_vals))
            R_reached_med = float(np.median([r.get("R_true_reached", np.nan) for r in rx]))
            regime_label = rx[0].get("regime_label", f"R={reg:g}")

            for method in methods:
                rm = [r for r in rx if r["method"] == method]
                if not rm:
                    continue
                out.append({
                    x_key: x,
                    "method": method,
                    "target_R_regime": float(reg),
                    "regime_label": regime_label,
                    "median_R_true_reached": R_reached_med,
                    "median_angle_deg": float(np.median([r["angle_deg"] for r in rm])),
                    "median_log_KKc_error": float(np.median([r["log_KKc_error"] for r in rm])),
                    "median_relErr_Fhat": rel_err_med,
                })
    return out

# 8) CSV helpers

def save_rows_csv(rows, path):
    if len(rows) == 0: return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# 9) Main: write data only

if __name__ == "__main__":
    outdir = "outputs_synthetic_figures"
    os.makedirs(outdir, exist_ok=True)

    rows_all, summary_M, summary_N, summary_T = run_all_scaling_experiments()

    raw_path = os.path.join(outdir, "Figure_S3_all_scaling_raw_rows.csv")
    m_path = os.path.join(outdir, "Figure_S3_scaling_M_summary.csv")
    n_path = os.path.join(outdir, "Figure_S3_scaling_N_summary.csv")
    t_path = os.path.join(outdir, "Figure_S3_scaling_T_summary.csv")

    save_rows_csv(rows_all, raw_path)
    save_rows_csv(summary_M, m_path)
    save_rows_csv(summary_N, n_path)
    save_rows_csv(summary_T, t_path)

    print("\nSaved scaling data:")
    print("  raw rows :", raw_path)
    print("  M summary:", m_path)
    print("  N summary:", n_path)
    print("  T summary:", t_path)
    print("\nNow run fig6_scaling_plot_from_csv_spyder.py to make the figure.")
