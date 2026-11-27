# -*- coding: utf-8 -*-
"""
Figure S4: Causal tracking of time-varying non-normality

Standalone script. No nndr package required.

Figure layout:
    4 rows x 1 column, single-column width

Columns:
    left  : decreasing kappa(t), 200 -> 1
    right : increasing kappa(t), 1 -> 200

Rows:
    top    : imposed kappa(t), instantaneous and trailing-window mean
    bottom : inferred R(t) = K/Kc(Delta), compared with trailing-window reference

Spyder usage:
    %runfile /path/to/figure_S4_timevarying_kappa_tracking.py --wdir /path/to/project_folder
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker


# 1) Publication style and method styles

def set_publication_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 7.8,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.05,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
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
        "lw": 1.15,
        "alpha": 0.95,
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


def despine(ax):
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)


def format_axis(ax, nbx=5, nby=5):
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbx))
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
        -0.14,
        1.08,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_method_curve(ax, x, y, method, label=True, markevery=None):
    st = METHOD_STYLE[method]
    ax.plot(
        x,
        y,
        color=st["color"],
        linestyle=st["linestyle"],
        marker=st["marker"],
        markersize=3.8,
        markerfacecolor="white" if method != "TRUE" else st["color"],
        markeredgecolor=st["color"],
        markeredgewidth=0.85,
        linewidth=st["lw"],
        alpha=st["alpha"],
        label=st["label"] if label else None,
        markevery=markevery,
    )


# 2) Basic utilities

def random_orthogonal(N, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


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
    R = K / (Kc + 1e-15)
    return Delta, kappa2D, K, Kc, R


def principal_angle_deg(Q1, Q2):
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.degrees(np.arccos(s[-1])))


# 3) Controlled non-normal operator family

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

    return U @ Sigma @ V.T @ Lambda @ V @ Sigma_inv @ U.T


# 4) Operator extraction methods

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


def method2_optimization(A, iters=250, seed=0):
    return extract_nr_by_optimization(A, iters=iters, seed=seed)


def method3_commutator(A):
    B = A @ A.T - A.T @ A
    evals, evecs = np.linalg.eigh(B)

    w_plus = evecs[:, -1]
    w_minus = evecs[:, 0]

    Q, _ = np.linalg.qr(np.column_stack([w_plus, w_minus]))
    return Q


# 5) VAR simulation and ridge estimation

def simulate_timevarying_var(F_seq, noise_bank):
    """
    Simulate M trajectories:
        x_{t+1} = F_t x_t + eta_t

    Parameters
    ----------
    F_seq : ndarray, shape (T, N, N)
    noise_bank : ndarray, shape (M, T, N)

    Returns
    -------
    S_list : list of ndarray
        Each array has shape (N, T+1), including x_0.
    """
    M, T, N = noise_bank.shape
    S_list = []

    for m in range(M):
        S = np.zeros((N, T + 1))
        x = np.zeros(N)

        for t in range(T):
            x = F_seq[t] @ x + noise_bank[m, t]
            S[:, t + 1] = x

        S_list.append(S)

    return S_list


def stack_window_pairs(S_list, t0, t1):
    """
    Stack one-step pairs from a window [t0, t1).

    Predictors:
        x_t for t = t0, ..., t1-1

    Responses:
        x_{t+1} for t = t0, ..., t1-1
    """
    Xcols = []
    Ycols = []

    for S in S_list:
        Xcols.append(S[:, t0:t1])
        Ycols.append(S[:, t0 + 1:t1 + 1])

    X = np.concatenate(Xcols, axis=1)
    Y = np.concatenate(Ycols, axis=1)
    return X, Y


def ridge_fit_F(X, Y, lam):
    N = X.shape[0]
    XXT = X @ X.T
    YXT = Y @ X.T
    A = XXT + float(lam) * np.eye(N)
    return np.linalg.solve(A.T, YXT.T).T


def select_lambda_ridge(X, Y, lambdas, frac_train=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = X.shape[1]

    idx = np.arange(n_samples)
    rng.shuffle(idx)

    ntr = max(2, int(frac_train * n_samples))
    itr = idx[:ntr]
    iva = idx[ntr:]

    if iva.size == 0:
        iva = itr

    Xtr = X[:, itr]
    Ytr = Y[:, itr]
    Xva = X[:, iva]
    Yva = Y[:, iva]

    mse = []
    for lam in lambdas:
        Fhat = ridge_fit_F(Xtr, Ytr, lam)
        R = Yva - Fhat @ Xva
        mse.append(float(np.mean(R * R)))

    mse = np.asarray(mse)
    j = int(np.argmin(mse))
    return float(lambdas[j])


# 6) Time-varying experiment

def make_kappa_schedule(T, kappa_start, kappa_end):
    """
    Smooth log-linear variation from kappa_start to kappa_end.
    """
    log_start = np.log(kappa_start)
    log_end = np.log(kappa_end)
    return np.exp(np.linspace(log_start, log_end, T))


def run_timevarying_case(
    case_name,
    kappa_start,
    kappa_end,
    N=100,
    M=25,
    Ttotal=600,
    W=80,
    step=10,
    rho=0.95,
    delta_sigma=0.05,
    sigma_noise=0.05,
    cfg_seed=7,
    noise_seed=1234,
    opt_iters=250,
):
    """
    Causal trailing-window tracking experiment.
    """
    print(f"\nRunning time-varying case: {case_name}")

    cfg = make_fixed_config(
        N=N,
        rho=rho,
        delta_sigma=delta_sigma,
        seed=cfg_seed,
    )

    kappa_t = make_kappa_schedule(Ttotal, kappa_start, kappa_end)

    F_seq = np.zeros((Ttotal, N, N))
    for t in range(Ttotal):
        F_seq[t] = build_F_from_config(cfg, kappa_t[t])

    rng = np.random.default_rng(noise_seed)
    noise_bank = sigma_noise * rng.standard_normal((M, Ttotal, N))

    S_list = simulate_timevarying_var(F_seq, noise_bank)

    # A slightly smaller lambda grid keeps the standalone script fast.
    lambdas = np.logspace(-6, 1, 12)

    rows = []

    window_ends = np.arange(W, Ttotal + 1, step)

    for count, tend in enumerate(window_ends, start=1):
        t0 = tend - W
        t1 = tend

        X, Y = stack_window_pairs(S_list, t0, t1)

        lam_star = select_lambda_ridge(
            X,
            Y,
            lambdas=lambdas,
            frac_train=0.8,
            seed=333 + tend,
        )

        Fhat = ridge_fit_F(X, Y, lam_star)

        Fbar = np.mean(F_seq[t0:t1], axis=0)

        Qref = method2_optimization(Fbar, iters=opt_iters, seed=0)
        _, _, _, _, R_ref = compute_indices_from_Q(Fbar, Qref)

        Qm2 = method2_optimization(Fhat, iters=opt_iters, seed=0)
        _, _, _, _, R_m2 = compute_indices_from_Q(Fhat, Qm2)

        Qm3 = method3_commutator(Fhat)
        _, _, _, _, R_m3 = compute_indices_from_Q(Fhat, Qm3)

        angle_m2 = principal_angle_deg(Qref, Qm2)
        angle_m3 = principal_angle_deg(Qref, Qm3)

        relErr_bar = float(
            np.linalg.norm(Fhat - Fbar, ord="fro") /
            (np.linalg.norm(Fbar, ord="fro") + 1e-15)
        )

        rows.append({
            "case": case_name,
            "t_end": int(tend),
            "t0": int(t0),
            "t1": int(t1),
            "kappa_end": float(kappa_t[tend - 1]),
            "kappa_window_mean": float(np.mean(kappa_t[t0:t1])),
            "lambda_star": float(lam_star),
            "R_trailing": float(R_ref),
            "R_M2": float(R_m2),
            "R_M3": float(R_m3),
            "dR_M2": float(R_m2 - R_ref),
            "dR_M3": float(R_m3 - R_ref),
            "angle_M2_deg": float(angle_m2),
            "angle_M3_deg": float(angle_m3),
            "relErr_Fhat_to_Fbar": float(relErr_bar),
        })

        if count % 10 == 0 or count == 1 or count == len(window_ends):
            print(
                f"  window {count:03d}/{len(window_ends):03d}, "
                f"t_end={tend}, "
                f"kappa_mean={np.mean(kappa_t[t0:t1]):.2f}, "
                f"R_ref={R_ref:.2f}, R_M2={R_m2:.2f}, R_M3={R_m3:.2f}"
            )

    return rows


# 7) CSV helper

def save_rows_csv(rows, path):
    if len(rows) == 0:
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rows_to_arrays(rows):
    keys = rows[0].keys()
    out = {}
    for key in keys:
        try:
            out[key] = np.asarray([float(r[key]) for r in rows])
        except Exception:
            out[key] = np.asarray([r[key] for r in rows])
    return out


# 8) Plot time-varying tracking figure, four-panel version


def _nice_symmetric_ylim(values, pad=1.15, min_span=0.15):
    """Return symmetric y-limits around zero for signed errors."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    m = float(np.nanmax(np.abs(vals)))
    m = max(m * pad, min_span)
    return -m, m


def _sparse_markevery(n, target=12):
    """Return a clean markevery value for dense time curves."""
    if n <= target:
        return 1
    return max(1, int(np.ceil(n / target)))


def _set_time_axis(ax, t):
    ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    despine(ax)


def _plot_R_equals_one(ax):
    ax.axhline(
        1.0,
        color="0.45",
        linestyle="--",
        linewidth=0.95,
        zorder=0,
    )
    ax.text(
        0.985,
        1.0,
        r"$R=1$",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.35",
    )


def plot_figure_timevarying_fourpanel(rows, outdir, cfg):
    """
    Four-panel time-varying tracking figure arranged in a 2x2 grid:

    [ Top-Left (a)  ]   [ Top-Right (b) ]
    [ Bottom-Left (c) ]   [ Bottom-Right (d) ]
    """
    set_publication_style()
    os.makedirs(outdir, exist_ok=True)

    arr = rows_to_arrays(rows)

    t = arr["t_end"]
    kappa_end = arr["kappa_end"]
    kappa_mean = arr["kappa_window_mean"]
    R_ref = arr["R_trailing"]
    R_m2 = arr["R_M2"]
    R_m3 = arr["R_M3"]
    dR_m2 = arr["dR_M2"]
    dR_m3 = arr["dR_M3"]

    markevery = _sparse_markevery(len(t), target=13)

    # Changed to 2 rows x 2 columns. Adjusted figure size for a 2x2 layout.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(6.85, 4.25),
        constrained_layout=False,
        sharex=True, # Shares x-axis across columns dynamically
    )

    # (a) Top-Left: Imposed kappa(t)
    ax = axes[0, 0]
    ax.plot(
        t,
        kappa_end,
        color="0.10",
        linewidth=1.15,
        label=r"inst. $\kappa(t)$",
    )
    ax.plot(
        t,
        kappa_mean,
        color="0.55",
        linewidth=1.00,
        linestyle="--",
        label=r"window mean $\kappa$",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"input $\kappa$")
    legend = ax.legend(loc="upper left", handlelength=1.5, borderpad=0.25, labelspacing=0.25)
    style_legend(legend)
    add_panel_label(ax, "a")
    _set_time_axis(ax, t)

    # (b) Top-Right: causal trailing-window reference R_trailing(t)
    ax = axes[0, 1]
    ax.plot(
        t,
        R_ref,
        color=METHOD_STYLE["TRUE"]["color"],
        linestyle=METHOD_STYLE["TRUE"]["linestyle"],
        linewidth=1.15,
        marker=METHOD_STYLE["TRUE"]["marker"],
        markersize=2.8,
        markevery=markevery,
        label=r"$R_{\rm trailing}$",
    )
    _plot_R_equals_one(ax)
    ax.set_yscale("log")
    ax.set_ylabel(r"$R_{\rm trailing}$")
    add_panel_label(ax, "b")
    _set_time_axis(ax, t)

    # (c) Bottom-Left: inferred tracking curves overlaid
    ax = axes[1, 0]
    ax.plot(
        t,
        R_ref,
        color=METHOD_STYLE["TRUE"]["color"],
        linestyle="-",
        linewidth=1.10,
        marker="o",
        markersize=2.6,
        markevery=markevery,
        label=r"$R_{\rm trailing}$",
        zorder=3,
    )
    plot_method_curve(ax, t, R_m2, "M2_OPT", label=True, markevery=markevery)
    plot_method_curve(ax, t, R_m3, "M3_COMM", label=True, markevery=markevery)
    _plot_R_equals_one(ax)
    ax.set_yscale("log")
    ax.set_ylabel(r"inferred $\widehat R$")
    ax.set_xlabel(r"window end time $t_{\rm end}$")
    add_panel_label(ax, "c")
    _set_time_axis(ax, t)

    # (d) Bottom-Right: signed residuals
    ax = axes[1, 1]
    plot_method_curve(ax, t, dR_m2, "M2_OPT", label=True, markevery=markevery)
    plot_method_curve(ax, t, dR_m3, "M3_COMM", label=True, markevery=markevery)
    ax.axhline(0.0, color="0.35", linestyle="-", linewidth=0.85, zorder=0)
    ymin, ymax = _nice_symmetric_ylim(np.r_[dR_m2, dR_m3], pad=1.20, min_span=0.20)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r"$\widehat R-R_{\rm trailing}$")
    ax.set_xlabel(r"window end time $t_{\rm end}$")
    add_panel_label(ax, "d")
    _set_time_axis(ax, t)

    # Clean grids
    for row in axes:
        for ax in row:
            ax.grid(False)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        handlelength=1.7,
        borderpad=0.25,
        labelspacing=0.25,
        columnspacing=1.0,
    )
    style_legend(legend)

    # Optimized spacing parameters for 2x2 layout
    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.21,
        top=0.94,
        wspace=0.28,
        hspace=0.22,
    )

    png_path = os.path.join(outdir, "fig6_timevarying_tracking_2x2_layout.png")
    pdf_path = os.path.join(outdir, "fig6_timevarying_tracking_2x2_layout.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.show()

    print("\nSaved 2x2 time-varying tracking figure:")
    print("  PNG:", png_path)
    print("  PDF:", pdf_path)

# 9) Main

if __name__ == "__main__":

    outdir = "outputs_synthetic_figures"
    os.makedirs(outdir, exist_ok=True)

    N = 100
    Ttotal = 500
    W = 80
    step = 10

    kappa_min = 1.0
    kappa_max = 200.0

    # CASE 1: M = 25 (current manuscript setting)
    M = 25

    rows_M25 = run_timevarying_case(
        case_name="increasing_M25",
        kappa_start=kappa_min,
        kappa_end=kappa_max,
        N=N,
        M=M,
        Ttotal=Ttotal,
        W=W,
        step=step,
        cfg_seed=7,
        noise_seed=5678,
    )

    save_rows_csv(
        rows_M25,
        os.path.join(outdir, "fig6_timevarying_increasing_M25_rows.csv"),
    )

    plot_figure_timevarying_fourpanel(
        rows=rows_M25,
        outdir=outdir,
        cfg={"N": N, "M": M, "Ttotal": Ttotal, "W": W,
             "step": step, "kappa_min": kappa_min, "kappa_max": kappa_max},
    )

    # CASE 2: M = 1 (referee-requested identifiability check)
    M = 1

    rows_M1 = run_timevarying_case(
        case_name="increasing_M1",
        kappa_start=kappa_min,
        kappa_end=kappa_max,
        N=N,
        M=M,
        Ttotal=Ttotal,
        W=W,
        step=step,
        cfg_seed=7,
        noise_seed=5678,
    )

    save_rows_csv(
        rows_M1,
        os.path.join(outdir, "fig6_timevarying_increasing_M1_rows.csv"),
    )

    # Optional but recommended (strong reviewer response):
    # reuse same plotting routine for direct visual comparison
    plot_figure_timevarying_fourpanel(
        rows=rows_M1,
        outdir=outdir,
        cfg={"N": N, "M": M, "Ttotal": Ttotal, "W": W,
             "step": step, "kappa_min": kappa_min, "kappa_max": kappa_max},
    )

    print("\nCompleted both ensemble (M=25) and single-realization (M=1) validation.")
