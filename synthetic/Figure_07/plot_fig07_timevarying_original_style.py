# -*- coding: utf-8 -*-
"""
Time-varying manuscript figure (submitted Fig. 7) in the original style,
updated to use corrected R_A and companion R_sigma. Includes both M=25 and
M=1 on the same comparison panels, with M=1 shown in lighter tones.
"""
from __future__ import annotations

import os, csv, json, hashlib, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from scipy.linalg import logm, schur
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, 'outputs_synthetic_figures')
os.makedirs(OUTDIR, exist_ok=True)


def lighten_color(color, amount=0.55):
    c = np.array(to_rgb(color))
    return tuple((1 - amount) * c + amount * np.ones(3))


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
    ax.text(-0.14, 1.08, f"({label})", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="bottom", ha="left")


def plot_method_curve(ax, x, y, method, label=None, markevery=None, tone='full', rsigma=False):
    st = METHOD_STYLE[method]
    color = st['color']
    alpha = st['alpha']
    lw = st['lw']
    linestyle = st['linestyle']
    marker = st['marker']
    markersize = 3.8
    mew = 0.85
    if tone == 'light':
        color = lighten_color(color, amount=0.55)
        alpha = 0.95
    if rsigma:
        linestyle = ':'
        lw = max(0.95, lw - 0.15)
        markersize = 3.0
        mew = 0.75
        if tone == 'full':
            color = lighten_color(st['color'], amount=0.35)
            alpha = 0.80
        else:
            color = lighten_color(st['color'], amount=0.72)
            alpha = 0.90
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker,
            markersize=markersize,
            markerfacecolor='white' if method != 'TRUE' else color,
            markeredgecolor=color, markeredgewidth=mew,
            linewidth=lw, alpha=alpha, label=label, markevery=markevery)

# 2) Basic utilities and corrected ratios

def random_orthogonal(N, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def principal_angle_deg(Q1, Q2):
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.degrees(np.arccos(s[-1])))


def corrected_reduced_ratios(Gamma, delta_t=1.0):
    G = np.asarray(Gamma, dtype=float)
    eigenvalues = np.linalg.eigvals(G)
    if max(abs(eigenvalues)) >= 1:
        return np.nan, np.nan
    def departure(A):
        T, _ = schur(A, output='real')
        if abs(T[1,0]) <= 1e-14 * max(1.0, np.linalg.norm(A)):
            return abs(float(T[0,1]))
        if T[0,1]*T[1,0] < 0:
            return abs(abs(float(T[0,1])) - abs(float(T[1,0])))
        eig = np.linalg.eigvals(A)
        return float(np.sqrt(max(0., np.linalg.norm(A, 'fro')**2 - np.sum(abs(eig)**2))))
    # R_sigma
    try:
        margin = float(np.sqrt(np.prod([1-abs(z)**2 for z in eigenvalues])))
        Rsigma = float(departure(G) / margin)
    except Exception:
        Rsigma = np.nan
    # R_A
    try:
        if np.any(abs(eigenvalues) <= 1e-15) or np.any((abs(eigenvalues.imag) <= 1e-12) & (eigenvalues.real < 0)):
            raise ValueError('invalid log domain')
        Lc = logm(G)
        if np.max(abs(Lc.imag)) > 1e-8 * max(1.0, np.linalg.norm(Lc.real)):
            raise ValueError('complex principal log')
        L = Lc.real / delta_t
        rates = np.linalg.eigvals(L)
        if np.max(abs(rates.imag)) <= 1e-12:
            g1, g2 = sorted([-float(z.real) for z in rates])
            if g1 <= 0:
                raise ValueError('unstable generator')
            denominator = (np.sqrt(g1)+np.sqrt(g2))*np.sqrt(2*np.sqrt(g1*g2))
        else:
            gamma = -float(np.mean(rates.real))
            omega = float(np.max(abs(rates.imag)))
            if gamma <= 0:
                raise ValueError('unstable generator')
            denominator = 2*np.sqrt(gamma*(np.hypot(gamma,omega)+gamma))
        RA = float(departure(L)/denominator)
    except Exception:
        RA = np.nan
    return RA, Rsigma


def compute_indices_from_Q_corrected(A, Q):
    Gamma = Q.T @ A @ Q
    RA, Rsigma = corrected_reduced_ratios(Gamma)
    return RA, Rsigma

# 3) Controlled non-normal operator family and extraction methods

def make_fixed_config(N, rho=0.95, delta_sigma=0.05, seed=1):
    rng = np.random.default_rng(seed)
    U = random_orthogonal(N, rng)
    V = random_orthogonal(N, rng)
    sigma_base = 1.0 + delta_sigma * rng.standard_normal(N - 1)
    sigma_base = np.clip(sigma_base, 0.1, 3.0)
    lam = rng.uniform(0.2 * rho, 0.95 * rho, size=N)
    lam[0] = rho
    Lambda = np.diag(lam)
    return {"N": N, "rho": rho, "delta_sigma": delta_sigma, "seed": seed,
            "U": U, "V": V, "Lambda": Lambda, "sigma_base": sigma_base}


def build_F_from_config(cfg, kappa):
    U, V, Lambda, N = cfg['U'], cfg['V'], cfg['Lambda'], cfg['N']
    sigma = np.ones(N)
    sigma[:-1] = cfg['sigma_base']
    sigma[-1] = 1.0 / float(kappa)
    Sigma = np.diag(sigma)
    Sigma_inv = np.diag(1.0 / sigma)
    return U @ Sigma @ V.T @ Lambda @ V @ Sigma_inv @ U.T


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
            v = rng.standard_normal(N); v /= np.linalg.norm(v) + 1e-15; continue
        u = Av / nu
        Atu = A.T @ u
        Atu = Atu - u * (u @ Atu)
        nv = np.linalg.norm(Atu)
        if nv < 1e-12:
            v = rng.standard_normal(N); v /= np.linalg.norm(v) + 1e-15; continue
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
    Xcols, Ycols = [], []
    for S in S_list:
        Xcols.append(S[:, t0:t1]); Ycols.append(S[:, t0 + 1:t1 + 1])
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
    idx = np.arange(n_samples); rng.shuffle(idx)
    ntr = max(2, int(frac_train * n_samples))
    itr = idx[:ntr]; iva = idx[ntr:]
    if iva.size == 0: iva = itr
    Xtr = X[:, itr]; Ytr = Y[:, itr]; Xva = X[:, iva]; Yva = Y[:, iva]
    mse = []
    for lam in lambdas:
        Fhat = ridge_fit_F(Xtr, Ytr, lam)
        R = Yva - Fhat @ Xva
        mse.append(float(np.mean(R * R)))
    mse = np.asarray(mse)
    return float(lambdas[int(np.argmin(mse))])

# 6) Time-varying experiment

def make_kappa_schedule(T, kappa_start, kappa_end):
    return np.exp(np.linspace(np.log(kappa_start), np.log(kappa_end), T))


def run_timevarying_case(case_name, kappa_start, kappa_end, N=100, M=25, Ttotal=600, W=80, step=10,
                         rho=0.95, delta_sigma=0.05, sigma_noise=0.05, cfg_seed=7, noise_seed=1234, opt_iters=250):
    cfg = make_fixed_config(N=N, rho=rho, delta_sigma=delta_sigma, seed=cfg_seed)
    kappa_t = make_kappa_schedule(Ttotal, kappa_start, kappa_end)
    F_seq = np.zeros((Ttotal, N, N))
    for t in range(Ttotal):
        F_seq[t] = build_F_from_config(cfg, kappa_t[t])
    rng = np.random.default_rng(noise_seed)
    noise_bank = sigma_noise * rng.standard_normal((M, Ttotal, N))
    S_list = simulate_timevarying_var(F_seq, noise_bank)
    lambdas = np.logspace(-6, 1, 12)
    rows = []
    window_ends = np.arange(W, Ttotal + 1, step)
    for tend in window_ends:
        t0 = tend - W; t1 = tend
        X, Y = stack_window_pairs(S_list, t0, t1)
        lam_star = select_lambda_ridge(X, Y, lambdas=lambdas, frac_train=0.8, seed=333+tend)
        Fhat = ridge_fit_F(X, Y, lam_star)
        Fbar = np.mean(F_seq[t0:t1], axis=0)
        Qref = method2_optimization(Fbar, iters=opt_iters, seed=0)
        RA_ref, Rs_ref = compute_indices_from_Q_corrected(Fbar, Qref)
        Qm2 = method2_optimization(Fhat, iters=opt_iters, seed=0)
        RA_m2, Rs_m2 = compute_indices_from_Q_corrected(Fhat, Qm2)
        Qm3 = method3_commutator(Fhat)
        RA_m3, Rs_m3 = compute_indices_from_Q_corrected(Fhat, Qm3)
        angle_m2 = principal_angle_deg(Qref, Qm2)
        angle_m3 = principal_angle_deg(Qref, Qm3)
        relErr_bar = float(np.linalg.norm(Fhat - Fbar, ord='fro') / (np.linalg.norm(Fbar, ord='fro') + 1e-15))
        rows.append({
            'case': case_name, 'M': int(M), 't_end': int(tend), 't0': int(t0), 't1': int(t1),
            'kappa_end': float(kappa_t[tend-1]), 'kappa_window_mean': float(np.mean(kappa_t[t0:t1])),
            'lambda_star': float(lam_star),
            'R_A_trailing': RA_ref, 'R_A_M2': RA_m2, 'R_A_M3': RA_m3,
            'R_sigma_trailing': Rs_ref, 'R_sigma_M2': Rs_m2, 'R_sigma_M3': Rs_m3,
            'dR_A_M2': (RA_m2 - RA_ref) if np.isfinite(RA_m2) and np.isfinite(RA_ref) else np.nan,
            'dR_A_M3': (RA_m3 - RA_ref) if np.isfinite(RA_m3) and np.isfinite(RA_ref) else np.nan,
            'dR_sigma_M2': (Rs_m2 - Rs_ref) if np.isfinite(Rs_m2) and np.isfinite(Rs_ref) else np.nan,
            'dR_sigma_M3': (Rs_m3 - Rs_ref) if np.isfinite(Rs_m3) and np.isfinite(Rs_ref) else np.nan,
            'angle_M2_deg': angle_m2, 'angle_M3_deg': angle_m3,
            'relErr_Fhat_to_Fbar': relErr_bar,
        })
    return rows


def save_rows_csv(rows, path):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def rows_to_arrays(rows):
    out = {}
    keys = rows[0].keys()
    for key in keys:
        try:
            out[key] = np.asarray([float(r[key]) for r in rows])
        except Exception:
            out[key] = np.asarray([r[key] for r in rows])
    return out

# 8) Plotting

def _nice_symmetric_ylim(values, pad=1.15, min_span=0.15):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return -1.0, 1.0
    m = float(np.nanmax(np.abs(vals)))
    m = max(m * pad, min_span)
    return -m, m


def _sparse_markevery(n, target=12):
    if n <= target: return 1
    return max(1, int(np.ceil(n / target)))


def _set_time_axis(ax, t):
    ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', top=True, right=True, length=2)
    ax.tick_params(which='major', direction='in', top=True, right=True, length=4)
    despine(ax)


def _plot_horizontal_threshold(ax, y=1.0, label=r'$R_A=1$', color='0.45'):
    ax.axhline(y, color=color, linestyle='--', linewidth=0.95, zorder=0)
    ax.text(0.985, y, label, transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7.5, color='0.35')


def plot_figure_timevarying_fourpanel(rows_M25, rows_M1, outdir, include_rsigma=False):
    set_publication_style(); os.makedirs(outdir, exist_ok=True)
    arr25, arr1 = rows_to_arrays(rows_M25), rows_to_arrays(rows_M1)
    t = arr25['t_end']; markevery = _sparse_markevery(len(t), target=13)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.85, 4.25), constrained_layout=False, sharex=True)

    # panel a
    ax = axes[0,0]
    ax.plot(t, arr25['kappa_end'], color='0.10', linewidth=1.15, label=r'inst. $\kappa(t)$')
    ax.plot(t, arr25['kappa_window_mean'], color='0.55', linewidth=1.00, linestyle='--', label=r'window mean $\kappa$')
    ax.set_yscale('log'); ax.set_ylabel(r'input $\kappa$')
    legend = ax.legend(loc='upper left', handlelength=1.5, borderpad=0.25, labelspacing=0.25); style_legend(legend)
    add_panel_label(ax,'a'); _set_time_axis(ax,t)

    # panel b
    ax = axes[0,1]
    yref = arr25['R_A_trailing'] if not include_rsigma else arr25['R_A_trailing']
    ax.plot(t, yref, color=METHOD_STYLE['TRUE']['color'], linestyle='-', linewidth=1.15,
            marker='o', markersize=2.8, markevery=markevery, label=r'$R_{\rm trailing}$')
    if include_rsigma:
        ax.plot(t, arr25['R_sigma_trailing'], color=lighten_color('#000000',0.45), linestyle=':', linewidth=1.0,
                marker='o', markerfacecolor='white', markersize=2.4, markevery=markevery, label=r'$R_{\sigma,\rm trailing}$')
    _plot_horizontal_threshold(ax, y=1.0, label=r'$R_A=1$')
    ax.set_yscale('log'); ax.set_ylabel(r'$R_{\rm trailing}$'); add_panel_label(ax,'b'); _set_time_axis(ax,t)

    # panel c inferred tracking curves
    ax = axes[1,0]
    ax.plot(t, arr25['R_A_trailing'], color='k', linestyle='-', linewidth=1.10, marker='o', markersize=2.6, markevery=markevery,
            label=r'$R_{A,\rm trailing}$', zorder=4)
    plot_method_curve(ax, t, arr25['R_A_M2'], 'M2_OPT', label='M2, $M=25$', markevery=markevery, tone='full')
    plot_method_curve(ax, t, arr25['R_A_M3'], 'M3_COMM', label='M3, $M=25$', markevery=markevery, tone='full')
    plot_method_curve(ax, t, arr1['R_A_M2'], 'M2_OPT', label='M2, $M=1$', markevery=markevery, tone='light')
    plot_method_curve(ax, t, arr1['R_A_M3'], 'M3_COMM', label='M3, $M=1$', markevery=markevery, tone='light')
    if include_rsigma:
        plot_method_curve(ax, t, arr25['R_sigma_M2'], 'M2_OPT', label=None, markevery=markevery, tone='full', rsigma=True)
        plot_method_curve(ax, t, arr25['R_sigma_M3'], 'M3_COMM', label=None, markevery=markevery, tone='full', rsigma=True)
        plot_method_curve(ax, t, arr1['R_sigma_M2'], 'M2_OPT', label=None, markevery=markevery, tone='light', rsigma=True)
        plot_method_curve(ax, t, arr1['R_sigma_M3'], 'M3_COMM', label=None, markevery=markevery, tone='light', rsigma=True)
    _plot_horizontal_threshold(ax, y=1.0, label=r'$R_A=1$')
    ax.set_yscale('log'); ax.set_ylabel(r'inferred $\widehat R$'); ax.set_xlabel(r'window end time $t_{\rm end}$')
    add_panel_label(ax,'c'); _set_time_axis(ax,t)

    # panel d residuals
    ax = axes[1,1]
    plot_method_curve(ax, t, arr25['dR_A_M2'], 'M2_OPT', label='M2, $M=25$', markevery=markevery, tone='full')
    plot_method_curve(ax, t, arr25['dR_A_M3'], 'M3_COMM', label='M3, $M=25$', markevery=markevery, tone='full')
    plot_method_curve(ax, t, arr1['dR_A_M2'], 'M2_OPT', label='M2, $M=1$', markevery=markevery, tone='light')
    plot_method_curve(ax, t, arr1['dR_A_M3'], 'M3_COMM', label='M3, $M=1$', markevery=markevery, tone='light')
    if include_rsigma:
        plot_method_curve(ax, t, arr25['dR_sigma_M2'], 'M2_OPT', label=None, markevery=markevery, tone='full', rsigma=True)
        plot_method_curve(ax, t, arr25['dR_sigma_M3'], 'M3_COMM', label=None, markevery=markevery, tone='full', rsigma=True)
        plot_method_curve(ax, t, arr1['dR_sigma_M2'], 'M2_OPT', label=None, markevery=markevery, tone='light', rsigma=True)
        plot_method_curve(ax, t, arr1['dR_sigma_M3'], 'M3_COMM', label=None, markevery=markevery, tone='light', rsigma=True)
    ax.axhline(0.0, color='0.35', linestyle='-', linewidth=0.85, zorder=0)
    ymin, ymax = _nice_symmetric_ylim(np.r_[arr25['dR_A_M2'], arr25['dR_A_M3'], arr1['dR_A_M2'], arr1['dR_A_M3']], pad=1.20, min_span=0.20)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r'$\widehat R-R_{\rm trailing}$'); ax.set_xlabel(r'window end time $t_{\rm end}$')
    add_panel_label(ax,'d'); _set_time_axis(ax,t)

    for row in axes:
        for ax in row:
            ax.grid(False)

    # legend includes lighter M=1 entries
    handles = [
        Line2D([0],[0], color='k', linestyle='-', marker='o', markersize=3, linewidth=1.1, label=r'$R_{A,\rm trailing}$'),
        Line2D([0],[0], color=METHOD_STYLE['M2_OPT']['color'], linestyle='-', marker='o', markerfacecolor='white', markersize=4, linewidth=1.25, label=r'M2, $M=25$'),
        Line2D([0],[0], color=METHOD_STYLE['M3_COMM']['color'], linestyle='--', marker='^', markerfacecolor='white', markersize=4, linewidth=1.25, label=r'M3, $M=25$'),
        Line2D([0],[0], color=lighten_color(METHOD_STYLE['M2_OPT']['color'],0.55), linestyle='-', marker='o', markerfacecolor='white', markersize=4, linewidth=1.1, label=r'M2, $M=1$'),
        Line2D([0],[0], color=lighten_color(METHOD_STYLE['M3_COMM']['color'],0.55), linestyle='--', marker='^', markerfacecolor='white', markersize=4, linewidth=1.1, label=r'M3, $M=1$'),
    ]
    if include_rsigma:
        handles.append(Line2D([0],[0], color='0.55', linestyle=':', linewidth=1.0, label=r'faint dotted companion $R_\sigma$'))
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,-0.01), ncol=3,
                        handlelength=1.7, borderpad=0.25, labelspacing=0.25, columnspacing=1.0)
    style_legend(legend)

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.22, top=0.94, wspace=0.28, hspace=0.22)
    tag = 'RA_Rsigma' if include_rsigma else 'RA_only'
    png_path = os.path.join(outdir, f'Fig07_timevarying_{tag}.png')
    pdf_path = os.path.join(outdir, f'Fig07_timevarying_{tag}.pdf')
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def sha256_of_file(path):
    h = hashlib.sha256();
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    set_publication_style(); os.makedirs(BASE_DIR, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)
    N = 100; Ttotal = 500; W = 80; step = 10; kappa_min = 1.0; kappa_max = 200.0
    rows_M25 = run_timevarying_case('increasing_M25', kappa_min, kappa_max, N=N, M=25, Ttotal=Ttotal, W=W, step=step, cfg_seed=7, noise_seed=5678)
    rows_M1  = run_timevarying_case('increasing_M1',  kappa_min, kappa_max, N=N, M=1,  Ttotal=Ttotal, W=W, step=step, cfg_seed=7, noise_seed=5678)
    save_rows_csv(rows_M25, os.path.join(OUTDIR, 'Fig07_timevarying_increasing_M25_rows_corrected.csv'))
    save_rows_csv(rows_M1,  os.path.join(OUTDIR, 'Fig07_timevarying_increasing_M1_rows_corrected.csv'))
    ra_png, ra_pdf = plot_figure_timevarying_fourpanel(rows_M25, rows_M1, OUTDIR, include_rsigma=False)
    rs_png, rs_pdf = plot_figure_timevarying_fourpanel(rows_M25, rows_M1, OUTDIR, include_rsigma=True)
    # copy source/provenance
    user_src = os.path.join(BASE_DIR, 'original_fig8a_timevarying_clan_UNCHANGED.py')
    if os.path.exists(user_src) and os.path.abspath(user_src) != os.path.abspath(os.path.join(BASE_DIR, 'original_fig8a_timevarying_clan_UNCHANGED.py')): shutil.copy2(user_src, os.path.join(BASE_DIR, 'original_fig8a_timevarying_clan_UNCHANGED.py'))
    # Source script already resides in BASE_DIR.
    # readme/caption/manifest
    with open(os.path.join(BASE_DIR, 'FIG07_caption.tex'),'w') as f:
        f.write(r'''\caption{Time-varying causal tracking experiment in the original manuscript layout. Panel (a) shows the imposed instantaneous and trailing-window mean $\kappa(t)$. Panel (b) shows the corrected trailing-window reference ratio, using $R_A$ as the primary threshold diagnostic; in the companion version, the dotted black curve shows the companion $R_\sigma$. Panel (c) compares inferred $\widehat R$ for M2 and M3, with $M=25$ in full color and $M=1$ in lighter tones. Panel (d) shows signed residuals relative to the trailing-window reference. The horizontal dashed threshold corresponds to $R_A=1$ in the primary plots.}''')
    with open(os.path.join(BASE_DIR, 'README.md'),'w') as f:
        f.write("""# NNDR time-varying manuscript Figure 7 package

This folder contains the time-varying moving-window figure in the original style of `fig8a_timevarying_clan.py`, updated to the corrected thresholds.

## Versions
- `outputs_synthetic_figures/Fig07_timevarying_RA_only.pdf/png`
- `outputs_synthetic_figures/Fig07_timevarying_RA_Rsigma.pdf/png`

## Visual convention
- `M=25` is shown with the original full method colors.
- `M=1` is shown in lighter shades of the same method colors.
- In the companion version, faint dotted overlays show the companion `R_sigma` quantities.

## Source files
- `plot_fig07_timevarying_original_style.py` — standalone Spyder-ready script.
- `original_fig8a_timevarying_clan_UNCHANGED.py` — user-provided source.
- corrected CSV outputs for `M=25` and `M=1`.

Run the plotting script in Spyder with F5 to regenerate the figures.
""")
    manifest = {
        'figure': 'Fig07_timevarying',
        'outputs': {
            'RA_only_pdf': {'path': ra_pdf, 'sha256': sha256_of_file(ra_pdf)},
            'RA_only_png': {'path': ra_png, 'sha256': sha256_of_file(ra_png)},
            'RA_Rsigma_pdf': {'path': rs_pdf, 'sha256': sha256_of_file(rs_pdf)},
            'RA_Rsigma_png': {'path': rs_png, 'sha256': sha256_of_file(rs_png)},
        },
        'n_rows_M25': len(rows_M25), 'n_rows_M1': len(rows_M1)
    }
    with open(os.path.join(BASE_DIR, 'manifest.json'),'w') as f: json.dump(manifest, f, indent=2)
    print('Generated time-varying Fig07 package in', BASE_DIR)

if __name__ == '__main__':
    main()
