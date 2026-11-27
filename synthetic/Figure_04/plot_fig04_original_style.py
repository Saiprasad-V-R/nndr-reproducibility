# -*- coding: utf-8 -*-
"""
Figure 4 original-style synthetic observable-alignment figure.
Preserves the plotting style from the user's original fig3_4_synthetic_transition.py,
while replacing the threshold definition with corrected R_A and optionally adding
R_sigma as a light secondary x-axis.
"""
from __future__ import annotations

import os, json, csv, hashlib, shutil
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.linalg import logm, schur

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, 'outputs_synthetic_figures')
os.makedirs(OUTDIR, exist_ok=True)

# Reproducibility parameters
N = 200
RHO = 0.95
DELTA_SIGMA = 0.05
CFG_SEED = 1

# Alignment curve parameters
SIGMA_NOISE = 0.05
T_ALIGN = 1000
N_ALIGN_REPEATS = 16
NOISE_SEED_ALIGN = 2468

KAPPA_GRID = np.unique(np.r_[
    np.linspace(1.00, 7.00, 46),
    np.linspace(7.25, 20.00, 28),
    np.geomspace(20.5, 450.0, 42)
].astype(float))

# Style cloned from original source
COLOR_BLUE   = "#2563EB"
COLOR_ORANGE = "#EA580C"
COLOR_THRESH = "#94A3B8"
COLOR_CURVE  = "#334155"
COLOR_DOT    = "#64748B"
COLOR_BG_STRIP = "#F8FAFC"
COLOR_RSIGMA = "#A8A29E"  # warm light grey for companion axis


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
        if T[0, 1] * T[1, 0] < 0:
            return abs(abs(float(T[0, 1])) - abs(float(T[1, 0])))
        eig = np.linalg.eigvals(A)
        return float(np.sqrt(max(0., np.linalg.norm(A, 'fro')**2 - np.sum(abs(eig)**2))))

    if np.max(abs(rates.imag)) <= 1e-12:
        g1, g2 = sorted([-float(z.real) for z in rates])
        if g1 <= 0:
            raise ValueError("Unstable continuous-time generator")
        denominator = (np.sqrt(g1) + np.sqrt(g2)) * np.sqrt(2 * np.sqrt(g1 * g2))
    else:
        gamma = -float(np.mean(rates.real))
        omega = float(np.max(abs(rates.imag)))
        if gamma <= 0:
            raise ValueError("Unstable continuous-time generator")
        denominator = 2 * np.sqrt(gamma * (np.hypot(gamma, omega) + gamma))
    RA = float(departure(L) / denominator)
    margin = float(np.sqrt(np.prod([1 - abs(z)**2 for z in eigenvalues])))
    Rsigma = float(departure(G) / margin)
    return RA, Rsigma


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
    R_A: float
    R_sigma: float
    F: np.ndarray
    n: np.ndarray
    r: np.ndarray

@dataclass
class AlignmentPoint:
    kappa_internal: float
    R_A: float
    R_sigma: float
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
    RA, Rsigma = corrected_reduced_ratios(Gamma)
    return GeometryResult(float(kappa_internal), float(RA), float(Rsigma), F, n, r)


def compute_alignment_point(geo: GeometryResult, rng: np.random.Generator) -> AlignmentPoint:
    vals = []
    for _ in range(N_ALIGN_REPEATS):
        noise = SIGMA_NOISE * rng.standard_normal((T_ALIGN, N))
        X = simulate_var_with_fixed_noise(geo.F, noise)
        vals.append(abs(corr(X.mean(axis=0), geo.r @ X)))
    vals = np.asarray(vals, dtype=float)
    return AlignmentPoint(
        kappa_internal=geo.kappa_internal,
        R_A=geo.R_A,
        R_sigma=geo.R_sigma,
        corr_mean=float(vals.mean()),
        corr_median=float(np.median(vals)),
        corr_q25=float(np.quantile(vals, 0.25)),
        corr_q75=float(np.quantile(vals, 0.75)),
        corr_std=float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        n_repeats=int(vals.size),
    )


def make_ticks_ra():
    return np.array([1e-2, 1e-1, 1e0, 1e1, 1e2], dtype=float)


def format_tick_label(v: float) -> str:
    if v <= 0:
        return ''
    lv = np.log10(v)
    if abs(lv - round(lv)) < 1e-10:
        n = int(round(lv))
        if n == 0:
            return '1'
        return rf'$10^{{{n}}}$'
    if v < 0.1:
        return f'{v:.2f}'
    if v < 1:
        return f'{v:.1f}'
    if v < 10:
        return f'{v:.1f}'
    if v < 100:
        return f'{v:.0f}'
    return rf'$10^{{{int(round(lv))}}}$'


def rsigma_for_ra_ticks(RA_sorted, Rsigma_sorted, ticks):
    x = np.log10(RA_sorted)
    y = np.log10(Rsigma_sorted)
    vals = []
    for t in ticks:
        if t < RA_sorted.min() or t > RA_sorted.max():
            vals.append(np.nan)
        else:
            vals.append(10**np.interp(np.log10(t), x, y))
    return np.array(vals)


def plot_alignment(points: list[AlignmentPoint], include_rsigma: bool = False):
    set_publication_style()

    RAvals = np.array([p.R_A for p in points], dtype=float)
    Rsvals = np.array([p.R_sigma for p in points], dtype=float)
    med = np.array([p.corr_median for p in points], dtype=float)
    q25 = np.array([p.corr_q25 for p in points], dtype=float)
    q75 = np.array([p.corr_q75 for p in points], dtype=float)

    order = np.argsort(RAvals)
    RAvals, Rsvals, med, q25, q75 = RAvals[order], Rsvals[order], med[order], q25[order], q75[order]
    keep = (RAvals >= 1e-2) & (RAvals <= 1e2)

    fig, ax = plt.subplots(figsize=(4.95, 3.45), facecolor="white")
    ax.set_facecolor(COLOR_BG_STRIP)
    despine(ax)

    ax.fill_between(RAvals[keep], q25[keep], q75[keep],
                    color=COLOR_DOT, alpha=0.16, linewidth=0,
                    label="Central 50% over noise trials")
    ax.plot(RAvals[keep], med[keep], color=COLOR_CURVE, lw=2.15,
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

    if include_rsigma:
        ticks_ra = make_ticks_ra()
        ticks_rs = rsigma_for_ra_ticks(RAvals[keep], Rsvals[keep], ticks_ra)
        secax = ax.secondary_xaxis('top')
        secax.set_xscale('log')
        secax.set_xlim(ax.get_xlim())
        secax.set_xticks(ticks_ra)
        labels = []
        for v in ticks_rs:
            if np.isfinite(v):
                if v < 0.1:
                    labels.append(f'{v:.2f}')
                elif v < 1:
                    labels.append(f'{v:.2f}')
                elif v < 10:
                    labels.append(f'{v:.2f}')
                elif v < 100:
                    labels.append(f'{v:.1f}')
                else:
                    labels.append(f'{v:.0f}')
            else:
                labels.append('')
        secax.set_xticklabels(labels)
        secax.tick_params(which='major', direction='in', top=True, bottom=False,
                          length=4, width=0.8, pad=1.5, colors=COLOR_RSIGMA, labelsize=7)
        secax.tick_params(which='minor', direction='in', top=True, bottom=False,
                          length=2, width=0.6, colors=COLOR_RSIGMA)
        secax.spines['top'].set_linewidth(0.8)
        secax.spines['top'].set_color(COLOR_RSIGMA)
        secax.set_xlabel(r"companion $R_\sigma$", color=COLOR_RSIGMA, labelpad=3, fontsize=8)

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.18, top=0.90)
    tag = 'RA_Rsigma' if include_rsigma else 'RA_only'
    out_png = os.path.join(OUTDIR, f'Fig04_{tag}.png')
    out_pdf = os.path.join(OUTDIR, f'Fig04_{tag}.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, transparent=True, bbox_inches='tight')
    plt.close(fig)
    return out_png, out_pdf, RAvals, Rsvals, med, q25, q75, keep


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    set_publication_style()
    cfg = make_fixed_config(N_=N, rho=RHO, delta_sigma=DELTA_SIGMA, seed=CFG_SEED)
    geometries = [compute_geometry(cfg, float(k)) for k in KAPPA_GRID]
    rng_align = np.random.default_rng(NOISE_SEED_ALIGN)
    alignment_points = [compute_alignment_point(geo, rng_align) for geo in geometries]

    ra_png, ra_pdf, RAvals, Rsvals, med, q25, q75, keep = plot_alignment(alignment_points, include_rsigma=False)
    rs_png, rs_pdf, *_ = plot_alignment(alignment_points, include_rsigma=True)

    # Save numeric data
    csv_path = os.path.join(OUTDIR, 'Fig04_alignment_points_corrected_ratios.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kappa_internal', 'R_A', 'R_sigma', 'corr_mean', 'corr_median', 'corr_q25', 'corr_q75', 'corr_std', 'n_repeats'])
        for p in alignment_points:
            writer.writerow([p.kappa_internal, p.R_A, p.R_sigma, p.corr_mean, p.corr_median, p.corr_q25, p.corr_q75, p.corr_std, p.n_repeats])

    # copy original source for provenance
    original_src = os.path.join(BASE_DIR, 'original_fig3_4_synthetic_transition_UNCHANGED.py')
    if os.path.exists(original_src) and os.path.abspath(original_src) != os.path.abspath(os.path.join(BASE_DIR, 'original_fig3_4_synthetic_transition_UNCHANGED.py')):
        shutil.copy2(original_src, os.path.join(BASE_DIR, 'original_fig3_4_synthetic_transition_UNCHANGED.py'))
    # Source script already resides in BASE_DIR.

    # caption stub
    caption = r'''\caption{Observable alignment across the synthetic transition. The curve shows the median and central 50\% range of $|\mathrm{corr}(m_k,\widehat{\mathbf{r}}^{\top}\mathbf{x}_k)|$ across 16 independent noise realizations for each synthetic operator in the fixed family. The horizontal axis uses the corrected reduced non-normality ratio $R_A=K/K_{c,A}$; the dashed line marks $R_A=1$. In the companion version, a light grey top axis shows the corresponding $R_\sigma$ values for the same operator family.}'''
    with open(os.path.join(BASE_DIR, 'FIG04_caption.tex'), 'w') as f:
        f.write(caption)

    readme = f'''# NNDR Figure 4 original-style package

This folder contains the Figure 4 observable-alignment figure in the exact visual style of the user's original `fig3_4_synthetic_transition.py`.

## Files
- `plot_fig04_original_style.py` — standalone Spyder-ready script.
- `outputs_synthetic_figures/Fig04_RA_only.pdf/png` — primary corrected $R_A$ version.
- `outputs_synthetic_figures/Fig04_RA_Rsigma.pdf/png` — companion version with a light-grey top axis for $R_\sigma$.
- `outputs_synthetic_figures/Fig04_alignment_points_corrected_ratios.csv` — frozen numerical values used in the curve.
- `original_fig3_4_synthetic_transition_UNCHANGED.py` — untouched source provided by the user.
- `FIG04_caption.tex` — caption stub.

## Notes
- All layout, fonts, axis styling, inward ticks, and overall appearance follow the original code.
- The only scientific change is the corrected threshold definition using $R_A$ and, in the companion version, the addition of $R_\sigma$ as a light secondary axis.
- Run `plot_fig04_original_style.py` in Spyder with F5 to regenerate both figure versions.
'''
    with open(os.path.join(BASE_DIR, 'README.md'), 'w') as f:
        f.write(readme)

    manifest = {
        'figure': 'Fig04',
        'base_dir': BASE_DIR,
        'parameters': {
            'N': N, 'RHO': RHO, 'DELTA_SIGMA': DELTA_SIGMA, 'CFG_SEED': CFG_SEED,
            'SIGMA_NOISE': SIGMA_NOISE, 'T_ALIGN': T_ALIGN, 'N_ALIGN_REPEATS': N_ALIGN_REPEATS,
            'NOISE_SEED_ALIGN': NOISE_SEED_ALIGN,
            'n_kappa_grid': int(KAPPA_GRID.size),
        },
        'outputs': {
            'Fig04_RA_only_pdf': {'path': ra_pdf, 'sha256': sha256_of_file(ra_pdf)},
            'Fig04_RA_only_png': {'path': ra_png, 'sha256': sha256_of_file(ra_png)},
            'Fig04_RA_Rsigma_pdf': {'path': rs_pdf, 'sha256': sha256_of_file(rs_pdf)},
            'Fig04_RA_Rsigma_png': {'path': rs_png, 'sha256': sha256_of_file(rs_png)},
            'csv': {'path': csv_path, 'sha256': sha256_of_file(csv_path)},
        },
        'curve_summary': {
            'min_R_A_kept': float(np.min(RAvals[keep])),
            'max_R_A_kept': float(np.max(RAvals[keep])),
            'min_R_sigma_kept': float(np.min(Rsvals[keep])),
            'max_R_sigma_kept': float(np.max(Rsvals[keep])),
            'n_points_kept': int(np.sum(keep)),
        }
    }
    with open(os.path.join(BASE_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print('Generated Figure 4 package in', BASE_DIR)

if __name__ == '__main__':
    main()
