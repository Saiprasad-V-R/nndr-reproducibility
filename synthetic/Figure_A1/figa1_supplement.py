# -*- coding: utf-8 -*-
"""
Supplementary figure: stable 2D non-normal transient growth.

Spyder-ready.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from figure_style import box_axes, configure_style, panel_label, precision_ticks


# User settings

OUTDIR = Path(__file__).resolve().parent / "supp_2d_transient_growth"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAVE_PNG = True
SAVE_PDF = True
SAVE_SVG = True
DPI = 600

a = 0.97
b = 0.75

TMAX = 80
t_grid = np.arange(TMAX + 1)

NQ = 600
QMAX_FACTOR = 2.4


# Publication style

def set_publication_style() -> None:
    configure_style(
        font_size=8.0,
        axes_labelsize=9.0,
        axes_titlesize=10.0,
        legend_fontsize=8.0,
        lines_linewidth=1.2,
        savefig_dpi=DPI,
        figure_dpi=150,
    )


COLORS = {
    "q_weak": "#0072B2",   # Blue
    "q_mid": "#D55E00",    # Vermilion/Orange
    "q_strong": "#009E73", # Green
    "main": "#0072B2",     # Blue
    "ref": "#808080",
    "black": "#222222",
}


def despine(ax) -> None:
    box_axes(ax)


def add_precision_ticks(ax) -> None:
    precision_ticks(ax)


def add_panel_label(ax, label: str) -> None:
    panel_label(ax, label, x=-0.16, y=1.04, fontsize=11)


# Core formulas

def F_matrix(a_val: float, b_val: float, q_val: float) -> np.ndarray:
    return np.array(
        [[a_val, q_val * (a_val - b_val)],
         [0.0, b_val]],
        dtype=float,
    )


def response_norm_en(a_val: float, b_val: float, q_val: float, t_vals: np.ndarray) -> np.ndarray:
    t_vals = np.asarray(t_vals, dtype=float)
    first = q_val * (a_val ** t_vals - b_val ** t_vals)
    second = b_val ** t_vals
    return np.sqrt(first ** 2 + second ** 2)


def sigma_max_closed_form(a_val: float, b_val: float, q_val: float) -> float:
    tau = a_val ** 2 + b_val ** 2 + q_val ** 2 * (a_val - b_val) ** 2
    disc = max(0.0, tau ** 2 - 4.0 * (a_val ** 2) * (b_val ** 2))
    s2 = 0.5 * (tau + np.sqrt(disc))
    return float(np.sqrt(max(s2, 0.0)))


def qc_one_step(a_val: float, b_val: float) -> float:
    denom = abs(a_val - b_val)
    if denom < 1e-14:
        return np.inf
    return float(np.sqrt((1.0 - a_val ** 2) * (1.0 - b_val ** 2)) / denom)


def qc_peak_discrete(a_val: float, b_val: float, t_vals: np.ndarray) -> tuple[float, int]:
    candidates = []
    times = []

    for t_val in t_vals:
        if t_val < 1:
            continue

        denom = abs(a_val ** t_val - b_val ** t_val)
        if denom <= 1e-14:
            continue

        numer_sq = max(0.0, 1.0 - b_val ** (2 * t_val))
        qct = np.sqrt(numer_sq) / denom

        if np.isfinite(qct):
            candidates.append(qct)
            times.append(int(t_val))

    candidates = np.asarray(candidates, dtype=float)
    times = np.asarray(times, dtype=int)

    idx = int(np.argmin(candidates))
    return float(candidates[idx]), int(times[idx])


def continuous_peak_time(a_val: float, b_val: float) -> float:
    if not (0.0 < b_val < a_val < 1.0):
        return np.nan
    ratio = np.log(b_val) / np.log(a_val)
    if ratio <= 0.0:
        return np.nan
    return float(np.log(ratio) / np.log(a_val / b_val))


def peak_gain_en(a_val: float, b_val: float, q_val: float, t_vals: np.ndarray) -> float:
    return float(np.max(response_norm_en(a_val, b_val, q_val, t_vals)))


# Compute curves

qc1 = qc_one_step(a, b)
qc_peak, t_peak_disc = qc_peak_discrete(a, b, t_grid)
t_peak_cont = continuous_peak_time(a, b)

qmax = QMAX_FACTOR * max(qc_peak, qc1)
q_grid = np.linspace(0.0, qmax, NQ)

peak_gain_curve = np.array([peak_gain_en(a, b, q, t_grid) for q in q_grid])
sigma_curve = np.array([sigma_max_closed_form(a, b, q) for q in q_grid])

q_weak = 0.50 * qc_peak
q_mid = 0.90 * qc_peak
q_strong = 1.45 * qc_peak

resp_weak = response_norm_en(a, b, q_weak, t_grid)
resp_mid = response_norm_en(a, b, q_mid, t_grid)
resp_strong = response_norm_en(a, b, q_strong, t_grid)

idx_peak_strong = int(np.argmax(resp_strong))
t_peak_strong = int(t_grid[idx_peak_strong])
gain_peak_strong = float(resp_strong[idx_peak_strong])


# Figure

set_publication_style()

# Switched to a compact 2-panel layout
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(7.5, 3.2),
    layout="constrained",
)

axA, axB = axes

# a. Transient response
axA.plot(
    t_grid,
    resp_weak,
    color=COLORS["q_weak"],
    lw=1.35,
    label=rf"$q=0.50\,q_{{c,\mathrm{{peak}}}}$",
)
axA.plot(
    t_grid,
    resp_mid,
    color=COLORS["q_mid"],
    lw=1.35,
    label=rf"$q=0.90\,q_{{c,\mathrm{{peak}}}}$",
)
axA.plot(
    t_grid,
    resp_strong,
    color=COLORS["q_strong"],
    lw=1.45,
    label=rf"$q=1.45\,q_{{c,\mathrm{{peak}}}}$",
)

axA.axhline(1.0, color=COLORS["ref"], lw=0.8, ls="--")
axA.plot(
    [t_peak_strong],
    [gain_peak_strong],
    marker="o",
    ms=3.8,
    color=COLORS["black"],
    zorder=5,
)

axA.text(
    t_peak_strong + 1.5,
    gain_peak_strong,
    rf"$t_\mathrm{{peak}}={t_peak_strong}$",
    fontsize=7.5,
    va="center",
    color=COLORS["black"],
)

axA.set_xlabel(r"Time step $t$")
axA.set_ylabel(r"$\|F^t e_n\|_2$")
axA.set_title("Delayed transient response", pad=8, fontweight="bold")
axA.set_xlim(0, TMAX)
axA.set_ylim(0, max(1.35, 1.10 * np.max(resp_strong)))
axA.legend(frameon=False, loc="upper right", handlelength=1.6, borderpad=0.2)
despine(axA)
add_precision_ticks(axA)
add_panel_label(axA, "a")


# b. Amplification thresholds (Combined peak gain & one-step sigma)

# Plot Peak Transient Gain
axB.plot(
    q_grid,
    peak_gain_curve,
    color=COLORS["main"],
    lw=1.5,
    label=r"Peak transient: $\max_t \|F^t e_n\|_2$"
)

# Plot One-step Optimal Gain
axB.plot(
    q_grid,
    sigma_curve,
    color=COLORS["q_mid"],
    lw=1.5,
    label=r"One-step optimal: $\sigma_{\max}(F)$"
)

axB.axhline(1.0, color=COLORS["ref"], lw=0.8, ls="--")

# Color-coded vertical line and text for Peak Transient threshold
axB.axvline(qc_peak, color=COLORS["main"], lw=1.0, ls=":", alpha=0.8)
axB.text(
    qc_peak * 1.03,
    1.01,
    rf"$q_{{c,\mathrm{{peak}}}}={qc_peak:.2f}$",
    rotation=90,
    va="bottom",
    ha="left",
    fontsize=7.5,
    color=COLORS["main"],
)

# Color-coded vertical line and text for One-step threshold
axB.axvline(qc1, color=COLORS["q_mid"], lw=1.0, ls=":", alpha=0.8)
axB.text(
    qc1 * 1.03,
    0.98,
    rf"$q_{{c,1}}={qc1:.2f}$",
    rotation=90,
    va="bottom",
    ha="left",
    fontsize=7.5,
    color=COLORS["q_mid"],
)

axB.set_xlabel(r"Non-normal coupling $q$")
axB.set_ylabel(r"Amplification factor") # General y-label for both curves
axB.set_title("Amplification thresholds", pad=8, fontweight="bold")
axB.set_xlim(0, qmax)

# Dynamic tight y-bounds covering both curves
max_y = max(np.max(peak_gain_curve), np.max(sigma_curve))
axB.set_ylim(0.8, max_y * 1.05)

axB.legend(frameon=False, loc="upper left", fontsize=8.5)
despine(axB)
add_precision_ticks(axB)
add_panel_label(axB, "b")


# Save

base = OUTDIR / "fig_supp_2d_transient_growth_combined_panels"

if SAVE_PNG:
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")

if SAVE_PDF:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=True)

if SAVE_SVG:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=True)

plt.show()


# Print values

print("\n" + "=" * 78)
print("Supplementary 2D transient-growth figure complete")
print("=" * 78)
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"Eigenvalues: lambda_1 = {a:.6f}, lambda_2 = {b:.6f}")
print(f"One-step threshold q_c,1 = {qc1:.6f}")
print(f"Peak-transient threshold q_c,peak = {qc_peak:.6f}")
print(f"Discrete t_peak at q_c,peak = {t_peak_disc:d}")
print(f"Continuous approximation t_* = {t_peak_cont:.6f}")
print("Representative q values:")
print(f"  q_weak   = {q_weak:.6f}")
print(f"  q_mid    = {q_mid:.6f}")
print(f"  q_strong = {q_strong:.6f}")
print("Strong response peak:")
print(f"  t_peak_strong = {t_peak_strong:d}")
print(f"  max_t ||F^t e_n|| = {gain_peak_strong:.6f}")
print("Saved to:")
print(f"  {base.with_suffix('.png')}")
print(f"  {base.with_suffix('.pdf')}")
print(f"  {base.with_suffix('.svg')}")
print("=" * 78 + "\n")
