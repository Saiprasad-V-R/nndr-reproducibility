from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.linalg import expm


OUT_PNG = Path(__file__).resolve().parent / "fig1_motivation_quiver_aligned_python.png"
OUT_PDF = Path(__file__).resolve().parent / "fig1_motivation_quiver_aligned_python.pdf"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11.5,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.facecolor": "white",
    })


def add_curve_arrow(ax, x: np.ndarray, idx: int, color, head_len: float, head_wid: float) -> None:
    n_pts = x.shape[1]
    idx = max(3, min(idx, n_pts - 4))
    p = x[:, idx]
    tangent = x[:, idx + 3] - x[:, idx - 3]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
    normal = np.array([-tangent[1], tangent[0]])

    tip = p + head_len * tangent
    base = p - 0.55 * head_len * tangent
    tri = np.column_stack([
        tip,
        base + 0.5 * head_wid * normal,
        base - 0.5 * head_wid * normal,
    ])
    ax.add_patch(Polygon(tri.T, closed=True, facecolor=color, edgecolor="none", zorder=5))


def simulate_linear_trajectories() -> dict[str, np.ndarray | float | int]:
    a_nn = np.array([[-1.0, 10.0], [0.0, -2.0]])
    a_no = np.array([[-1.0, 0.0], [0.0, -2.0]])
    x0 = np.array([0.70, 0.92])

    t = np.linspace(0.0, 7.0, 900)
    x_nn = np.column_stack([expm(a_nn * tk) @ x0 for tk in t])
    x_no = np.column_stack([expm(a_no * tk) @ x0 for tk in t])

    n_nn = np.linalg.norm(x_nn, axis=0)
    n_no = np.linalg.norm(x_no, axis=0)
    i_peak = int(np.argmax(n_nn))
    peak_nn = float(n_nn[i_peak])

    return {
        "A_nn": a_nn,
        "A_no": a_no,
        "x0": x0,
        "t": t,
        "Xnn": x_nn,
        "Xno": x_no,
        "nNN": n_nn,
        "nNO": n_no,
        "iPeak": i_peak,
        "peakNN": peak_nn,
    }


def simulate_noise_burst(a_nn: np.ndarray, a_no: np.ndarray) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(8)
    dt = 0.01
    t_noise = 50.0
    tn = np.arange(0.0, t_noise + dt, dt)
    n_steps = tn.size
    sigma = 0.55

    xst_nn = np.zeros((2, n_steps))
    xst_no = np.zeros((2, n_steps))
    for k in range(n_steps - 1):
        d_w = np.sqrt(dt) * rng.standard_normal(2)
        xst_nn[:, k + 1] = xst_nn[:, k] + a_nn @ xst_nn[:, k] * dt + sigma * d_w
        xst_no[:, k + 1] = xst_no[:, k] + a_no @ xst_no[:, k] * dt + sigma * d_w

    norm_st_nn = np.linalg.norm(xst_nn, axis=0)
    norm_st_no = np.linalg.norm(xst_no, axis=0)
    idx_burst = int(np.argmax(norm_st_nn))
    rhat = xst_nn[:, idx_burst] / (np.linalg.norm(xst_nn[:, idx_burst]) + 1e-12)
    proj_r = np.abs(rhat @ xst_nn)

    return {
        "tn": tn,
        "Xst_nn": xst_nn,
        "Xst_no": xst_no,
        "norm_st_nn": norm_st_nn,
        "norm_st_no": norm_st_no,
        "idxBurst": idx_burst,
        "proj_r": proj_r,
    }


def build_figure() -> plt.Figure:
    configure_style()

    c_nn = np.array([170, 20, 55]) / 255.0
    c_nor = np.array([40, 95, 150]) / 255.0
    c_gray = np.array([0.80, 0.80, 0.80])
    c_mag = np.array([220, 0, 170]) / 255.0
    c_blk = np.array([0.05, 0.05, 0.05])
    c_r = np.array([0.35, 0.35, 0.35])

    lw_main = 2.5

    deterministic = simulate_linear_trajectories()
    stochastic = simulate_noise_burst(deterministic["A_nn"], deterministic["A_no"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    ax1, ax2, ax3 = axes

    # Panel (a)
    ax1.axhline(0.0, color=[0.85, 0.85, 0.85], lw=1.5, zorder=0)
    ax1.axvline(0.0, color=[0.85, 0.85, 0.85], lw=1.5, zorder=0)

    xg, yg = np.meshgrid(np.linspace(-0.45, 4.25, 28), np.linspace(-0.30, 1.30, 24))
    u = deterministic["A_nn"][0, 0] * xg + deterministic["A_nn"][0, 1] * yg
    v = deterministic["A_nn"][1, 0] * xg + deterministic["A_nn"][1, 1] * yg
    speed = np.sqrt(u**2 + v**2)
    line_width = 0.45 + 0.65 * (speed / (np.max(speed) + 1e-12))
    ax1.streamplot(
        xg,
        yg,
        u,
        v,
        density=1.1,
        color=[0.78, 0.78, 0.78],
        linewidth=line_width,
        arrowsize=0.9,
        minlength=0.15,
        maxlength=4.0,
        zorder=0,
    )

    x_nn = deterministic["Xnn"]
    x_no = deterministic["Xno"]
    ax1.plot(x_nn[0], x_nn[1], "-", color=c_nn, lw=lw_main)
    ax1.plot(x_no[0], x_no[1], ":", color=c_nor, lw=lw_main)
    add_curve_arrow(ax1, x_nn, 250, c_nn, 0.22, 0.075)
    add_curve_arrow(ax1, x_no, 145, c_nor, 0.18, 0.065)

    x0 = deterministic["x0"]
    ax1.plot(x0[0], x0[1], "o", ms=8.5, mfc="white", mec=c_blk, mew=1.5)
    ax1.text(x0[0] + 0.09, x0[1] + 0.04, r"$\mathbf{x}(0)$", fontsize=12, color=c_blk)

    ax1.plot(x_nn[0, -1], x_nn[1, -1], "s", ms=7.5, mfc=c_nn, mec="white", mew=1.0)
    ax1.plot(x_no[0, -1], x_no[1, -1], "s", ms=7.5, mfc=c_nor, mec="white", mew=1.0)

    slow_v = np.array([1.0, 0.0])
    fast_v = np.array([-10.0, 1.0])
    fast_v /= np.linalg.norm(fast_v)
    ax1.quiver(0.0, 0.0, 3.25 * slow_v[0], 3.25 * slow_v[1], angles="xy", scale_units="xy", scale=1, color=c_mag, lw=3.2)
    ax1.quiver(0.0, 0.0, -1.65 * fast_v[0], -1.65 * fast_v[1], angles="xy", scale_units="xy", scale=1, color=c_mag, lw=3.2)

    ax1.text(0.22, 0.62, "normal trajectory", color=c_nor, fontsize=12, rotation=68, ha="center", va="center")
    ax1.text(2.18, 0.86, "non-normal excursion", color=c_nn, fontsize=12, rotation=-31, ha="center", va="center")
    ax1.plot(0.0, 0.0, "o", ms=7.5, mfc=c_blk, mec="white", mew=1.2)
    ax1.text(
        0.02,
        -0.075,
        "stable fixed point",
        color=c_blk,
        fontsize=10.8,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.15),
    )
    ax1.text(1.30, -0.18, "fast eigendirection", color=c_mag, fontsize=11, rotation=-6, ha="center", va="top")
    ax1.text(2.70, 0.04, "slow eigendirection", color=c_mag, fontsize=11, ha="center", va="bottom")

    origin = np.array([0.8, 0.25])
    ax1.quiver(origin[0], origin[1], 0.0, 0.35, angles="xy", scale_units="xy", scale=1, color=c_blk, lw=2.6)
    ax1.quiver(origin[0], origin[1], 1.1, 0.0, angles="xy", scale_units="xy", scale=1, color=c_r, lw=2.6)
    ax1.text(origin[0] - 0.08, origin[1] + 0.35, r"$\hat{\mathbf{n}}$", fontsize=13, color=c_blk, ha="center")
    ax1.text(origin[0] + 1.15, origin[1] - 0.05, r"$\hat{\mathbf{r}}$", fontsize=13, color=c_r, ha="center")

    ax1.set_xlim(-0.55, 4.25)
    ax1.set_ylim(-0.35, 1.30)
    ax1.set_xlabel(r"State $x_1$")
    ax1.set_ylabel(r"State $x_2$")
    ax1.set_title("(a) Phase-space geometry", fontweight="normal")
    ax1.grid(True, alpha=0.15)

    # Panel (b)
    t = deterministic["t"]
    n_nn = deterministic["nNN"]
    n_no = deterministic["nNO"]
    i_peak = deterministic["iPeak"]
    peak_nn = deterministic["peakNN"]

    ax2.plot(t, n_nn, "-", color=c_nn, lw=lw_main, label=r"Non-normal $\|\mathbf{x}(t)\|_2$")
    ax2.plot(t, n_no, ":", color=c_nor, lw=lw_main, label=r"Normal $\|\mathbf{x}(t)\|_2$")
    ax2.plot(t[0], n_nn[0], "o", mfc="white", mec=c_blk, ms=7.5, mew=1.2)
    ax2.plot(t[-1], n_nn[-1], "s", mfc=c_nn, mec="white", ms=7.5, mew=1.0)
    ax2.plot(t[-1], n_no[-1], "s", mfc=c_nor, mec="white", ms=7.5, mew=1.0)
    ax2.plot(t[i_peak], peak_nn, "o", mfc=c_nn, mec="white", ms=7.5, mew=1.0)
    ax2.text(t[i_peak] + 0.20, peak_nn + 0.05, "Transient excursion", fontsize=12, color=c_nn)

    ax2.set_xlabel(r"Time $t$")
    ax2.set_ylabel(r"Magnitude $\|\mathbf{x}(t)\|_2$")
    ax2.set_title("(b) Deterministic amplification", fontweight="normal")
    ax2.legend(loc="upper right", frameon=True, edgecolor=tuple(c_gray))
    ax2.set_xlim(0.0, 7.0)
    ax2.set_ylim(0.0, float(np.max(n_nn) * 1.15))
    ax2.grid(True, alpha=0.15)

    # Panel (c)
    tn = stochastic["tn"]
    norm_st_nn = stochastic["norm_st_nn"]
    norm_st_no = stochastic["norm_st_no"]
    proj_r = stochastic["proj_r"]
    idx_burst = stochastic["idxBurst"]

    ax3.plot(
        tn,
        norm_st_no,
        ":",
        color=c_nor,
        lw=1.6,
        alpha=0.95,
        label=r"Normal $\|\mathbf{x}\|_2$",
    )
    ax3.plot(
        tn,
        norm_st_nn,
        "-",
        color=c_nn,
        lw=1.95,
        alpha=0.92,
        label=r"Non-normal $\|\mathbf{x}\|_2$",
    )
    ax3.plot(
        tn,
        proj_r,
        linestyle=(0, (5.0, 2.2)),
        color=c_blk,
        lw=1.35,
        alpha=0.98,
        label=r"Projection $|\hat{\mathbf{r}}^\top\mathbf{x}|$",
    )
    ax3.plot(tn[idx_burst], norm_st_nn[idx_burst], "o", mfc=c_nn, mec="white", ms=8.0, mew=1.0)
    ax3.annotate(
        r"Burst aligns with $\hat{\mathbf{r}}$",
        xy=(tn[idx_burst], norm_st_nn[idx_burst]),
        xytext=(tn[idx_burst] + 4.0, norm_st_nn[idx_burst] + 0.16),
        fontsize=11.0,
        color=c_blk,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=c_blk, lw=1.0, shrinkA=2, shrinkB=4),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25),
        zorder=6,
    )
    ax3.set_xlabel(r"Time $t$")
    ax3.set_ylabel("Magnitude")
    ax3.set_title("(c) Stochastic burst alignment", fontweight="normal")
    ax3.legend(
        loc="upper right",
        frameon=True,
        edgecolor=tuple(c_gray),
        facecolor="white",
        framealpha=0.92,
        fontsize=9.8,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=2.0,
    )
    ax3.set_xlim(0.0, 30.0)
    ax3.set_ylim(0.0, float(np.max(norm_st_nn) * 1.18))
    ax3.grid(True, alpha=0.15)

    for ax in axes:
        ax.minorticks_on()

    return fig


def main() -> None:
    fig = build_figure()
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
