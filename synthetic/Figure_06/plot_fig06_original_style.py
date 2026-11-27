# -*- coding: utf-8 -*-
"""
Figure 6 plotting script adapted from the author-provided Figure S3 style, using frozen V19 outputs.

This is the plotting companion to:
    V19 corrected T01 figure-six CSVs (included in data/).

It does NOT recompute synthetic trajectories. It imports:
    outputs_synthetic_figures/Figure_S3_all_scaling_raw_rows.csv
    outputs_synthetic_figures/Figure_S3_scaling_M_summary.csv
    outputs_synthetic_figures/Figure_S3_scaling_N_summary.csv
    outputs_synthetic_figures/Figure_S3_scaling_T_summary.csv

Spyder usage:
    %runfile /path/to/fig6_scaling_plot_from_csv_spyder.py --wdir
"""

import os
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# User settings

BASE = Path(__file__).resolve().parent
OUTDIR = str(BASE / "outputs_synthetic_figures")
DATA_DIR = BASE / "data"
INCLUDE_RSIGMA = False
RAW_ROWS_CSV = os.path.join(OUTDIR, "Figure_S3_all_scaling_raw_rows.csv")
SUMMARY_M_CSV = os.path.join(OUTDIR, "Figure_S3_scaling_M_summary.csv")
SUMMARY_N_CSV = os.path.join(OUTDIR, "Figure_S3_scaling_N_summary.csv")
SUMMARY_T_CSV = os.path.join(OUTDIR, "Figure_S3_scaling_T_summary.csv")

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


# CSV loading

_STRING_FIELDS = {"method", "regime_label", "scaling_axis"}

def _parse_csv_value(key, value):
    """Convert numeric CSV fields back to int/float while preserving labels."""
    if key in _STRING_FIELDS:
        return value

    if value is None:
        return np.nan

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan

    # Keep LaTeX/text fields if any slipped through.
    if any(tok in s for tok in ("$", "\\", "approx")):
        return s

    try:
        f = float(s)
    except ValueError:
        return value

    # Convert known integer fields back to int for comparisons/tick labels.
    if key in {"N", "M", "T_train"} and np.isfinite(f):
        return int(round(f))

    return float(f)


def read_rows_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required CSV: {path}\n"
            "Run fig6_scaling_write_data_spyder.py first."
        )

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: _parse_csv_value(k, v) for k, v in row.items()})

    return rows


# 9) Plot Figure S3

METHOD_STYLE = {
    "TRUE": {"label": "Reference", "color": "#000000", "linestyle": "-", "marker": "o", "lw": 1.45, "alpha": 1.0},
    "M1_BF": {"label": "Eigenbasis (M1)", "color": "#0072B2", "linestyle": ":", "marker": "s", "lw": 1.15, "alpha": 0.95},
    "M2_OPT": {"label": "Optimization (M2)", "color": "#D55E00", "linestyle": "-", "marker": "o", "lw": 1.35, "alpha": 1.0},
    "M3_COMM": {"label": "Commutator (M3)", "color": "#009E73", "linestyle": "--", "marker": "^", "lw": 1.35, "alpha": 1.0},
}

REGIME_STYLE = {
    0.20: {"label": r"$R_{A,\rm ref}\approx0.2$", "color": "#4C78A8"},
    1.00: {"label": r"$R_{A,\rm ref}\approx1$", "color": "#F58518"},
    10.0: {"label": r"$R_{A,\rm ref}\approx10$", "color": "#54A24B"},
}

def extract_summary_arrays(summary, x_key):
    methods = ["M1_BF", "M2_OPT", "M3_COMM"]
    regimes = sorted(set(row["target_R_regime"] for row in summary))
    x_values = sorted(set(row[x_key] for row in summary))
    data = {}

    for reg in regimes:
        data[reg] = {}
        for method in methods:
            xs, angle, kkcerr, rserr, relerr, rtrue = [], [], [], [], [], []
            for x in x_values:
                match = [r for r in summary if r[x_key] == x and r["method"] == method and r["target_R_regime"] == reg]
                if not match:
                    continue
                match = match[0]
                xs.append(x)
                angle.append(match["median_angle_deg"])
                kkcerr.append(match["median_log_KKc_error"])
                rserr.append(match["sigma_log_error"])
                relerr.append(match["median_relErr_Fhat"])
                rtrue.append(match["median_R_true_reached"])

            data[reg][method] = {
                "x_raw": np.array(xs, dtype=float),
                "x_pos": np.arange(len(xs), dtype=float),
                "angle": np.array(angle, dtype=float),
                "kkcerr": np.array(kkcerr, dtype=float),
                "rserr": np.array(rserr, dtype=float),
                "relerr": np.array(relerr, dtype=float),
                "rtrue": np.array(rtrue, dtype=float),
            }

    data["_x_values"] = x_values
    data["_regimes"] = regimes
    return data

def set_ordered_ticks(ax, all_values, show_values=None):
    """Use categorical positions but keep all computed x-values visible.

    all_values gives the full data order used by the curves. show_values controls
    which tick labels are printed. This avoids silently clipping intermediate
    N-values while still keeping the axis visually compact.
    """
    all_values = list(all_values)
    if show_values is None:
        show_values = all_values

    tick_pos = []
    tick_lab = []
    for v in show_values:
        if v in all_values:
            tick_pos.append(all_values.index(v))
            tick_lab.append(str(int(v)))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_xlim(-0.35, len(all_values) - 0.65)

def plot_method_curve(ax, x, y, method, label=True, markersize=4.0, lw_scale=1.0):
    st = METHOD_STYLE[method]
    ax.plot(
        x, y,
        color=st["color"],
        linestyle=st["linestyle"],
        marker=st["marker"],
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=st["color"],
        markeredgewidth=0.85,
        linewidth=st["lw"] * lw_scale,
        alpha=st["alpha"],
        label=st["label"] if label else None,
    )


def plot_regime_method_curve(ax, x, y, reg, method, label=True, markersize=3.2, linewidth=1.25):
    """Color encodes R regime; line style encodes method."""
    color = REGIME_STYLE.get(float(reg), {"color": "0.3"})["color"]
    method_ls = {"M2_OPT": "-", "M3_COMM": "--", "M1_BF": ":"}[method]
    method_marker = {"M2_OPT": "o", "M3_COMM": "^", "M1_BF": "s"}[method]
    method_name = {"M2_OPT": "M2", "M3_COMM": "M3", "M1_BF": "M1"}[method]
    reg_name = REGIME_STYLE.get(float(reg), {"label": f"R={reg:g}"})["label"]
    ax.plot(
        x, y,
        color=color,
        linestyle=method_ls,
        marker=method_marker,
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.8,
        linewidth=linewidth,
        alpha=0.98,
        label=(f"{method_name}, {reg_name}" if label else None),
    )

def nice_ylim_from_methods(data, ykey, methods=("M2_OPT", "M3_COMM"), pad_frac=0.18, lower_zero=True):
    vals = []
    for reg in data["_regimes"]:
        for method in methods:
            vals.extend(list(np.asarray(data[reg][method][ykey], dtype=float)))

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0.0, 1.0

    ymin = float(np.min(vals))
    ymax = float(np.max(vals))

    if lower_zero:
        ymin = 0.0

    span = max(ymax - ymin, 1e-9)
    ymax = ymax + pad_frac * span

    if not lower_zero:
        ymin = ymin - pad_frac * span

    if ymax <= ymin:
        ymax = ymin + 1.0

    return ymin, ymax

def tight_ylim_for_kkcerr(data):
    vals = []
    for reg in data["_regimes"]:
        vals.extend(list(np.asarray(data[reg]["M2_OPT"]["kkcerr"], dtype=float)))
        vals.extend(list(np.asarray(data[reg]["M3_COMM"]["kkcerr"], dtype=float)))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0.0, 1.0

    ymax = float(np.max(vals))
    ymax = max(1e-3, 1.12 * ymax)
    return 0.0, ymax

def add_fixed_parameter_note(ax, note):
    ax.text(
        0.02,
        0.96,
        note,
        transform=ax.transAxes,
        fontsize=7.0,
        color="0.25",
        ha="left",
        va="top",
        zorder=50,
        bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5),
    )

def add_factor_two_reference_if_relevant(ax, ymin, ymax, x_pos=0.1, label="factor-of-two error", ha="left"):
    y_thresh = np.log10(2.0)

    # ---- HARD CLEAN: remove previous copies (THIS IS THE KEY FIX) ----
    for t in list(ax.texts):
        if "factor-of-two" in t.get_text():
            t.remove()

    # ---- draw once ----
    if y_thresh < 0.92 * ymax:
        ax.axhline(
            y_thresh,
            color="0.70",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )

        ax.text(
            x_pos,
            y_thresh + 0.018 * (ymax - ymin),
            label,
            transform=ax.get_yaxis_transform(),
            fontsize=7.4,
            color="0.45",
            va="bottom",
            ha=ha,
        )

def main_regimes_for_plot(data):
    """Return all evaluated regimes for the main axes."""
    return list(data["_regimes"])

def regime_ylim(data, ykey, regimes, methods=("M2_OPT", "M3_COMM"), pad_frac=0.18, lower_zero=True):
    vals = []
    for reg in regimes:
        if reg not in data:
            continue
        for method in methods:
            vals.extend(list(np.asarray(data[reg][method][ykey], dtype=float)))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    if lower_zero:
        ymin = 0.0
    span = max(ymax - ymin, 1e-9)
    ymax = ymax + pad_frac * span
    if not lower_zero:
        ymin = ymin - pad_frac * span
    if ymax <= ymin:
        ymax = ymin + 1.0
    return ymin, ymax


def kkcerr_ylim_for_regimes(data, regimes):
    vals = []
    for reg in regimes:
        if reg not in data:
            continue
        vals.extend(list(np.asarray(data[reg]["M2_OPT"]["kkcerr"], dtype=float)))
        vals.extend(list(np.asarray(data[reg]["M3_COMM"]["kkcerr"], dtype=float)))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    ymax = max(1e-3, 1.14 * float(np.max(vals)))
    return 0.0, ymax


def plot_scaling_row_revised(ax_left, ax_right, data, x_label, tick_labels, fixed_note):

    main_methods = ["M2_OPT", "M3_COMM"]
    main_regimes = main_regimes_for_plot(data)

    # LEFT PANEL
    for reg in main_regimes:
        for method in main_methods:
            plot_regime_method_curve(
                ax_left,
                data[reg][method]["x_pos"],
                data[reg][method]["angle"],
                reg,
                method,
                label=True,
            )

    ax_left.axhspan(0.0, 5.0, color="0.93", zorder=-10)
    ax_left.axhline(5.0, color="0.70", linestyle="--", linewidth=0.8)
    ax_left.text(0.03, 0.115, r"$<5^\circ$", transform=ax_left.transAxes,
                 fontsize=7.4, color="0.40")

    set_ordered_ticks(ax_left, data["_x_values"], tick_labels)
    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel(r"Plane error $\theta$ (deg)")
    ax_left.set_ylim(*regime_ylim(data, "angle", main_regimes))
    despine(ax_left)
    format_axis(ax_left, set_x_locator=False)
    add_fixed_parameter_note(ax_left, fixed_note)

    # RIGHT PANEL (FINAL FIX)
    for reg in main_regimes:
        for method in main_methods:
            plot_regime_method_curve(
                ax_right,
                data[reg][method]["x_pos"],
                data[reg][method]["kkcerr"],
                reg,
                method,
                label=True,
            )

    if INCLUDE_RSIGMA:
        # Companion diagnostic on the SAME corrected operator and same selected plane.
        # Error is measured relative to the corresponding population R_sigma for
        # that N and reference R_A regime, not relative to R_A.
        for reg in main_regimes:
            for method in main_methods:
                ax_right.plot(
                    data[reg][method]["x_pos"],
                    data[reg][method]["rserr"],
                    color=REGIME_STYLE[float(reg)]["color"],
                    linestyle=(0,(1.2,2.0)),
                    linewidth=0.92,
                    alpha=0.34,
                    zorder=1,
                )

    set_ordered_ticks(ax_right, data["_x_values"], tick_labels)
    ax_right.set_xlabel(x_label)
    ax_right.set_ylabel(r"Log-ratio error $\mathcal{E}_{R_A}$")

    ymin, ymax = kkcerr_ylim_for_regimes(data, main_regimes)
    if INCLUDE_RSIGMA:
        vals = np.array([v for reg in main_regimes for method in main_methods
                         for v in data[reg][method]["rserr"]], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ymax = max(ymax, 1.14 * float(np.max(vals)))
    ax_right.set_ylim(ymin, ymax)

    despine(ax_right)
    format_axis(ax_right, set_x_locator=False)
    add_fixed_parameter_note(ax_right, fixed_note)

    # Dynamically place the factor-of-two text to avoid colliding with insets or curves
    if "dimension" in x_label:
        # Panel (d): Inset at top right, threshold near top. Left is perfectly clear.
        add_factor_two_reference_if_relevant(ax_right, ymin, ymax, x_pos=0.02, ha="left")
    elif "horizon" in x_label:
        # Panel (f): Inset at top right. Left is blocked by curves. Place right before inset.
        add_factor_two_reference_if_relevant(ax_right, ymin, ymax, x_pos=0.9, ha="right")
    else:
        # Panel (b): No inset. Left is blocked by curves. Middle is clear.
        add_factor_two_reference_if_relevant(ax_right, ymin, ymax, x_pos=0.50, ha="left")

    # --- Secondary axis: relative reconstruction error for middle regime only ---
    ax_rel = ax_left.twinx()
    mid_reg = min(data["_regimes"], key=lambda r: abs(float(r) - 1.0))
    ax_rel.plot(
        data[mid_reg]["M2_OPT"]["x_pos"],
        data[mid_reg]["M2_OPT"]["relerr"],
        color="0.7", linestyle=":", linewidth=0.8, alpha=0.9, zorder=-2,
    )
    rel = np.asarray(data[mid_reg]["M2_OPT"]["relerr"], dtype=float)
    ymin_rel = max(0.0, float(np.min(rel)) * 0.80)
    ymax_rel = float(np.max(rel)) * 1.15
    if ymax_rel <= ymin_rel:
        ymax_rel = ymin_rel + 0.1
    ax_rel.set_ylim(ymin_rel, ymax_rel)
    ax_rel.set_ylabel(r"relErr$(\widehat F)$", color="0.58", labelpad=4)
    ax_rel.tick_params(axis="y", which="both", colors="0.58", labelsize=6.4)
    ax_rel.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax_rel.tick_params(which="major", direction="in", top=True, right=True, length=4)
    ax_rel.spines["right"].set_color("0.78")
    ax_rel.spines["right"].set_linewidth(0.7)
    ax_rel.grid(False)


def _inset_row_values(rows_all, x_key, x_values, metric, M_value, scaling_axis, method, regime=1.0):
    """Collect one inset series from raw rows."""
    xs = []
    ys = []
    for x in x_values:
        rr = [
            r for r in rows_all
            if r.get("scaling_axis", "") == scaling_axis
            and int(r["M"]) == int(M_value)
            and r["method"] == method
            and abs(float(r["target_R_regime"]) - float(regime)) < 1e-12
            and int(r[x_key]) == int(x)
        ]
        if not rr:
            ys.append(np.nan)
            xs.append(x)
            continue
        xs.append(x)
        numeric = np.array([r[metric] for r in rr], dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        ys.append(float(np.median(numeric)) if numeric.size else np.nan)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _ordered_inset_positions(x_values):
    x_values = list(x_values)
    return np.arange(len(x_values), dtype=float)


def add_m1_only_inset(
    ax,
    rows_all,
    x_key,
    x_values,
    metric,
    ylim_hint=None,
    loc="upper right",
    regime=1.0,
    duplicate_M3=False,
):
    """
    Inset showing M=1 consistency check.
    If duplicate_M3=True → plots M3 twice with different styling.
    """

    x_values = list(x_values)
    xpos = np.arange(len(x_values), dtype=float)

    if x_key == "N":
        axis_M1 = "N_M1_inset"
    elif x_key == "T_train":
        axis_M1 = "T_train_M1_inset"
    else:
        raise ValueError("x_key must be 'N' or 'T_train'")

    axins = inset_axes(
        ax,
        width="31%",
        height="31%",
        loc=loc,
        borderpad=0.75,
    )

    plotted_vals = []

    # CASE 1: NORMAL (M2 + M3)
    if not duplicate_M3:

        method_specs = {
            "M2_OPT": {"color": METHOD_STYLE["M2_OPT"]["color"], "marker": "o"},
            "M3_COMM": {"color": METHOD_STYLE["M3_COMM"]["color"], "marker": "^"},
        }

        for method, spec in method_specs.items():

            _, y = _inset_row_values(
                rows_all,
                x_key,
                x_values,
                metric,
                M_value=1,
                scaling_axis=axis_M1,
                method=method,
                regime=regime,
            )

            plotted_vals.extend(list(y[np.isfinite(y)]))

            axins.plot(
                xpos,
                y,
                color=spec["color"],
                linestyle=(0, (2.2, 1.4)),
                marker=spec["marker"],
                markersize=2.6,
                markerfacecolor=spec["color"],
                markeredgecolor=spec["color"],
                linewidth=0.95,
                alpha=0.95,
            )

    else:

        # M3 VERSION 1 (ORANGE, BACK)
        _, y2 = _inset_row_values(
            rows_all,
            x_key,
            x_values,
            metric,
            M_value=1,
            scaling_axis=axis_M1,
            method="M3_COMM",
            regime=regime,
        )

        plotted_vals.extend(list(y2[np.isfinite(y2)]))

        axins.plot(
            xpos,
            y2,
            color="#D55E00",
            linestyle=(0, (2.2, 1.4)),
            marker="o",
            markersize=2.6,
            markerfacecolor="#D55E00",
            markeredgecolor="#D55E00",
            linewidth=0.95,
            alpha=0.60,   # faded background
            zorder=2,
        )

        # M3 VERSION 2 (GREEN, FRONT)
        _, y1 = _inset_row_values(
            rows_all,
            x_key,
            x_values,
            metric,
            M_value=1,
            scaling_axis=axis_M1,
            method="M3_COMM",
            regime=regime,
        )

        plotted_vals.extend(list(y1[np.isfinite(y1)]))

        axins.plot(
            xpos,
            y1,
            color=METHOD_STYLE["M3_COMM"]["color"],
            linestyle=(0, (2.2, 1.4)),
            marker="^",   # triangle (as requested)
            markersize=2.6,
            markerfacecolor=METHOD_STYLE["M3_COMM"]["color"],
            markeredgecolor=METHOD_STYLE["M3_COMM"]["color"],
            linewidth=0.95,
            alpha=0.95,
            zorder=3,
        )

    # Styling
    axins.set_facecolor("white")

    for side in ("left", "right", "bottom", "top"):
        axins.spines[side].set_linewidth(0.55)
        axins.spines[side].set_color("0.25")

    axins.set_xticks([0, len(x_values) - 1])
    axins.set_xticklabels([str(int(x_values[0])), str(int(x_values[-1]))])

    if plotted_vals:
        vals = np.asarray(plotted_vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size:
            lo = 0.0 if metric in ("angle_deg", "log_KKc_error") else float(np.nanmin(vals))
            hi = float(np.nanmax(vals))

            if ylim_hint is not None:
                hi = max(hi, float(ylim_hint))

            span = max(hi - lo, 1e-9)
            axins.set_ylim(lo, hi + 0.12 * span)

    axins.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))

    axins.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=2.0,
        width=0.55,
        labelsize=5.6,
        pad=1.0,
    )

    axins.minorticks_on()
    axins.grid(False)

    axins.text(
        0.04, 0.94,
        r"$M=1$",
        transform=axins.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color="0.25",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3),
    )

    return axins



def plot_M1_validity_companion(rows_all, outdir):
    """Two clean panels for the sparse M=1, R_A, threshold-near-one checks.

    Unlike tiny insets, this grid faithfully reports all tested configurations:
    a number is the true finite absolute log10 error, and an en dash means that
    no admissible R_A value exists. No interpolation, zero-filling, or pooling.
    """
    set_publication_style()
    fig, axs = plt.subplots(1, 2, figsize=(7.25, 2.55))
    runs = [
        ('N_M1_inset', 'N', [50,100,150,200,250,300,350,400], r'Dimension $N$'),
        ('T_train_M1_inset','T_train',[3,4,10,25,50,100,200,400,800],
         r'Training horizon $T_{\rm train}$'),
    ]
    export=[]
    for panel,(ax,(axis_name,field,x_values,xlabel)) in enumerate(zip(axs,runs)):
        xloc = np.arange(len(x_values))
        for method,ypos,shape,color,short in [
            ('M2_OPT',1.0,'o',METHOD_STYLE['M2_OPT']['color'],'M2'),
            ('M3_COMM',0.0,'^',METHOD_STYLE['M3_COMM']['color'],'M3')
        ]:
            raw=[]
            for x in x_values:
                matches=[r for r in rows_all if r['scaling_axis']==axis_name
                         and r['method']==method and int(r['M'])==1
                         and int(r[field])==int(x)
                         and abs(float(r['target_R_regime'])-1.)<1e-12]
                if len(matches)!=1:
                    raise ValueError(f'Expected exactly one {axis_name}/{method}/{x} row, got {len(matches)}')
                r=matches[0]
                val=float(r['log_KKc_error'])
                is_valid=(int(r['R_A_valid'])==1 and np.isfinite(val))
                raw.append(is_valid)
                xi=x_values.index(x)
                # Alternating visual notation ensures inadmissibility is not
                # mistaken for numerical zero.
                if is_valid:
                    ax.plot([xi],[ypos],marker=shape,linestyle='None',
                            markersize=6.9,markerfacecolor='white',
                            markeredgecolor=color,markeredgewidth=1.35,zorder=4)
                    ax.text(xi,ypos+.18,f'{val:.2f}',ha='center',va='bottom',
                            color=color,fontsize=7.4,zorder=5)
                else:
                    ax.text(xi,ypos,'--',color='0.67',fontsize=8.5,
                            ha='center',va='center',zorder=3)
                export.append({'scan':axis_name,'x':x,'method':short,
                               'admissible':int(is_valid),'E_RA':val if is_valid else np.nan,
                               'status':r['R_A_status']})
            expected={
                'N_M1_inset':{'M2_OPT':1,'M3_COMM':2},
                'T_train_M1_inset':{'M2_OPT':2,'M3_COMM':5},
            }
            assert sum(raw)==expected[axis_name][method],(axis_name,method,sum(raw))
        for yy in (-.28,.5,1.5):
            ax.axhline(yy,color='0.92',lw=.6,zorder=-1)
        ax.set_xlim(-.55,len(x_values)-.45)
        ax.set_ylim(-.38,1.63)
        ax.set_yticks([0,1]);ax.set_yticklabels(['M3','M2'])
        ax.set_xticks(xloc);ax.set_xticklabels([str(x) for x in x_values],fontsize=7.8)
        ax.set_xlabel(xlabel,labelpad=6,fontsize=9.0)
        ax.tick_params(direction='in',top=True,right=True,length=3.4,width=.8)
        ax.tick_params(which='minor',bottom=False,top=False,left=False,right=False)
        for spine in ax.spines.values(): spine.set_linewidth(.8)
        # Identification outside the data cell region; not a plot title.
        ax.text(-.11,1.08,f'({"ab"[panel]})',transform=ax.transAxes,
                fontweight='bold',fontsize=10.5,ha='left',va='bottom')
        ax.text(.98,.95,r'$M=1,\ R_{A,\rm ref}\simeq1$',transform=ax.transAxes,
                ha='right',va='top',fontsize=7.5,color='0.33')
    # A single compact encoding note. All exact values are also in the CSV.
    fig.text(.5,.025,
             r'Numbers: $\mathcal{E}_{R_A}=|\log_{10}(\widehat R_A/R_{A,\rm ref})|$'
             r'     |     --  inadmissible $R_A$ (not zero)',
             ha='center',va='bottom',fontsize=8.1,color='0.35')
    fig.subplots_adjust(left=.085,right=.985,top=.83,bottom=.34,wspace=.26)
    stem=Path(outdir)/'Fig06_M1_admissibility_and_error'
    fig.savefig(str(stem)+'.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(str(stem)+'.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    import csv
    dest=Path(outdir)/'Fig06_M1_admissibility_and_error.csv'
    with dest.open('w',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    return dest

def plot_figure_S3(summary_M, summary_N, summary_T, rows_all, outdir):
    set_publication_style()
    os.makedirs(outdir, exist_ok=True)

    data_M = extract_summary_arrays(summary_M, "M")
    data_N = extract_summary_arrays(summary_N, "N")
    data_T = extract_summary_arrays(summary_T, "T_train")

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(7.25, 6.85),
        constrained_layout=False,
    )

    plot_scaling_row_revised(
        axes[0, 0],
        axes[0, 1],
        data_M,
        r"number of trajectories $M$",
        [1, 2, 5, 10, 15, 25, 50, 100],
        r"fixed $N=100$, $T_{\rm train}=200$",
    )

    plot_scaling_row_revised(
        axes[1, 0],
        axes[1, 1],
        data_N,
        r"dimension $N$",
        [50, 100, 200, 400],
        r"fixed $M=25$, $T_{\rm train}=200$",
    )

    plot_scaling_row_revised(
        axes[2, 0],
        axes[2, 1],
        data_T,
        r"training horizon $T_{\rm train}$",
        [3, 4, 10, 25, 50, 100, 200, 400, 800],
        r"fixed $M=25$, $N=100$",
    )

    # Compact Didier check: M=1 versus M=25 at R_true≈1.
    # These insets appear only in panels (c)--(f); main curves remain M=25.
    add_m1_only_inset(
        axes[1, 0], rows_all,
        x_key="N", x_values=data_N["_x_values"],
        metric="angle_deg", loc="upper right",
    )
    # Detailed M=1 validity/error data are moved to an unobstructed separate panel.
    add_m1_only_inset(
        axes[2, 0], rows_all,
        x_key="T_train", x_values=data_T["_x_values"],
        metric="angle_deg", loc="upper right",
    )
    # No (f) overlay: preserve the entire full-information scaling curve.

    if INCLUDE_RSIGMA:
        axes[0, 1].text(0.98, 0.91, r"faint dotted: $R_\sigma$",
                        color="0.55", fontsize=7.0, ha="right", va="top",
                        transform=axes[0, 1].transAxes)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        handlelength=2.2,
        columnspacing=1.2,
    )

    style_legend(legend)

    panel_labels = ["a", "b", "c", "d", "e", "f"]
    idx = 0

    for i in range(3):
        for j in range(2):
            add_panel_label(axes[i, j], panel_labels[idx])
            idx += 1

    fig.subplots_adjust(
        left=0.088,
        right=0.92,
        bottom=0.105,
        top=0.86,
        wspace=0.55,
        hspace=0.35,
    )

    variant = "RA_Rsigma" if INCLUDE_RSIGMA else "RA_only"
    png_path = os.path.join(outdir, f"Fig06_{variant}.png")
    pdf_path = os.path.join(outdir, f"Fig06_{variant}.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")

    plt.close(fig)

# Main: load data and plot

def prepare_verified_inputs():
    """Convert frozen V19 corrected results into original plotting-script CSV schema.

    This only copies/aggregates the already completed V19 synthetic experiment;
    it NEVER runs simulations or changes the population, plane, or diagnostics.
    One realization exists per condition, so the inherited 'median' keys are
    compatibility names, not claims of uncertainty estimation.
    """
    import math
    raw = read_rows_csv(str(DATA_DIR / "T01_fig06_full_results.csv"))
    refs = read_rows_csv(str(DATA_DIR / "T01_fig06_target_kappas.csv"))
    ref_sig = {(int(r["N"]), round(float(r["target_R"]), 8)):float(r["reference_Rsigma"])
               for r in refs}
    map_axis={"M":"M", "N":"N", "T":"T_train", "N_M1":"N_M1_inset", "T_M1":"T_train_M1_inset"}
    method_map={"M2":"M2_OPT", "M3":"M3_COMM"}
    processed=[]
    for r in raw:
        n = int(r["N"])
        regime = float(r["target_R"])
        rs_ref = ref_sig[(n, round(regime, 8))]
        rs_hat = float(r["R_sigma"])
        sigma_error = float(abs(np.log10(rs_hat / rs_ref))) if rs_hat > 0 and rs_ref>0 and np.isfinite(rs_hat) else np.nan
        log_error = float(r["absolute_log_ratio"])
        if not (int(r["valid"]) and np.isfinite(log_error)):
            log_error = np.nan
        processed.append({
            "N":n, "M":int(r["M"]), "T_train":int(r["T_train"]),
            "target_R_regime":regime, "regime_label":rf"RA_ref≈{regime:g}",
            "method":method_map[r["method"]],
            "scaling_axis":map_axis[r["axis"]],
            "angle_deg":float(r["angle_deg"]),
            "log_KKc_error":log_error,
            "sigma_log_error":sigma_error,
            "relErr_Fhat":float(r["operator_error_F"]),
            "R_true_reached":float(r["R_ref"]),
            "R_A_hat":float(r["R_hat"]),
            "R_sigma_hat":rs_hat, "R_sigma_ref":rs_ref,
            "R_A_status":str(r["status"]), "R_A_valid":int(r["valid"]),
            "source_axis":r["axis"], "source_x":int(r["x"]),
        })
    def save_csv(rows, path):
        if not rows: raise ValueError("Empty derived data")
        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_csv(processed, RAW_ROWS_CSV)
    # Main panel data: one frozen realization per setting; no further pooling.
    for axis, filename in (("M",SUMMARY_M_CSV),("N",SUMMARY_N_CSV),("T_train",SUMMARY_T_CSV)):
        rows=[]
        for r in processed:
            if r["scaling_axis"] != axis: continue
            rows.append({
                **r,
                "median_angle_deg":r["angle_deg"],
                "median_log_KKc_error":r["log_KKc_error"],
                "median_relErr_Fhat":r["relErr_Fhat"],
                "median_R_true_reached":r["R_true_reached"],
            })
        save_csv(rows,filename)
    return len(raw),len(processed)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(OUTDIR, exist_ok=True)
    original_count, converted_count = prepare_verified_inputs()
    rows_all=read_rows_csv(RAW_ROWS_CSV)
    summary_M=read_rows_csv(SUMMARY_M_CSV)
    summary_N=read_rows_csv(SUMMARY_N_CSV)
    summary_T=read_rows_csv(SUMMARY_T_CSV)
    print(f"V19 corrected rows: {original_count}, derived rows: {converted_count}")
    print("Rate-invalid evaluations retained as missing.")
    INCLUDE_RSIGMA=False
    plot_figure_S3(summary_M,summary_N,summary_T,rows_all,OUTDIR)
    INCLUDE_RSIGMA=True
    plot_figure_S3(summary_M,summary_N,summary_T,rows_all,OUTDIR)
    companion_csv = plot_M1_validity_companion(rows_all,OUTDIR)
    print("Figure 6: both primary variants and M=1 detail exported to",OUTDIR)
    print("Validation table:", companion_csv)
