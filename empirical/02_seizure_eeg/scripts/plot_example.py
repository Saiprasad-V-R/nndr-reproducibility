from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = ROOT / "results"
w = pd.read_csv(r / "window_diagnostics.csv")
a = pd.read_csv(r / "representative_EEG_meanfield.csv")
fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
ax[0].plot(a.t_rel_s, a.mean_eeg_microvolt, color="0.35", lw=.6)
ax[0].set_ylabel("Mean EEG (µV)")
for method, color in (("M2", "#d55e00"), ("M3", "#009e73")):
    for axis, col, label in zip(ax[1:], ("R_A", "R_sigma", "support"), (r"$R_A$", r"$R_\sigma$", "Support")):
        axis.plot(w.tau_end_s, w[f"{method}_{col}"], color=color, lw=1, label=method)
        axis.set_ylabel(label)
for axis in ax:
    axis.axvspan(0, 40, color="0.85", alpha=.5)
    axis.tick_params(direction="in", top=True, right=True)
    axis.set_xlim(-120, 70)
ax[1].set_ylim(0, 3)
ax[2].set_ylim(0, 3)
for method, color in (("M2", "#d55e00"), ("M3", "#009e73")):
    outside = w[(w.tau_end_s >= -120) & (w.tau_end_s <= 70) & (w[f"{method}_R_sigma"] > 3)]
    ax[2].scatter(outside.tau_end_s, [2.92] * len(outside), marker="^", s=22,
                  facecolors="none", edgecolors=color, linewidths=.9, zorder=5)
ax[2].text(.02, .95, "5 values > 3", transform=ax[2].transAxes,
           ha="left", va="top", fontsize=8)
ax[1].legend(frameon=False, ncol=2)
ax[-1].set_xlabel("Time from seizure onset (s)")
fig.savefig(r / "eeg_example.pdf")
fig.savefig(r / "eeg_example.png", dpi=300)
