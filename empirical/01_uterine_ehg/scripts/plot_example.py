from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = ROOT / "results"
w = pd.read_csv(r / "records" / "ice039_p_3of6" / "windows.csv")
a = pd.read_csv(r / "activity_envelope.csv")
p = pd.read_csv(r / "detected_activity_peaks.csv")
fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
ax[0].plot(a.time_sec, a.activity_envelope_uV, color="0.35", lw=.55)
for peak in p.peak_time_sec:
    ax[0].axvline(peak, color="#c43c39", lw=.6, alpha=.6)
ax[0].set_ylabel("Activity (µV)")
for method, color in (("M2", "#d55e00"), ("M3", "#009e73")):
    z = w[w.method == method]
    for axis, col, label in zip(ax[1:], ("RA", "R_sigma", "support"), (r"$R_A$", r"$R_\sigma$", "Support")):
        axis.plot(z.time_end_sec, z[col], lw=1, color=color, label=method)
        axis.set_ylabel(label)
for axis in ax:
    axis.tick_params(direction="in", top=True, right=True)
ax[2].set_yscale("symlog", linthresh=1)
ax[1].legend(frameon=False, ncol=2)
ax[-1].set_xlabel("Time (s)")
fig.savefig(r / "ehg_example.pdf")
fig.savefig(r / "ehg_example.png", dpi=300)
