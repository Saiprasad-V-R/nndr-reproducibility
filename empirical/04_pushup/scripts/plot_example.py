from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = ROOT / "results"
t = pd.read_csv(r / "activity_timeseries.csv")
response = pd.read_csv(r / "response_M3.csv")
points = pd.read_csv(r / "pushup_subset_final" / "pushup_subset_point_estimates.csv")
intervals = pd.read_csv(r / "pushup_subset_final" / "pushup_subset_bootstrap_summary.csv")
shifts = pd.read_csv(r / "circular_shift_correlations.csv")
summary = pd.read_csv(r / "circular_shift_summary.csv")
fig, ax = plt.subplots(4, 1, figsize=(8, 9), constrained_layout=True)
ax[0].plot(t.time_sec, t.acceleration_x_demeaned, color="0.35", lw=.7)
ax[0].set_ylabel("Acceleration")
ax[1].plot(t.time_sec, t.activity_envelope, color="#d55e00", lw=1, label="Activity")
other = ax[1].twinx()
other.plot(t.time_sec, response.response_coordinate_abs, color="#009e73", lw=.8, label="|Response|")
ax[1].set_ylabel("Activity")
other.set_ylabel("|Response|")
raw = points[points.variant == "raw_demeaned"]
ci = intervals[intervals.variant == "raw_demeaned"]
for j, method in enumerate(("M2", "M3")):
    for k, regime in enumerate(("low", "high")):
        row = raw[(raw.method == method) & (raw.regime == regime)].iloc[0]
        bounds = ci[(ci.method == method) & (ci.regime == regime)].iloc[0]
        for offset, metric, prefix, marker in ((-.09, "RA", "RA", "o"), (.09, "RB", "RB", "s")):
            x = j*3+k+offset
            y = row[metric]
            ax[2].errorbar(x, y, yerr=[[max(0,y-bounds[f"{prefix}_ci95_low"])],[max(0,bounds[f"{prefix}_ci95_high"]-y)]], fmt=marker, color="#d55e00" if metric=="RA" else "#0072b2", capsize=2)
ax[2].set_xticks([0,1,3,4], ["M2 low","M2 high","M3 low","M3 high"])
ax[2].set_ylabel(r"$R_A$, $R_\sigma$")
for method,color in (("M2","#d55e00"),("M3","#009e73")):
    z=shifts[shifts.method==method]
    ax[3].hist(z.correlation, bins=30, histtype="step", color=color, label=method)
    obs=summary[summary.method==method].observed_correlation.iloc[0]
    ax[3].axvline(obs,color=color,ls="--")
ax[3].set_xlabel("Shifted activity-response correlation")
ax[3].set_ylabel("Count")
ax[3].legend(frameon=False)
fig.savefig(r / "pushup_example.pdf")
fig.savefig(r / "pushup_example.png", dpi=300)
