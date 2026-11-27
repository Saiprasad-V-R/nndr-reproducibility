from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = ROOT / "results"
w = pd.read_csv(r / "window_diagnostics.csv")
a = pd.read_csv(r / "acceleration_magnitude.csv")
onset, offset = 1316.422, 1326.640625
fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
ax[0].plot(a.time_sec-onset, a.acceleration_magnitude_mg/1000, color="0.35", lw=.6)
ax[0].set_ylabel("Magnitude (g)")
for method, color in (("M2", "#d55e00"), ("M3", "#009e73")):
    z = w[w.method == method]
    for axis, col, gate, label in zip(ax[1:], ("RA", "R_sigma", "support"), ("good_RA", "good_sigma", None), (r"$R_A$", r"$R_\sigma$", "Support")):
        values = z[col].where(z[gate]) if gate else z[col]
        axis.plot(z.t_end-onset, values, color=color, lw=1, label=method)
        axis.set_ylabel(label)
for axis in ax:
    axis.axvspan(0, offset-onset, color="0.88")
    axis.tick_params(direction="in", top=True, right=True)
    axis.set_xlim(-15, 16)
ax[1].legend(frameon=False, ncol=2)
ax[-1].set_xlabel("Time from freeze onset (s)")
fig.savefig(r / "fog_example.pdf")
fig.savefig(r / "fog_example.png", dpi=300)
