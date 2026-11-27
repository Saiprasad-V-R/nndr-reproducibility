# Figure 4 mean-field alignment

**Calculation coverage:** final rate-based generator present.

**Numerical generation:** `plot_fig04_original_style.py; calculation/01_baseline_reproduction/fig04/run.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_fig04_original_style.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

The corrected script computes R_A from log(Gamma) and 116 settings × 16 noise draws. Only /mnt/data and self-copy code were changed for portability. Running plot_fig04_original_style.py recomputes the scan; use the saved PDF for immediate use.
