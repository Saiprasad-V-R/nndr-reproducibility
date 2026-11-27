# Figure 8 broad spectra

**Calculation coverage:** V19 calculation source confirmed.

**Numerical generation:** `calculation/01_baseline_reproduction/fig08/run.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_broad_original_style_Spyder.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

The frozen full-results CSV is SHA-256 identical to V19 output. The original_fig7 script is historical; V19 run.py uses rate-calibrated R_A. Figure number follows the revised manuscript, regardless of the older source filename.
