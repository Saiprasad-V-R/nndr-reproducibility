# Figure 3 synthetic trajectories

**Calculation coverage:** final rate-based generator present.

**Numerical generation:** `plot_fig03_original_style.py; calculation/01_baseline_reproduction/fig03/run.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_fig03_original_style.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

The corrected script evaluates R_A using log(Gamma) and saves the four-case CSV. The V19 archive holds 600-step coordinate arrays, common innovations, operators, config and seed information. The historical original_fig3_4 script is retained for comparison only.
