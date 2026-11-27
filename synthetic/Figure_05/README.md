# Figure 5 method recovery

**Calculation coverage:** V19 calculation source confirmed.

**Numerical generation:** `calculation/01_baseline_reproduction/fig05/run.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_fig05_original_style.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

The frozen T01_fig05_full_results_V19.csv is SHA-256 identical to the V19 run output. The original_fig5 script is historical and not the authoritative corrected generator. The plotting script reads archived CSV without simulations.
