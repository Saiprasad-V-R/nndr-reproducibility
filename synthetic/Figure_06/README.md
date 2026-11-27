# Figure 6 scaling

**Calculation coverage:** V19 calculation source confirmed.

**Numerical generation:** `calculation/01_baseline_reproduction/fig06/run.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_fig06_original_style.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

The frozen full-results CSV is SHA-256 identical to V19 output. Historical writer and plotter are retained for provenance; use the V19 run.py for numerical generation and the separate Ultimate Figures plot script for archived-data rendering.
