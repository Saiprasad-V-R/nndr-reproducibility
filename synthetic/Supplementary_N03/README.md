# Supplementary_N03

**Calculation coverage:** calculation modules found for archived tables.

**Numerical generation:** `calculation/08_adequacy_calibration/run.py, calculation/12_bootstrap_solver/run.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_revfig03.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

V19 T08 and T12 source modules are included. Plotting uses frozen adequacy and interval CSVs; the selected plot tables are not rerun here.
