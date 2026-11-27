# Supplementary_N01

**Calculation coverage:** calculation modules found for archived tables.

**Numerical generation:** `calculation/02_N1_comparators/run.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_revfig01.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

The supplied V19 source modules generate the corresponding experiment tables. The final plot script reads the archived CSVs in data/.
