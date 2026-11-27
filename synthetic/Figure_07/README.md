# Figure 7 time-varying VAR

**Calculation coverage:** final corrected M=25/M=1 calculation source confirmed.

**Numerical generation:** `plot_fig07_timevarying_original_style.py`. Do not run an expensive simulation merely to redraw the archived figure. Run calculation scripts in a separate copy if fresh results are needed; they write into their calculation subtree.

**Figure from existing data:** `plot_fig07_timevarying_original_style.py`. Where this is a combined generator and plotter, running it also recomputes numerical data; use the frozen figure files for immediate manuscript assembly.

Its main() sets N=100, T=500, W=80, step=10 and saves both corrected 43-row window tables. The V19 fig07 run.py is an older M=1/T=600 calculation kept as archive context, not the final generator. Only /mnt/data and self-copy code were changed for portability.
