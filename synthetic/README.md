# NNDR revised manuscript: synthetic reproduction

This working copy preserves the manuscript calculations, parameters, seeds, numerical tables, plotting scripts and reference figures. The archived scripts are the scientific source for reproduction; the toolkit is a comparison reference. See [MATHEMATICAL_CONSISTENCY.md](MATHEMATICAL_CONSISTENCY.md) for the source audit and each figure folder's README for file-level details.

## Requirements and use

Python 3.10+ with `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn` and `threadpoolctl`; see `requirements.txt`. Run commands from the indicated figure directory. Calculation commands below are documented for future reproduction and were **not** run during this housekeeping. They may overwrite local outputs or take substantial time; use a disposable copy for fresh calculations. Plot commands are likewise documented, not executed here. Some final plotting scripts also calculate results, as marked.

| Figure | Experiment and archived numerical files | Calculation command | Plotting command |
|---|---|---|---|
| 1 | Conceptual motivation; no numerical table | `python fig1_motivation.py` | Same script |
| 3 | Four VAR trajectories; V19 operator/trajectory CSV and common-innovation NPZ under `calculation/.../outputs/`, corrected ratio CSV under `outputs_synthetic_figures/` | `python calculation/01_baseline_reproduction/fig03/run.py`; final corrected script also calculates | `python plot_fig03_original_style.py` (also calculates) |
| 4 | Alignment scan; 116-setting and 1,856-replicate V19 CSVs, corrected plotting CSV | `python calculation/01_baseline_reproduction/fig04/run.py`; final corrected script also calculates | `python plot_fig04_original_style.py` (also calculates) |
| 5 | M1/M2/M3 recovery; `T01_fig05_full_results.csv`, ridge-CV CSV and corrected plotting table | `python calculation/01_baseline_reproduction/fig05/run.py` | `python plot_fig05_original_style.py` |
| 6 | M, N and training-horizon scans; full-results CSV and inset-validity data | `python calculation/01_baseline_reproduction/fig06/run.py` | `python plot_fig06_original_style.py` |
| 7 | Time-varying benchmark; corrected M=25 and M=1 window CSVs under `outputs_synthetic_figures/` | `python plot_fig07_timevarying_original_style.py` (final M=25/M=1 generator) | Same script (also calculates) |
| 8 | Broad singular spectra; trial-level CSV, spectra CSV and group summaries | `python calculation/01_baseline_reproduction/fig08/run.py` | `python plot_broad_original_style_Spyder.py` |
| A.1 | Analytic two-dimensional transient example; source PDF | `python figa1_supplement.py` | Same script |
| N01 | Comparison methods; `data/T02_*.csv` | `python calculation/02_N1_comparators/run.py` | `python plot_revfig01.py` |
| N02 | Factorial and ablation controls; `data/T04_*.csv`, `data/T05_*.csv` | `python calculation/04_N1_factorial/run.py`; `python calculation/05_N2_ablation/run.py` | `python plot_revfig02.py` |
| N03 | Adequacy and interval coverage; `data/T08_*.csv`, `data/T12_*.csv` | `python calculation/08_adequacy_calibration/run.py`; `python calculation/12_bootstrap_solver/run.py` | `python plot_revfig03.py` |
| N04 | Coloured forcing; `data/T06_*.csv` and transcribed `data/R28_*_counts_transcribed.csv` | `python calculation/06_N3_negative_controls/run.py` (T06 only) | `python plot_revfig04_coloured_forcing.py` |
| S01 | Lag geometry; `data/T03_*.csv` | `python calculation/03_N1_lag_geometry/run.py` | `python plot_suppfig01.py` |
| S02 | Nonlinear controls; `data/T09_*.csv` | `python calculation/09_N4_nonlinear/run.py` | `python plot_suppfig02.py` |
| S03 | Time-varying ground truth; `data/T10_*.csv` | `python calculation/10_timevarying_groundtruth/run.py` | `python plot_suppfig03.py` |
| S04 | Sample efficiency; `data/T11_*.csv` | `python calculation/11_N5_sample_efficiency/run.py` | `python plot_suppfig04.py` |

The `calculation/shared/` modules and per-experiment JSON configurations are local dependencies. `original_*_UNCHANGED.py` files document historical calculations and must not replace the corrected final generators. Archived CSVs retain trial-level or plotting-level precision as supplied; NPZ files retain arrays such as shared innovations. Reference PDFs and PNGs are included for comparison. No experimental dataset downloads are needed for these synthetic experiments.

**Limits:** N04's two R28 aggregate-count CSVs are transcriptions; original fit-level numerical files and their generator have not been recovered. Figure 1 and A.1 source-to-PDF equivalence has not been checked. The documented M3 directional ordering and near-degenerate K reporting conventions remain unchanged.
