# NNDR revised manuscript reproduction resources

Local reproduction resources for **Inferring Non-Normal Amplification Geometry from Multivariate Time Series**, using the clean revised manuscript dated 27 November 2026. This directory is separate from the NNDR toolkit and from the original research workspace. It contains [synthetic experiments](synthetic/README.md) and [empirical resources](empirical/README.md): four representative examples plus compact EHG and seizure EEG cohort archives. The archives omit the full raw databases and some intermediate window tables.

## Install

Use Python 3.10 or newer in a clean environment. Install dependencies for the intended part:

```bash
python -m pip install -r synthetic/requirements.txt
python -m pip install -r empirical/requirements.txt
```

The archived calculation scripts retain their manuscript implementations. The [general-purpose NNDR toolkit](https://github.com/Saiprasad-V-R/nndr-toolkit) is a mathematical comparison reference, not a replacement execution dependency. Dataset attribution and redistribution terms are recorded in [empirical/DATA_LICENSES.md](empirical/DATA_LICENSES.md).

## Run

For a representative empirical example, change to its folder and run its `scripts/run_example.py` (the push-up primary calculation uses `scripts/pushup_subset_final_5s_bootstrap.py` and `scripts/run_timing_example.py`), then `scripts/plot_example.py`. Each folder's README gives exact commands and outputs. These commands recompute diagnostics from the included **single** recording. Archived CSVs in `reference_data/` remain read-only comparison material.

For synthetic experiments, use the command inventory in [synthetic/README.md](synthetic/README.md). Some archived synthetic plotting scripts also regenerate numerical data; use a disposable copy for expensive calculations. Existing archived PDFs and tables can be inspected without running them.

## Scientific scope

Both parts retain the revised rate-based variance-response ratio \(R_A\) and the discrete one-step ratio \(R_\sigma\), with the appropriate stability and principal-log admissibility checks. Extraction methods, regularization, shrinkage, response-support definitions and directional conventions are experiment-specific and preserved in source. The [synthetic consistency audit](synthetic/MATHEMATICAL_CONSISTENCY.md), [empirical verification notes](empirical/README.md) and [reproducibility guide](REPRODUCIBILITY.md) state the numerical checks and limits.

The empirical part contains four representative examples: [uterine EHG](empirical/01_uterine_ehg/README.md), [seizure EEG](empirical/02_seizure_eeg/README.md), [freezing of gait](empirical/03_freezing_of_gait/README.md), and [push-up motion](empirical/04_pushup/README.md). It also contains archived EHG and seizure EEG cohort summary tables and source scripts. The full cohorts were not rerun here; see the cohort guides for the archived-summary route and its limits. The representative FoG and push-up examples do not establish population-wide mechanisms.

Didier, the original push-up recording owner, authorized publication; the complete push-up example is included. No specific data license has been assigned. The reproduction repository has no approved software license yet; the MIT license of `nndr-toolkit` does not automatically license this separate archive. `CITATION.cff` identifies the manuscript and repository without claiming an unissued software version or DOI.
