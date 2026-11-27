# T08 — Five-part outcome-blind model-adequacy calibration

This module implements the synthetic five-part local-linear adequacy checks in
`FULL_TEST_SPEC.md` and compares pre-screen and accepted-screen outcomes from
T06 (negative controls) and T07 (matched positive/normal controls).

## Run

Run `python run.py` from anywhere (or open `run.py` in Spyder and press Run).
The module uses only local inputs in `../06_N3_negative_controls` and
`../07_N3_positive_controls` and regenerates all physical time series from the
original stored seeds. No command-line arguments, downloads or external files.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for reproducible performance.

An optional `reuse_replay_cache=true` is provided for quick figure/caption
regeneration after the raw diagnostics are computed. **The saved default is
`false`**, so ordinary reruns replay the source generators and cannot silently
use a stale cache after source changes. Inputs and their SHA-256 hashes are
recorded in `outputs/T08_source_manifest.json`.

## Five screens

A. Out-of-time 20% holdout full VAR(1) one-step prediction versus fitted
   diagonal VAR(1) (`log(MSE_full/MSE_diag)`).
B. Held-out direct-lag covariance discrepancy at h=2,4 versus powers of Fhat.
C. Lag-2 and lag-4 directly fitted maps versus Fhat^2 and Fhat^4. Complex
   principal-frequency inconsistency is recorded separately, not counted as
   a sixth test.
D. Correlation of mean squared residual energy relative to multivariate
   phase-randomized Fourier surrogates with cross-spectra preserved.
E. Six-lag portmanteau whiteness on the 2D leading singular left/right plane
   of the *training* map (method-agnostic; not claimed identical to M2).

The five statistical cutoffs are empirical 99th percentiles, fixed on correctly
specified white-innovation VAR(1) simulations in even-indexed subjects,
including normal, non-normal, short/oscillatory and undisturbed baseline
systems. They never depend on fitted R, event classification or detection
success. All odd-indexed physical subjects are reserved for the main reported
performance. Baseline/event and normal/positive recordings from the same
subject share innovations; uncertainty therefore resamples subjects.

The narrower Gaussian VAR(1) only calibration has been **retained explicitly
as a sensitivity check** in `outputs/narrow_Gaussian_VAR1_pilot/` and is NOT
used for the main T08 scientific conclusions.

## Result interpretation

Model adequacy is not a classifier of physical non-normality. Failed recordings
are **unassessable**, not negative. When screening removes many true-positive
recordings, accepted-only conditional accuracy can become misleading. All
reports include rejection fractions, raw detections per all, screened
detections per all, and detection conditional on passing; invalid R_A fits
remain flagged. Screening cutoffs are synthetic, record-length-dependent and
**not calibrated for the user's empirical streams**.

## Key outputs

- `outputs/T08_FigN3_five_adequacy_calibration.pdf/.png`
- `outputs/T08_locked_calibration_cutoffs.csv`
- `outputs/T08_record_level_adequacy.csv`
- `outputs/T08_all_method_results_with_gate.csv`
- `outputs/T08_holdout_power_FPR_and_rejection.csv`
- `outputs/T08_baseline_event_rejection_by_family.csv`
- `outputs/T08_holdout_diagnostic_rejection_by_family.csv`
- `outputs/T08_subject_cluster_bootstrap_intervals.csv`
- `outputs/T08_manifest.json`, `outputs/T08_source_manifest.json`
- `outputs/T08_report.md` (reviewer-focused interpretation)
- `FigN3_adequacy_revised_caption.tex`

`tests/test_t08.py` has 22 dedicated tests. The complete suite up to T08
passes 213 tests.
