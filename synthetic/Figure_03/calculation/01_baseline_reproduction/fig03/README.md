# T01 / Revised submitted Fig. 3: two documented protocols

## Run, no arguments

From any directory:

```bash
python path/to/NNDR_synthetic_revision_suite/01_baseline_reproduction/fig03/run.py
```

From suite root:

```bash
python -m pytest -q 01_baseline_reproduction/fig03/tests/
```

Requires `numpy`, `scipy`, `pandas`, `matplotlib`, and `pytest` for tests. `OPENBLAS_NUM_THREADS=1` improves predictability on large multicore hosts, but is optional.

## Figure identity

This is submitted manuscript Fig. 3, PDF printed page 9, Eqs. (43)–(44). The revised response letter moves the detailed old Figs. 3 and 4 to Supplement S1, so the final figure numbering is not yet frozen.

## Fixed historical operators versus recalibrated operators

1. **Faithful fixed-operator revision** (`T01_fig03_original_operators_revised_R.*`): holds the original four internal κ values, F matrices, fixed Λ, and signal model; uses revised continuous-time-rate R, multistart M2, and a genuinely common Gaussian noise realization in all four cases. This is the **recommended plot** for a scientifically transparent correction to Fig. 3.
2. **Optional new targeted R redraw** (`T01_fig03_recalibrated_targets.*`): creates **different** F matrices at newly selected κ values so that corrected R hits the old nominal target values 0.1, 1, 9.8, 30. The recalibration cannot be described as merely recomputing R for the original Figure 3.

The legacy script accidentally generated a new random-noise sequence for each operator, contrary to the submitted Fig. 3 caption. The revised experiment uses exactly the same noise input for every F. The plotted results are **M=1 individual trajectories**, not an average over M=25. We additionally simulate 25 independent noise paths and report the distribution of response/input RMS ratios and mean-field alignments without incorrectly averaging noisy trajectories.

## Acceptance checks and scientific limits

- Eq. (44) independently reconstructed and numerically identical to the original construction.
- The full F spectrum and its radius 0.95 are invariant across κ; every Γ has a real positive spectrum here.
- M2 planes are orthonormal; projected singular gain does not exceed full singular gain.
- All corrected rates-based R and exact one-step R_sigma values are finite and independently computed by T00.
- The second historical case crosses R=1 after the correction: an original near-threshold case cannot be called subcritical under the new threshold.
- The 25-trajectory table is descriptive, not a confidence interval for R or a finite-data inference test.
- The illustrative comparison is not evidence that M2 or M3 outperforms snapshot or SVD baselines; those tests belong to N1/T02.

## Main numerical assets

All under `outputs/`: the two plotted figures as PNG/PDF, `T01_fig03_operator_and_single_trajectory_summary.csv`, `T01_fig03_25_trajectory_replicates.csv`, `T01_fig03_25_trajectory_summary.csv`, and exact `.npz` records of the operator family, the tested F/Q/Γ, trajectories, and common-noise arrays. The report is `T01_fig03_report.md` and the hash manifest is `T01_fig03_run_manifest.json`.
