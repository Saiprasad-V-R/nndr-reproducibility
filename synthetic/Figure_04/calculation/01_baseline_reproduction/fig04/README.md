# T01 / submitted Fig. 4 — mean-field alignment after corrected threshold

**Status:** complete numerical reproduction under the working rebuttal Option 1.

Run `OPENBLAS_NUM_THREADS=1 python run.py` here (no additional arguments). The script imports only shared modules from the revision suite, creates its own fixed-spectrum family with the submitted seed, and writes its outputs to `outputs/`.

## Frozen design

- `N=200`, `rho(F)=0.95`, `sigma=0.05`, `T=1000`, 116 internal `kappa` values, and 16 independent white-noise realizations per `kappa`.
- Exact same 116 `kappa` values and **same physical matrices** as the legacy Fig. 4 source (relative matrix differences <2e-15 at tested kappa values).
- A shared random-noise bank across `kappa` produces a **paired parameter comparison**; each of its 16 realizations is independent of the other 15.
- Response directions are obtained from the **known operator**, via 24-start M2, not from finite-data inference.
- Rates-based `R` (primary) and exact one-step `R_sigma` (companion) are both saved. The old multiplier-based ratio is retained in the results table ONLY for auditing differences.
- 16-run median and interquartile range; no bootstrap significance claim is made.

## Numerical validity

The simulation is implemented using the exact diagonal similarity transform (much faster than multiplying dense 200x200 matrices every step). This is verified against the dense recurrence at kappa=1,4,60,450. The independent analytical population correlation from the stationary covariance is verified directly by solving the full 200D discrete Lyapunov equation, agreeing at ~1e-14. The original legacy matrices were checked independently (max relative difference ~2e-15).

## Scientific interpretation

The full-spectrum stability is fixed. Corrected R spans approximately 0.056–118.08. Correlation is not monotonically rising for all subthreshold R: there is a dip around R~0.87 before a pronounced increase for R>1. An R=1 crossing is **not** a theoretically predicted sharp crossing of correlation. This validates an association for this particular fixed family, not universal causal/equality statements and not finite-data recovery.

## Outputs

- `outputs/T01_fig04_alignment_rates_revised.pdf/png`: revised main-text figure, correct rates-based x-axis, sample IQR.
- `outputs/T01_fig04_alignment_with_stationary_reference.pdf/png`: independent population validation overlay.
- `outputs/T01_fig04_116point_alignment.csv`: complete plotted data, both threshold conventions and population curve.
- `outputs/T01_fig04_1856_replicates.csv`: all trial-wise absolute correlations.
- `outputs/T01_fig04_planes_and_reduced_operators.npz`: 116 extracted reference planes and 2D compressed operators.
- `outputs/T01_fig04_direct_simulation_checks.json` and `outputs/T01_fig04_full_lyapunov_checks.json`.
- `outputs/T01_fig04_run_manifest.json` with configuration hashes, random seed and environment.
- `Fig04_revised_caption.tex`; `tests/test_fig04.py`; `outputs/T01_fig04_report.md`.
