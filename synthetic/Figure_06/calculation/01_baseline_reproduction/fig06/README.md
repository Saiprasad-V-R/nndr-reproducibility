# T01 / Fig. 6 — corrected scaling benchmark

`run.py` (no arguments) regenerates the entire original three-axis grid (M: 8, N: 8, T: 9 values; 3 spectrum/geometry regimes) and the M=1 inset tests. Results go to `outputs/`.

**Experimental changes:** Original Fig. 6's target values were set under an invalid multiplier-based threshold. Here kappa is **recalibrated** for each N and each target using the rates-based variance-response criterion. Thus the physical operators are not identical to the submitted Fig. 6; compare this figure at fixed target *corrected* R, not as a fixed-operator reproduction. In contrast, Figs. 3–5 retain historical kappa.

**Estimator changes:** 8-start M2, 14-start population M2, relative-ridge cross-validation on the last 20% of each trajectory, and invalid negative-real reduced multipliers treated as missing. No forced stabilization. M3 uses the symmetric commutator. Input/response columns orthonormal.

**Statistical scope:** A single random realization per condition; no confidence bands or population repeatability inferred. The reference is M2(true F), so results assess estimation accuracy only, not extraction-rule superiority. The planted-plane benchmark is T02/N1.

**Sample count:** `n_pairs` in CSV counts informative (nonzero-predictor) pairs: M*(T-1), because x0=0. Sufficient identifiability for an unregularized linear map is generically `n_pairs >= N` with full-rank predictors, **not N² independent state samples**. Sample efficiency is an error-versus-operator-perturbation and commutator-gap question. The N-scan stores the direct perturbation and Davis–Kahan bound for a later N5 analysis.

**Result files:** `T01_fig06_scaling_revised.{png,pdf}`, `T01_fig06_full_results.csv`, `T01_fig06_target_kappas.csv`, `T01_fig06_summary.json`, `T01_fig06_report.md`, and `T01_fig06_run.log`.
