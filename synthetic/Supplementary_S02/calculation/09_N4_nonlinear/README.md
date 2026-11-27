# T09 — Fig. N4: nonlinear dynamics with independently known Jacobians

Run `python run.py` (no arguments; runnable from Spyder or any working directory).
For reproducibility, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before starting Python.

## Protocol

- Four 8D systems: 4 directed FitzHugh–Nagumo units, 4 directed Stuart–Landau oscillators below and above Hopf, and an 8-coordinate symmetric-gradient coupled double-well SDE.
- Eight independent subjects/system, ten estimation trajectories/subject, and 64 independent trajectories used only for an ensemble statistical reference. Each observed trajectory has 300 transitions.
- Stochastic Euler–Maruyama `h=0.01`, output sampled at `Δt=0.2`; additive independent Brownian noise in each physical state coordinate. Model-specific noise amplitudes and all generator coefficients in `config.json` / `run.py`.
- Window length 60 transitions, stride 20. FHN pulse [23,30] model-time units. `before_pulse`, `overlaps_pulse`, `after_pulse` based on the **full support** of each window, not the center.
- Ground truth local map is the mean of ordered products of `(I+h J[x(t)])` over all microsteps **of each observation step**. A separate, independent large-ensemble `C1 C0^{-1}` is computed from observations within each matching time window. Both are independent of the estimated operator.
- Fit finite data using centered lag covariance, trace-relative ridge, extract M2 (seven multistart restarts) and M3; compare angles versus known-physics tangent M2 plane and independent statistical M2 plane, threshold values, known full gain, and gain capture.
- Apply the five T08 fixed cutoffs as **exploratory model-applicability transfer only**: they were calibrated on distinct, largely linear models of different window lengths; do not call their nominal false-positive rates validated on N4.
- Independent-subject cluster bootstrap preserves all correlated moving windows. No pooled-window binomial confidence intervals.

## Primary outputs

`outputs/T09_FigN4_nonlinear_known_jacobians.pdf/.png`: 6-panel draft Fig. N4.

`outputs/T09_nonlinear_FHN_gradient_sensitivity.pdf/.png`: direct FHN ratio/geometry/time comparison and the gradient adverse case.

`outputs/T09_window_truth_and_fit.csv`: one row per physical subject/window with known and fitted threshold/gain, independent statistical target, admissibility flags, and five adequacy scores.

`outputs/T09_all_method_results.csv`: paired M2/M3 results, fitted and reference plane errors, and full gain captured.

`outputs/T09_condition_summary.csv`: before/overlap/after and below/above-Hopf counts.

`outputs/T09_spurious_detections_and_gate.csv`: true-subthreshold denominator, number of spurious above-one fits, adequacy gate caught/not-caught.

`outputs/T09_subject_cluster_CI.csv`: cluster-bootstrap 95% uncertainty; only eight independent subjects per family, so interpret intervals cautiously.

`outputs/T09_numerical_jacobian_checks.csv`: finite-difference drift-J and full one-observation-step ordered-propagator tests.

`outputs/T09_report.md`, `outputs/T09_manifest.json`: explanatory audit, source hashes, provenance and invalid counts.

## Critical cautions

The mean ordered tangent map, the exponent of the mean instantaneous Jacobian, and the statistical linear predictor are different estimands in a nonlinear stochastic system. A gradient drift has a symmetric instantaneous Jacobian but this **does not** imply its finite-time ordered product or `C1 C0^{-1}` is Euclidean normal. No rate-based R value is assigned if a 2D compression is spectrally unstable, has a negative-real multiplier, or otherwise falls outside the specified principal logarithm domain. Also, a gain-maximizing plane can be poorly identified in nearly normal systems, making a principal angle intrinsically unstable. The complete tables preserve the gradient-system spurious high-R case and the stricter adequacy-screen rejection fractions.
