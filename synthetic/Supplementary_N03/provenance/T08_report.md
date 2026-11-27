# T08 — Calibrated five-part local-linear model adequacy

## Study and calibration protocol

- Replayed physical recordings exactly from T06 (864) and T07 (3456) reference generators and original stored seeds; no re-use of preliminary two-rule gate.
- A. Last 20% one-step mean-square predictive ratio, full fitted map relative to diagonal regularized VAR1; an upper-tail log ratio indicates prediction failure.
- B. Held-out lag covariance discrepancy at lags 2 and 4 versus the powers of the trained VAR1 map.
- C. Semigroup discrepancy: directly trained lag-2/lag-4 maps against powers of the one-step map. Complex principal-frequency alias suspicion is a *separate flag*, not an additional calibrated rejection rule.
- D. Cross-channel mean squared residual energy autocorrelation, contrasted with multivariate Fourier phase-randomized surrogate trajectories. Each surrogate applies the SAME phase to all variables at a given frequency within a trajectory, preserving its cross-spectrum. We calibrate the observed-minus-surrogate-mean difference, not an asymptotic p-value.
- E. Six-lag two-dimensional projected residual portmanteau. Projection is the leading right/left singular plane of the TRAINING operator, not outcome-selected M2/M3. This makes the gate method-agnostic; no claim of exact M2 projection.
- Lock all five upper cutoffs at empirical 99% quantiles among 900 correctly specified VAR1 records from 432 independent synthetic subject clusters (even replicate indices); includes Gaussian normal/nonnormal, short and oscillatory exact VAR1, and unperturbed baselines of burst/noise/switch/VAR2-event protocols. Within-subject label/phase records are correlated, and are not counted as independent clusters. No RA label or future event outcome used.
- Evaluate on odd replicate indices only. Even-replicate cases are saved for audit but NOT counted in held-out power/FPR.
- Five separately calibrated 1% upper-tail screens are combined with an OR rule (nominal Bonferroni familywise bound 5% under ideal calibration); the 1% empirical quantiles are finite-sample estimates, not exact finite-sample guarantees.
- Invalid reduced R_A fits are unassessable and still included in full-cohort denominators. Failed model-adequacy windows are also unassessable, not automatically classified as true negatives.

## Locked upper-tail cutoffs

| Test | Statistic | Cutoff |
|---|---|---:|
| A | `a_predictive_log_mse_ratio` | 0.085969 |
| B | `b_lag_covariance_mismatch` | 0.60614 |
| C | `c_semigroup_discrepancy` | 0.34383 |
| D | `d_energy_excess_vs_surrogate` | 0.13791 |
| E | `e_projected_portmanteau` | 45.2 |

## Held-out evaluation

- Independent held-out physical records: 2160; correctly specified VAR1 recordings: 900, jointly rejected 33/900 (3.67%). These are paired recordings from fewer independent subject clusters.
- T07 true positive (R_star>1) raw M2 / M3 detection: 92.13% / 92.13%; after screening detected per ALL: 27.78% / 27.78%.
- T07 matched-normal raw M2 false positive: 0.93%; screened false positive per ALL: 0.00%.
- T06 event spatially coloured forcing adequacy rejection: 100.00%.
- Screened detection rate among accepted windows has a different denominator from detected-and-accepted rate among all. Both saved per condition and method.
- Primary uncertainty intervals are formed by resampling independent subject clusters, never time samples or multiple paired records as independent. When zero events are observed, percentile-bootstrap bounds degenerate at zero, so descriptive Wilson binomial bounds are also given; neither is a universal error guarantee. See T08_subject_cluster_bootstrap_intervals.csv.
- These estimates describe this synthetic design, not a universal validated rate on arbitrary empirical data. No empirical thresholds are tuned or tested here.

## Full-cohort versus conditional-accepted denominators

- Detailed T08_holdout_power_FPR_and_rejection.csv reports RAW detections / all, accepted detections / all, and accepted detections / accepted, as distinct estimands. A high accepted-only power after rejecting most positives is NOT evidence of an effective operational detector.
- T08_baseline_event_rejection_by_family.csv reports all baseline and event rejection fractions independently, with subject-level split labels.

## Important limitations

- The gate includes a 2D leading-singular-plane portmanteau, not a recovered M2/M3 plane. The gate is shared by both inference methods and should be named accordingly in the rebuttal.
- Validation is split by independent Monte Carlo replicate, not a test on entirely new physical matrix families. Generalization to new dynamics remains outstanding.
- The five checks do not prove local linearity or physical causation; a model may pass yet still be misspecified. Aliasing and invalid estimated matrix logarithms require separate caution.
- The entire waveform is available to tests B–E (A is strictly held out), including the post-event period. In observational use, screens must be recalibrated for record length and sampling.

## Reproduction

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 08_adequacy_calibration/run.py`
Input: original T06/T07 scripts, configuration files, CSV result identifiers, and random seeds. Outputs include complete record-level tests, five locked cutoffs, method-level joined gates, held-out family power/rejection, figures, and tests.
