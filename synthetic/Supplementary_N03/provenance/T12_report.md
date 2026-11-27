# T12 — Dependence-preserving bootstrap calibration and M2 solver audit

## Design

- 288 independent synthetic subject-recordings; 6 spectral families; 120 moving-block resamples per record; 144 solver-reference audits with 24 single starts each.
- Trajectory-stratified bootstrap resamples whole independent trajectories and contiguous blocks of (x_t,x_(t+1)) pairs. Block boundaries are NOT interpreted as valid transitions. The terminal block is trimmed so exactly M*T transition pairs are used; this is a block-pair bootstrap.
- Stationary burn-in precedes all estimation windows; no artificial independence of adjacent transitions.
- Oracle: planted Qstar fixed in all bootstrap draws; population truth from Qstar.T @ F @ Qstar.
- Selected M2: reoptimized in every bootstrap; population truth uses high-start M2 of exact F, not the planted plane unless they coincide.
- M2 numerical strong reference uses independent multistart L-BFGS-B on the single-vector reduced objective, plus high-start M2. It is NOT a global optimum certificate.
- Percentile and *bias-corrected percentile* (BC, not BCa) evaluated at nominal 90% and 95%. BC uses z0 only, no acceleration.
- A ratio interval is estimable only if the point target is admissible and >=90% of bootstrap replicates yield that ratio; invalids are not clipped.
- Primary coverage denominator is all independent subjects; missing/inadmissible intervals count as noncoverage. Conditional coverage and Wilson bounds also reported.
- Heuristic effective sample size uses a positive-sequence autocorrelation estimate of the fixed planted response projection; not an exact likelihood or independent-sample count.

## Unconditional 95% percentile CI coverage for R_A

| Sampling | Method | Covered / all subjects | Estimable | Unconditional | Conditional |
|---|---|---:|---:|---:|---:|
| M1_T90 | oracle | 80/96 | 92/96 | 0.833 | 0.870 |
| M1_T90 | selected_M2 | 27/96 | 41/96 | 0.281 | 0.659 |
| M1_T240 | oracle | 86/96 | 96/96 | 0.896 | 0.896 |
| M1_T240 | selected_M2 | 49/96 | 75/96 | 0.510 | 0.653 |
| M8_T240 | oracle | 93/96 | 96/96 | 0.969 | 0.969 |
| M8_T240 | selected_M2 | 70/96 | 89/96 | 0.729 | 0.787 |

## Solver sensitivity

- Median single-start objective-hit fraction within 1% of independent strong numerical reference: 0.208.
- Median over audited datasets of the maximum single-start objective gap: 0.16277.
- Median single-start plane spread relative to best-start plane: 5.09 degrees.

## Interpretation

Bootstrap coverage is specific to this finite ensemble and the stated block length / ridge / fit protocol. Strongly non-normal regimes, repeated spectra and nearly tied commutator modes may produce nonregular estimates; standard percentile/BC intervals should not be advertised as universally calibrated. M2 restarts are a numerical uncertainty diagnostic; close objectives do not guarantee the same physical plane when the landscape is nearly degenerate.

## Files

- `T12_subject_point_estimates.csv` — subject-level estimates, statuses, ESS and plane angles.
- `T12_bootstrap_draws.csv` — **every bootstrap replicate** including invalid R statuses.
- `T12_interval_results.csv` — every nominal CI and coverage indicator.
- `T12_coverage_by_family_regime.csv` — conditional/unconditional coverage and Wilson bounds.
- `T12_solver_audit.csv` — independent numerical strong-reference comparisons.
- `T12_known_population_references.csv` — each planted versus selected-M2 population truth.
- Two figures PNG/PDF, code, config, tests, run manifest.

## Exploratory dimension sensitivity (separate cohort)

72 additional independent subject-recordings across N=6,12,24; two spectral families, two regimes and six subjects per cell. The 60-replicate intervals are exploratory, not pooled with the main cohort; Wilson intervals are wide. See T12_dimension_summary.csv and its figure.
