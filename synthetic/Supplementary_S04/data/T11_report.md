# T11 — Fig. N5: Sample-efficiency and commutator-gap validation

## Study and provenance

New controlled experiment: 18 independent base operator/innovation subjects, 72 physical operators (four paired commutator strengths per base subject), 432 finite-data fits. N=[16, 32, 64]; competing commutator fractions [0.0, 0.5, 0.9, 0.98]; 6 sample-size/horizon regimes, including M=25, T=3/4 and M=1, and 6 independent seeds per physical design. The same orthogonal embedding and innovations are reused at all four competing commutator strengths for each base subject, as well as nested across sampling regimes. Per-condition bootstrap resamples independent base subjects, not overlapping pairs. Existing T01 Figure 6's 184 stored method rows are reused **without simulation rerun** and separately audited.

## Exact perturbation facts

Let E=F_hat-F and C(F)=FF^T-F^TF. Then C(F+E)-C(F)=FE^T+EF^T+EE^T-F^TE-E^TF-E^TE, so ||delta C||_2 <=4||F||_2||E||_2+2||E||_2^2. For simple extreme commutator eigenvalues separated by g_C from the nearest interior eigenvalues, the M3 plane has

    sin(theta_max) <= min(1, sqrt(2)*delta/(g_C-delta)), delta<g_C; else <=1.

This conservative bound follows by applying the residual equation to each extreme eigenvector and combining both with a Frobenius bound. The often-plotted 2*delta/g_C is a *first-order proxy*, **not asserted as a certified global bound**.

All 432 controlled fits satisfy both the commutator polynomial bound and the conservative Davis-Kahan inequality. Only 36 give a nonvacuous (<1) certified bound; the remaining fits do **not** contradict perturbation theory. Population M3 exactly spans an independent planted Qstar because the primary block has larger extreme commutator eigenvalue than the competing block. The ensemble deliberately decreases g_C while keeping all eigenvalues fixed.

## Empirical outcomes

- Cases with informative n=M(T-1)<N: 48/432. This affects unregularized full-operator identifiability. It is not an N^2 threshold.
- Inadmissible rate-based ratios: M2 175/432; M3 135/432. Plane angles include **all** fits regardless of R admissibility.
- Maximum observed relative difference in separately recomputed original Fig. 6 commutator gaps: 2.13e-14 absolute. Original Fig. 6 M3 angles are to M2(known F), so **must not** be used as a Davis-Kahan test against population M3 eigenvectors.

### High-information subset M16,T240

| Competitor fraction | Median M3 angle (degrees) | Median M2 angle (degrees) | Nonvacuous D-K bounds / 18 |
|---|---:|---:|---:|
| 0 | 9.014 | 15.184 | 18/18 |
| 0.5 | 9.115 | 15.336 | 12/18 |
| 0.9 | 12.480 | 19.377 | 0/18 |
| 0.98 | 19.211 | 23.013 | 0/18 |

## Scientific qualifications

The scatter plot in panel (c) marks perturbation ratios >10 at the right boundary as triangles, retaining all such cases visually. A full operator with n>=N informative transitions need not yield a well-estimated commutator plane if its g_C is small. Conversely, plane recovery may succeed when the Frobenius error of F_hat is non-negligible, if the extreme commutator gap is large enough. The mathematical inequality is deliberately conservative, and its frequent vacuity is not a finding of failed recovery. M2 is a nonconvex numerical extractor, and a Davis-Kahan theorem for the symmetric commutator does not automatically apply to its optimized plane. Re-estimation under the same stochastic data is not an independent validation replicate; all interval denominators count seeded operators. The exact original Figure 6 N=50–400, M=1 insets and T=3/4 were previously generated under T01 and are **analyzed here, not newly simulated**.

## Files

`T11_full_gap_sample_rows.csv` (all fitted diagnostics and exclusions), `T11_cluster_bootstrap_summary.csv` (6 independent seeded operator replicates per N × condition, clustered by subject), `T11_original_Fig06_reanalysis.csv` (existing original scan), `T11_FigN5_gap_sample_efficiency.pdf`, `T11_original_Fig06_gap_reanalysis.pdf`, `T11_summary.json`.
