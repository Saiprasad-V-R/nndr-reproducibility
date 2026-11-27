# T01 — Submitted Figure 5 (finite-data M1/M2/M3 comparison)

## Execution

From any directory:

```bash
python /path/to/NNDR_synthetic_revision_suite/01_baseline_reproduction/fig05/run.py
```

Dependencies: Python 3, NumPy, SciPy, Pandas, Matplotlib. All outputs go to `outputs/`.

From suite root, validate:

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q 00_threshold_validation/tests 01_baseline_reproduction/fig03/tests 01_baseline_reproduction/fig04/tests 01_baseline_reproduction/fig05/tests
```

## Connection to the submitted manuscript

This reproduces the experimental operator family behind *submitted* Fig. 5, PDF page 11, with N=100, rho(F)=0.95, Gaussian innovation SD=0.05, 29 original kappa settings, T_train=300, M=1/25 and T_test=1200. The extraction-independent planted-plane N1 study is **not** this figure.

## Revision choices (recorded, not silent)

- Same Eq. (44) operators and fixed physical kappa settings, with seed=1.
- The old multiplier-based R is replaced by rates-based R_A, and R_sigma is saved separately. `T01_fig05_known_operator_old_vs_new.csv` documents the shift without changing the physical operators.
- 17 trace-normalized ridge levels from 1e-8 to 1.0, selected by 80/20 **chronological within-trajectory** validation. The old absolute ridge with shuffled 80/20 pairs is intentionally not reproduced; this is the revision's normalization and a less leakage-prone split.
- Twelve initializations of M2, 300 projected iterations per initialization. The best attained objective is used; it is NOT a certified global optimum.
- M3 extracts the most positive and most negative eigenvectors of the symmetric commutator.
- M1 follows the legacy eigenbasis-SVD rule. For complex fitted eigenvectors it chooses the dominant real or imaginary part; its stability near repeated or complex modes remains a limitation.
- The old `cplane` code optimized covariance rather than Pearson correlation. The corrected exact multiple-correlation formula is used here and independently checked against a brute-force plane sweep.
- Every M=1 and M=25 fit uses the *same test time series* at each kappa and paired simulation noise across kappa. This improves paired comparisons but is not a repeated-Monte-Carlo uncertainty analysis.
- No spectrum is forcibly stabilized. 19/174 inferred plane/rate diagnostics are unsupported; all 19 are in M=1, and they are omitted only from R-based lines, not from plane-angle or correlation panels. The source statuses are in the CSV.

## Evidence and limits

The known-operator reference plane is M2 applied to the known F, so the M2 recovery angle isolates estimation noise; it is not an unbiased comparison of extraction-rule quality. The revision explicitly promises an independent Q* planted benchmark with snapshot and SVD comparators in T02/N1.

The robust result is that on this constructed family, with 25 trajectories, M2/M3 have about 4.8/4.5 degree median angle to the population-M2 plane. With one trajectory, angle medians are about 30.4/30.6 degrees and the ratio may be invalid. These medians average over nonuniform kappa, not independent Monte Carlo trials.

## Main assets

- `run.py`: fully standalone driver (finds its shared modules relative to this directory)
- `config.json`: frozen parameters and random seeds
- `tests/test_fig05.py`: 14 tests of the data construction, CV, plane correlation, method invariants and saved results
- `outputs/T01_fig05_revised_rates.pdf` and `.png`: four-panel plot
- `outputs/T01_fig05_full_results.csv`: all methods and both data budgets with exact ratios and validity flags
- `outputs/T01_fig05_ridge_cv.csv`: selected regularization and fit-quality metrics
- `outputs/T01_fig05_known_operator_old_vs_new.csv`: numerical ratio correction at identical physical operators
- `outputs/T01_fig05_results.json`, `outputs/T01_fig05_report.md`, `outputs/T01_fig05_manifest.json`
- `Fig05_revised_caption.tex`: LaTeX caption

The project tracker remains **T01 IN PROGRESS** until submitted Figs. 6–8 are also regenerated.
