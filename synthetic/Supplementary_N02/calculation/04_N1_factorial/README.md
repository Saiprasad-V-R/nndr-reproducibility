# T04 — N1e: spectrum-versus-geometry factorial diagnostic benchmark

Run with no command-line arguments from anywhere:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py
```

This script creates an independently controlled *paired intervention* ensemble (90 anchors, 180 interventions), compares two-feature classifiers with group-held-out evaluation, tests a **known planted two-dimensional rate-block** and an **unconstrained M2 plane from finite data**, includes spectrum-matched normal operators, numerical derivative-of-impulse-variance truth, a documented lower-bound Kreiss grid, and two-independent-noise estimator stability checks.

**Interpretation restriction:** A fitted-classifier distinction between an engineered geometry-only versus spectrum-only change is not a demonstrated clinical/event classifier or evidence of causation from observation alone. Results favor standard diagnostics over inferred NNDR K/Δ in this tested regime; see `outputs/T04_report.md`. Population oracle figures establish theoretical factorization, not general finite-data superiority.

For a fair method-to-method comparison, use `outputs/T04_common_support_summary.csv` with the same 140 admissible intervention pairs and group fold assignments for all seven methods. Per-method larger-denominator scores in `T04_cv_summary.csv` are reported as separate coverage diagnostics, not pairwise accuracy claims.

Re-run regression:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/
```
