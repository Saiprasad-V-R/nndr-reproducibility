# T01 / submitted Figure 7 — Reproduction audit (22 October 2026)

## Scientific status

**VALIDATED COMPUTATION — ORIGINAL M=1 SUCCESS CLAIM NOT REPRODUCED.**
This is a critical audit outcome, not a passing tracking benchmark.
The time-varying true operator is simulated faithfully with N=100, M=1,
T=600, W=80, window step=10, rho(F_t)=0.95, Gaussian forcing sigma=0.05,
and a logarithmic ramp kappa(t)=1 -> 200. The inferred R is recomputed on
continuous-time logarithmic rates, per rebuttal Appendix B/C.

- `run.py` generates the faithful M=1 benchmark and mathematical reference.
- `run_M25_sensitivity.py` **only** tests whether adding 25 trajectories
  explains the legacy discrepancy. It is NOT the submitted M=1 benchmark.
- `run_seed_sensitivity.py` repeats the main M=1 analysis for five independent
  realizations (5678–5682; 5678 reuses primary outputs).
- `plot_M25_comparison.py` visualizes the single-record vs ensemble contrast.
- `tests/test_fig07.py` checks simulation, exact operator averaging, rank,
  schedule, window indexing, invalid-log handling and outputs.

## Reproduction command

From this folder or any working directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_M25_sensitivity.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_seed_sensitivity.py
python plot_M25_comparison.py
```

Figures and full results are saved in `outputs/`. This script never forces
stability and never assigns a rates-based R to a reduced operator with negative
real multipliers under the principal-log convention.

## Explicit differences from legacy code

1. The original plot-producing legacy file *sets M=25*, despite the published
   Fig. 7 caption specifying M=1. An alternate legacy script plots both but
   overwrites files and uses T=500, not T=600.
2. Primary ratio computed from rates through the real principal matrix log.
3. Relative ridge parameter selected with the last 20% of each 80-pair window
   as validation, not a random train/validation split. The change is disclosed;
   exact legacy ridge selection should not be described as reproduced.
4. M2 is multistart. No forced stabilization; negative-real reduced multipliers
   are explicitly excluded from R; R_sigma remains available when reduced map
   is stable.
5. Arithmetic mean of true F_t is calculated *exactly*; because F(kappa)
   depends on both kappa and kappa^-1, it is NOT F(mean kappa).

## Limitations

Reference is M2 applied to the true time-averaged operator (as in the submitted
Fig.7). It is not an independent physically planted Q*. Tests of true planted
plane/time variation are reserved for new T10. M25 tracking is still poor
at the earliest times, when the true non-normal mechanism is weak.

Do not use this as a publication-ready success figure without revising the
scientific claim and discussing the invalid-window fractions.
