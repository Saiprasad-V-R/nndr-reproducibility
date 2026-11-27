# T01 — Submitted Fig. 8 (broad singular-value spectra): revised numerical audit

## Reproduction

From this directory run `python run.py`, then `python run_population_check.py`, then `python plot_diagnostics.py`. Python requires NumPy, SciPy, pandas and matplotlib; pytest verifies the saved outputs. There are no command-line arguments and all filenames are fixed. For bitwise more consistent numerical behavior, use `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## Frozen geometry and declared scientific correction

- `N=80`; `M=18`; legacy `T_train=220` denotes **220 one-step pairs per trajectory** (the script evolves 221 state observations including the initial zero), a legacy code/caption ambiguity.
- Fixed absolute ridge `1e-4`; Gaussian innovations SD `.035`; 8 seeds; five background exponents `0.25,0.75,1.25,1.75,2.25`.
- Known planted plane `Q_star=O[:,:2]`; `F=O diag(B_R,D_beta) O.T`, `D_beta, jj = 0.74*j^-beta`.
- Embedded multipliers `a=.945` and `b=.455` are retained exactly; they give multiplier splitting `.35`, but **rates splitting `.8659515`**.
- Coupling `B_R[0,1]=R_target*Kc_rates*(a-b)` is **recalibrated** so the new *rates-based* reference R values are exactly `1,3,10,30`; this changes the physical operator from the originally submitted fixed-old-R operators, by design and explicitly.
- Shared reduced diagnostics compute `R_variance` from the principal real matrix logarithm and `R_sigma` as the separate one-step measure. No clipping of invalid reduced eigenvalues and no forced stability rescaling.
- M2 is approximate multistart projected alternating optimization (12 restarts, 220 iterations); M3 uses the symmetric commutator's extreme eigenvectors.

## Key finding

Strong regimes (`R_star=10,30`) have small principal-angle and rate-R errors for both methods. At `R_star=1`, however, M2 fails to recover the planted plane in a broad background **even on the exact population operator**: at beta >= .75 its optimal full-operator transverse-transfer objective is 0.0099 to 0.0165 greater than the maximum objective attainable in Qstar. Thus the reference target differs from the M2 objective's preferred plane, independently of sampling noise. See `T01_fig08_population_reference_check.csv`.

In finite data, for Rstar=1, M2 has 32/40 invalid reduced ratios (31 unstable reduced maps and one negative real multiplier), with every corresponding full fitted map spectrally stable. M3 has 40/40 admissible ratios in that subgroup but median plane-angle error 25.5 deg. These cases invalidate a blanket claim that both methods recover the planted plane accurately across all beta and Rstar. Panel (c) and the log scatter (d) report only admissible R estimates; **one must report 32 omissions explicitly**, not interpret those panels without the validity count.

## Outputs

- `outputs/T01_fig08_broad_spectrum_revised.pdf/.png`: four-panel regeneration of original Fig. 8.
- `outputs/T01_fig08_population_and_validity_diagnostic.pdf/.png`: complementary population and invalid-fit evidence.
- `outputs/T01_fig08_full_results.csv`: every simulation x method row including denominator and status.
- `outputs/T01_fig08_population_reference_check.csv`: zero-noise population benchmark and planted-plane objective.
- `outputs/T01_fig08_summary.json`, `outputs/T01_fig08_run_manifest.json`, `outputs/T01_fig08_report.md`.
- `tests/test_fig08.py`: frozen parameter checks, physical/orthogonal coordinate consistency, independent planted truth, invalid-fit bookkeeping, negative result and strong-regime recovery.

**Classification**: numerical reproduction/audit complete; revision of the original unconditional positive claim is required. This is not a completed N1 independent comparator suite; T02 is still pending.

## Exactly unchanged legacy physical operators — additional control

`run_unchanged_legacy_operator_control.py` preserves the *submitted* off-diagonal
coupling and uses the revised rates threshold for the reference and fitted R.
Its old labels 1,3,10,30 correspond to corrected true R
3.8474,11.5423,38.4742,115.4225. Both methods recover these physically
stronger matrices much more accurately; all 320 rates ratios are admissible.
**The positive historical result is therefore compatible with the newly
observed failure at true corrected R=1.** See `T01_fig08_original_vs_recalibrated.pdf`.
