# T05 — Fig. N2a,b: synthetic plane and threshold ablation

Reproduce with no command-line arguments from any working directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python 05_N2_ablation/run.py
```

## Fixed experiment

- `N=32`; `M=8` independent trajectories for each condition, with 240 transitions per trajectory; isotropic innovation noise `0.035`; trace-scaled ridge `0.001`; 10 seeded paired systems per design cell.
- Physical families: isolated planted non-normal 2×2 block, same block plus a competing full-system amplifier, and spectrum-matched normal controls.
- Independent interventions: change eigenvector non-normal geometry only; change eigenvalues while holding eigenvectors fixed; rotate the *location of the planted plane* through 0.58 radians; combined intervention; normal stable baseline/control.
- Planes: estimated M2/M3, snapshot PCA, baseline-frozen M2 and M3, independent current planted plane, and baseline-frozen planted plane. The planted reference is **never** used by the estimator.
- Full-system one-step gain `sigma_max(F)` and all *known-operator* plane-capture fractions are computed with the true full `F`, not with the fitted operator. Separately record fitted-plane `R_A`, `R_sigma`, `K`, `Kc`, `d_F`, and `K/K_NA`, including invalid-domain flags. The full operator is stable in every trial.
- Every experimental subject provides both baseline and event operator conditions. The same latent innovations are used across the pair; when the planted plane rotates, the innovations are orthogonally transformed with the entire system. Differences across paired conditions are controlled physical interventions, not clinical/event labels.

## Principal outputs

- `outputs/T05_FigN2ab_plane_threshold_ablation.pdf`, `.png`: pooled four-panel ablation **with restrictions**. Do not interpret a pooled planted-plane angle as an overall quality measure in the competing family.
- `outputs/T05_family_stratified_ablation.pdf`, `.png`: isolates the competing-amplifier effect; recommended companion figure.
- `outputs/T05_all_phase_methods.csv`: **all** per-method rows and their full-system, true-compression and fitted-compression diagnostics.
- `outputs/T05_planted_reference.csv`: independent physical ground truths.
- `outputs/T05_all_paired_contrasts.csv`: paired log contrasts (event minus baseline), invalid cases retained.
- `outputs/T05_log_decomposition_by_method.csv`: means-of-logs decomposition on identical eligible subjects.
- `outputs/T05_estimand_errors_all_rows.csv`: explicit metric error versus the *planted* physical estimand. In a competing family, estimators may recover the true larger amplifier instead; do not misinterpret planted error as global-growth error.
- `outputs/T05_method_summary.csv`: 95% independent-subject percentile bootstrap interval for median plane angle and known-operator capture, plus full-gain miss counts.
- `outputs/T05_report.md`, `outputs/T05_manifest.json`, and `FigN2ab_revised_caption.tex`.

## Interpretation and limitations

`R_A=K/Kc` is the **delayed continuous-time impulse variance** threshold; `R_sigma` is the **discrete one-step** threshold. All formulas are applied on **2D compressions**, not asserted as full-operator thresholds. A stable full operator can yield an unstable 2D compression. Full one-step gain can be missed by PCA or a frozen plane; conversely a plane with a large error relative to the *designated* planted block may still capture a stronger non-planted full-system amplifier. The mean-of-log contrast factorization is exact for positive K and Kc, but must not be applied to median-based contrasts. Normal K near machine zero is explicitly excluded from log-ratio calculations.

Re-run all tests with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q 05_N2_ablation/tests/
```
