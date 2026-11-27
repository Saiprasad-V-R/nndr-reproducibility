# T06 — Fig. N3 negative controls

Run `python run.py` or open `run.py` in Spyder and Run; no arguments. `config.json` fixes the complete simulation and inference design. `outputs/T06_report.md` contains the full scientific interpretation and limits.

Raw data: `T06_subject_method_results.csv` (all 1,728 rows), `T06_paired_event_contrasts.csv` (864 pairs), `T06_typeI_Wilson_summary.csv`, `T06_cohort_permutation_tests.csv`, `T06_large_ensemble_references.csv` (exact stationary and independently estimated projections), `T06_manifest.json`.

Figures: `T06_FigN3_negative_controls.pdf/.png` and `T06_oscillator_sampling_sensitivity.pdf/.png`.

All 29 new tests are in `tests/test_t06.py`. Crucial qualification: the preliminary screens are *not* a replacement for the calibration planned for T08, and all rejected/invalid cases stay in denominators.
