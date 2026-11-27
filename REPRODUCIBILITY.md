# Reproduction guide and verification

## Source of truth

The authoritative scientific reference is the 27 November 2026 clean revised manuscript, `NNDR_revised_clean.pdf`. The archived experiment scripts in this directory reproduce the manuscript calculations; `nndr-toolkit` was used for mathematical comparison. No toolkit API was substituted into a calculation when that could change archived results.

## Execution map

| Part | Calculation | Plotting | Outputs and checks |
|---|---|---|---|
| Synthetic Figures 1, 3–8, A.1 and supplementary N01–N04/S01–S04 | Figure-specific commands in [synthetic/README.md](synthetic/README.md) | Same inventory | Original numerical tables, source figures and local configurations; see [mathematical audit](synthetic/MATHEMATICAL_CONSISTENCY.md). |
| EHG `ice039_p_3of6` | From `empirical/01_uterine_ehg/`: `python scripts/run_example.py` | `python scripts/plot_example.py` | 679 windows; finite R_A >1.56. M3 support basis variation documented in [direction audit](empirical/01_uterine_ehg/EHG_DIRECTION_AUDIT.md). |
| EEG `chb01_03` | From `empirical/02_seizure_eeg/`: `python scripts/run_example.py` | `python scripts/plot_example.py` | 201 endpoints; R_A, R_sigma and entropy support match archived columns to ≤6.9e-7. |
| FoG `S05R01_E28` | From `empirical/03_freezing_of_gait/`: `python scripts/run_example.py` | `python scripts/plot_example.py` | 62 overlapping endpoints per method match archived R_A to <1e-13. Manuscript aggregation is in `results/manuscript_event_summary.csv`. |
| Push-up | From `empirical/04_pushup/`: `python scripts/pushup_subset_final_5s_bootstrap.py` and `python scripts/run_timing_example.py` | `python scripts/plot_example.py` | 1,913/338 pairs; raw point estimates match archived values to <6e-14. |

Install with `python -m pip install -r synthetic/requirements.txt` and `python -m pip install -r empirical/requirements.txt`. Empirical scripts read the recording in their local `data/` folder and write newly calculated results to `results/`; they do not use `reference_data/` as numerical input. Run one example at a time. The full-cohort helper files retained inside some examples are imported for their validated functions; their cohort entry points are never called by the example runners.

The four representative examples are [uterine EHG](empirical/01_uterine_ehg/README.md), [seizure EEG](empirical/02_seizure_eeg/README.md), [freezing of gait](empirical/03_freezing_of_gait/README.md), and [push-up motion](empirical/04_pushup/README.md).

## Mathematical conventions

R_A is the rate-based variance-response ratio of the real principal logarithm of an admissible, stable reduced one-step map. R_sigma is the exact discrete one-step amplification ratio and may be defined when R_A is not. Real and complex rate thresholds, one-step multiplier splitting, full/reduced stability and undefined estimates remain distinct. Relative ridge is scaled by the state-covariance trace and dimension; shrinkage preserves the operator trace. M2 and M3 directional ordering follows each archived experiment. EHG and FoG use normalized L1 response-direction support; EEG uses entropy support. The push-up primary subset analysis uses 5 s circular blocks and 2,000 bootstrap replicates; its separate moving-window normal control uses 1 s windows at 0.1 s steps.

## Numerical verification

- **EHG:** 679 endpoints and eight detected activity peaks. M2 R_A maximum difference versus the original cache is 1.5e-7; M3 R_A 4.8e-6. M3 response support differs in 100 windows because the within-plane directional basis is sensitive to the coordinate-dependent optimizer; the plane and ratio invariants remain close. Original and recomputed support are both retained.
- **EEG:** 201 endpoints; maximum raw-column discrepancy is 6.9e-7. The original 2996–3036 s annotation is used. The illustrative plot's R_A and R_sigma axes are 0–3; five in-view R_sigma values above 3 are marked while all values remain in CSV.
- **FoG:** event 28 begins at 1316.422 s and ends at 1326.641 s. Section S16's 15 s pre-onset and full-interval endpoint sets yield M3 median R_A 0.543/1.370, R_sigma 0.755/1.694; M2 median R_A 0.579/1.368, R_sigma 0.801/1.710. The archived eight-window matched-baseline/first-four-second contrast gives different numbers; see its README.
- **Push-up:** the local revision `pushup_subset_report.txt`, Figure S21 and Table S18 all specify 1,913/338 class pairs, 5 s circular blocks and 2,000 replicates. All 12 recomputed point-estimate rows match the archived table to <6e-14; bootstrap summary rows match to <3.4e-12. The raw M2/M3 shift correlations are 0.312622/0.321402, each with 1,000 shifts and upper-tail p=0.000999, matching the archived timing table. The 441-window normal-reference control and phase-adaptive variants are archived but were not rerun for this representative preparation.

The numerical CSVs have not been rounded or altered to match a plot. `empirical/FILE_SHA256.csv` inventories the empirical files. The 54 recorded empirical numerical, raw-data and figure checksums remained unchanged during comment-only cleaning.

## Limits and release gate

The synthetic N04 R28 aggregate-count tables are transcriptions without recovered fit-level sources. Figures 1 and A.1 have not had exact source-to-PDF equivalence checked. The EHG M3 direction-dependent support is environment-sensitive as described above. The four empirical examples are representative analyses. Compact EHG and seizure EEG cohort archives now supply final summary tables, but the full raw datasets and some intermediate final window tables remain outside the package. Public recordings carry the original dataset attribution terms in [DATA_LICENSES.md](empirical/DATA_LICENSES.md). Didier authorized publication of the push-up recording; no specific data license has been assigned.

## Main empirical cohorts

The [EHG cohort guide](empirical/01_uterine_ehg/cohort/REPRODUCIBILITY.md) covers Fig. 10(a), Figs. S13–S14 and Table S17; the [seizure EEG cohort guide](empirical/02_seizure_eeg/cohort/REPRODUCIBILITY.md) covers Fig. 10(b), Figs. S17–S18 and Table S16. From each cohort directory, `python scripts/verify_archived.py` verifies archived manuscript checkpoints without processing raw recordings. Each guide separately describes the original full-data scripts, required input layout, aggregation, licensing, and missing intermediate data. The EHG association does not identify a causal mechanism; the seizure cohort does not show a consistent onset increase or establish an early-warning indicator.

## Public distribution status

The complete push-up example, including its original recording and derived results, is included with Didier's publication authorization. No specific reuse license has been assigned. The other empirical recordings retain their source-specific attribution and terms in `empirical/DATA_LICENSES.md`. Archived-table verification is distinct from end-to-end numerical reproduction: the EHG common-gate window cache and final EEG event-level normalized ratio table are not bundled, and the EEG final pooled aggregation source was not recovered. The separate [general-purpose `nndr-toolkit`](https://github.com/Saiprasad-V-R/nndr-toolkit) is MIT-licensed; no software license has yet been approved for this reproduction repository. Add a repository license only after the rights holders choose one.
