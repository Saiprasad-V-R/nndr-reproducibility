# Empirical NNDR reproduction resources

Each folder contains one original recording in `data/`, untouched archived tables in `reference_data/`, the original calculation modules and a single-recording entry point in `scripts/`, and newly recomputed CSVs and illustrative PDF/PNG plots in `results/`. Install Python 3.10+ dependencies with `python -m pip install -r requirements.txt`, then follow each folder's README. Scripts were run once on the supplied local recordings; no cohort calculation was run.

| Example | Manuscript location | Calculation entry point | Plot entry point | Main numerical check |
|---|---|---|---|---|
| `01_uterine_ehg` | Figure 9(a,c), S11 | `python scripts/run_example.py` | `python scripts/plot_example.py` | 679 endpoints; finite R_A >1.56; M3 support has a documented directional discrepancy |
| `02_seizure_eeg` | Figure 9(b,d) | `python scripts/run_example.py` | `python scripts/plot_example.py` | 201 endpoints; raw ratios/support reproduce archived columns to ≤6.9e-7 |
| `03_freezing_of_gait` | S19 | `python scripts/run_example.py` | `python scripts/plot_example.py` | 62 overlapping endpoints per method match archived R_A to <1e-13; manuscript medians reproduced with its 15 s/full-interval selection |
| `04_pushup` | S21 | `python scripts/pushup_subset_final_5s_bootstrap.py` then `python scripts/run_timing_example.py` | `python scripts/plot_example.py` | 1,913/338 pairs; R_A/R_sigma point estimates match archived values to <6e-14 |

The EHG and CHB-MIT recordings are available from PhysioNet under Open Data Commons Attribution 1.0: [EHGDB](https://physionet.org/content/ehgdb/1.0.0/) and [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/). The Daphnet S05R01 recording is available from [UCI](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) under CC BY 4.0. These public inputs are included locally for immediate execution; their checksums and sizes are in `FILE_SHA256.csv`. The user-supplied push-up raw recording has no documented public license and needs an explicit publication decision before upload. Dataset citations and exact execution details are in the example READMEs.

The original mathematical implementations are retained. R_A uses rate-based stable reduced dynamics and a real principal logarithm; R_sigma uses the discrete stable reduced map. Relative ridge, isotropic trace-preserving shrinkage, M2/M3, stability and admissibility gates follow each experiment's source. EHG and FoG use normalized L1 response support; EEG uses entropy support. Multiplier splitting remains separate from rate splitting. The plots are illustrations, not copies of Virgile's publication layout. The EEG R_A and R_sigma display ranges are 0–3, with out-of-range R_sigma points marked; stored values remain unchanged.

The November 27 clean revised manuscript was located in the local Downloads revision package and consulted directly. The FoG manuscript medians reproduce once its endpoint selection is used; the archived eight-window/first-four-second contrast is a different summary. EHG M3 direction-dependent support is not identical in 100 windows, as documented in `01_uterine_ehg/EHG_DIRECTION_AUDIT.md`. No general mechanism is inferred from the selected FoG or push-up examples.

## Cohort archives

- [Uterine EHG cohort](01_uterine_ehg/cohort/REPRODUCIBILITY.md): compact 123-recording/45-subject tables, corrected 44-subject contrasts, contraction-marker controls and archived calculation scripts.
- [Seizure EEG cohort](02_seizure_eeg/cohort/REPRODUCIBILITY.md): final 182-event pooled curves, screening/count tables, 10-versus-40-second matched sensitivity tables and archived calculation scripts.

Run `python scripts/verify_archived.py` inside either cohort directory for read-only numerical checkpoints. The EHG common-gate full window cache and final EEG event-level normalized ratio table are not bundled; consult the cohort guides before claiming raw-to-manuscript reproduction. The representative examples above remain unchanged.
