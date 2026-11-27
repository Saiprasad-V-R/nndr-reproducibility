# T02 — New Fig N1a, N1b and N1d, independent comparators

Run from any folder:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /path/to/NNDR_synthetic_revision_suite/02_N1_comparators/run.py
```

Regenerates results/plots. The primary accuracy benchmark uses the independently planted 2D block, not an M2-defined reference. The Eq44 family does **not** have such a planted plane; only method-relative descriptive comparison is included for that family. Uses 8 paired seeds, 4 families, 2 sample sizes and 6 direct snapshot horizons. Direct-lag snapshot fitting is an explicit matched-ridge baseline and is *not* asserted to reproduce the specialized methods of the two external snapshot publications.

See `outputs/T02_report.md` for interpretation and caveats; see `config.json` for all numerical settings.
