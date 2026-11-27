# T11 / Figure N5 — spectral-gap and sample-efficiency validation

Run `python run.py` from any directory (or run the script directly in Spyder).
It uses the shared rate-based R_A, avoids forced stabilization and preserves all
inadmissible reduced-log statuses. The same operator orientation and white innovations are paired across
competing-geometry strengths; recordings are nested across M and T regimes for *paired* sample-efficiency comparisons.

**Scientific estimand:** M3 extreme commutator-eigenspace relative to the
independently planted leading 2D block, with a deliberately controlled gap to
another non-normal amplifier. M2 results are recorded separately; the M3
Davis-Kahan certification is **not** claimed for M2.

The previously regenerated submitted Fig. 6 from T01 is not rerun or overwritten:
its M/N/T, N50--400, M1 and T3/T4 data are audited in a separate file. Its M3
error compares to M2's known-operator reference (not M3 eigenvectors) and thus
cannot be treated as a direct check of the commutator's Davis-Kahan theorem.
