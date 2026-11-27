# T03 — Fig N1c, rates-based lag/geometry/spectrum benchmark

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py` (no arguments).

Exact two-dimensional continuous generator tests versus embedded N=40 analogues, with varying spectra at fixed eigenvectors and varying eigenvector geometry at fixed spectra. For each lag tau=1,2,4,8 compare population M2(F^tau) with direct lagged-covariance M2(C_tau C0^-1) inferred from M=1 and M=25 noisy trajectories. The complex-spectrum control records a known true frequency and the alias of its principal logarithm at tau=8. Outputs report every inadmissible reduced rate instead of clipping.
