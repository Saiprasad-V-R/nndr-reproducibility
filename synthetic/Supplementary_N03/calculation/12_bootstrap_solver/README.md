# T12 — Moving-block bootstrap and M2 optimizer robustness

Run from Spyder or a terminal without any command-line arguments:

```bash
python run.py
```

The script resolves paths from its own location. It regenerates all T12 CSV
records, both PDF/PNG figures, and a JSON run manifest in `outputs/`.
Use Python 3.11+ with numpy, scipy, pandas, matplotlib and pytest installed.
For reproducibility, set BLAS threads to one; `run.py` does so before imports.

## What is validated

* Six independently specified six-dimensional VAR(1) operator families: weak
  and strong real spectra, weak and strong non-normal complex conjugate pairs,
  a repeated-real-rate Jordan block, and almost competing commutator modes.
* Three data conditions: (M=1,T=90), (M=1,T=240), (M=8,T=240).
* Sixteen independently generated stationary subjects per family and sampling
  condition. Subject observations have a 300-step burn-in and a fixed physical
  innovation standard deviation of 0.05.
* 120 subject-specific, trajectory-stratified moving-block bootstrap draws
  per fitted data set. Moving blocks contain **paired consecutive transitions**
  and never reinterpret a concatenation boundary as a physical transition.
  Resampling includes trajectory-level cluster selection. The final block is
  shorter where necessary to give exactly M*T transitions per draw.
* For each draw, the ridge map is refitted and M2 is **reoptimized**.
  A fixed independently known planted-plane analysis provides a diagnostic
  oracle benchmark, not a deployed method.
* 90% and 95% percentile and bias-corrected percentile (BC) CIs for R_A,
  R_sigma and the largest reduced singular value. BC is **not BCa**: it does
  not estimate jackknife acceleration.
* If the estimate is inadmissible or fewer than 90% of bootstrap ratios are
  admissible, the interval is marked **nonestimable**. Missing intervals
  contribute zero to unconditional coverage; conditional-on-estimability
  coverage and numerator/denominator are given separately.
* Independent strong numerical reference for M2 based on multistart
  L-BFGS-B optimization of the equivalent single-vector problem. This is
  a numerical lower bound on the global objective, **not a certificate of
  global optimality**. Single-start objective and plane variability measured
  in separate solver experiments.

## Statistical caution

The bootstrap resamples observed transition pairs. Its effective sample size
is therefore limited by temporal dependence and available independent
trajectories. The supplied ESS is an autocorrelation *heuristic*, not an exact
number of independent data points. Coverage is a finite-design Monte Carlo
property, not a general guarantee, and is estimated from just 16 independent
subjects per individual family/sampling cell. Uncertainty for such coverage
estimates is correspondingly large; Wilson intervals are provided. Repeated
and near-tied eigenspaces are nonregular cases and can frustrate standard
percentile bootstrap inference.

All original T00–T11 results are retained without alteration in the V19
cumulative ZIP. The empirical workstream is separate.
