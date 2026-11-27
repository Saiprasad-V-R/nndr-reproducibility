# T06 — Negative controls and false-positive audit

**Numerical scope:** 18 control settings × 24 independent synthetic subjects = 432 paired baseline/event subjects. Per subject N=12, M=6 independent trajectories, time horizon 180, except explicit W=80 oscillator settings; ridge=0.001 × trace(XᵀX)/N; genuine M2 multistart/M3 inference. 1,728 method-phase results (864 event rows). Event and baseline share innovations and the M2 initialization seed, but different subjects have independent random seeds. All selections are fixed in `config.json`.

**Ground truth:** Generating VAR(1) operators are normal strict contractions for Gaussian, bursts, coloured forcing, observation noise, switches, nonlinear saturation, and Stuart–Landau below Hopf/normal oscillators. Switching operators are normal at each time and products cannot amplify in spectral norm. For VAR(2) both A and B are individually normal, but the augmented-order dynamics need not be normal: these are specifically misspecification controls, NOT proven nonamplifying augmented systems.

**Diagnostics:** `R_A` is the continuous-time delayed-variance criterion; `R_sigma` is the exact discrete-time one-step criterion. Inadmissible reduced principal logarithms remain missing, not forced to zero. Per-subject event exceedance is reported against the full negative-generating cohort of 24 and a 95% Wilson binomial CI (inadmissible cases counted in denominator and separately flagged). Paired cohort tests use a one-sided 16,384-draw random sign permutation of mean event-minus-baseline differences on fully admissible subject pairs; BH correction separately within criterion. An observed threshold exceedance proportion is not a hypothesis test type-I level.

## Main results

- **Raw event threshold:** R_A>1 in **43/864** cases (4.98%), R_sigma>1 in **59/864** cases (6.83%). Of all phases, **3/1728** R_A cases are inadmissible and **0** R_sigma are inadmissible.
- For *spatially correlated coloured forcing*, both M2 and M3 falsely exceed R_A>1 in **21/24** event subjects; their one-step R_sigma>1 rates are **24/24**.
- With coarse sampling (nominal beta=20, Δt=4, W=180), M3 yields **1/24** false R_A exceedance. Several short-window normal-oscillator variants have R_sigma>1 due to finite-data fitting.
- For the remaining 16 settings there are no event R_A>1 detections in this particular parameter grid. **0/24 is not zero risk**: the Wilson 95% upper limit is 13.8%.

## Exact stationary population analysis (independent of estimated sample F)

The coloured-noise augmented state is (x_t,ξ_{t−1}); with ξ_t=Φξ_{t−1}+Bε_t, its transition is [[F,Φ],[0,Φ]] and shocks [B;B]ε. Solving its stationary discrete Lyapunov equation gives the exact effective VAR(1) predictor F_eff=(F C_xx+Φ C_{ξx}) C_xx⁻¹. The original **physical F is symmetric/normal**, but the effective F_eff for anisotropic spatially correlated coloured forcing need not be. This non-normality is a *population model-misspecification effect*, not merely finite-sample noise. Analytic stationary effective maps are also computed for observation error and VAR(2), and exact generating F is retained. Independently fitted 250-trajectory windowed estimates are **finite-window proxies, not exact population limits**.

- Spatially coloured event, M2: exact effective R_A=1.307, R_sigma=2.080; independent large-ensemble R_A proxy=1.273.
- Spatially coloured event, M3: exact effective R_A=1.311, R_sigma=2.089; independent large-ensemble R_A proxy=1.274.

## Spurious cohort event contrasts

BH-significant positive R_A mean contrasts occur for both M2 and M3 in each of four settings (8 tests total among 36 R_A method–setting comparisons):

| Family | Median ΔR_A M2 | Median ΔR_A M3 | Event threshold exceedances |
|---|---:|---:|---:|
| isotropic bursts | 0.095 | 0.110 | 0/48 |
| common mode bursts | 0.149 | 0.159 | 0/48 |
| coloured spatial | 0.684 | 0.697 | 42/48 |
| observation noise | 0.332 | 0.307 | 0/48 |

**Crucial distinction:** A statistically significant positive event-minus-baseline shift can occur *even if all event ratios remain below unity*. An observed contrast does not imply non-normal physical dynamics. Inadmissible pairs are excluded only from contrast hypothesis tests with explicit pair counts; they remain in raw denominators.

## Preliminary screening — NOT the calibrated T08 five-test gate

Two exploratory checks are computed on residuals: maximum same-coordinate lag-1 residual correlation compared to 3.5/√[M(W−1)], and ratio of maximum/minimum residual variance across temporal thirds compared to 1.8. These are descriptive, **not outcome-blind calibrated decision thresholds**; T08 is still pending. Screening rejects all event recordings for the four burst/coloured-noise families; thus their *post-screen* threshold proportion is **undefined (zero passers)**, not 0%. It retains the single coarse-oscillator M3 false R_A event. Raw and screened denominators are retained separately; screen sensitivity and specificity remain unestablished.

## Interpretation and limits

1. Normal driving with large amplitude alone did not produce R_A>1 for tested bursts, but its apparent event contrast did shift significantly.
2. Spatially correlated coloured innovations created substantial R_A and R_sigma false alarms, also in an exact stationary population projection: a serious model-adequacy issue rather than proof of latent non-normality.
3. Some coarse-sampled normal oscillators give finite-data false alarms; a principal logarithm does not solve aliasing.
4. Tested switching and VAR(2) conditions did not exceed threshold; these negative findings apply only to the specified contraction/variance parameters, not universally.
5. Future T07 must quantify positive-control power/ROC and matched normal nulls. T08 must independently calibrate all five adequacy gates, including screen-retention and rejection frequencies. This T06 result is a **completed numerical audit**, not a fully validated final Fig. N3.

**Reproduction:** `python 06_N3_negative_controls/run.py` (no arguments; also runs in Spyder). Detailed cases, exact references, excluded fits, bins and paired p-values are in the named output CSVs. Use single-threaded BLAS for numerical consistency. Only synthetic data are used.
