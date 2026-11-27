# T01 — Revised original Figure 6: Scaling of plane and rate-based R inference

**Status:** full prescribed scan executed (one stochastic realization per setting), with eight dedicated regression tests passed.

## Design

- Synthetic family: same Eq. (44) similarity construction, ρ(F)=0.95, Gaussian innovation σ=0.05.
- Axes exactly as submitted: M = {1,2,5,10,15,25,50,100}; N = {50,100,150,200,250,300,350,400}; T_train = {3,4,10,25,50,100,200,400,800}.
- Regimes **recalibrated** to reference R≈{0.2,1,10} using the corrected continuous-time rates; this changes κ and hence the physical operators relative to the original Fig. 6.
- The reference plane is M2 on the known operator. Unlike independent planted-plane experiments, this validates estimation accuracy only.
- Relative trace-normalized ridge, chronological validation inside each trajectory, 8-start M2 fit and 14-start M2 population reference, no rescaling or spectral clipping.
- Corrected **informative** state pair count M×(T_train−1) owing to initial x0=0; the zero-input transition still appears in the regression.

## Representative numerical results

| Scan / regime | M2 largest angle | M3 largest angle |
|---|---|---|
| N=400, M=25, T=200, R≈0.2 | 89.83° (R̂=0.478, relErr=0.432) | 81.99° (R̂=0.426, relErr=0.432) |
| N=400, M=25, T=200, R≈1 | 21.97° (R̂=0.905, relErr=0.431) | 20.42° (R̂=0.884, relErr=0.431) |
| N=400, M=25, T=200, R≈10 | 2.22° (R̂=8.901, relErr=0.367) | 2.21° (R̂=8.890, relErr=0.367) |
| M=100, N=100, T=200, R≈1 | 4.29° (R̂=0.975, relErr=0.099) | 5.39° (R̂=0.956, relErr=0.099) |
| T=800, M=25, N=100, R≈1 | 4.31° (R̂=0.974, relErr=0.098) | 5.39° (R̂=0.955, relErr=0.098) |

## Explicit missing and underdetermined results

- Total method-results: **184**, valid rate-based R: **146**; invalid: **38**.
- M: 43/48 valid. Reason(s): {'negative_real_multiplier_principal_log_not_real': 5, 'ok': 43}.
- N: 48/48 valid. Reason(s): {'ok': 48}.
- T: 45/54 valid. Reason(s): {'negative_real_multiplier_principal_log_not_real': 9, 'ok': 45}.
- N_M1: 3/16 valid. Reason(s): {'negative_real_multiplier_principal_log_not_real': 13, 'ok': 3}.
- T_M1: 7/18 valid. Reason(s): {'negative_real_multiplier_principal_log_not_real': 11, 'ok': 7}.
- The invalid cases were assigned NaN, not clipped or declared successful. Many single-trajectory fits produce negative real reduced multipliers; the chosen principal-logarithm, real-generator convention cannot assign R.
- At N=400 and M=1, only 199 informative sample pairs are available for a 400-dimensional operator. Ridge yields a fit but does not restore full unregularized identifiability.

## Scientific interpretation and limits

- **Strong non-normality** (R≈10) produces accurate plane recovery even when the 400×400 operator is not estimated accurately in relative Frobenius norm.
- **Intermediate regime** (R≈1) has errors of order 20° at N=400, so claims of negligible plane error across all dimensions would be incorrect.
- **Weak non-normality** (R≈0.2) has near-orthogonal estimated/reference planes at N=400 and should not be described as identified.
- In the time-horizon and M scans, the accuracy of the strongly non-normal plane improves with more data; results from one synthetic trajectory per setting do not quantify uncertainty or repeatability.
- Sample-efficiency should be explained through perturbation norm and commutator eigengap (Davis–Kahan), **not** by claiming that N² independent state samples are needed for identification. The perturbation bound is explicitly recorded and can be vacuous.
- These results are not yet the full new Fig. N5 requested by Reviewer 3: N5 additionally needs the cross-run perturbation-vs-gap diagnostic and a more systematic uncertainty assessment.

## Reproduction

From the project root:
```bash
python 01_baseline_reproduction/fig06/run.py
pytest -q 01_baseline_reproduction/fig06/tests
```

Outputs: 184 rows in T01_fig06_full_results.csv, full target-kappa table, revised six-panel figure (PDF/PNG), source hashes and logs.
