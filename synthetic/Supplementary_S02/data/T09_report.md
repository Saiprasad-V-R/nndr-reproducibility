# T09 — nonlinear dynamics with known Jacobians: numerical audit

## What is independently known

- Four 8D physical systems; 8 independent subjects per family; 10 estimation trajectories/subject; 64 independent reference trajectories per family.
- Euler–Maruyama integration step h=0.01, observation Δt=0.2; W=60, stride=20, T=300.
- Brownian forcing: independent state-coordinate shocks with standard deviation `noise_sd*sqrt(h)` per numerical microstep. Physical coupling is directed in FHN and Stuart–Landau; the double well has symmetric coupling and is a gradient SDE.
- The window mean of *ordered one-sample variational maps* is the Jacobian/tangent reference. The map is differentiated from the Euler–Maruyama scheme, including all time-ordering.
- The independent ensemble statistical predictor uses centered C1*C0^(-1) from entirely separate replicated physical trajectories in matching time windows. These references differ, especially for nonstationary forcing; this is not an estimation failure of the method per se.
- No real rates ratio is assigned to an unstable or negative-multiplier compression. `R_sigma` and full-system gain are reported separately. Above-Hopf time windows may have non-stable local tangent compressions.
- The five-test T08 gate is applied unchanged (no threshold fitting to T09 outcomes), but its numerical calibration was not trained for W=60 8D nonlinear SDEs; all gate rates are exploratory transfer diagnostics, NOT calibrated type-I errors.

## Numerical derivative checks

| family                 |   instantaneous_J_max_abs_error |   sample_variational_max_abs_error |   gradient_J_asymmetry |
|:-----------------------|--------------------------------:|-----------------------------------:|-----------------------:|
| FHN_directed_spikes    |                         2.7e-11 |                            2.3e-10 |                    1.1 |
| SL_directed_below_Hopf |                         1.4e-11 |                            4.5e-11 |                    2.2 |
| SL_directed_above_Hopf |                         9.2e-12 |                            3.5e-11 |                    2.2 |
| double_well_gradient   |                         6.9e-11 |                            1.1e-10 |                    0   |

## All-window status and reference discrepancies

| family                 |   windows |   true_gain |   tangent_R_valid |   fitted_M2_R_valid |   fitted_M2_R_gt1 |   gate_reject |   rel_stat_diff |   rel_naive_diff |
|:-----------------------|----------:|------------:|------------------:|--------------------:|------------------:|--------------:|----------------:|-----------------:|
| FHN_directed_spikes    |       104 |         104 |               104 |                  60 |                24 |           104 |          0.123  |          0.00154 |
| SL_directed_above_Hopf |       104 |           6 |               104 |                 104 |                 0 |             4 |          0.0183 |          0.00172 |
| SL_directed_below_Hopf |       104 |           0 |               104 |                 104 |                 0 |             6 |          0.013  |          0.00129 |
| double_well_gradient   |       104 |           0 |               104 |                  97 |                 1 |            44 |          0.311  |          0.00528 |

## Median largest principal-angle error (degrees) vs known tangent M2 plane

| family                 |    M2 |    M3 |
|:-----------------------|------:|------:|
| FHN_directed_spikes    | 77.27 | 78.19 |
| SL_directed_above_Hopf | 40.12 | 67.18 |
| SL_directed_below_Hopf | 50.50 | 64.12 |
| double_well_gradient   | 72.51 | 76.52 |

## Spurious fitted R_A > 1 versus known tangent subthreshold and adequacy gate

| family                 | condition            | method   |   windows |   reference_RA_le1 |   inferred_RA_valid |   inferred_RA_gt1 |   spurious_RA_gt1_on_known_subthreshold |   spurious_caught_by_gate |   spurious_accepted_by_gate |   gate_rejected |
|:-----------------------|:---------------------|:---------|----------:|-------------------:|--------------------:|------------------:|----------------------------------------:|--------------------------:|----------------------------:|----------------:|
| FHN_directed_spikes    | before_pulse         | M2       |        24 |                  0 |                  24 |                18 |                                       0 |                         0 |                           0 |              24 |
| FHN_directed_spikes    | before_pulse         | M3       |        24 |                  0 |                  24 |                18 |                                       0 |                         0 |                           0 |              24 |
| FHN_directed_spikes    | overlaps_pulse       | M2       |        40 |                 32 |                   0 |                 0 |                                       0 |                         0 |                           0 |              40 |
| FHN_directed_spikes    | overlaps_pulse       | M3       |        40 |                 32 |                   1 |                 1 |                                       1 |                         1 |                           0 |              40 |
| FHN_directed_spikes    | after_pulse          | M2       |        40 |                 40 |                  36 |                 6 |                                       6 |                         6 |                           0 |              40 |
| FHN_directed_spikes    | after_pulse          | M3       |        40 |                 40 |                  37 |                 7 |                                       7 |                         7 |                           0 |              40 |
| SL_directed_below_Hopf | below_Hopf           | M2       |       104 |                104 |                 104 |                 0 |                                       0 |                         0 |                           0 |               6 |
| SL_directed_below_Hopf | below_Hopf           | M3       |       104 |                104 |                 104 |                 0 |                                       0 |                         0 |                           0 |               6 |
| SL_directed_above_Hopf | above_Hopf           | M2       |       104 |                104 |                 104 |                 0 |                                       0 |                         0 |                           0 |               4 |
| SL_directed_above_Hopf | above_Hopf           | M3       |       104 |                104 |                 104 |                 0 |                                       0 |                         0 |                           0 |               4 |
| double_well_gradient   | gradient_equilibrium | M2       |       104 |                104 |                  97 |                 1 |                                       1 |                         0 |                           1 |              44 |
| double_well_gradient   | gradient_equilibrium | M3       |       104 |                104 |                 104 |                 0 |                                       0 |                         0 |                           0 |              44 |

The spurious-rate denominator is strictly windows with known tangent rates R_A valid and <=1; unknown or unstable local tangent references are NOT silently treated as negatives. The gradient false-positive above threshold is *accepted*, not caught, by the transferred gate.

## Subject-level uncertainty

T09_subject_cluster_CI.csv contains 95% percentile intervals from resampling the eight independent subjects within each family/condition, preserving all overlapping windows of each subject. The window-level counts are not independent Bernoulli trials.

## Interpretation and key limitations

- Gradient SDE: instantaneous Jacobians are symmetric by construction. Products of noncommuting symmetric instantaneous updates and/or C1*C0^(-1) can be nonnormal; do NOT claim K=0 for a statistical linearization or for finite-time products.
- Report the fraction of windows with valid fitted and true reduced R separately from the fraction above threshold. A fitted high-R window is not necessarily evidence of a high-R local Jacobian in these nonlinear/nonstationary systems.
- FHN before/overlapping/after conditions are defined by actual complete window support, not centers; windows intersecting pulse time [23,30] are classified as overlapping.
- The almost-normal Stuart–Landau and gradient tangent maps can have weakly identified gain planes; a large principal angle alone does not imply a large error in physically meaningful gain.
- The gradient reference is the local tangent operator; its true full gain stays below one in all sampled windows. Despite this, one M2 compressed fit exceeds the corrected threshold, and the transferred adequacy gate does not flag that case. This is an adverse control result.
- State-coordinate plane estimates are not planted geometric invariants; the independent reference here is a known-physics variational-map optimizer, not a unique invariant nonlinear manifold.
- Different RNG seeds for independent statistical reference and estimation; no reference labels enter fitting or the T08 adequacy cutoff selection.

## Next work

T10: moving-window independent time-varying planted-plane control; T11: gap-dependent sample efficiency; T12: dependence-preserving bootstrap and solver robustness. Empirical analyses remain separate.
