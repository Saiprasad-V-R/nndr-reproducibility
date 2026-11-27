# T10 independent time-varying planted-plane benchmark

## Methods

N=10, T=400, window=80, stride=40, independent subjects=6, nested M=[1, 12].
True plane is Q*(t)=[cos(theta)e1+sin(theta)e3, cos(theta)e2+sin(theta)e4],
with exact orthogonal complement, true 2x2 upper triangular block, and a normal diagonal background.
Instantaneous R_A is evaluated on the planted B(t), independently of M2/M3.
The center reference uses B(t_center); the window-geometry reference is Q*(t_center)^T mean(F_t) Q*(t_center).
The latter is NOT the population statistical least-squares window coefficient. No equivalence is assumed.
All time-varying models are stable in eigenvalues, but switching products can amplify transiently.
Noise jump varies innovation standard deviation alone, keeping the physical operator fixed.
Fitting uses ridge normal equations to all window pairs pooled across M trajectories.
For M=1 only 80 pairs identify 10 columns; numerical conditioning remains a limitation.
M2 is multistart approximate optimization, not certified global; M3 uses extremal commutator eigenvectors.

Rows: 1080; inadmissible RA: 85; crossings inferred only from adjacent valid windows.

## Summary by case, M and method

| case                |   M | method   |   observations |   valid |   inadmissible |   median_angle_mid_deg |   median_angle_instant_deg |   median_abs_R_err_center_valid |   median_abs_R_err_instant_valid |   detected_R_greater_1 |   truth_R_greater_1_center |   rel_F_error_median |   median_orientation_lag_steps |
|:--------------------|----:|:---------|---------------:|--------:|---------------:|-----------------------:|---------------------------:|--------------------------------:|---------------------------------:|-----------------------:|---------------------------:|---------------------:|-------------------------------:|
| stationary          |   1 | M2       |             54 |      40 |             14 |                 35.322 |                     35.322 |                           0.375 |                            0.375 |                     38 |                         54 |                0.675 |                        nan     |
| stationary          |   1 | M3       |             54 |      51 |              3 |                 27.654 |                     27.654 |                           0.331 |                            0.331 |                     48 |                         54 |                0.675 |                        nan     |
| stationary          |  12 | M2       |             54 |      54 |              0 |                  9.359 |                      9.359 |                           0.105 |                            0.105 |                     54 |                         54 |                0.175 |                        nan     |
| stationary          |  12 | M3       |             54 |      54 |              0 |                  7.614 |                      7.614 |                           0.094 |                            0.094 |                     54 |                         54 |                0.175 |                        nan     |
| slow_rotation       |   1 | M2       |             54 |      37 |             17 |                 35.213 |                     35.588 |                           0.231 |                            0.231 |                     37 |                         54 |                0.692 |                        -26.158 |
| slow_rotation       |   1 | M3       |             54 |      52 |              2 |                 26.818 |                     26.754 |                           0.211 |                            0.211 |                     52 |                         54 |                0.692 |                        -15.797 |
| slow_rotation       |  12 | M2       |             54 |      54 |              0 |                  8.814 |                      9.503 |                           0.056 |                            0.056 |                     54 |                         54 |                0.172 |                        -12.918 |
| slow_rotation       |  12 | M3       |             54 |      54 |              0 |                  7.439 |                      8.419 |                           0.071 |                            0.071 |                     54 |                         54 |                0.172 |                         -9.670 |
| fast_rotation       |   1 | M2       |             54 |      43 |             11 |                 33.776 |                     37.366 |                           0.210 |                            0.210 |                     43 |                         54 |                0.689 |                          1.784 |
| fast_rotation       |   1 | M3       |             54 |      51 |              3 |                 27.532 |                     32.153 |                           0.229 |                            0.229 |                     49 |                         54 |                0.689 |                          1.203 |
| fast_rotation       |  12 | M2       |             54 |      54 |              0 |                  8.745 |                     17.399 |                           0.061 |                            0.061 |                     54 |                         54 |                0.221 |                          0.051 |
| fast_rotation       |  12 | M3       |             54 |      54 |              0 |                  7.285 |                     16.778 |                           0.059 |                            0.059 |                     54 |                         54 |                0.221 |                         -0.099 |
| threshold_crossing  |   1 | M2       |             54 |      45 |              9 |                 37.530 |                     37.530 |                           0.298 |                            0.384 |                     36 |                         42 |                0.684 |                        nan     |
| threshold_crossing  |   1 | M3       |             54 |      50 |              4 |                 29.289 |                     29.289 |                           0.227 |                            0.394 |                     31 |                         42 |                0.684 |                        nan     |
| threshold_crossing  |  12 | M2       |             54 |      54 |              0 |                 10.597 |                     10.597 |                           0.093 |                            0.231 |                     40 |                         42 |                0.180 |                        nan     |
| threshold_crossing  |  12 | M3       |             54 |      54 |              0 |                  8.477 |                      8.477 |                           0.070 |                            0.218 |                     40 |                         42 |                0.180 |                        nan     |
| noise_variance_jump |   1 | M2       |             54 |      36 |             18 |                 34.829 |                     34.829 |                           0.150 |                            0.150 |                     35 |                         54 |                0.689 |                        nan     |
| noise_variance_jump |   1 | M3       |             54 |      50 |              4 |                 28.124 |                     28.124 |                           0.301 |                            0.301 |                     48 |                         54 |                0.689 |                        nan     |
| noise_variance_jump |  12 | M2       |             54 |      54 |              0 |                  8.161 |                      8.161 |                           0.044 |                            0.044 |                     54 |                         54 |                0.178 |                        nan     |
| noise_variance_jump |  12 | M3       |             54 |      54 |              0 |                  6.992 |                      6.992 |                           0.045 |                            0.045 |                     54 |                         54 |                0.178 |                        nan     |

## Individual crossing timing and misses

|   subject |   M | method   |   true_center_crossing |   estimated_center_crossing |   timing_error | crossing_recovered   |   n_invalid |
|----------:|----:|:---------|-----------------------:|----------------------------:|---------------:|:---------------------|------------:|
|         0 |   1 | M2       |                117.780 |                      78.302 |        -39.479 | True                 |           0 |
|         1 |   1 | M2       |                117.780 |                     nan     |        nan     | False                |           1 |
|         2 |   1 | M2       |                117.780 |                      66.782 |        -50.998 | True                 |           2 |
|         3 |   1 | M2       |                117.780 |                      95.696 |        -22.084 | True                 |           1 |
|         4 |   1 | M2       |                117.780 |                     nan     |        nan     | False                |           2 |
|         5 |   1 | M2       |                117.780 |                     nan     |        nan     | False                |           3 |
|         0 |   1 | M3       |                117.780 |                     162.162 |         44.382 | True                 |           0 |
|         1 |   1 | M3       |                117.780 |                      97.841 |        -19.940 | True                 |           0 |
|         2 |   1 | M3       |                117.780 |                     nan     |        nan     | False                |           1 |
|         3 |   1 | M3       |                117.780 |                     194.726 |         76.946 | True                 |           1 |
|         4 |   1 | M3       |                117.780 |                     172.996 |         55.215 | True                 |           1 |
|         5 |   1 | M3       |                117.780 |                     159.237 |         41.456 | True                 |           1 |
|         0 |  12 | M2       |                117.780 |                      95.281 |        -22.500 | True                 |           0 |
|         1 |  12 | M2       |                117.780 |                     107.186 |        -10.595 | True                 |           0 |
|         2 |  12 | M2       |                117.780 |                     110.722 |         -7.058 | True                 |           0 |
|         3 |  12 | M2       |                117.780 |                     124.930 |          7.150 | True                 |           0 |
|         4 |  12 | M2       |                117.780 |                     121.893 |          4.113 | True                 |           0 |
|         5 |  12 | M2       |                117.780 |                     105.880 |        -11.900 | True                 |           0 |
|         0 |  12 | M3       |                117.780 |                      97.608 |        -20.172 | True                 |           0 |
|         1 |  12 | M3       |                117.780 |                     109.309 |         -8.471 | True                 |           0 |
|         2 |  12 | M3       |                117.780 |                     116.090 |         -1.691 | True                 |           0 |
|         3 |  12 | M3       |                117.780 |                     125.355 |          7.574 | True                 |           0 |
|         4 |  12 | M3       |                117.780 |                     121.364 |          3.584 | True                 |           0 |
|         5 |  12 | M3       |                117.780 |                     108.702 |         -9.079 | True                 |           0 |

## Subject-cluster bootstrap 95% intervals for plane-angle error

| case                |   M | method   |   subjects |   pooled_median_angle_deg |   cluster_bootstrap_95_lower |   cluster_bootstrap_95_upper |   bootstrap_draws |
|:--------------------|----:|:---------|-----------:|--------------------------:|-----------------------------:|-----------------------------:|------------------:|
| stationary          |   1 | M2       |          6 |                    35.322 |                       32.470 |                       38.896 |               400 |
| stationary          |   1 | M3       |          6 |                    27.654 |                       24.657 |                       30.651 |               400 |
| stationary          |  12 | M2       |          6 |                     9.359 |                        8.366 |                       10.287 |               400 |
| stationary          |  12 | M3       |          6 |                     7.614 |                        6.707 |                        8.172 |               400 |
| slow_rotation       |   1 | M2       |          6 |                    35.213 |                       30.823 |                       39.881 |               400 |
| slow_rotation       |   1 | M3       |          6 |                    26.818 |                       24.440 |                       28.547 |               400 |
| slow_rotation       |  12 | M2       |          6 |                     8.814 |                        8.104 |                       10.112 |               400 |
| slow_rotation       |  12 | M3       |          6 |                     7.439 |                        6.237 |                        8.009 |               400 |
| fast_rotation       |   1 | M2       |          6 |                    33.776 |                       33.089 |                       35.665 |               400 |
| fast_rotation       |   1 | M3       |          6 |                    27.532 |                       26.419 |                       29.576 |               400 |
| fast_rotation       |  12 | M2       |          6 |                     8.745 |                        8.427 |                       10.202 |               400 |
| fast_rotation       |  12 | M3       |          6 |                     7.285 |                        6.713 |                        7.919 |               400 |
| threshold_crossing  |   1 | M2       |          6 |                    37.530 |                       33.148 |                       39.671 |               400 |
| threshold_crossing  |   1 | M3       |          6 |                    29.289 |                       28.460 |                       31.571 |               400 |
| threshold_crossing  |  12 | M2       |          6 |                    10.597 |                        8.960 |                       13.570 |               400 |
| threshold_crossing  |  12 | M3       |          6 |                     8.477 |                        6.480 |                       10.730 |               400 |
| noise_variance_jump |   1 | M2       |          6 |                    34.829 |                       32.435 |                       40.307 |               400 |
| noise_variance_jump |   1 | M3       |          6 |                    28.124 |                       25.751 |                       30.389 |               400 |
| noise_variance_jump |  12 | M2       |          6 |                     8.161 |                        7.792 |                        9.136 |               400 |
| noise_variance_jump |  12 | M3       |          6 |                     6.992 |                        6.275 |                        7.654 |               400 |

## Interpretation and limitations

A window estimate cannot recover an instantaneous operator without temporal-mixing bias, even with infinitely many trajectories.
No R_A value is imputed for an invalid principal logarithm or unstable reduced spectrum.
A crossing is reported as missing, rather than scoring zero timing error, if no two adjacent valid window results bracket 1.
Within-subject window observations are overlapping and not independent; assess uncertainty at subject level.
This benchmark does not substitute for T11 gap perturbation or T12 bootstrap calibration.
