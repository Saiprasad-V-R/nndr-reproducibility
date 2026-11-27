# Scientific audit: RevFig01

**Source:** V19 `02_N1_comparators` (`T02_group_summary.csv`, `T02_angle_bootstrap_intervals.csv`, and `T02_snapshot_horizons.csv`). No new simulations, interpolation, or smoothing.

## Panel mapping

| Panel | Numerical population | Quantity | Takeaway |
|:--|:--|:--|:--|
| (a) | Dominant embedded block; 8 seeds per method and M; M=1 and M=18 | Median maximum principal angle to independently planted `Q_*` | M2, M3 and SVD improve with more trajectories and agree closely. |
| (b) | Stronger competing non-normal background; 8 seeds per method and M | Same as (a) | The extracted planes can be far from the designated planted plane even with more data. |
| (c) | Three planted families; direct lagged-snapshot regressions; M=18, 8 seeds per horizon and family | Median planted-plane angle vs horizon T | Horizon selection changes which amplification geometry is recovered. |
| (d) | Dominant embedded block; M=18, 8 seeds per method | Median one-step singular-gain capture | A method can capture large gain without having a unique advantage in planted-plane error. |

## Reference and statistics

- `Q_*` is the independently planted plane, **not** the plane recovered by M2 using the generating matrix.
- The angle is the **largest principal angle** of two orthonormal two-dimensional planes, in degrees.
- The open/filled markers show **medians** for M=1 and M=18. Only M=18 gets displayed **95% seed-bootstrap confidence interval whiskers**; each interval is constructed from 8 independent seeds, hence uncertainty is exploratory.
- Panel (c) uses **M=18 only**. The previous bar-figure script mixed M=1 and M=18 while aggregating by T, which was not a scientifically matched horizon comparison.
- Panel (d) uses `median_relative_growth_capture` from the frozen T02 output: `sigma_max(Q.T @ F @ Q) / sigma_max(F)`. The line at 1 marks full one-step gain capture, not an NNDR threshold.
- The stronger competing block can legitimately dominate total amplification even when its plane is not the planted target. An angle near 90° is a mismatch to the target, not necessarily a failure to find any amplifier.
- The direct lagged-snapshot comparator is *not* a full implementation of the specialized published regularizers requested by the referees. This remains a limitation.
- The fixed-spectrum Eq. 44 family has **no independent planted `Q_*`** and is properly excluded from planted-plane error panels.

## Source values checked

Dominant block, M=18: M2 median 1.77°, M3 1.77°, SVD 1.77°. Competing block, M=18: M2 89.77°, M3 89.87°, SVD 89.79°. Source seeds: 8 per family/method/M.
