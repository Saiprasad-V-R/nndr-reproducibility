from pathlib import Path
import numpy as np
import pandas as pd
import fig15_pushup_final_defensible as core

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "raw_samples_Pushup_duo_27Dec2025..csv"
t, signal, fs, dt = core.read_pushup_csv(source)
variants, aux = core.build_variants(t, signal, fs)
envelope, high, threshold = core.build_common_activity_labels(aux["X_raw_demeaned"], fs)
X = variants["raw_demeaned"]["X"]
pd.DataFrame({"time_sec": t, "acceleration_x_demeaned": X[:, 0], "activity_envelope": envelope}).to_csv(ROOT / "results" / "activity_timeseries.csv", index=False)
rows = []
summary = []
rng = np.random.default_rng(core.SEED + 9001)
min_shift = max(1, int(round(core.MIN_SHIFT_SEC * fs)))
for method in ("M2", "M3"):
    result = core.infer_global_plane(X, method)
    reaction = np.abs(X @ result["r"])
    observed = core.corr1d(envelope, reaction)
    allowed = np.arange(min_shift, len(reaction) - min_shift + 1, dtype=int)
    values = []
    for index in range(core.N_SHIFT_NULL):
        shift = int(rng.choice(allowed))
        value = core.corr1d(envelope, np.roll(reaction, shift))
        rows.append({"method": method, "replicate": index, "shift_samples": shift, "correlation": value})
        values.append(value)
    values = np.asarray(values)
    summary.append({"method": method, "observed_correlation": observed, "p_upper": (1 + np.sum(values >= observed)) / (1 + len(values)), "n_shifts": len(values)})
    pd.DataFrame({"time_sec": t, "response_coordinate_abs": reaction}).to_csv(ROOT / "results" / f"response_{method}.csv", index=False)
pd.DataFrame(rows).to_csv(ROOT / "results" / "circular_shift_correlations.csv", index=False)
pd.DataFrame(summary).to_csv(ROOT / "results" / "circular_shift_summary.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
