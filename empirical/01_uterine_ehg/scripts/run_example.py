from pathlib import Path
import json
import pandas as pd
import ehg_final_v2_full_cohort as ehg

ROOT = Path(__file__).resolve().parents[1]
STEM = "ice039_p_3of6"

ehg.DATA_DIR = ROOT / "data"
ehg.OUTDIR = ROOT / "results"
ehg.FORCE_RECOMPUTE = True
windows, metadata = ehg.save_record(STEM, representative=False)
waveform = ehg.read_ehg(STEM, representative=False)
output = ROOT / "results"
pd.DataFrame({"time_sec": waveform["time"], "activity_envelope_uV": waveform["env"]}).to_csv(output / "activity_envelope.csv", index=False)
ehg.peak_times(windows).to_csv(output / "detected_activity_peaks.csv", index=False)
(output / "single_record_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
geometry = []
X = waveform["X"]
win = int(round(ehg.WIN_SEC * ehg.FS_ANALYSIS))
step = int(round(ehg.STEP_SEC * ehg.FS_ANALYSIS))
for wi, start in enumerate(range(0, len(X)-win+1, step)):
    block = X[start:start+win]
    F, fit = ehg.fit_map(block)
    if F is None:
        continue
    for method in ("M2", "M3"):
        Q, _ = (ehg.m2_plane(F, ehg.SEED+wi) if method == "M2"
                else ehg.m3_plane(F, ehg.SEED+wi+100000))
        if Q is None:
            continue
        response = block @ Q[:, 0]
        meanfield = block.mean(axis=1)
        r2 = float(pd.Series(response).corr(pd.Series(meanfield)) ** 2)
        geometry.append({"method": method, "time_end_sec": float(windows.loc[(windows.method == method) & (windows.time_start_sec == waveform["time"][start]), "time_end_sec"].iloc[0]), "meanfield_R2": r2})
pd.DataFrame(geometry).to_csv(output / "meanfield_reconstruction.csv", index=False)
print(f"{STEM}: {metadata['n_rolling_windows']} windows, {len(windows)} method rows")
