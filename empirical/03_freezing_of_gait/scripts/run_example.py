from pathlib import Path
import numpy as np
import pandas as pd
import fog_nndr_revised_v2 as fog

ROOT = Path(__file__).resolve().parents[1]
rec = fog.load_recording(ROOT / "data" / "S05R01.txt")
events = fog.get_events(rec["anno"])
event_no, onset_index, offset_index = events[27]
assert event_no == 28
step = int(round(fog.STEP_SEC * fog.NOMINAL_FS))
first = max(0, ((onset_index - int(30 * fog.NOMINAL_FS)) // step) * step)
last = min(len(rec["x"]), offset_index + int(20 * fog.NOMINAL_FS))
segment = dict(rec)
for key in ("time", "x", "anno", "bad_gap", "raw_magnitude"):
    segment[key] = rec[key][first:last]
windows = fog.windows_one_record(segment)
for col in ("i_start", "i_end"):
    windows[col] += first
windows.to_csv(ROOT / "results" / "window_diagnostics.csv", index=False)
t0 = rec["time"][onset_index]
t1 = rec["time"][offset_index-1] + 1/fog.NOMINAL_FS
summaries = []
for method in ("M2", "M3"):
    own = windows[windows.method == method]
    for name, mask in (("pre_15s", (own.t_end >= t0-15) & (own.t_end < t0)),
                       ("within_freeze", (own.t_end >= t0) & (own.t_end < t1))):
        selected = own[mask]
        summaries.append({"method": method, "period": name, "n_endpoints": len(selected),
                          "n_RA_admissible": int(selected.good_RA.sum()),
                          "n_R_sigma_admissible": int(selected.good_sigma.sum()),
                          "median_RA": selected.RA.where(selected.good_RA).median(),
                          "median_R_sigma": selected.R_sigma.where(selected.good_sigma).median()})
pd.DataFrame(summaries).to_csv(ROOT / "results" / "manuscript_event_summary.csv", index=False)
raw = pd.DataFrame({"time_sec": rec["time"][first:last], "acceleration_magnitude_mg": rec["raw_magnitude"][first:last], "annotation": rec["anno"][first:last]})
raw.to_csv(ROOT / "results" / "acceleration_magnitude.csv", index=False)
print(f"S05R01 event 28: {rec['time'][onset_index]:.3f}–{rec['time'][offset_index-1]+1/fog.NOMINAL_FS:.3f} s; {len(windows)} method rows")
