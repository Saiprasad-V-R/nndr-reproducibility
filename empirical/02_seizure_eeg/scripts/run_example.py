from pathlib import Path
import numpy as np
import pandas as pd
import nndr_single_seizure_plotcore as core
import eeg_geometry as geom

ROOT = Path(__file__).resolve().parents[1]
core.OUTPUT.mkdir(parents=True, exist_ok=True)
(core.OUTPUT / "representative_40s_corrected_diagnostics_checkpoint.csv").unlink(missing_ok=True)
rows, wave, duration, metadata = core.compute(core.find_edf())
X, _ = core.s6.read_eeg(core.find_edf())
onset = metadata["onset_s"]
size = int(round(core.WINDOW_SEC * core.s6.FS))
geometry = []
for tau in rows.tau_end_s:
    stop = int(round((onset + tau) * core.s6.FS))
    result = geom.geom_one_window(X[stop-size:stop], core.s6)
    geometry.append({"tau_end_s": tau, **result})
geometry = pd.DataFrame(geometry)
rows = rows.merge(geometry, on="tau_end_s", validate="one_to_one")
rows.to_csv(ROOT / "results" / "window_diagnostics.csv", index=False)
print(f"chb01_03: {len(rows)} window endpoints, {duration:g} s annotated seizure")
