from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
a = pd.read_csv(DATA / 'individual_eeg_activity.csv')
assert a[['recording', 'event_id']].drop_duplicates().shape[0] == 182
assert a.recording.nunique() == 135
assert a.independent_patient_id.nunique() == 23
meta = pd.read_csv(DATA / 'pooled_plot_metadata.csv').iloc[0]
assert meta.median_annotated_seizure_duration_s == 46.5
curves = pd.read_csv(DATA / 'pooled_manuscript_diagnostics.csv')
for method, expected_median, expected_max in [('M2', 1.02, 1.12), ('M3', 1.04, 1.10)]:
    x = curves[(curves.method == method) & (curves.diagnostic == 'R_A') & (curves.value_kind == 'baseline_relative') & curves.window_end_relative_s.between(0, 46.5)]
    assert round(x.central_estimate.median(), 2) == expected_median
    assert round(x.central_estimate.max(), 2) == expected_max
    print(method, x.central_estimate.median(), x.central_estimate.max())
counts = pd.read_csv(DATA / 'Seizure_EEG_counts_FINAL.csv')
for diagnostic, expected in [('R_A', {'M2': 20468, 'M3': 22587}), ('R_sigma', {'M2': 20534, 'M3': 22621})]:
    for method, n in expected.items():
        x = counts[(counts.analysis_version == 'final_pooled_conservative_40s') & (counts.method == method) & (counts.diagnostic == diagnostic) & (counts.value_kind == 'baseline_relative')]
        assert len(x) == 1 and int(x.iloc[0].total_event_window_rows) == 22878
        assert int(x.iloc[0].admissible_windows) == n
s = pd.read_csv(DATA / 'EEG_window_sensitivity_summary.csv')
for measure, expected in [('R_A', {'M2': 79, 'M3': 91}), ('R_sigma', {'M2': 83, 'M3': 93})]:
    rows = s[s.measure == measure]
    for method, n in expected.items():
        assert n in set(rows.loc[rows.method == method, 'n_matched_events'])
assert set(s.n_independent_patients) == {21}
print('EEG archived checkpoints passed')
