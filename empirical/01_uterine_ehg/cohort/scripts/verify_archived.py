from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
for name, expected in [('subject_RA.csv', {'M2': (0.53, 39), 'M3': (0.45, 36)}), ('subject_Rsigma.csv', {'M2': (1.21, 40), 'M3': (1.09, 39)})]:
    table = pd.read_csv(DATA / name)
    for method, (median, positive) in expected.items():
        values = table.loc[table.method == method, 'difference'].dropna()
        assert len(values) == 44
        assert round(values.median(), 2) == median
        assert (values > 0).sum() == positive
        print(name, method, len(values), values.median(), positive)
coverage = pd.read_csv(DATA / 'dataset_coverage.csv')
assert coverage.record.nunique() == 123
assert coverage.subject_id.nunique() == 45
marker = pd.read_csv(DATA / 'marker_primary_M3.csv')
assert set(marker.n_markers) == {137}
assert set(marker.n_subjects) == {30}
assert set(marker.n_recordings) == {59}
print('EHG archived checkpoints passed')
