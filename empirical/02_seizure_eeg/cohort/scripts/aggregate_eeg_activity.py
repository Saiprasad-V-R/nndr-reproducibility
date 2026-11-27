#!/usr/bin/env python3
"""Rebuild EEG-01B activity summaries from saved event trajectories only."""
from pathlib import Path
import numpy as np
import pandas as pd
DATA=Path(__file__).resolve().parents[1]/'data'
e=pd.read_csv(DATA/'individual_eeg_activity.csv')
assert e[['recording','event_id']].drop_duplicates().shape[0]==182
assert e.recording.nunique()==135 and e.patient.nunique()==24
assert e.loc[e.eeg_activity.notna(),['recording','event_id']].drop_duplicates().shape[0]==179
e['independent_patient_id']=e.patient.replace({'chb21':'chb01'})
assert e.independent_patient_id.nunique()==23
assert not e.duplicated(['recording','event_id','window_end_relative_s']).any()
e.to_csv(DATA/'individual_eeg_activity.csv',index=False,float_format='%.17g')
u=e.groupby(['independent_patient_id','window_end_relative_s'],sort=True).eeg_activity.agg(eeg_activity_patient_median='median',n_valid_events='count').reset_index()
u.to_csv(DATA/'independent_patient_eeg_activity.csv',index=False,float_format='%.17g')
g=u.groupby('window_end_relative_s').eeg_activity_patient_median
q=g.quantile([.25,.75]).unstack()
p=pd.DataFrame({'window_end_relative_s':g.median().index,'eeg_activity_median':g.median().values,'eeg_activity_q25':q[.25].values,'eeg_activity_q75':q[.75].values,'n_patients':g.count().values})
p['eeg_activity_iqr_width']=p.eeg_activity_q75-p.eeg_activity_q25
p['n_seizures']=p.window_end_relative_s.map(e.groupby('window_end_relative_s').eeg_activity.count()).astype(int)
assert (p.loc[p.window_end_relative_s.between(-50,100),'n_patients']==23).all()
p.to_csv(DATA/'pooled_eeg_activity.csv',index=False,float_format='%.17g')
p.to_csv(DATA/'final_plotted_eeg_activity_timeseries.csv',index=False,float_format='%.17g')
print('Rebuilt activity summaries from 182 saved events, 23 identities.')
