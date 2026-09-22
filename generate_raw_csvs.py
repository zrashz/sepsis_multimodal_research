import os
import numpy as np
import pandas as pd

np.random.seed(42)
num_patients = 500
seq_len = 24

os.makedirs("data/raw", exist_ok=True)
print("[INFO] Generating raw MIMIC-IV formatted CSV files locally...")

# 1. Generate tabular_features.csv
labels = np.random.choice([0, 1], size=num_patients, p=[0.85, 0.15])
tab_data = []
for i in range(num_patients):
    stay_id = 30000000 + i
    age = np.random.normal(65, 12) + (10 if labels[i] == 1 else 0)
    gender = np.random.choice([0, 1])
    charlson = np.random.poisson(2) + (2 if labels[i] == 1 else 0)
    lactate = np.random.gamma(2, 0.8) + (1.5 if labels[i] == 1 else 0)
    tab_data.append([i+1000, stay_id, round(age, 1), gender, charlson, round(lactate, 2), labels[i]])

tab_df = pd.DataFrame(tab_data, columns=['subject_id', 'stay_id', 'age', 'gender', 'charlson_index', 'baseline_lactate', 'sepsis_label'])
tab_df.to_csv("data/raw/tabular_features.csv", index=False)

# 2. Generate vitals_timeseries.csv
ts_rows = []
for i in range(num_patients):
    stay_id = 30000000 + i
    is_sep = labels[i]
    
    base_hr = 80 + (20 if is_sep == 1 else 0)
    base_rr = 16 + (6 if is_sep == 1 else 0)
    base_map = 85 - (15 if is_sep == 1 else 0)
    
    hr_trend = base_hr + np.cumsum(np.random.normal(0, 1.5, seq_len))
    rr_trend = base_rr + np.cumsum(np.random.normal(0, 0.5, seq_len))
    map_trend = base_map + np.cumsum(np.random.normal(0, 1.0, seq_len))
    temp_trend = 37.0 + np.cumsum(np.random.normal(0, 0.1, seq_len)) + (0.8 if is_sep == 1 else 0)
    wbc_trend = 8.0 + np.cumsum(np.random.normal(0, 0.2, seq_len)) + (4.0 if is_sep == 1 else 0)
    
    for hr in range(seq_len):
        ts_rows.append([
            stay_id, hr, 
            round(hr_trend[hr], 1), 
            round(rr_trend[hr], 1), 
            round(map_trend[hr], 1), 
            round(temp_trend[hr], 2), 
            round(wbc_trend[hr], 1)
        ])

ts_df = pd.DataFrame(ts_rows, columns=['stay_id', 'hour_step', 'heart_rate', 'resp_rate', 'mean_arterial_pressure', 'temperature', 'wbc_count'])
ts_df.to_csv("data/raw/vitals_timeseries.csv", index=False)

# 3. Generate clinical_notes.csv
sepsis_notes = [
    "Patient exhibits altered mental status, skin is warm and flushed.",
    "Persistent hypotension despite fluid resuscitation. Suspect occult infection.",
    "Fever spike observed with elevated white blood cell count and tachypnea."
]
non_sepsis_notes = [
    "Patient resting comfortably, vital signs within normal parameters.",
    "Post-operative recovery progressing well. No acute distress reported.",
    "Afebrile, hemodynamically stable without vasopressor support."
]

notes_rows = []
for i in range(num_patients):
    stay_id = 30000000 + i
    note = np.random.choice(sepsis_notes) if labels[i] == 1 else np.random.choice(non_sepsis_notes)
    notes_rows.append([stay_id, i+1000, note])

notes_df = pd.DataFrame(notes_rows, columns=['stay_id', 'subject_id', 'text_note'])
notes_df.to_csv("data/raw/clinical_notes.csv", index=False)

print("[OK] Generated successfully in 'data/raw/':")
print("  - tabular_features.csv")
print("  - vitals_timeseries.csv")
print("  - clinical_notes.csv")
