import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.config import Config

def generate_synthetic_data(num_patients=500, seq_len=Config.SEQ_LEN):
    """
    Generates synthetic MIMIC-IV aligned multimodal datasets for testing.
    """
    np.random.seed(Config.SEED)
    labels = np.random.choice([0, 1], size=num_patients, p=[0.85, 0.15])
    
    # 1. Tabular Static Features
    tabular_data = []
    for i in range(num_patients):
        age = np.random.normal(65, 12) + (10 if labels[i] == 1 else 0)
        gender = np.random.choice([0, 1])
        charlson = np.random.poisson(2) + (2 if labels[i] == 1 else 0)
        lactate = np.random.gamma(2, 0.8) + (1.5 if labels[i] == 1 else 0)
        tabular_data.append([age, gender, charlson, lactate])
        
    tabular_df = pd.DataFrame(tabular_data, columns=['age', 'gender', 'charlson_index', 'baseline_lactate'])
    
    # 2. Time-Series Vital Signs (N, 24, 5)
    time_series_data = np.zeros((num_patients, seq_len, 5))
    for i in range(num_patients):
        base_hr = 80 + (20 if labels[i] == 1 else 0)
        base_rr = 16 + (6 if labels[i] == 1 else 0)
        base_map = 85 - (15 if labels[i] == 1 else 0)
        
        hr_trend = base_hr + np.cumsum(np.random.normal(0, 1.5, seq_len))
        rr_trend = base_rr + np.cumsum(np.random.normal(0, 0.5, seq_len))
        map_trend = base_map + np.cumsum(np.random.normal(0, 1.0, seq_len))
        temp_trend = 37.0 + np.cumsum(np.random.normal(0, 0.1, seq_len)) + (0.8 if labels[i] == 1 else 0)
        wbc_trend = 8.0 + np.cumsum(np.random.normal(0, 0.2, seq_len)) + (4.0 if labels[i] == 1 else 0)
        
        time_series_data[i] = np.column_stack([hr_trend, rr_trend, map_trend, temp_trend, wbc_trend])
        
    # 3. Clinical Text Narratives
    text_corpus = []
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
    
    for i in range(num_patients):
        phrase = np.random.choice(sepsis_notes) if labels[i] == 1 else np.random.choice(non_sepsis_notes)
        text_corpus.append(phrase)
        
    return tabular_df, time_series_data, text_corpus, labels

if __name__ == "__main__":
    tab, ts, txt, y = generate_synthetic_data()
    print(f"[OK] Data Loader Ready: Generated {len(y)} patient records.")
