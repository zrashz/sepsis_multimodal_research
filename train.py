import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import shap

from src.config import Config
from src.data_loader import generate_synthetic_data
from src.models_text import BioClinicalBERTClassifier

np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

# Temporal LSTM Model
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=1):
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        out = torch.sigmoid(self.fc(h_n[-1]))
        return out

def main():
    print("="*65)
    print("   MULTIMODAL SEPSIS PIPELINE: 5-FOLD CV + BIOCLINICALBERT   ")
    print("="*65)

    # 1. Load Data
    print("\n[1/4] Loading Cohort Dataset...")
    tabular_df, time_series_data, text_corpus, labels = generate_synthetic_data(num_patients=300)

    # 2. Extract BioClinicalBERT Embeddings
    print("\n[2/4] Extracting BioClinicalBERT Embeddings from Notes...")
    bert_module = BioClinicalBERTClassifier()
    text_embeddings = bert_module.extract_embeddings(text_corpus)
    print(f" -> Text Embeddings Shape: {text_embeddings.shape}")

    # 3. Stratified 5-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    
    cv_scores = {
        'tabular_auroc': [],
        'temporal_auroc': [],
        'text_auroc': [],
        'fusion_auroc': [],
        'fusion_auprc': []
    }

    print("\n[3/4] Running 5-Fold Stratified Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(tabular_df, labels), 1):
        y_train, y_val = labels[train_idx], labels[val_idx]

        # --- Tabular Branch ---
        X_tab_train, X_tab_val = tabular_df.iloc[train_idx], tabular_df.iloc[val_idx]
        scaler = StandardScaler()
        X_tab_train_s = scaler.fit_transform(X_tab_train)
        X_tab_val_s = scaler.transform(X_tab_val)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=Config.SEED, class_weight='balanced')
        rf_model.fit(X_tab_train_s, y_train)
        pred_tab_train = rf_model.predict_proba(X_tab_train_s)[:, 1]
        pred_tab_val = rf_model.predict_proba(X_tab_val_s)[:, 1]

        # --- Temporal LSTM Branch ---
        X_ts_train_t = torch.tensor(time_series_data[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_ts_val_t = torch.tensor(time_series_data[val_idx], dtype=torch.float32)

        lstm_model = TemporalLSTM()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
        criterion = nn.BCELoss()

        lstm_model.train()
        for epoch in range(12):
            optimizer.zero_grad()
            out = lstm_model(X_ts_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            pred_ts_train = lstm_model(X_ts_train_t).numpy().flatten()
            pred_ts_val = lstm_model(X_ts_val_t).numpy().flatten()

        # --- Text BioClinicalBERT Branch ---
        X_text_train, X_text_val = text_embeddings[train_idx], text_embeddings[val_idx]
        text_clf = LogisticRegression(max_iter=500, class_weight='balanced')
        text_clf.fit(X_text_train, y_train)
        pred_text_train = text_clf.predict_proba(X_text_train)[:, 1]
        pred_text_val = text_clf.predict_proba(X_text_val)[:, 1]

        # --- Late Fusion Meta-Learner ---
        meta_train = np.column_stack([pred_tab_train, pred_ts_train, pred_text_train])
        meta_val = np.column_stack([pred_tab_val, pred_ts_val, pred_text_val])

        meta_learner = LogisticRegression()
        meta_learner.fit(meta_train, y_train)
        pred_fusion = meta_learner.predict_proba(meta_val)[:, 1]

        # Metrics Recording
        cv_scores['tabular_auroc'].append(roc_auc_score(y_val, pred_tab_val))
        cv_scores['temporal_auroc'].append(roc_auc_score(y_val, pred_ts_val))
        cv_scores['text_auroc'].append(roc_auc_score(y_val, pred_text_val))
        cv_scores['fusion_auroc'].append(roc_auc_score(y_val, pred_fusion))
        cv_scores['fusion_auprc'].append(average_precision_score(y_val, pred_fusion))

        print(f"  Fold {fold}/5 -> Tabular: {cv_scores['tabular_auroc'][-1]:.4f} | "
              f"Temporal: {cv_scores['temporal_auroc'][-1]:.4f} | "
              f"Text(BERT): {cv_scores['text_auroc'][-1]:.4f} | "
              f"Fusion AUROC: {cv_scores['fusion_auroc'][-1]:.4f}")

    print("\n" + "="*65)
    print("           CROSS-VALIDATION EVALUATION RESULTS           ")
    print("="*65)
    print(f" Tabular AUROC:       {np.mean(cv_scores['tabular_auroc']):.4f} ± {np.std(cv_scores['tabular_auroc']):.4f}")
    print(f" Temporal LSTM AUROC: {np.mean(cv_scores['temporal_auroc']):.4f} ± {np.std(cv_scores['temporal_auroc']):.4f}")
    print(f" BioClinicalBERT:     {np.mean(cv_scores['text_auroc']):.4f} ± {np.std(cv_scores['text_auroc']):.4f}")
    print(f" Multimodal Fusion:   {np.mean(cv_scores['fusion_auroc']):.4f} ± {np.std(cv_scores['fusion_auroc']):.4f}")
    print(f" Multimodal AUPRC:    {np.mean(cv_scores['fusion_auprc']):.4f} ± {np.std(cv_scores['fusion_auprc']):.4f}")

    # 4. Generate SHAP Plot
    print("\n[4/4] Generating SHAP Explanation Plot...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_tab_val_s)
    shap_vals_class1 = shap_values.values[:, :, 1] if len(shap_values.shape) == 3 else shap_values.values

    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_vals_class1, features=X_tab_val, feature_names=X_tab_val.columns.tolist(), show=False)
    save_path = os.path.join(Config.OUTPUT_FIGURES_DIR, "shap_summary.png")
    plt.title("SHAP Feature Importance (Structured Vitals/Labs)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] Updated SHAP plot saved to: {save_path}")

if __name__ == "__main__":
    main()
