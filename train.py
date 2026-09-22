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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import shap

from src.config import Config
from src.data_loader import load_real_mimic_data
from src.models_text import BioClinicalBERTClassifier

np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

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
    print("   THESIS EVALUATION PIPELINE: 5-FOLD CV & ROC CURVE EXPORT   ")
    print("="*65)

    # 1. Ingest Data
    tabular_df, time_series_data, text_corpus, labels = load_real_mimic_data()

    # 2. Extract BioClinicalBERT Features
    bert_module = BioClinicalBERTClassifier()
    text_embeddings = bert_module.extract_embeddings(text_corpus)

    # 3. Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    
    metrics = {'Tabular': [], 'Temporal LSTM': [], 'BioClinicalBERT': [], 'Late Fusion': []}
    auprc_metrics = {'Tabular': [], 'Temporal LSTM': [], 'BioClinicalBERT': [], 'Late Fusion': []}
    
    last_val_y = None
    last_preds = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(tabular_df, labels), 1):
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Tabular
        scaler = StandardScaler()
        X_tab_train_s = scaler.fit_transform(tabular_df.iloc[train_idx])
        X_tab_val_s = scaler.transform(tabular_df.iloc[val_idx])

        rf = RandomForestClassifier(n_estimators=100, random_state=Config.SEED, class_weight='balanced')
        rf.fit(X_tab_train_s, y_train)
        pred_tab_tr = rf.predict_proba(X_tab_train_s)[:, 1]
        pred_tab_val = rf.predict_proba(X_tab_val_s)[:, 1]

        # Temporal LSTM
        X_ts_tr = torch.tensor(time_series_data[train_idx], dtype=torch.float32)
        y_ts_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_ts_val = torch.tensor(time_series_data[val_idx], dtype=torch.float32)

        lstm = TemporalLSTM()
        opt = torch.optim.Adam(lstm.parameters(), lr=0.005)
        crit = nn.BCELoss()

        lstm.train()
        for epoch in range(12):
            opt.zero_grad()
            loss = crit(lstm(X_ts_tr), y_ts_tr)
            loss.backward()
            opt.step()

        lstm.eval()
        with torch.no_grad():
            pred_ts_tr = lstm(X_ts_tr).numpy().flatten()
            pred_ts_val = lstm(X_ts_val).numpy().flatten()

        # Text BioClinicalBERT
        text_clf = LogisticRegression(max_iter=500, class_weight='balanced')
        text_clf.fit(text_embeddings[train_idx], y_train)
        pred_txt_tr = text_clf.predict_proba(text_embeddings[train_idx])[:, 1]
        pred_txt_val = text_clf.predict_proba(text_embeddings[val_idx])[:, 1]

        # Late Fusion
        meta_tr = np.column_stack([pred_tab_tr, pred_ts_tr, pred_txt_tr])
        meta_val = np.column_stack([pred_tab_val, pred_ts_val, pred_txt_val])

        meta = LogisticRegression()
        meta.fit(meta_tr, y_train)
        pred_fusion = meta.predict_proba(meta_val)[:, 1]

        # Record metrics
        preds = {'Tabular': pred_tab_val, 'Temporal LSTM': pred_ts_val, 'BioClinicalBERT': pred_txt_val, 'Late Fusion': pred_fusion}
        for name, p in preds.items():
            metrics[name].append(roc_auc_score(y_val, p))
            auprc_metrics[name].append(average_precision_score(y_val, p))

        if fold == 5:
            last_val_y = y_val
            last_preds = preds

    # Plot Multi-Model ROC Curves
    plt.figure(figsize=(8, 6))
    for name, p in last_preds.items():
        fpr, tpr, _ = roc_curve(last_val_y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUROC = {np.mean(metrics[name]):.3f})", lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(Config.OUTPUT_FIGURES_DIR, "roc_comparison.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] ROC Comparison Plot exported to: {roc_path}")

    # Generate LaTeX Table Code for Thesis
    print("\n" + "="*65)
    print("           LaTeX TABLE CODE FOR THESIS DISSERTATION           ")
    print("="*65)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance comparison of single modalities vs. Late Fusion baseline on Sepsis Prediction.}")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Model Branch} & \\textbf{AUROC (Mean \\pm SD)} & \\textbf{AUPRC (Mean \\pm SD)} \\\\")
    print("\\hline")
    for name in metrics:
        auroc_m, auroc_s = np.mean(metrics[name]), np.std(metrics[name])
        auprc_m, auprc_s = np.mean(auprc_metrics[name]), np.std(auprc_metrics[name])
        print(f"{name:<20} & {auroc_m:.4f} \\pm {auroc_s:.4f} & {auprc_m:.4f} \\pm {auprc_s:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
