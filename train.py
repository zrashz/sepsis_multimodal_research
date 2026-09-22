import os
import sys

# Set non-interactive Matplotlib backend BEFORE importing pyplot to prevent Tcl/Tk errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import shap

from src.config import Config
from src.data_loader import generate_synthetic_data

# Set random seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

# ==========================================
# 1. TEMPORAL LSTM MODULE
# ==========================================
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=1):
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = torch.sigmoid(self.fc(last_hidden))
        return out, last_hidden

# ==========================================
# 2. MAIN TRAINING PIPELINE
# ==========================================
def main():
    print("="*60)
    print("   SEPSIS MULTIMODAL LATE FUSION PIPELINE RUN   ")
    print("="*60)

    # Step A: Load Data
    print("\n[1/5] Loading Multimodal Dataset...")
    tabular_df, time_series_data, text_corpus, labels = generate_synthetic_data(num_patients=600)
    
    # Split indices (80% train, 20% test)
    indices = np.arange(len(labels))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, labels, test_size=0.2, random_state=Config.SEED, stratify=labels
    )

    # --- BRANCH 1: TABULAR MODEL (Random Forest) ---
    print("\n[2/5] Training Tabular Branch (Random Forest)...")
    X_tab_train, X_tab_test = tabular_df.iloc[idx_train], tabular_df.iloc[idx_test]
    scaler = StandardScaler()
    X_tab_train_scaled = scaler.fit_transform(X_tab_train)
    X_tab_test_scaled = scaler.transform(X_tab_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=Config.SEED, class_weight='balanced')
    rf_model.fit(X_tab_train_scaled, y_train)

    train_tab_probs = rf_model.predict_proba(X_tab_train_scaled)[:, 1]
    test_tab_probs = rf_model.predict_proba(X_tab_test_scaled)[:, 1]
    print(f" -> Tabular AUROC: {roc_auc_score(y_test, test_tab_probs):.4f}")

    # --- BRANCH 2: TEMPORAL TIME-SERIES MODEL (LSTM) ---
    print("\n[3/5] Training Temporal Branch (PyTorch LSTM)...")
    X_ts_train_t = torch.tensor(time_series_data[idx_train], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_ts_test_t = torch.tensor(time_series_data[idx_test], dtype=torch.float32)

    lstm_model = TemporalLSTM()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

    lstm_model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        preds, _ = lstm_model(X_ts_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        train_ts_probs, _ = lstm_model(X_ts_train_t)
        test_ts_probs, _ = lstm_model(X_ts_test_t)
    
    train_ts_probs = train_ts_probs.numpy().flatten()
    test_ts_probs = test_ts_probs.numpy().flatten()
    print(f" -> Temporal AUROC: {roc_auc_score(y_test, test_ts_probs):.4f}")

    # --- BRANCH 3: TEXT NLP MODEL (TF-IDF Baseline) ---
    print("\n[4/5] Training Text Branch (TF-IDF Narrative Vectorizer)...")
    train_texts = [text_corpus[i] for i in idx_train]
    test_texts = [text_corpus[i] for i in idx_test]

    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    X_text_train = vectorizer.fit_transform(train_texts).toarray()
    X_text_test = vectorizer.transform(test_texts).toarray()

    text_model = RandomForestClassifier(n_estimators=50, random_state=Config.SEED, class_weight='balanced')
    text_model.fit(X_text_train, y_train)

    train_text_probs = text_model.predict_proba(X_text_train)[:, 1]
    test_text_probs = text_model.predict_proba(X_text_test)[:, 1]
    print(f" -> Text NLP AUROC: {roc_auc_score(y_test, test_text_probs):.4f}")

    # --- MULTIMODAL LATE FUSION META-LEARNER ---
    print("\n[5/5] Combining Modalities via Late Fusion Meta-Learner...")
    X_fusion_train = np.column_stack([train_tab_probs, train_ts_probs, train_text_probs])
    X_fusion_test = np.column_stack([test_tab_probs, test_ts_probs, test_text_probs])

    meta_learner = LogisticRegression()
    meta_learner.fit(X_fusion_train, y_train)

    final_probs = meta_learner.predict_proba(X_fusion_test)[:, 1]

    print("\n" + "="*60)
    print("             FINAL MULTIMODAL EVALUATION             ")
    print("="*60)
    print(f"Final Multimodal AUROC: {roc_auc_score(y_test, final_probs):.4f}")
    print(f"Final Multimodal AUPRC: {average_precision_score(y_test, final_probs):.4f}")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, (final_probs > 0.5).astype(int)))

    # --- SHAP EXPLAINABILITY ---
    print("\nGenerating SHAP Feature Attribution Plot...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_tab_test_scaled)
    
    shap_vals_class1 = shap_values.values[:, :, 1] if len(shap_values.shape) == 3 else shap_values.values

    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_vals_class1, 
        features=X_tab_test, 
        feature_names=X_tab_test.columns.tolist(),
        show=False
    )
    
    save_path = os.path.join(Config.OUTPUT_FIGURES_DIR, "shap_summary.png")
    plt.title("SHAP Feature Importance (Structured Vitals/Labs)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] SHAP Explanation saved to: {save_path}")

if __name__ == "__main__":
    main()
