import os
import torch

class Config:
    SEED = 42

    # Absolute paths based on repository root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoints")

    # Prediction task setup
    SEQ_LEN = 24             # 24-hour time-series observation window
    PREDICTION_HORIZON = 6   # Predict sepsis 6 hours early
    MAX_TEXT_LEN = 256       # Token length for BioClinicalBERT

    # Hardware device allocation
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directories exist automatically
os.makedirs(Config.OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.DATA_PROCESSED_DIR, exist_ok=True)
