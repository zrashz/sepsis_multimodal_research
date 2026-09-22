import torch
import transformers
import sklearn
import shap
from src.config import Config

def run_check():
    print("="*55)
    print("      RESEARCH ENVIRONMENT SANITY CHECK      ")
    print("="*55)
    print(f"[OK] PyTorch Version:     {torch.__version__}")
    print(f"[OK] CUDA/GPU Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    -> Active GPU:       {torch.cuda.get_device_name(0)}")
    else:
        print("    -> Running on CPU mode")
    
    print(f"[OK] Transformers Version:{transformers.__version__}")
    print(f"[OK] Scikit-Learn Version:{sklearn.__version__}")
    print(f"[OK] SHAP Version:        {shap.__version__}")
    print(f"[OK] Target Hardware:     {Config.DEVICE}")
    print(f"[OK] Root Directory Path: {Config.BASE_DIR}")
    print("="*55)
    print("Environment verification complete. Ready for research execution!")

if __name__ == "__main__":
    run_check()
