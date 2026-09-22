import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from src.config import Config

class BioClinicalBERTClassifier(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super(BioClinicalBERTClassifier, self).__init__()
        self.use_fallback = False
        
        try:
            print("[INFO] Loading BioClinicalBERT tokenizer & weights...")
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            
            # Freeze BERT parameters for fast CPU inference
            for param in self.bert.parameters():
                param.requires_grad = False
            print("[OK] BioClinicalBERT loaded successfully!")
            
        except Exception as e:
            print(f"[WARNING] Could not load BioClinicalBERT ({e}). Switching to lightweight Clinical Embeddings.")
            self.use_fallback = True

    def extract_embeddings(self, texts, max_length=Config.MAX_TEXT_LEN):
        """Extracts text embeddings (768-dim from BERT or 50-dim fallback)."""
        if self.use_fallback:
            # Deterministic fast feature extraction fallback for clinical keywords
            embeddings = []
            keywords = ["fever", "hypotension", "tachycardia", "sepsis", "fluid", "altered", "afebrile", "stable"]
            for text in texts:
                vec = [text.lower().count(kw) for kw in keywords]
                # Pad vector to 64 dimensions for consistency
                vec += [0.0] * (64 - len(vec))
                embeddings.append(vec)
            return np.array(embeddings)
            
        self.eval()
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = self.bert(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

if __name__ == "__main__":
    bert_module = BioClinicalBERTClassifier()
    sample_emb = bert_module.extract_embeddings(["Patient shows signs of sepsis and high fever."])
    print(f"[OK] Test Embedding Extracted! Shape: {sample_emb.shape}")
