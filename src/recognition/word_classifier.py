"""
PyTorch Word-Level Sign Classifier for direct word recognition
"""
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
WORD_MODEL_PATH = ROOT_DIR / "models" / "word_model.pt"
WORD_META_PATH = ROOT_DIR / "models" / "word_meta.json"

from src.preprocessing.feature_extractor import FeatureExtractor

class WordNet(nn.Module):
    def __init__(self, input_dim=63, hidden_dims=[128, 64], num_classes=20, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class WordClassifier:
    def __init__(self, model_path=WORD_MODEL_PATH, metadata_path=WORD_META_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            self.class_names = checkpoint["class_names"]
            self.num_classes = checkpoint.get("num_classes", len(self.class_names))
            self.input_dim = checkpoint.get("input_dim", 63)
            scaler_dict = checkpoint.get("scaler", {})
            mean = scaler_dict.get("mean")
            scale = scaler_dict.get("scale")
            model_config = checkpoint.get("model_config", {})
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.class_names = metadata["class_names"]
            self.num_classes = len(self.class_names)
            self.input_dim = metadata.get("input_dim", 63)
            mean = metadata["scaler"]["mean"]
            scale = metadata["scaler"]["scale"]
            model_config = metadata.get("model_config", {})
            state_dict = checkpoint

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}

        self.feature_extractor = FeatureExtractor(mean=mean, scale=scale)

        hidden_dims = model_config.get("hidden_dims", [128, 64])
        dropout = model_config.get("dropout", 0.2)

        self.model = WordNet(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            num_classes=self.num_classes,
            dropout=dropout
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, raw_landmarks: np.ndarray):
        """
        Takes raw 21x3 hand landmarks, normalizes them, and returns:
        - top_class: predicted sign word string (e.g. 'Hello', 'Thank You', 'Water', 'Help')
        - confidence: float confidence score between 0.0 and 1.0
        - all_probs: dictionary mapping word name -> probability
        """
        if raw_landmarks is None:
            return None, 0.0, {}

        features = self.feature_extractor.normalize_landmarks(raw_landmarks)
        if features is None:
            return None, 0.0, {}

        tensor_in = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        confidence = float(probs[top_idx])
        all_probs = {self.class_names[i]: float(probs[i]) for i in range(self.num_classes)}

        return top_class, confidence, all_probs
