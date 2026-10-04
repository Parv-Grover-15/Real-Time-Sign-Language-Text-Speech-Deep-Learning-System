"""
PyTorch LSTM + Self-Attention Sequence Classifier (from Sign-Bridge contribution)
"""
import torch
import torch.nn as nn
import numpy as np

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        return self.layer_norm(x + attn_out)

class LSTMAttentionNet(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, num_heads=4, num_classes=26, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = SelfAttentionBlock(embed_dim=hidden_dim * 2, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        attn_out = self.attention(lstm_out)  # (batch_size, seq_len, hidden_dim * 2)
        # Global average pooling over time dimension
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim * 2)
        logits = self.fc(pooled)
        return logits

class LSTMAttentionClassifier:
    def __init__(self, input_dim=63, num_classes=26, class_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or [chr(i) for i in range(65, 91)]
        self.num_classes = len(self.class_names)
        self.model = LSTMAttentionNet(input_dim=input_dim, num_classes=self.num_classes).to(self.device)
        self.model.eval()

    def predict_sequence(self, sequence: list):
        """
        Takes a list of sequence feature vectors of length T and returns:
        - top_class: predicted sign string
        - confidence: float score
        """
        if not sequence:
            return None, 0.0

        seq_arr = np.array(sequence, dtype=np.float32)
        tensor_in = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        confidence = float(probs[top_idx])

        return top_class, confidence
