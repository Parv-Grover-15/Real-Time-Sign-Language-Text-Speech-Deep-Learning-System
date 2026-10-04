"""
Training script for PyTorch LSTM + Self-Attention Sequence Model
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.recognition.lstm_classifier import LSTMAttentionNet

MODEL_OUT_PATH = ROOT_DIR / "models" / "lstm_attention.pt"

def train_lstm_demo(epochs=5, batch_size=16):
    print("Initializing PyTorch LSTM + Attention Model Training...")
    num_samples = 100
    seq_len = 20
    input_dim = 63
    num_classes = 26

    X = torch.randn(num_samples, seq_len, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAttentionNet(input_dim=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / num_samples
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

    # Save trained checkpoint
    torch.save(model.state_dict(), MODEL_OUT_PATH)
    print(f"LSTM + Attention Model saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    train_lstm_demo()
