import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("traffic_dataset.csv", header=None)

X = df.iloc[:, 0:8].values
y = df.iloc[:, 8].values

# -----------------------
# Normalize
# -----------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# -----------------------
# Create sequences
# -----------------------
SEQ_LENGTH = 20

def create_sequences(X, y, seq_len):
    sequences = []
    labels = []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        labels.append(y[i+seq_len])
    return np.array(sequences), np.array(labels)

X_seq, y_seq = create_sequences(X, y, SEQ_LENGTH)

# -----------------------
# Train/Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

# -----------------------
# Model
# -----------------------
class ZTAModel(nn.Module):
    def __init__(self, input_dim):
        super(ZTAModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

model = ZTAModel(input_dim=8)

# -----------------------
# Class Weight Handling
# -----------------------
pos_weight_value = (len(y_seq) - sum(y_seq)) / sum(y_seq)
pos_weight = torch.tensor([pos_weight_value])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Training
# -----------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "zta_model.pth")
print("Training complete. Model saved.")
