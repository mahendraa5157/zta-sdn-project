import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ===============================
# CONFIG
# ===============================
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
FEATURE_DIM = 8
SEQ_LENGTH = 20

# ===============================
# Load Data
# ===============================
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===============================
# Model Definition
# ===============================
class ZTAModel(nn.Module):
    def __init__(self, input_dim):
        super(ZTAModel, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (B, T, F)
        x = x.permute(0, 2, 1)        # (B, F, T)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)        # (B, T, C)
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # last timestep
        return self.fc(out).squeeze() # logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ZTAModel(input_dim=FEATURE_DIM).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===============================
# Training Loop
# ===============================
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds)
    rec = recall_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    auc = roc_auc_score(all_true, all_probs)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {total_loss/len(train_loader):.4f} | "
          f"Acc: {acc:.4f} | "
          f"F1: {f1:.4f} | "
          f"AUC: {auc:.4f}")

# ===============================
# Save Model
# ===============================
torch.save(model.state_dict(), "zta_model.pth")

print("Training complete. Model saved as zta_model.pth")
