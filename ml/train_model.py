import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------------
# Load Data
# -------------------------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# -------------------------------
# Model Definition
# -------------------------------
class ZTAModel(nn.Module):
    def __init__(self, input_dim):
        super(ZTAModel, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(32, 64, batch_first=True)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # (B, T, C)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep

        out = self.fc(out)
        return torch.sigmoid(out).squeeze()

model = ZTAModel(input_dim=X_train.shape[2])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------
EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_preds = (val_outputs > 0.5).float()

        acc = accuracy_score(y_val.numpy(), val_preds.numpy())
        prec = precision_score(y_val.numpy(), val_preds.numpy())
        rec = recall_score(y_val.numpy(), val_preds.numpy())
        f1 = f1_score(y_val.numpy(), val_preds.numpy())
        auc = roc_auc_score(y_val.numpy(), val_outputs.numpy())

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {loss.item():.4f} | "
          f"Val Acc: {acc:.4f} | "
          f"F1: {f1:.4f} | "
          f"AUC: {auc:.4f}")

# Save model
torch.save(model.state_dict(), "zta_model.pth")

print("Training complete. Model saved as zta_model.pth")
