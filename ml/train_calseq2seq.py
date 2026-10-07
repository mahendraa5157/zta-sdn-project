import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# --------------------------------
# Load Dataset
# --------------------------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# --------------------------------
# Model Definition
# --------------------------------
class CALSeq2Seq(nn.Module):
    def __init__(self, feat_dim, cnn_channels=64, lstm_hidden=128, attn_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(feat_dim, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True
        )

        self.attn = nn.Linear(lstm_hidden, attn_dim)
        self.attn_score = nn.Linear(attn_dim, 1)

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.permute(0, 2, 1)  # (B, T, C)

        out, _ = self.lstm(x)

        e = torch.tanh(self.attn(out))
        a = F.softmax(self.attn_score(e), dim=1)

        context = (a * out).sum(dim=1)

        logits = self.fc(context).squeeze(1)
        return logits

# --------------------------------
# Initialize Model
# --------------------------------
model = CALSeq2Seq(feat_dim=X_train.shape[2]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------------------
# Training Loop
# --------------------------------
EPOCHS = 50
BATCH_SIZE = 64
best_val_loss = float("inf")
patience = 10
counter = 0

def batch_loader(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in batch_loader(X_train, y_train, BATCH_SIZE):
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val).item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "calseq2seq.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# --------------------------------
# Evaluation
# --------------------------------
model.load_state_dict(torch.load("calseq2seq.pth"))
model.eval()

with torch.no_grad():
    test_logits = model(X_test)
    probs = torch.sigmoid(test_logits)
    preds = (probs > 0.5).float()

acc = accuracy_score(y_test.cpu(), preds.cpu())
prec = precision_score(y_test.cpu(), preds.cpu())
rec = recall_score(y_test.cpu(), preds.cpu())
f1 = f1_score(y_test.cpu(), preds.cpu())
roc = roc_auc_score(y_test.cpu(), probs.cpu())

print("\n--- Test Metrics ---")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC AUC:", roc)
