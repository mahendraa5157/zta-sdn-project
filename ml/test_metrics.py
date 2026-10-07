import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# Load Test Data
# -----------------------------
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)

# -----------------------------
# Model Definition (same as training)
# -----------------------------
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
        out = self.fc(out)
        return torch.sigmoid(out).squeeze()

# -----------------------------
# Load Trained Model
# -----------------------------
model = ZTAModel(input_dim=X_test.shape[2])
model.load_state_dict(torch.load("zta_model.pth"))
model.eval()

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    probs = outputs.numpy()
    preds = (probs > 0.5).astype(int)

# -----------------------------
# Metrics
# -----------------------------
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print("\n--- Test Metrics ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")
