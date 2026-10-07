import torch
import torch.nn as nn
import numpy as np
import joblib
import math
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

SEQ_LENGTH = 20
FEATURE_DIM = 8

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ZTAModel(input_dim=FEATURE_DIM).to(device)
model.load_state_dict(torch.load("zta_model.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler.pkl")

trust_scores = {}
alpha = 0.85
gamma = 1.0

def compute_trust(host_id, anomaly_prob):
    similarity = math.exp(-gamma * anomaly_prob)

    if host_id not in trust_scores:
        trust_scores[host_id] = 0.8

    trust_scores[host_id] = (
        alpha * trust_scores[host_id]
        + (1 - alpha) * similarity
    )

    return trust_scores[host_id]

app = FastAPI()

class PredictionRequest(BaseModel):
    host_id: str
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):

    features = np.array(request.features)

    if features.shape != (SEQ_LENGTH, FEATURE_DIM):
        return {"error": f"Invalid feature shape: {features.shape}"}

    features = scaler.transform(features)

    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        anomaly_prob = torch.sigmoid(logits).item()

    trust = compute_trust(request.host_id, anomaly_prob)

    return {
        "anomaly_probability": float(anomaly_prob),
        "trust_score": float(trust)
    }

if __name__ == "__main__":
    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000)

