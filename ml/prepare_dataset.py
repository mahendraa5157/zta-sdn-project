import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SEQ_LENGTH = 20
FEATURE_DIM = 8

# ---------------------------------
# Load CSV
# ---------------------------------
df = pd.read_csv("traffic_dataset.csv", header=None)

print("Total rows:", len(df))

features = df.iloc[:, 0:8].values
labels = df.iloc[:, 8].values

# ---------------------------------
# Create sequences
# ---------------------------------
X = []
y = []

for i in range(len(features) - SEQ_LENGTH):
    X.append(features[i:i+SEQ_LENGTH])
    y.append(labels[i+SEQ_LENGTH-1])  # label from last row

X = np.array(X)
y = np.array(y)

print("Sequence dataset shape:", X.shape)
print("Labels shape:", y.shape)

# ---------------------------------
# Train / Validation split
# ---------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------
# Normalize (VERY IMPORTANT)
# ---------------------------------
scaler = StandardScaler()

# reshape for scaler (2D)
X_train_reshaped = X_train.reshape(-1, FEATURE_DIM)
X_val_reshaped = X_val.reshape(-1, FEATURE_DIM)

X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_val_scaled = scaler.transform(X_val_reshaped)

# reshape back to 3D
X_train = X_train_scaled.reshape(-1, SEQ_LENGTH, FEATURE_DIM)
X_val = X_val_scaled.reshape(-1, SEQ_LENGTH, FEATURE_DIM)

# ---------------------------------
# Save everything
# ---------------------------------
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)

joblib.dump(scaler, "scaler.pkl")

print("Dataset prepared successfully.")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
