import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

SEQ_LENGTH = 10

# --------------------------------
# Load Data (NO HEADER IN CSV)
# --------------------------------
df = pd.read_csv("../controller/flow_stats_log.csv")


# Clean possible N/A values
df = df.replace("N/A", "0.0.0.0")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

# Ensure numeric types
df['packet_count'] = pd.to_numeric(df['packet_count'], errors='coerce').fillna(0)
df['byte_count'] = pd.to_numeric(df['byte_count'], errors='coerce').fillna(0)
df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)

# --------------------------------
# Feature Engineering
# --------------------------------

# Avoid divide by zero
df['packet_count'] = df['packet_count'].replace(0, 1)

df['avg_pkt_size'] = df['byte_count'] / df['packet_count']
df['pkt_per_sec'] = df['packet_count'] / (df['duration'] + 1e-5)

# Encode src_ip numerically
df['src_ip_encoded'] = df['src_ip'].astype('category').cat.codes

# --------------------------------
# Labeling
# --------------------------------
# Attacker IP = 10.0.0.99
df['label'] = df['src_ip'].apply(lambda x: 1 if x == '10.0.0.3' else 0)


# --------------------------------
# Select Features
# --------------------------------
feature_columns = [
    'packet_count',
    'byte_count',
    'duration',
    'avg_pkt_size',
    'pkt_per_sec',
    'src_ip_encoded'
]

X = df[feature_columns].values
y = df['label'].values

# --------------------------------
# Normalize Features
# --------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# --------------------------------
# Sliding Window Sequence Builder
# --------------------------------
def create_sequences(X, y, seq_length):
    sequences = []
    labels = []

    for i in range(len(X) - seq_length):
        seq = X[i:i + seq_length]
        label = 1 if np.any(y[i:i + seq_length]) else 0
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)

X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LENGTH)

print("Sequences shape:", X_seq.shape)
print("Labels shape:", y_seq.shape)

# --------------------------------
# Train / Val / Test Split
# --------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# --------------------------------
# Save Processed Dataset
# --------------------------------
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Dataset saved successfully.")

