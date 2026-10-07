import requests
import json

features = [[0.1,0.2,0.3,0.4,0.5,0.6] for _ in range(50)]

payload = {
    "host_id": "10.0.0.1",
    "features": features
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)

print(response.json())
