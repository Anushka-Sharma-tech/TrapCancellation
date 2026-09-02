import numpy as np
import requests

URL = "https://anushka-8-trapcancellation-backend.hf.space/stream/analyze"

# 1 second of Float32 PCM at 16 kHz
audio = np.zeros(16000, dtype=np.float32)

response = requests.post(
    URL,
    data=audio.tobytes(),
    headers={
        "Content-Type": "application/octet-stream"
    },
    timeout=120,
)

print("Status:", response.status_code)
print("Response:", response.text)