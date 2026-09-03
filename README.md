

```markdown
# 🛡️ TrapCancellation

**TrapCancellation** is a full-stack voice fraud and deepfake detection system. It analyzes live microphone audio chunks using machine learning and contextual NLP to assign a real-time risk score to incoming calls, with special attention to Hinglish social-engineering scam patterns common in Indian banking fraud.

## ✨ Features

- **Real-time Audio Chunking** — Processes audio in 250ms chunks via the browser Web Audio API.
- **Deepfake Detection** — Wav2Vec2 transformer model (`garystafford/wav2vec2-deepfake-voice-detector`) detects acoustic spoofing artifacts.
- **Contextual Risk Engine** — Regex + Logistic Regression NLP layer flags urgency, financial pressure, and Hinglish scam phrases.
- **Live Risk Scoring UI** — Next.js frontend with real-time visual risk indicators.
- **ZeroGPU Deployment** — Lazy GPU loading on Hugging Face Spaces for cost-efficient inference.

## 🏗️ Architecture & Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, React, TypeScript (Vercel) |
| Backend | Python, FastAPI, Pydantic, Librosa (Hugging Face Spaces / ZeroGPU) |
| ML | PyTorch, Transformers, ONNX, scikit-learn |

## 🔌 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Server health check |
| `POST /stream/analyze` | Main production inference endpoint (audio chunk → JSON risk score) |
| `POST /voice/analyze` | Voice-only acoustic analysis |
| `POST /api/v1/analyze-call` | Full call analysis (acoustic + contextual) |
| `WS /ws/stream` | Streaming inference (disabled in production due to HF ZeroGPU proxy constraints) |

## 🧠 Machine Learning

- **Acoustic model fine-tuning**: `backend/ml/train_acoustic.py` fine-tunes the Wav2Vec2 model on an Indic accent dataset for improved regional detection accuracy.
- **Contextual intent model**: `backend/ml/train_contextual.py` trains a TF-IDF + Logistic Regression pipeline on bilingual (English/Hinglish) fraud phrases.
- **ONNX Export**: `model-export/export_wav2vec2.py` and `model-export/export_aasist.py` convert PyTorch models for browser-side inference.

## 🚀 Local Setup

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

## 🔮 Future Improvements

- Migrate from HTTP chunking to full WebRTC for lower latency.
- Implement rate-limiting and API key authentication.
- Upgrade contextual engine from Regex to an LLM-based intent analyzer.

## 📄 License

MIT License
```

---

## Notes

- I merged endpoint info from the Hugging Face Space metadata (`/voice/analyze`, `/api/v1/analyze-call`, `/ws/stream`) [5](#0-4)  with the original README's production endpoint list [6](#0-5) , since both appear to describe the same backend but at different documentation layers (root Space README vs. project README).
- I added a "Machine Learning" section referencing `train_acoustic.py` and `train_contextual.py`, which weren't documented in the original README but are clearly part of the model pipeline.
- This is a draft — I can't push it directly; you'd need to copy it into `README.md` yourself or ask me to make the edit in a full Devin session.

Wiki pages you might want to explore:
- [Glossary (Anushka-Sharma-tech/TrapCancellation)](/wiki/Anushka-Sharma-tech/TrapCancellation#6)

### Citations

**File:** README.md (L1-38)
```markdown
# TrapCancellation

TrapCancellation is a full-stack voice fraud and deepfake detection system. It analyzes live microphone audio chunks using machine learning and contextual natural language processing to assign a real-time risk score to incoming calls.

## Features
* **Real-time Audio Chunking:** Processes audio in 250ms chunks via browser Web Audio API.
* **Deepfake Detection:** Utilizes a Wav2Vec2 transformer model for acoustic anomaly detection.
* **Contextual Risk Engine:** Scans transcripts for urgency markers, financial keywords, and scam patterns.
* **Modern UI:** Next.js frontend with live visual risk indicators.

## Architecture & Technology Stack
* **Frontend:** Next.js, React, TypeScript. Deployed on Vercel.
* **Backend:** Python, FastAPI, Pydantic, Librosa. Deployed on Hugging Face Spaces (ZeroGPU).
* **Machine Learning:** PyTorch, Transformers, ONNX.

## API Endpoints (Production)
* `GET /health` - Server health check.
* `POST /stream/analyze` - Main inference endpoint. Accepts audio chunks and returns JSON risk scores.
* *(Note: `/ws/stream` is disabled in production due to HF ZeroGPU proxy constraints).*

## Local Setup
**Backend:**
1. `cd backend`
2. `pip install -r requirements.txt`
3. `uvicorn app.main:app --reload`

**Frontend:**
1. `cd frontend`
2. `npm install`
3. `npm run dev`

## Future Improvements
* Migrate from HTTP chunking to full WebRTC for lower latency.
* Implement rate-limiting and API key authentication.
* Upgrade contextual engine from Regex to an LLM-based intent analyzer.

## License
MIT License
```

**File:** backend/requirements.txt (L1-30)
```text
---
title: TrapCancellation Voice Security Engine
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12"
app_file: app.py
---

# 🛡️ TrapCancellation Voice Security Engine

Real-Time Voice Impersonation & Contextual Fraud Detection API.

## Endpoints

- `GET /health`
- `POST /voice/analyze`
- `POST /api/v1/analyze-call`
- `WS /ws/stream`

## ZeroGPU

The acoustic deepfake model runs using Hugging Face ZeroGPU.

The GPU-dependent function is declared at module level using:

```python
@spaces.GPU(duration=120)
```

**File:** backend/ml/train_contextual.py (L1-20)
```python
import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

OUTPUT_MODEL = os.path.join(os.path.dirname(__file__), "../models/contextual_intent.pkl")

# Training data: 1 = Social Engineering/Fraud, 0 = Normal Banking Request
TRAINING_DATA = [
    ("Transfer 5 lakhs immediately to the vendor account, it is an emergency.", 1),
    ("Do not tell your supervisor, this is a highly confidential acquisition.", 1),
    ("Please share the OTP you just received to verify your account freeze.", 1),
    ("I need to check my account balance for the savings account.", 0),
    ("Can you send me the account statement for last month?", 0),
    ("Jaldi se paise transfer karo, emergency hai.", 1),
    ("Call cut mat karna, secret transaction hai.", 1),
    ("Mujhe apni passbook update karni hai.", 0)
]
```

**File:** backend/ml/train_acoustic.py (L1-14)
```python
import os
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../models/indic-deepfake-detector")
```
