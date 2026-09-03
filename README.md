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