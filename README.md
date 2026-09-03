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

## 📘 Complete Technical Mastery Guide

Want to understand how TrapCancellation actually works, from the basics of programming and audio to FastAPI, Wav2Vec2, DSP, risk scoring, deployment, and the project's real limitations?

👉 **[Read / Download the TrapCancellation Technical Mastery Guide](./docs/TrapCancellation_Visual_Textbook_Edition.docx)**