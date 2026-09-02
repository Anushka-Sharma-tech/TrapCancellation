import io
import os
import numpy as np
import torch
import librosa
import soundfile as sf

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

app = FastAPI(title="TrapCancellation Voice API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv(
    "DEEPFAKE_MODEL",
    "garystafford/wav2vec2-deepfake-voice-detector",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def read_audio_bytes(audio_bytes: bytes):
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    return audio.astype(np.float32), 16000


def speech_activity(audio: np.ndarray):
    if len(audio) == 0:
        return False, 0.0, 0.0

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))

    intervals = librosa.effects.split(audio, top_db=28)
    speech_samples = sum(end - start for start, end in intervals)
    speech_ratio = speech_samples / max(1, len(audio))

    has_speech = rms > 0.012 and peak > 0.04 and speech_ratio > 0.18
    return has_speech, rms, speech_ratio


def dsp_scores(audio: np.ndarray, sr: int):
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    pitches, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )

    voiced = pitches[~np.isnan(pitches)]

    if len(voiced) < 8:
        prosody = 0.0
    else:
        pitch_std = float(np.std(voiced))
        pitch_jump = float(np.mean(np.abs(np.diff(voiced))) / max(1.0, np.mean(voiced)))
        prosody = clamp01((pitch_std / 180.0) * 0.45 + pitch_jump * 1.4)

    artifact = clamp01(flatness * 4.0 + zcr * 1.8)

    return artifact, prosody


def model_fake_probability(audio: np.ndarray, sr: int):
    min_len = int(sr * 2.5)
    max_len = int(sr * 8.0)

    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    audio = audio[:max_len]

    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    return float(probs[1].item())


@app.post("/voice/analyze")
async def analyze_voice(request: Request):
    audio_bytes = await request.body()
    audio, sr = read_audio_bytes(audio_bytes)

    has_speech, rms, speech_ratio = speech_activity(audio)

    if not has_speech:
        return {
            "acousticArtifact": 0.0,
            "prosodyAnomaly": 0.0,
            "speakerDrift": 0.0,
            "behavioralRisk": 0.0,
            "reasons": ["No clear speech detected. Silence and room noise ignored."],
        }

    fake_prob = model_fake_probability(audio, sr)
    artifact_dsp, prosody = dsp_scores(audio, sr)

    acoustic_artifact = clamp01(fake_prob * 0.75 + artifact_dsp * 0.25)

    reasons = []
    if fake_prob > 0.65:
        reasons.append("Anti-spoofing model found synthetic or cloned voice patterns.")
    if artifact_dsp > 0.45:
        reasons.append("Spectral texture suggests acoustic artifacts.")
    if prosody > 0.45:
        reasons.append("Pitch rhythm and prosody look irregular.")
    if not reasons:
        reasons.append("Speech detected, but no strong spoofing indicators found.")

    return {
        "acousticArtifact": acoustic_artifact,
        "prosodyAnomaly": prosody,
        "speakerDrift": 0.0,
        "behavioralRisk": 0.0,
        "reasons": reasons,
    }