import os
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

class AcousticSpoofDetector:
    def __init__(self, model_path_or_id: str | None = None):
        self.model_id = model_path_or_id or os.getenv(
            "DEEPFAKE_MODEL", 
            "garystafford/wav2vec2-deepfake-voice-detector"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Acoustic Deepfake Model: {self.model_id} on {self.device}")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        # Resolve fake label index dynamically
        self.fake_index = self._resolve_fake_index()

    def _resolve_fake_index(self) -> int:
        id2label = getattr(self.model.config, "id2label", {0: "fake", 1: "real"})
        for idx, label in id2label.items():
            if str(label).lower() in ["fake", "spoof", "spoofed", "generated", "deepfake"]:
                return int(idx)
        return 0  # Fallback default

    def predict_risk(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> float:
        """
        Runs acoustic classification and returns fake probability as a float (0.0 to 1.0).
        """
        if len(audio_array) == 0:
            return 0.0

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 5  # Analyze 5-second slice
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        return float(probs[self.fake_index].item())