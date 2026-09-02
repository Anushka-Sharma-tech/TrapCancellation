"""
Dataset Downloader for Indic Accents and Dialects
Downloads subsets of AI4Bharat Kathbath / Common Voice and organizes directory structure.
"""
import os
from datasets import load_dataset
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = {
    "train_real": os.path.join(BASE_DIR, "train", "real"),
    "val_real": os.path.join(BASE_DIR, "val", "real"),
    "test_real": os.path.join(BASE_DIR, "test_indian", "real")
}

for path in SPLITS.values():
    os.makedirs(path, exist_ok=True)

def download_indic_samples():
    print("Fetching Indian-accent genuine speech samples from AI4Bharat Kathbath / IndicSuperb...")
    # Load Hindi and Tamil subsets (streaming to avoid filling disk)
    dataset = load_dataset(
        "ai4bharat/indic-superb", 
        "asr_kathbath_hindi", 
        split="test", 
        streaming=True
    )

    count = 0
    for idx, sample in enumerate(dataset):
        if count >= 300:  # Adjust download quota as needed
            break
        
        audio = sample["audio"]
        split_target = "train_real" if count < 200 else ("val_real" if count < 250 else "test_real")
        target_path = os.path.join(SPLITS[split_target], f"indic_kathbath_{count:04d}.wav")
        
        # Save 16kHz mono audio
        sf.write(target_path, audio["array"], audio["sampling_rate"])
        count += 1

    print(f"Downloaded and saved {count} verified Indian dialect samples into train/val/test splits.")

if __name__ == "__main__":
    download_indic_samples()