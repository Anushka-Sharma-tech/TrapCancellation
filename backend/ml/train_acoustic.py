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

def train():
    print("Loading Indic dataset for acoustic fine-tuning...")
    # Loads the train/val/test folders created by download_indic.py
    dataset = load_dataset("audiofolder", data_dir=DATA_DIR)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        audio = batch["audio"]
        inputs = feature_extractor(
            audio["array"],
            sampling_rate=16000,
            max_length=16000 * 5,
            truncation=True,
            padding="max_length",
        )
        batch["input_values"] = inputs["input_values"][0]
        return batch

    print("Extracting acoustic features...")
    encoded = dataset.map(preprocess, remove_columns=["audio"], num_proc=1)
    
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=eval_pred.labels)

    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
    model.freeze_feature_encoder()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("val", encoded["train"]),
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning on Indian accents...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    feature_extractor.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}. Update DEEPFAKE_MODEL env var to this path.")

if __name__ == "__main__":
    train()