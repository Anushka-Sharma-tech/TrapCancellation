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

def train_intent_model():
    print("Training Contextual Intent NLP Model...")
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    # Pipeline: Convert text to frequency vectors -> Train Logistic Regression
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(class_weight="balanced")
    )
    
    pipeline.fit(texts, labels)
    
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(pipeline, f)
        
    print(f"Contextual NLP model saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    train_intent_model()