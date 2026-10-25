
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "logistic_regression_tfidf_linguistic.pkl"
)

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
vectorizer = artifact["vectorizer"]
scaler = artifact["scaler"]
FEATURE_COLS = artifact["feature_cols"]

app = FastAPI(
    title="Fake News Detection API",
    description="Predicts news veracity and authorship origin.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsRequest(BaseModel):
    text: str


def extract_features(text: str) -> list[float]:
    # Keep this logic consistent with the training feature extractor.
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    word_count = len(words)
    unique_words = len({word.lower() for word in words})
    sentence_count = len(sentences)

    alphabetic_count = sum(char.isalpha() for char in text)
    uppercase_count = sum(char.isupper() for char in text)

    lexical_diversity = (
        unique_words / word_count if word_count > 0 else 0.0
    )
    capitalization_ratio = (
        uppercase_count / alphabetic_count
        if alphabetic_count > 0
        else 0.0
    )

    character_count = len(text)
    punctuation_ratio = (
        sum(char in ".,!?;:" for char in text) / character_count
        if character_count > 0
        else 0.0
    )

    avg_sentence_length = (
        word_count / sentence_count if sentence_count > 0 else 0.0
    )

    feature_values = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "lexical_diversity": lexical_diversity,
        "punctuation_ratio": punctuation_ratio,
        "capitalization_ratio": capitalization_ratio,
        "avg_sentence_length": avg_sentence_length,
    }

    return [float(feature_values[name]) for name in FEATURE_COLS]


@app.get("/")
def home():
    return {
        "message": "Fake News Detection API is running",
        "model": MODEL_PATH.name,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": [str(label) for label in model.classes_],
    }


@app.post("/predict")
def predict_news(request: NewsRequest):
    news_text = request.text.strip()

    if not news_text:
        raise HTTPException(
            status_code=400,
            detail="Please enter some news text.",
        )

    # Transform raw text and linguistic features exactly as trained.
    text_matrix = vectorizer.transform([news_text])
    raw_features = np.asarray(
        extract_features(news_text), dtype=float
    ).reshape(1, -1)
    scaled_features = scaler.transform(raw_features)

    combined_matrix = hstack(
        [text_matrix, csr_matrix(scaled_features)],
        format="csr",
    )

    prediction = str(model.predict(combined_matrix)[0])
    probabilities = model.predict_proba(combined_matrix)[0]

    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    veracity, authorship = prediction.split("/", 1)

    veracity_confidence = sum(
        probability
        for label, probability in class_probabilities.items()
        if label.split("/", 1)[0] == veracity
    )

    authorship_confidence = sum(
        probability
        for label, probability in class_probabilities.items()
        if label.split("/", 1)[1] == authorship
    )

    overall_confidence = class_probabilities[prediction]

    return {
        "prediction": prediction,
        "veracity": veracity,
        "authorship": authorship,
        "confidence": round(overall_confidence * 100, 2),
        "veracity_confidence": round(veracity_confidence * 100, 2),
        "authorship_confidence": round(authorship_confidence * 100, 2),
        "class_probabilities": {
            label: round(probability * 100, 2)
            for label, probability in class_probabilities.items()
        },
    }
