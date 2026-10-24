
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "naive_bayes_tuned_tfidf.pkl"
)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Fake News Detection API",
    description="Predicts news veracity and authorship origin.",
    version="1.0.0",
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


@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
def predict_news(request: NewsRequest):
    news_text = request.text.strip()

    if not news_text:
        raise HTTPException(
            status_code=400,
            detail="Please enter some news text.",
        )

    prediction = str(model.predict([news_text])[0])

    probabilities = model.predict_proba([news_text])[0]
    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "veracity": prediction.split("/")[0],
        "authorship": prediction.split("/")[1],
        "confidence": round(confidence * 100, 2),
    }
