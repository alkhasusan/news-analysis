#train_deployment_model.py
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
FEATURE_PATH = ROOT / "data" / "processed" / "final_dataset_with_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_PATH = MODEL_DIR / "logistic_regression_tfidf_linguistic.pkl"

FEATURE_COLS = [
    "word_count",
    "sentence_count",
    "lexical_diversity",
    "punctuation_ratio",
    "capitalization_ratio",
    "avg_sentence_length",
]

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    feature_df = pd.read_csv(FEATURE_PATH).set_index("news_id")

    required = {"news_id", "text", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    missing_features = set(FEATURE_COLS) - set(feature_df.columns)
    if missing_features:
        raise ValueError(
            f"Feature dataset is missing: {sorted(missing_features)}"
        )

    if df["text"].isna().any() or df["class"].isna().any():
        raise ValueError("Text or class labels contain missing values.")

    # Match features to the dataset by article ID.
    features = feature_df.loc[df["news_id"], FEATURE_COLS].to_numpy()

    # Fit TF-IDF on the complete dataset for final deployment.
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    text_matrix = vectorizer.fit_transform(df["text"].astype(str))

    # Use the same standardization approach as the evaluated LR model.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    combined_matrix = hstack(
        [text_matrix, csr_matrix(scaled_features)],
        format="csr",
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(combined_matrix, df["class"])

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "classes": model.classes_.tolist(),
        "feature_type": "TF-IDF + Linguistic",
        "training_rows": len(df),
    }

    joblib.dump(artifact, OUTPUT_PATH)

    # Reload the saved artifact and test the complete prediction path.
    saved = joblib.load(OUTPUT_PATH)
    sample_text = str(df.iloc[0]["text"])
    sample_features = feature_df.loc[
        df.iloc[0]["news_id"], FEATURE_COLS
    ].to_numpy(dtype=float).reshape(1, -1)

    sample_text_matrix = saved["vectorizer"].transform([sample_text])
    sample_scaled = saved["scaler"].transform(sample_features)
    sample_combined = hstack(
        [sample_text_matrix, csr_matrix(sample_scaled)],
        format="csr",
    )
    prediction = saved["model"].predict(sample_combined)[0]

    print("\nDeployment model saved successfully.")
    print(f"Path: {OUTPUT_PATH}")
    print(f"Training rows: {len(df)}")
    print(f"Classes: {saved['classes']}")
    print(f"Sample prediction: {prediction}")
    print("Existing Naive Bayes model was not modified.")

if __name__ == "__main__":
    main()
