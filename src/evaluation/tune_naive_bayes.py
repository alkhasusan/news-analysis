
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
AI_PATH = ROOT / "data" / "generated" / "ai_rewrites_final.csv"
RESULTS_PATH = ROOT / "results"
MODELS_PATH = ROOT / "models"

RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
ai_df = pd.read_csv(AI_PATH)

ai_mapping = ai_df.drop_duplicates("news_id").set_index("news_id")["original_id"].to_dict()

df["group_id"] = df["news_id"]
ai_mask = df["authorship"].str.lower().eq("ai")
df.loc[ai_mask, "group_id"] = df.loc[ai_mask, "news_id"].map(ai_mapping)

if df["group_id"].isna().any():
    raise ValueError("Some AI articles could not be matched to their original articles.")

groups = df["group_id"].drop_duplicates().to_numpy()
train_groups, test_groups = train_test_split(
    groups, test_size=0.2, random_state=42
)

train_df = df[df["group_id"].isin(train_groups)].copy()
test_df = df[df["group_id"].isin(test_groups)].copy()

print("Training class counts:")
print(train_df["class"].value_counts())
print("\nTest class counts:")
print(test_df["class"].value_counts())

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nb", MultinomialNB())
])

param_grid = {
    "tfidf__max_features": [10000, 20000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "nb__alpha": [0.01, 0.1, 0.5, 1.0]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    verbose=1,
    refit=True
)

search.fit(
    train_df["text"],
    train_df["class"],
    groups=train_df["group_id"]
)

cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results[
    ["rank_test_score", "mean_test_score", "std_test_score", "params"]
].sort_values("rank_test_score")

cv_results.to_csv(
    RESULTS_PATH / "naive_bayes_tuning_cv_results.csv",
    index=False
)

best_model = search.best_estimator_
predictions = best_model.predict(test_df["text"])

print("\nBest parameters:")
print(search.best_params_)

print(f"\nBest cross-validation macro F1: {search.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(test_df['class'], predictions):.4f}")
print(f"Test macro F1: {f1_score(test_df['class'], predictions, average='macro'):.4f}")
print(f"Test weighted F1: {f1_score(test_df['class'], predictions, average='weighted'):.4f}")

print("\nTest classification report:")
print(classification_report(
    test_df["class"],
    predictions,
    zero_division=0
))

joblib.dump(
    best_model,
    MODELS_PATH / "naive_bayes_tuned_tfidf.pkl"
)

print("\nSaved cross-validation results and tuned model.")
