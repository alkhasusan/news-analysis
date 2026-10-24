
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "final_dataset_with_features.csv"
AI_PATH = ROOT / "data" / "generated" / "ai_rewrites_final.csv"
OUTPUT_DIR = ROOT / "results" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text"
TARGET_COL = "class"

LINGUISTIC_FEATURES = [
    "word_count",
    "sentence_count",
    "lexical_diversity",
    "punctuation_ratio",
    "capitalization_ratio",
    "avg_sentence_length"
]

df = pd.read_csv(DATA_PATH)
ai_df = pd.read_csv(AI_PATH)

df["news_id"] = df["news_id"].astype(str)
ai_df["news_id"] = ai_df["news_id"].astype(str)
ai_df["original_id"] = ai_df["original_id"].astype(str)

ai_to_original = dict(
    zip(ai_df["news_id"], ai_df["original_id"])
)

df["group_id"] = (
    df["news_id"]
    .map(ai_to_original)
    .fillna(df["news_id"])
)

df = df.dropna(
    subset=[TEXT_COL, TARGET_COL] + LINGUISTIC_FEATURES
).reset_index(drop=True)

groups = df["group_id"].unique()

train_groups, test_groups = train_test_split(
    groups,
    test_size=0.2,
    random_state=42
)

train_mask = df["group_id"].isin(train_groups)
test_mask = df["group_id"].isin(test_groups)

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_text = vectorizer.fit_transform(train_df[TEXT_COL])
X_test_text = vectorizer.transform(test_df[TEXT_COL])

scaler = StandardScaler()

X_train_ling = scaler.fit_transform(
    train_df[LINGUISTIC_FEATURES]
)

X_test_ling = scaler.transform(
    test_df[LINGUISTIC_FEATURES]
)

X_train = hstack([
    X_train_text,
    csr_matrix(X_train_ling)
]).tocsr()

X_test = hstack([
    X_test_text,
    csr_matrix(X_test_ling)
]).tocsr()

y_train = train_df[TARGET_COL].astype(str)
y_test = test_df[TARGET_COL].astype(str)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

feature_names = (
    vectorizer.get_feature_names_out().tolist()
    + LINGUISTIC_FEATURES
)

rng = np.random.default_rng(42)

background_indices = rng.choice(
    X_train.shape[0],
    size=min(50, X_train.shape[0]),
    replace=False
)

explain_indices = rng.choice(
    X_test.shape[0],
    size=min(50, X_test.shape[0]),
    replace=False
)

background = X_train[background_indices].toarray()
X_explain = X_test[explain_indices].toarray()

explainer = shap.LinearExplainer(
    model,
    background
)

raw_shap_values = explainer.shap_values(X_explain)

if isinstance(raw_shap_values, list):
    shap_by_class = np.stack(raw_shap_values, axis=-1)
else:
    shap_by_class = np.asarray(raw_shap_values)

if shap_by_class.ndim != 3:
    raise ValueError(
        f"Unexpected SHAP output shape: {shap_by_class.shape}"
    )

if shap_by_class.shape[1] != len(feature_names):
    if shap_by_class.shape[0] == len(model.classes_):
        shap_by_class = np.moveaxis(shap_by_class, 0, -1)
    else:
        raise ValueError(
            f"Cannot match SHAP dimensions: {shap_by_class.shape}"
        )

if shap_by_class.shape[-1] != len(model.classes_):
    raise ValueError(
        f"Unexpected class dimension: {shap_by_class.shape}"
    )

ling_start = len(feature_names) - len(LINGUISTIC_FEATURES)

summary_rows = []

for class_idx, class_name in enumerate(model.classes_):
    class_values = shap_by_class[:, :, class_idx]
    ling_values = class_values[:, ling_start:]

    for feature_idx, feature_name in enumerate(LINGUISTIC_FEATURES):
        summary_rows.append({
            "class": class_name,
            "feature": feature_name,
            "mean_abs_shap": np.mean(
                np.abs(ling_values[:, feature_idx])
            )
        })

    order = np.argsort(
        np.mean(np.abs(ling_values), axis=0)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(
        np.array(LINGUISTIC_FEATURES)[order],
        np.mean(np.abs(ling_values), axis=0)[order]
    )
    plt.xlabel("Mean absolute SHAP value (log-odds scale)")
    plt.ylabel("Linguistic feature")
    plt.title(f"SHAP importance: {class_name}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"shap_linguistic_{class_name.replace('/', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTPUT_DIR / "shap_linguistic_importance.csv",
    index=False
)

overall = (
    summary_df.groupby("feature")["mean_abs_shap"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 5))
plt.barh(overall.index, overall.values)
plt.xlabel("Mean absolute SHAP value (log-odds scale)")
plt.ylabel("Linguistic feature")
plt.title("Overall SHAP importance across four classes")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_overall_linguistic_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("SHAP analysis completed.")
print(f"Training articles: {len(train_df)}")
print(f"Test articles: {len(test_df)}")
print(f"Explained test articles: {len(explain_indices)}")
print(f"Classes: {list(model.classes_)}")
print(f"Results saved to: {OUTPUT_DIR}")
print(summary_df.to_string(index=False))
