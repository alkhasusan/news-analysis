
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
FEATURE_PATH = ROOT / "data" / "processed" / "final_dataset_with_features.csv"
AI_PATH = ROOT / "data" / "generated" / "ai_rewrites_final.csv"
RESULTS_PATH = ROOT / "results"
MODELS_PATH = ROOT / "models"
MATRIX_PATH = RESULTS_PATH / "confusion_matrices"
FEATURE_PLOT_PATH = RESULTS_PATH / "feature_analysis"

MATRIX_PATH.mkdir(parents=True, exist_ok=True)
FEATURE_PLOT_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_df = pd.read_csv(FEATURE_PATH)
ai_df = pd.read_csv(AI_PATH)

feature_cols = [
    "word_count",
    "sentence_count",
    "lexical_diversity",
    "punctuation_ratio",
    "capitalization_ratio",
    "avg_sentence_length"
]

feature_labels = [
    "Word count",
    "Sentence count",
    "Lexical diversity",
    "Punctuation ratio",
    "Capitalization ratio",
    "Average sentence length"
]

ai_mapping = (
    ai_df.drop_duplicates("news_id")
    .set_index("news_id")["original_id"]
    .to_dict()
)

df["group_id"] = df["news_id"]
ai_mask = df["authorship"].str.lower().eq("ai")
df.loc[ai_mask, "group_id"] = df.loc[ai_mask, "news_id"].map(ai_mapping)

if df["group_id"].isna().any():
    raise ValueError("Some AI articles could not be matched to their originals.")

groups = df["group_id"].drop_duplicates().to_numpy()
train_groups, test_groups = train_test_split(
    groups,
    test_size=0.2,
    random_state=42
)

train_df = df[df["group_id"].isin(train_groups)].copy().reset_index(drop=True)
test_df = df[df["group_id"].isin(test_groups)].copy().reset_index(drop=True)

feature_df = feature_df.set_index("news_id")

train_features = feature_df.loc[
    train_df["news_id"], feature_cols
].to_numpy()

test_features = feature_df.loc[
    test_df["news_id"], feature_cols
].to_numpy()

y_train = train_df["class"]
y_test = test_df["class"]

classes = ["Fake/AI", "Fake/Human", "Real/AI", "Real/Human"]

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_text = vectorizer.fit_transform(train_df["text"])
X_test_text = vectorizer.transform(test_df["text"])

results = []
fitted_models = {}
predictions_by_model = {}

def evaluate(name, feature_type, model, X_train, X_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Features": feature_type,
        "Accuracy": accuracy_score(y_test, predictions),
        "Weighted Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Weighted Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
    })

    cm = confusion_matrix(y_test, predictions, labels=classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} - {feature_type}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    filename = (
        f"{name.lower().replace(' ', '_')}_"
        f"{feature_type.lower().replace(' ', '_').replace('+', 'plus')}.png"
    )
    fig.savefig(MATRIX_PATH / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{name} | {feature_type}")
    print(classification_report(
        y_test, predictions, labels=classes, zero_division=0
    ))

    fitted_models[(name, feature_type)] = model
    predictions_by_model[(name, feature_type)] = predictions
    return model


tfidf_models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20, random_state=42
    )
}

for name, model in tfidf_models.items():
    evaluate(name, "TF-IDF", model, X_train_text, X_test_text)

standard_scaler = StandardScaler()
train_standard = standard_scaler.fit_transform(train_features)
test_standard = standard_scaler.transform(test_features)

minmax_scaler = MinMaxScaler()
train_minmax = minmax_scaler.fit_transform(train_features)
test_minmax = minmax_scaler.transform(test_features)

X_train_combined_nb = hstack([
    X_train_text,
    csr_matrix(train_minmax)
])
X_test_combined_nb = hstack([
    X_test_text,
    csr_matrix(test_minmax)
])

X_train_combined_standard = hstack([
    X_train_text,
    csr_matrix(train_standard)
])
X_test_combined_standard = hstack([
    X_test_text,
    csr_matrix(test_standard)
])

combined_models = {
    "Naive Bayes": (
        X_train_combined_nb,
        X_test_combined_nb,
        MultinomialNB(alpha=0.1)
    ),
    "Logistic Regression": (
        X_train_combined_standard,
        X_test_combined_standard,
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Decision Tree": (
        X_train_combined_standard,
        X_test_combined_standard,
        DecisionTreeClassifier(max_depth=20, random_state=42)
    )
}

for name, (X_train, X_test, model) in combined_models.items():
    evaluate(name, "TF-IDF + Linguistic", model, X_train, X_test)

tuned_path = MODELS_PATH / "naive_bayes_tuned_tfidf.pkl"

if tuned_path.exists():
    tuned_model = joblib.load(tuned_path)
    tuned_predictions = tuned_model.predict(test_df["text"])

    results.append({
        "Model": "Naive Bayes (Tuned)",
        "Features": "TF-IDF",
        "Accuracy": accuracy_score(y_test, tuned_predictions),
        "Weighted Precision": precision_score(
            y_test, tuned_predictions, average="weighted", zero_division=0
        ),
        "Weighted Recall": recall_score(
            y_test, tuned_predictions, average="weighted", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test, tuned_predictions, average="weighted", zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, tuned_predictions, average="macro", zero_division=0
        )
    })

    cm = confusion_matrix(y_test, tuned_predictions, labels=classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Naive Bayes Tuned - TF-IDF")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        MATRIX_PATH / "naive_bayes_tuned_tfidf.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nNaive Bayes (Tuned) | TF-IDF")
    print(classification_report(
        y_test, tuned_predictions, labels=classes, zero_division=0
    ))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Weighted F1", ascending=False)

results_df.to_csv(
    RESULTS_PATH / "final_model_comparison.csv",
    index=False
)

print("\nFINAL RESULTS")
print(results_df.round(4).to_string(index=False))


feature_index = {label: i for i, label in enumerate(feature_labels)}

class_means = pd.DataFrame(
    train_minmax,
    columns=feature_labels
)
class_means["class"] = y_train.to_numpy()

class_means = (
    class_means.groupby("class")[feature_labels]
    .mean()
    .reindex(classes)
)

class_means.to_csv(
    FEATURE_PLOT_PATH / "naive_bayes_class_feature_means.csv"
)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(feature_labels))
width = 0.19

for i, class_name in enumerate(classes):
    ax.bar(
        x + (i - 1.5) * width,
        class_means.loc[class_name].values,
        width,
        label=class_name
    )

ax.set_xticks(x, feature_labels, rotation=25, ha="right")
ax.set_ylabel("Mean Min-Max-Scaled Feature Value")
ax.set_title("Naive Bayes: Linguistic Feature Means by Class")
ax.legend()
fig.tight_layout()
fig.savefig(
    FEATURE_PLOT_PATH / "figure6a_naive_bayes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


lr_model = fitted_models[("Logistic Regression", "TF-IDF + Linguistic")]
lr_coefficients = lr_model.coef_[:, -len(feature_cols):]

lr_df = pd.DataFrame(
    lr_coefficients,
    index=lr_model.classes_,
    columns=feature_labels
)
lr_df.to_csv(
    FEATURE_PLOT_PATH / "logistic_regression_linguistic_coefficients.csv"
)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(feature_labels))
width = 0.19

for i, class_name in enumerate(lr_model.classes_):
    ax.bar(
        x + (i - 1.5) * width,
        lr_df.loc[class_name].values,
        width,
        label=class_name
    )

ax.axhline(0, linewidth=0.8)
ax.set_xticks(x, feature_labels, rotation=25, ha="right")
ax.set_ylabel("Coefficient (Standardized Features)")
ax.set_title("Logistic Regression: Linguistic Feature Coefficients")
ax.legend()
fig.tight_layout()
fig.savefig(
    FEATURE_PLOT_PATH / "figure6b_logistic_regression.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


dt_model = fitted_models[("Decision Tree", "TF-IDF + Linguistic")]
dt_importance = dt_model.feature_importances_[-len(feature_cols):]

dt_df = pd.DataFrame({
    "Feature": feature_labels,
    "Importance": dt_importance
}).sort_values("Importance", ascending=True)

dt_df.to_csv(
    FEATURE_PLOT_PATH / "decision_tree_linguistic_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(dt_df["Feature"], dt_df["Importance"])
ax.set_xlabel("Impurity-Based Feature Importance")
ax.set_title("Decision Tree: Linguistic Feature Importance")
fig.tight_layout()
fig.savefig(
    FEATURE_PLOT_PATH / "figure6c_decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(
    ["Naive Bayes", "Logistic Regression", "Decision Tree"],
    [
        results_df.loc[
            (results_df["Model"] == "Naive Bayes") &
            (results_df["Features"] == "TF-IDF"), "Accuracy"
        ].iloc[0] * 100,
        results_df.loc[
            (results_df["Model"] == "Logistic Regression") &
            (results_df["Features"] == "TF-IDF"), "Accuracy"
        ].iloc[0] * 100,
        results_df.loc[
            (results_df["Model"] == "Decision Tree") &
            (results_df["Features"] == "TF-IDF"), "Accuracy"
        ].iloc[0] * 100
    ]
)
axes[0].set_title("TF-IDF Only")
axes[0].set_ylabel("Accuracy (%)")
axes[0].tick_params(axis="x", rotation=25)

combined_accuracy = []
for model_name in ["Naive Bayes", "Logistic Regression", "Decision Tree"]:
    value = results_df.loc[
        (results_df["Model"] == model_name) &
        (results_df["Features"] == "TF-IDF + Linguistic"),
        "Accuracy"
    ].iloc[0] * 100
    combined_accuracy.append(value)

axes[1].bar(
    ["Naive Bayes", "Logistic Regression", "Decision Tree"],
    combined_accuracy
)
axes[1].set_title("TF-IDF + Linguistic")
axes[1].set_ylabel("Accuracy (%)")
axes[1].tick_params(axis="x", rotation=25)

model_names = ["Naive Bayes", "Logistic Regression", "Decision Tree"]
tfidf_accuracy = [
    results_df.loc[
        (results_df["Model"] == name) &
        (results_df["Features"] == "TF-IDF"), "Accuracy"
    ].iloc[0] * 100
    for name in model_names
]

x = np.arange(len(model_names))
width = 0.36
axes[2].bar(x - width / 2, tfidf_accuracy, width, label="TF-IDF")
axes[2].bar(x + width / 2, combined_accuracy, width, label="TF-IDF + Linguistic")
axes[2].set_xticks(x, model_names, rotation=25)
axes[2].set_title("Feature Comparison")
axes[2].set_ylabel("Accuracy (%)")
axes[2].legend()

fig.tight_layout()
fig.savefig(
    RESULTS_PATH / "figure4_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\nSaved Figure 4:")
print(RESULTS_PATH / "figure4_accuracy_comparison.png")

print("\nSaved Figure 6 diagrams and data:")
print(FEATURE_PLOT_PATH)
print("\nSaved confusion matrices:")
print(MATRIX_PATH)
print("\nSaved comparison table:")
print(RESULTS_PATH / "final_model_comparison.csv")
