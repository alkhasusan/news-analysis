
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("data/processed/final_dataset_with_features.csv")

features = [
    "word_count",
    "sentence_count",
    "lexical_diversity",
    "punctuation_ratio",
    "capitalization_ratio",
    "avg_sentence_length"
]

originals = pd.read_csv("data/processed/originals.csv")
ai = pd.read_csv("data/generated/ai_rewrites_final.csv")

df["group_id"] = df["news_id"]

ai_map = ai.set_index("news_id")["original_id"]
df.loc[df["authorship"] == "AI", "group_id"] = (
    df.loc[df["authorship"] == "AI", "news_id"].map(ai_map)
)

assert df["group_id"].notna().all()

groups = df[["group_id", "news_label"]].drop_duplicates("group_id")

train_groups, test_groups = train_test_split(
    groups["group_id"],
    test_size=0.2,
    random_state=42
)

train_ids = set(train_groups)
test_ids = set(test_groups)

train_df = df[df["group_id"].isin(train_ids)].copy()
test_df = df[df["group_id"].isin(test_ids)].copy()

print("Training class counts:")
print(train_df["class"].value_counts())

print("\nTesting class counts:")
print(test_df["class"].value_counts())

expected_classes = {"Real/Human", "Real/AI", "Fake/Human", "Fake/AI"}
assert set(df["class"].unique()) == expected_classes
assert set(test_df["class"].unique()) == expected_classes

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_text = vectorizer.fit_transform(train_df["text"])
X_test_text = vectorizer.transform(test_df["text"])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[features])
X_test_num = scaler.transform(test_df[features])

minmax = MinMaxScaler()
X_train_nb_num = minmax.fit_transform(train_df[features])
X_test_nb_num = minmax.transform(test_df[features])

X_train_combined = hstack([X_train_text, csr_matrix(X_train_num)])
X_test_combined = hstack([X_test_text, csr_matrix(X_test_num)])

X_train_nb = hstack([X_train_text, csr_matrix(X_train_nb_num)])
X_test_nb = hstack([X_test_text, csr_matrix(X_test_nb_num)])

models = {
    "Naive Bayes": (MultinomialNB(alpha=0.1), X_train_nb, X_test_nb),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_combined,
        X_test_combined
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=20, random_state=42),
        X_train_combined,
        X_test_combined
    )
}

results = []

for name, (model, X_train, X_test) in models.items():
    model.fit(X_train, train_df["class"])
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(test_df["class"], predictions),
        "Weighted Precision": precision_score(
            test_df["class"], predictions,
            average="weighted", zero_division=0
        ),
        "Weighted Recall": recall_score(
            test_df["class"], predictions,
            average="weighted", zero_division=0
        ),
        "Weighted F1": f1_score(
            test_df["class"], predictions,
            average="weighted", zero_division=0
        )
    })

    print(f"\n========== {name.upper()} ==========")
    print(classification_report(
        test_df["class"], predictions, zero_division=0
    ))

results_df = pd.DataFrame(results)
results_df.to_csv("results/linguistic_model_comparison.csv", index=False)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))
print("\nSaved: results/linguistic_model_comparison.csv")
