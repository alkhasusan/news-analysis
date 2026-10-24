
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("data/processed/final_dataset.csv")
ai = pd.read_csv("data/generated/ai_rewrites_final.csv")

df["group_id"] = df["news_id"]
ai_map = ai.set_index("news_id")["original_id"]

mask = df["authorship"] == "AI"
df.loc[mask, "group_id"] = df.loc[mask, "news_id"].map(ai_map)

assert df["group_id"].notna().all()

groups = df[["group_id"]].drop_duplicates()

train_groups, test_groups = train_test_split(
    groups["group_id"],
    test_size=0.2,
    random_state=42
)

train_ids = set(train_groups)
test_ids = set(test_groups)

train_df = df[df["group_id"].isin(train_ids)]
test_df = df[df["group_id"].isin(test_ids)]

print("Training classes:")
print(train_df["class"].value_counts())
print("\nTesting classes:")
print(test_df["class"].value_counts())

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42)
}

results = []

for name, model in models.items():
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
results_df.to_csv("results/tfidf_only_comparison.csv", index=False)

print("\n========== TF-IDF-ONLY COMPARISON ==========")
print(results_df.round(4).to_string(index=False))
print("\nSaved: results/tfidf_only_comparison.csv")
