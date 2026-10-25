#grouped_evaluation.py
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("data/final_dataset.csv")

group_info = df.groupby("news_id").first().reset_index()

train_groups, test_groups = train_test_split(
    group_info,
    test_size=0.2,
    random_state=42,
    stratify=group_info["news_label"]
)

train_ids = set(train_groups["news_id"])
test_ids = set(test_groups["news_id"])

train_df = df[df["news_id"].isin(train_ids)]
test_df = df[df["news_id"].isin(test_ids)]

X_train = train_df["text"]
y_train = train_df["authorship"]

X_test = test_df["text"]
y_test = test_df["authorship"]

print("========== SPLIT ==========")
print(f"Training articles: {len(train_df)}")
print(f"Testing articles:  {len(test_df)}")

print("\nTraining classes:")
print(y_train.value_counts())

print("\nTesting classes:")
print(y_test.value_counts())

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="AI")
recall = recall_score(y_test, y_pred, pos_label="AI")
f1 = f1_score(y_test, y_pred, pos_label="AI")

print("\n========== ORIGINAL-AWARE RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))