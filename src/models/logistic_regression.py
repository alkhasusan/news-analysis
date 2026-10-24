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

originals = pd.read_csv("data/originals.csv")
ai = pd.read_csv("data/ai_rewrites_final.csv")

human = pd.DataFrame({
    "group_id": originals["original_id"],
    "text": originals["text"],
    "class": originals["veracity"] + "/Human"
})

ai_data = pd.DataFrame({
    "group_id": ai["original_id"],
    "text": ai["text"],
    "class": ai["news_label"] + "/AI"
})

df = pd.concat([human, ai_data], ignore_index=True)

groups = df["group_id"].drop_duplicates()

train_groups, test_groups = train_test_split(
    groups,
    test_size=0.2,
    random_state=42
)

train_ids = set(train_groups)
test_ids = set(test_groups)

train_df = df[df["group_id"].isin(train_ids)]
test_df = df[df["group_id"].isin(test_ids)]

X_train = train_df["text"]
y_train = train_df["class"]

X_test = test_df["text"]
y_test = test_df["class"]

print("========== FOUR-CLASS SPLIT ==========")
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

print("\n========== LOGISTIC REGRESSION RESULTS ==========")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")

print(
    f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
)

print(
    f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
)

print(
    f"F1 Score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\n========== CONFUSION MATRIX ==========")

labels = [
    "Real/Human",
    "Real/AI",
    "Fake/Human",
    "Fake/AI"
]

print(
    pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=labels,
        columns=labels
    )
)