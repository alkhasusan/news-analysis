import pandas as pd

originals = pd.read_csv("data/originals.csv")
ai = pd.read_csv("data/ai_rewrites_final.csv")

human = pd.DataFrame({
    "news_id": originals["original_id"],
    "authorship": "Human",
    "news_label": originals["veracity"],
    "subject": originals["subject"],
    "date": originals["date"],
    "text": originals["text"]
})

ai_final = ai[[
    "news_id",
    "authorship",
    "news_label",
    "subject",
    "date",
    "text"
]].copy()

final = pd.concat([human, ai_final], ignore_index=True)

final["class"] = (
    final["news_label"] + "/" + final["authorship"]
)

final.to_csv("data/final_dataset.csv", index=False)

print("\n========== FINAL DATASET ==========")
print(f"Total articles: {len(final)}")

print("\n--- Authorship ---")
print(final["authorship"].value_counts())

print("\n--- Veracity ---")
print(final["news_label"].value_counts())

print("\n--- Classes ---")
print(final["class"].value_counts())

print("\n--- Missing values ---")
print(final.isna().sum())

print("\nSaved to: data/final_dataset.csv")