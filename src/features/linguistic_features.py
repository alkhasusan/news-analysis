import re
import pandas as pd


def extract_features(text):

    text = str(text)

    words = re.findall(r"\b[a-zA-Z]+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    unique_words = len(set(word.lower() for word in words))
    sentence_count = len(sentences)

    if word_count > 0:
        lexical_diversity = unique_words / word_count
        capitalization_ratio = sum(
            1 for c in text if c.isupper()
        ) / sum(
            1 for c in text if c.isalpha()
        )
    else:
        lexical_diversity = 0
        capitalization_ratio = 0

    character_count = len(text)

    if character_count > 0:
        punctuation_ratio = sum(
            1 for c in text if c in ".,!?;:"
        ) / character_count
    else:
        punctuation_ratio = 0

    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
    else:
        avg_sentence_length = 0

    return pd.Series({
        "word_count": word_count,
        "sentence_count": sentence_count,
        "lexical_diversity": lexical_diversity,
        "punctuation_ratio": punctuation_ratio,
        "capitalization_ratio": capitalization_ratio,
        "avg_sentence_length": avg_sentence_length
    })


df = pd.read_csv("data/processed/final_dataset.csv")

features = df["text"].apply(extract_features)

df = pd.concat([df, features], axis=1)

df.to_csv("data/processed/final_dataset_with_features.csv", index=False)

print("\n========== LINGUISTIC FEATURES ==========")

print(
    df[
        [
            "word_count",
            "sentence_count",
            "lexical_diversity",
            "punctuation_ratio",
            "capitalization_ratio",
            "avg_sentence_length"
        ]
    ].describe()
)

print("\n========== FEATURES BY CLASS ==========")

print(
    df.groupby("class")[
        [
            "word_count",
            "sentence_count",
            "lexical_diversity",
            "punctuation_ratio",
            "capitalization_ratio",
            "avg_sentence_length"
        ]
    ].mean().round(4)
)

print("\nSaved to: data/processed/final_dataset_with_features.csv")