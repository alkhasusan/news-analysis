import os
import time
import random
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# SETUP
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)

DATA = Path("data")

INPUT_FILE = DATA / "originals.csv"
OUTPUT_FILE = DATA / "ai_rewrites_final.csv"

MODEL = "gemini-3.5-flash-lite"

TARGET_REAL = 250
TARGET_FAKE = 250

MAX_RETRIES = 5


# =========================================================
# LOAD ORIGINALS
# =========================================================

print("=" * 70)
print("LOADING ORIGINAL DATA")
print("=" * 70)

originals = pd.read_csv(INPUT_FILE)

originals["original_id"] = (
    originals["original_id"]
    .astype(str)
)

originals["veracity"] = (
    originals["veracity"]
    .astype(str)
    .str.strip()
)

print(f"Total originals: {len(originals)}")

real = originals[
    originals["veracity"].str.lower() == "real"
].sample(
    n=TARGET_REAL,
    random_state=42
)

fake = originals[
    originals["veracity"].str.lower() == "fake"
].sample(
    n=TARGET_FAKE,
    random_state=42
)

selected = pd.concat(
    [real, fake],
    ignore_index=True
)

selected = selected.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(f"Selected Real: {len(real)}")
print(f"Selected Fake: {len(fake)}")
print(f"Total selected: {len(selected)}")


# =========================================================
# LOAD EXISTING PROGRESS
# =========================================================

if OUTPUT_FILE.exists():

    results = pd.read_csv(OUTPUT_FILE)

    completed_ids = set(
        results["original_id"]
        .astype(str)
    )

    print()
    print("=" * 70)
    print("RESUMING EXISTING RUN")
    print("=" * 70)

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

else:

    results = pd.DataFrame(
        columns=[
            "news_id",
            "original_id",
            "authorship",
            "news_label",
            "subject",
            "date",
            "text",
            "similarity",
            "quality"
        ]
    )

    completed_ids = set()


# =========================================================
# PROMPT
# =========================================================

def build_prompt(article, veracity):

    return f"""
You are rewriting a news article for a research dataset.

The original article is classified as: {veracity}

Rewrite this article so that it reads like it was naturally written
by a different human journalist.

STRICT RULES:

1. Preserve the original meaning.
2. Preserve the original factual claims.
3. Preserve the original veracity.
4. A Real article must remain Real.
5. A Fake article must remain Fake.
6. Do NOT invent facts.
7. Do NOT add names, dates, numbers, quotes, locations, events,
   statistics, or claims that are not present in the original.
8. Do NOT remove important factual information.
9. Change the wording substantially.
10. Change sentence structures substantially.
11. You may naturally reorganize sentences and paragraphs.
12. Do NOT summarize the article.
13. Do NOT drastically shorten the article.
14. Do NOT mention AI.
15. Do NOT explain the rewriting process.
16. Do NOT include a title unless the original article itself contains
   the title as part of the supplied text.
17. Output ONLY the rewritten article.

ORIGINAL ARTICLE:

{article}
"""


# =========================================================
# GENERATE ONE ARTICLE
# =========================================================

def generate_rewrite(article, veracity):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=build_prompt(
                    article,
                    veracity
                )
            )

            ai_text = (
                response.text.strip()
                if response.text
                else ""
            )

            if not ai_text:
                raise ValueError(
                    "Empty response."
                )

            if len(ai_text) < 200:
                raise ValueError(
                    "Response is suspiciously short."
                )

            return ai_text

        except Exception as e:

            print(
                f"    Attempt {attempt}/{MAX_RETRIES} failed:"
            )

            print(
                f"    {str(e)[:250]}"
            )

            if attempt < MAX_RETRIES:

                wait_time = min(
                    60,
                    5 * (2 ** (attempt - 1))
                )

                wait_time += random.uniform(0, 3)

                print(
                    f"    Retrying in "
                    f"{wait_time:.1f} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "    Giving up on this article."
                )

    return None


# =========================================================
# SIMILARITY
# =========================================================

def calculate_similarity(original, rewritten):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform([
        original,
        rewritten
    ])

    return cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]


def classify_quality(score):

    if score >= 0.80:
        return "TOO_SIMILAR"

    if score >= 0.60:
        return "CHECK"

    return "GOOD"


# =========================================================
# GENERATION LOOP
# =========================================================

print()
print("=" * 70)
print("STARTING GENERATION")
print("=" * 70)

for index, row in selected.iterrows():

    original_id = str(row["original_id"])

    if original_id in completed_ids:

        continue

    veracity = row["veracity"]

    article = str(row["text"])

    completed_count = len(completed_ids)

    print()
    print("=" * 70)

    print(
        f"Progress: "
        f"{completed_count}/{len(selected)}"
    )

    print(
        f"Article: {original_id}"
    )

    print(
        f"Veracity: {veracity}"
    )

    ai_text = generate_rewrite(
        article,
        veracity
    )

    if ai_text is None:

        print(
            "    FAILED — moving to next article."
        )

        continue

    similarity = calculate_similarity(
        article,
        ai_text
    )

    quality = classify_quality(
        similarity
    )

    # Only AI-authored article receives final news_id
    news_id = f"AI_{len(results) + 1:04d}"

    new_row = {
        "news_id": news_id,
        "original_id": original_id,
        "authorship": "AI",
        "news_label": veracity,
        "subject": row["subject"],
        "date": row["date"],
        "text": ai_text,
        "similarity": similarity,
        "quality": quality
    }

    results = pd.concat(
        [
            results,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )

    completed_ids.add(
        original_id
    )

    # SAVE IMMEDIATELY
    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        f"    Generated successfully."
    )

    print(
        f"    Original length: "
        f"{len(article)}"
    )

    print(
        f"    AI length: "
        f"{len(ai_text)}"
    )

    print(
        f"    Similarity: "
        f"{similarity:.4f}"
    )

    print(
        f"    Quality: "
        f"{quality}"
    )

    # Small delay between successful requests
    time.sleep(1)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)

if len(results) > 0:

    print(
        f"Total generated: {len(results)}"
    )

    print()
    print(
        results["news_label"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        results["quality"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Average similarity: "
        f"{results['similarity'].mean():.4f}"
    )

    print(
        f"Median similarity: "
        f"{results['similarity'].median():.4f}"
    )

else:

    print("No articles generated.")

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)