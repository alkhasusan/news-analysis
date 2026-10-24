import pandas as pd

originals = pd.read_csv("data/originals.csv")
ai = pd.read_csv("data/ai_rewrites_final.csv")

selected = pd.concat([
    originals[originals["veracity"] == "Real"].sample(250, random_state=42),
    originals[originals["veracity"] == "Fake"].sample(250, random_state=42)
])

selected_ids = set(selected["original_id"])
generated_ids = set(ai["original_id"])

missing = sorted(selected_ids - generated_ids)

print("\n========== DATASET QC ==========")

print(f"\nOriginals selected: {len(selected)}")
print(f"AI rewrites generated: {len(ai)}")

print("\n--- Balance ---")
print(ai["news_label"].value_counts())

print("\n--- Quality ---")
print(ai["quality"].value_counts())

print("\n--- Similarity ---")
print(f"Average: {ai['similarity'].mean():.4f}")
print(f"Median:  {ai['similarity'].median():.4f}")
print(f"Min:     {ai['similarity'].min():.4f}")
print(f"Max:     {ai['similarity'].max():.4f}")

print("\n--- Missing originals ---")
if missing:
    for x in missing:
        print(x)
else:
    print("NONE")

print("\n--- CHECK rows ---")
check = ai[ai["quality"] == "CHECK"]
if len(check):
    print(check[["news_id", "original_id", "news_label", "similarity"]].to_string(index=False))
else:
    print("NONE")

print("\n--- TOO_SIMILAR rows ---")
similar = ai[ai["quality"] == "TOO_SIMILAR"]
if len(similar):
    print(similar[["news_id", "original_id", "news_label", "similarity"]].to_string(index=False))
else:
    print("NONE")

print("\n--- Duplicate original IDs ---")
dupes = ai[ai["original_id"].duplicated(keep=False)]
if len(dupes):
    print(dupes[["news_id", "original_id"]].to_string(index=False))
else:
    print("NONE")

print("\n--- Empty texts ---")
empty = ai[ai["text"].isna() | (ai["text"].str.strip() == "")]
print(f"Empty AI texts: {len(empty)}")

print("\n========== END QC ==========")