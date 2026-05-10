from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import json


print("Loading data...")
products = pd.read_parquet("products_nlp.parquet")
reviews = pd.read_parquet("reviews_nlp.parquet")
rich_docs = pd.read_parquet("products_rich.parquet")

print(f"  Products:  {len(products):,}")
print(f"  Reviews:   {len(reviews):,}")
print(f"  Rich docs: {len(rich_docs):,}")


rich_lookup = dict(zip(rich_docs.parent_asin, rich_docs.rich_text))
print(f"  Rich lookup: {len(rich_lookup):,} items")


print("Pre-grouping reviews by ASIN...")
review_groups = {
    asin: group
    for asin, group in reviews.groupby("parent_asin")
}
print(f"  {len(review_groups):,} products have reviews")


ASPECT_COLS = {
    'gameplay': ['gameplay', 'mechanics', 'combat', 'fun',
                 'missions', 'levels', 'difficulty', 'progression'],
    'graphics': ['graphics', 'visuals', 'animation', 'textures',
                 'beautiful', 'stunning', 'art', 'lighting'],
    'story':    ['story', 'plot', 'narrative', 'character',
                 'ending', 'writing', 'dialogue', 'lore'],
    'controls': ['controls', 'controller', 'responsive', 'lag',
                 'button', 'keyboard', 'mouse', 'sensitivity'],
    'value':    ['price', 'worth', 'money', 'expensive', 'cheap',
                 'value', 'cost', 'dlc', 'content', 'hours'],
}


print("Pre-building product lookup...")
product_lookup = {
    row.parent_asin: row
    for row in products.itertuples()
}


def contains_aspect_keywords(text, keywords):
    text_lower = str(text).lower()
    return any(kw in text_lower for kw in keywords)


def build_pairs():
    pairs = []
    skipped = 0

    for asin in tqdm(products.parent_asin.unique(), desc="Building pairs"):

        if asin not in rich_lookup:
            skipped += 1
            continue
        anchor = rich_lookup[asin]

        product_reviews = review_groups.get(asin, None)
        if product_reviews is None or len(product_reviews) == 0:
            skipped += 1
            continue

        for aspect, keywords in ASPECT_COLS.items():
            mask = product_reviews["text"].apply(
                lambda t: contains_aspect_keywords(t, keywords)
            )
            aspect_revs = product_reviews[mask]

            if len(aspect_revs) >= 3:
                top_revs = aspect_revs.nlargest(5, "helpful_vote")
                positive = " ".join(top_revs["text"].fillna("").tolist())
                positive = positive[:1000]
                pairs.append({
                    "anchor":    anchor,
                    "positive":  positive,
                    "pair_type": f"aspect_{aspect}",
                    "asin":      asin,
                })

        all_text = " ".join(
            product_reviews.nlargest(10, "helpful_vote")
            ["text"].fillna("").tolist()
        )[:1500]
        pairs.append({
            "anchor":    anchor,
            "positive":  all_text,
            "pair_type": "all_reviews",
            "asin":      asin,
        })

        product_row = product_lookup.get(asin)
        if product_row is not None:
            metadata_pos = (
                f"Title: {getattr(product_row, 'title', '')}. "
                f"Category: {getattr(product_row, 'full_category_path', '')}. "
                f"Brand: {getattr(product_row, 'store', '')}."
            )
        else:
            metadata_pos = f"Title: unknown. Category: Video Games."

        pairs.append({
            "anchor":    anchor,
            "positive":  metadata_pos,
            "pair_type": "metadata",
            "asin":      asin,
        })

    print(f"  Skipped {skipped:,} products (no rich doc or no reviews)")
    return pairs


pairs = build_pairs()
print(f"\nTotal pairs built: {len(pairs):,}")


print("Saving training_pairs.jsonl...")
with open("training_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print("Saved training_pairs.jsonl")
print("\nPair counts by type:")
types = Counter(p["pair_type"] for p in pairs)
for t, c in sorted(types.items()):
    print(f"  {t}: {c:,}")
print(f"\nTotal: {sum(types.values()):,} training pairs")
