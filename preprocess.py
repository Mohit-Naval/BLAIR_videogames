import gzip
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

TRAIN_TS = 1_640_000_000
VAL_TS = 1_672_000_000
MIN_CHARS = 30


def load_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_metadata_dict(meta_path):
    meta = {}
    for item in tqdm(load_jsonl_gz(meta_path), desc="Loading metadata"):
        asin = item.get("parent_asin") or item.get("asin", "")
        parts = []
        if item.get("title"):
            parts.append(item["title"])
        if item.get("features"):
            parts.append(" ".join(item["features"][:3]))
        if item.get("description"):
            desc = item["description"]
            if isinstance(desc, list):
                desc = " ".join(desc)
            parts.append(desc[:300])
        metadata_str = " | ".join(p.strip() for p in parts if p.strip())
        if len(metadata_str) >= MIN_CHARS:
            meta[asin] = metadata_str
    return meta


def build_review_pairs(reviews_path, meta_dict):
    train, val, test = [], [], []
    user_histories = {}

    for rev in tqdm(load_jsonl_gz(reviews_path), desc="Processing reviews"):
        asin = rev.get("parent_asin") or rev.get("asin", "")
        if asin not in meta_dict:
            continue

        ts = rev.get("timestamp", 0)

        if ts > 1_700_000_000_000:
            ts = ts / 1000
        elif ts > 1_700_000_000:
            pass

        review_text = f"{rev.get('title', '').strip()} {rev.get('text', '').strip()}".strip(
        )
        if len(review_text) < MIN_CHARS:
            continue

        pair = {
            "user_id":   rev.get("user_id", ""),
            "asin":      asin,
            "review":    review_text,
            "metadata":  meta_dict[asin],
            "timestamp": ts,
            "rating":    rev.get("rating", 0),
        }

        if ts < TRAIN_TS:
            train.append(pair)
        elif ts < VAL_TS:
            val.append(pair)
        else:
            test.append(pair)

        uid = rev.get("user_id", "")
        if uid:
            user_histories.setdefault(uid, []).append(
                {"asin": asin, "timestamp": ts}
            )

    return train, val, test, user_histories


def run_preprocessing():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Building metadata dict...")
    meta = build_metadata_dict("data/raw/metadata_video_games.jsonl.gz")
    print(f"  {len(meta):,} items with valid metadata")

    print("Building review pairs...")
    train, val, test, histories = build_review_pairs(
        "data/raw/reviews_video_games.jsonl.gz", meta
    )
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    for name, data in [("train", train), ("val", val), ("test", test)]:
        pd.DataFrame(data).to_parquet(
            f"data/processed/{name}.parquet", index=False)

    sorted_histories = {
        uid: sorted(items, key=lambda x: x["timestamp"])
        for uid, items in histories.items()
        if len(items) >= 3
    }
    pd.to_pickle(sorted_histories, "data/processed/user_histories.pkl")
    pd.to_pickle(meta,            "data/processed/metadata.pkl")
    print("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
