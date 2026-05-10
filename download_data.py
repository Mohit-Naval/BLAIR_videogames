import os
import requests
from pathlib import Path

BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw"
FILES = {
    "reviews":  "review_categories/Video_Games.jsonl.gz",
    "metadata": "meta_categories/meta_Video_Games.jsonl.gz",
}

Path("data/raw").mkdir(parents=True, exist_ok=True)

for key, path in FILES.items():
    url = f"{BASE_URL}/{path}"
    dest = f"data/raw/{key}_video_games.jsonl.gz"
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping.")
        continue
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="")
    print(f"\nSaved to {dest}")
