"""快速构建真实商品图库（并行下载，默认约 2–5 分钟）。

用法（项目根目录）:
    python scripts/scripts/build_real_dataset_fast.py
    python scripts/scripts/build_real_dataset_fast.py --per-category 30
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from io import BytesIO

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"

# 每类用不同 seed，picsum 返回真实照片（非商品专用，但比色块强得多）
CATEGORIES = {
    "laptop": {"price": (4000, 10000), "brands": ["Dell", "Lenovo", "ASUS", "HP", "Xiaomi"]},
    "phone": {"price": (2000, 9000), "brands": ["Apple", "Samsung", "Xiaomi", "Huawei"]},
    "keyboard": {"price": (200, 1000), "brands": ["Logitech", "Razer", "Keychron"]},
}

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 15


def download_one(category: str, index: int) -> dict | None:
    seed = f"{category}-{index}"
    url = f"https://picsum.photos/seed/{seed}/400/400"
    path = IMAGE_DIR / f"{category}_{index}.jpg"

    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(path, quality=85)
    except Exception:
        return None

    meta = CATEGORIES[category]
    brand = random.choice(meta["brands"])
    types = ["Gaming", "Student", "Business", "Pro"]
    name = f"{brand} {random.choice(types)} {category.title()}"

    return {
        "name": name,
        "description": f"{brand} {category}, daily use",
        "image_path": str(path).replace("\\", "/"),
        "price": random.randint(*meta["price"]),
        "category": category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=20, help="每类商品数，默认 20")
    parser.add_argument("--workers", type=int, default=12, help="并行下载数")
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(cat, i) for cat in CATEGORIES for i in range(args.per_category)]

    print(f"Downloading {len(tasks)} images ({args.workers} workers)...")

    products = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, c, i): (c, i) for c, i in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            done += 1
            if row:
                products.append(row)
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}")

    for pid, row in enumerate(products, start=1):
        row["id"] = pid

    df = pd.DataFrame(products)
    df.to_csv(DATA_DIR / "products.csv", index=False)

    print(f"\nDone: {len(products)} products in {IMAGE_DIR}")
    print("Restart API to reload index.")


if __name__ == "__main__":
    main()
