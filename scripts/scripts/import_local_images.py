"""从本地图片文件夹秒级生成 products.csv（无需下载）。

把商品图片放进 data/import/ 下任意子文件夹，然后运行:
    python scripts/scripts/import_local_images.py

文件名/文件夹名会用于推断 category，也可在下方 CATEGORIES 里手动映射。
"""

from pathlib import Path
import pandas as pd
import random

IMPORT_DIR = Path("data/import")
IMAGE_DIR = Path("data/images")
OUTPUT_CSV = Path("data/products.csv")

BRANDS = {
    "laptop": ["Dell", "Lenovo", "ASUS", "HP", "Xiaomi"],
    "phone": ["Apple", "Samsung", "Xiaomi", "Huawei"],
    "keyboard": ["Logitech", "Razer", "Keychron"],
    "default": ["Generic"],
}

PRICE = {
    "laptop": (4000, 10000),
    "phone": (2000, 9000),
    "keyboard": (200, 1000),
    "default": (99, 9999),
}

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def guess_category(path: Path) -> str:
    text = f"{path.parent.name} {path.stem}".lower()
    for key in ("laptop", "phone", "keyboard", "headphone", "monitor", "bag", "shoe"):
        if key in text:
            return key
    return "default"


def main() -> None:
    if not IMPORT_DIR.exists():
        IMPORT_DIR.mkdir(parents=True)
        print(f"Created {IMPORT_DIR}")
        print("Put product images there, then run this script again.")
        return

    files = [f for f in IMPORT_DIR.rglob("*") if f.suffix.lower() in SUFFIXES]
    if not files:
        print(f"No images found in {IMPORT_DIR}")
        return

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    products = []

    for idx, src in enumerate(sorted(files), start=1):
        category = guess_category(src)
        brand = random.choice(BRANDS.get(category, BRANDS["default"]))
        low, high = PRICE.get(category, PRICE["default"])
        name = f"{brand} {src.stem.replace('_', ' ').title()}"

        dst = IMAGE_DIR / f"product_{idx}{src.suffix.lower()}"
        dst.write_bytes(src.read_bytes())

        products.append(
            {
                "id": idx,
                "name": name,
                "description": f"{name}, category: {category}",
                "image_path": str(dst).replace("\\", "/"),
                "price": random.randint(low, high),
                "category": category,
            }
        )

    pd.DataFrame(products).to_csv(OUTPUT_CSV, index=False)
    print(f"Done in seconds: {len(products)} products -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
