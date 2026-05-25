"""一键构建商品库：自动生成图片 + products.csv。

完全离线，不依赖京东爬虫、Hugging Face 或 Stable Diffusion。
适合本地开发多模态电商系统（CLIP 搜图 + LLM 推荐）。

用法（在项目根目录）:
    python scripts/scripts/build_product_library.py
    python scripts/scripts/build_product_library.py --per-category 100
    python scripts/scripts/build_product_library.py --categories laptop phone
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# =========================
# 商品模板
# =========================

CATALOG = {
    "laptop": {
        "brands": ["Dell", "Lenovo", "ASUS", "HP", "Xiaomi"],
        "types": ["Gaming", "Student", "Business"],
        "price_range": (4000, 10000),
        "hue": 120,
    },
    "phone": {
        "brands": ["Apple", "Samsung", "Xiaomi", "Huawei"],
        "types": ["Flagship", "Gaming", "Camera"],
        "price_range": (2000, 9000),
        "hue": 280,
    },
    "keyboard": {
        "brands": ["Logitech", "Razer", "Keychron"],
        "types": ["Mechanical", "Wireless"],
        "price_range": (200, 1000),
        "hue": 30,
    },
    "headphone": {
        "brands": ["Sony", "Bose", "Apple", "Sennheiser"],
        "types": ["Wireless", "Noise Cancelling", "Gaming"],
        "price_range": (300, 3000),
        "hue": 200,
    },
    "monitor": {
        "brands": ["Dell", "LG", "Samsung", "AOC"],
        "types": ["4K", "Gaming", "Office"],
        "price_range": (800, 5000),
        "hue": 180,
    },
}

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"


def _color(seed: int, hue_shift: int, sat: int = 140, light: int = 110) -> tuple[int, int, int]:
    """由 seed 生成稳定但不同的颜色。"""
    h = (hue_shift + seed * 37) % 360
    # 简易 HSL -> RGB
    c = (1 - abs(2 * light / 255 - 1)) * sat / 255
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = light / 255 - c / 2
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255),
    )


def _draw_product_shape(
    draw: ImageDraw.ImageDraw,
    category: str,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0

    if category == "laptop":
        screen_h = int(h * 0.55)
        draw.rounded_rectangle([x0, y0, x1, y0 + screen_h], radius=12, fill=fill, outline=accent, width=3)
        draw.rectangle([x0 + 8, y0 + screen_h, x1 - 8, y1], fill=accent)
        draw.line([x0 + 20, y0 + screen_h + 8, x1 - 20, y0 + screen_h + 8], fill=(255, 255, 255), width=2)

    elif category == "phone":
        draw.rounded_rectangle(box, radius=28, fill=fill, outline=accent, width=4)
        cam_y = y0 + int(h * 0.08)
        draw.ellipse([x0 + w // 2 - 8, cam_y, x0 + w // 2 + 8, cam_y + 16], fill=accent)

    elif category == "keyboard":
        draw.rounded_rectangle(box, radius=10, fill=fill, outline=accent, width=3)
        for row in range(4):
            for col in range(12):
                kw = w // 14
                kh = h // 6
                kx = x0 + 10 + col * (kw + 2)
                ky = y0 + 10 + row * (kh + 3)
                draw.rounded_rectangle([kx, ky, kx + kw, ky + kh], radius=2, fill=accent)

    elif category == "headphone":
        band_y = y0 + int(h * 0.15)
        draw.arc([x0, band_y, x1, y0 + h], start=200, end=340, fill=accent, width=8)
        ear_w = int(w * 0.22)
        draw.rounded_rectangle([x0, y0 + h // 3, x0 + ear_w, y1], radius=12, fill=fill, outline=accent, width=3)
        draw.rounded_rectangle([x1 - ear_w, y0 + h // 3, x1, y1], radius=12, fill=fill, outline=accent, width=3)

    elif category == "monitor":
        screen_h = int(h * 0.75)
        draw.rounded_rectangle([x0, y0, x1, y0 + screen_h], radius=8, fill=fill, outline=accent, width=3)
        stand_w = int(w * 0.15)
        cx = (x0 + x1) // 2
        draw.rectangle([cx - stand_w // 2, y0 + screen_h, cx + stand_w // 2, y1 - 10], fill=accent)
        draw.rectangle([x0 + w // 4, y1 - 10, x1 - w // 4, y1], fill=accent)


def generate_product_image(category: str, product_id: int, name: str) -> Image.Image:
    """生成类别可区分、同类有差异的商品图。"""
    size = 512
    seed = int(hashlib.md5(f"{category}-{product_id}".encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed)

    meta = CATALOG[category]
    bg = _color(seed, meta["hue"], sat=80, light=220)
    fill = _color(seed + 7, meta["hue"], sat=160, light=rng.randint(80, 130))
    accent = _color(seed + 13, meta["hue"] + 40, sat=180, light=rng.randint(50, 90))

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    margin = 60
    box = (margin, margin, size - margin, size - margin - 40)
    _draw_product_shape(draw, category, box, fill, accent)

    # 每张图独特的纹理，帮助 CLIP 区分同类商品
    for _ in range(30):
        px = rng.randint(0, size - 1)
        py = rng.randint(0, size - 1)
        c = _color(seed + px + py, meta["hue"] + 80, sat=100, light=rng.randint(160, 220))
        draw.point((px, py), fill=c)

    # 商品名标注
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, size - 32), name[:40], fill=(20, 20, 20), font=font)

    return img


def build_product_record(category: str, index: int, product_id: int) -> dict:
    meta = CATALOG[category]
    brand = random.choice(meta["brands"])
    ptype = random.choice(meta["types"])
    name = f"{brand} {ptype} {category.title()}"
    low, high = meta["price_range"]
    price = random.randint(low, high)

    image_name = f"{category}_{index}.jpg"
    image_path = IMAGE_DIR / image_name

    img = generate_product_image(category, product_id, name)
    img.save(image_path, quality=90)

    return {
        "id": product_id,
        "name": name,
        "description": f"{brand} {ptype} {category}, suitable for daily use",
        "image_path": str(image_path).replace("\\", "/"),
        "price": price,
        "category": category,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build product library offline")
    parser.add_argument(
        "--per-category",
        type=int,
        default=50,
        help="Number of products per category (default: 50)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATALOG.keys()),
        choices=list(CATALOG.keys()),
        help="Categories to generate",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing images before building",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for f in IMAGE_DIR.glob("*.jpg"):
            f.unlink()
        for f in IMAGE_DIR.glob("*.png"):
            f.unlink()

    products = []
    product_id = 1

    total = len(args.categories) * args.per_category
    print(f"Building {total} products across {len(args.categories)} categories...")

    for category in args.categories:
        print(f"\n[{category}] generating {args.per_category} items...")
        for i in range(args.per_category):
            record = build_product_record(category, i, product_id)
            products.append(record)
            product_id += 1
            if (i + 1) % 10 == 0 or i + 1 == args.per_category:
                print(f"  {i + 1}/{args.per_category} done")

    df = pd.DataFrame(products)
    df.to_csv(DATA_DIR / "products.csv", index=False)

    print(f"\nDone!")
    print(f"  products.csv  -> {len(products)} rows")
    print(f"  data/images/  -> {len(list(IMAGE_DIR.glob('*.jpg')))} images")
    print("\nNext: restart your API to reload the product index.")


if __name__ == "__main__":
    main()
