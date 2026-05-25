"""快速填充 data/ 目录：下载示例商品图片并生成 products.csv。

不依赖京东爬虫，也不需要下载大型 Hugging Face 数据集。
适合本地开发、测试 CLIP 搜图 API。

用法（在项目根目录）:
    python scripts/scripts/build_demo_dataset.py
"""

from pathlib import Path
import pandas as pd
import requests

IMAGE_DIR = Path("data/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 示例商品：名称、分类、价格、图片 URL（Wikimedia Commons，可公开访问）
SAMPLE_PRODUCTS = [
    {
        "name": "Student Notebook",
        "category": "laptop",
        "price": 2999,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/ASUS_Vivobook_15.jpg/640px-ASUS_Vivobook_15.jpg",
    },
    {
        "name": "Office PC",
        "category": "desktop",
        "price": 4999,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Dell_PowerEdge_R940.jpg/640px-Dell_PowerEdge_R940.jpg",
    },
    {
        "name": "Gaming Laptop",
        "category": "laptop",
        "price": 7999,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Lenovo_IdeaPad_Gaming_3.jpg/640px-Lenovo_IdeaPad_Gaming_3.jpg",
    },
    {
        "name": "Wireless Headphones",
        "category": "audio",
        "price": 899,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Beats_Electronics_logo.svg/640px-Beats_Electronics_logo.svg.png",
    },
    {
        "name": "Smartphone Pro",
        "category": "phone",
        "price": 5999,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/IPhone_15_Pro_Blue_Titanium.jpg/640px-IPhone_15_Pro_Blue_Titanium.jpg",
    },
    {
        "name": "Running Shoes",
        "category": "shoes",
        "price": 699,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Nike_Free_Run_2.jpg/640px-Nike_Free_Run_2.jpg",
    },
    {
        "name": "Leather Backpack",
        "category": "bag",
        "price": 459,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Backpack.jpg/640px-Backpack.jpg",
    },
    {
        "name": "Coffee Maker",
        "category": "kitchen",
        "price": 1299,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/640px-A_small_cup_of_coffee.JPG",
    },
    {
        "name": "Office Chair",
        "category": "furniture",
        "price": 1599,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Office_chair.jpg/640px-Office_chair.jpg",
    },
    {
        "name": "Mechanical Keyboard",
        "category": "accessory",
        "price": 599,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Mechanical_keyboard.jpg/640px-Mechanical_keyboard.jpg",
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def download_image(url: str, save_path: Path) -> bool:
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        save_path.write_bytes(response.content)
        return True
    except Exception as exc:
        print(f"  download failed: {exc}")
        return False


def main():
    products = []

    for idx, item in enumerate(SAMPLE_PRODUCTS, start=1):
        image_name = f"product_{idx}.jpg"
        image_path = IMAGE_DIR / image_name

        print(f"[{idx}/{len(SAMPLE_PRODUCTS)}] {item['name']}")

        if not download_image(item["image_url"], image_path):
            continue

        products.append(
            {
                "id": idx,
                "name": item["name"],
                "image_path": str(image_path).replace("\\", "/"),
                "price": item["price"],
                "category": item["category"],
            }
        )

    if not products:
        raise SystemExit("No images downloaded. Check your network connection.")

    df = pd.DataFrame(products)
    df.to_csv("data/products.csv", index=False)

    print(f"\nDone: {len(products)} products saved.")
    print(f"  CSV:   data/products.csv")
    print(f"  Images: data/images/")


if __name__ == "__main__":
    main()
