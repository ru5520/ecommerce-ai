"""从 Hugging Face 下载商品图片并生成 products.csv。

用法（在项目根目录）:
    python scripts/scripts/build_dataset.py
    python scripts/scripts/build_dataset.py --num 200
    python scripts/scripts/build_dataset.py --offline   # 无法联网时用本地生成

国内网络建议:
    set HF_ENDPOINT=https://hf-mirror.com
    python scripts/scripts/build_dataset.py

说明:
    旧版 `load_dataset("ashraq/...")` 在 Windows 上可能报:
    NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.
    本脚本改为通过镜像下载 parquet 再读取，绕过该问题。
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# 镜像（需在 import huggingface 相关库之前设置）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", os.environ["HF_ENDPOINT"])

import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image
from tqdm import tqdm

DATASET_REPO = "ashraq/fashion-product-images-small"
PARQUET_FILE = "data/train-00000-of-00002-6cff4c59f91661c3.parquet"
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

IMAGE_DIR = Path("data/images")
CSV_PATH = Path("data/products.csv")
CACHE_PARQUET = Path("data/cache/fashion_train.parquet")

# 本地脚本生成的占位图前缀（与 HF 真实图 product_*.jpg 区分）
SYNTHETIC_PREFIXES = ("laptop_", "phone_", "keyboard_", "headphone_", "monitor_", "desktop_")


def clean_image_directory(only_synthetic: bool = False) -> int:
    """删除 data/images 下的旧图。only_synthetic=True 时只删本地生成的假图。"""
    if not IMAGE_DIR.exists():
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        return 0

    removed = 0
    for path in IMAGE_DIR.iterdir():
        if not path.is_file():
            continue
        if only_synthetic:
            if not path.name.startswith(SYNTHETIC_PREFIXES):
                continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            print(f"Warning: could not delete {path}: {exc}")
    label = "synthetic" if only_synthetic else "all"
    print(f"Removed {removed} {label} image(s) from {IMAGE_DIR}")
    return removed


def mirror_parquet_url() -> str:
    return f"{MIRROR}/datasets/{DATASET_REPO}/resolve/main/{PARQUET_FILE}"


def download_parquet(dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size > 1_000_000:
        print(f"Using cached parquet: {dest}")
        return dest

    url = mirror_parquet_url()
    print(f"Downloading parquet from mirror ({url}) ...")
    print("(约 130MB，网速慢时请耐心等待，或 Ctrl+C 后使用 --offline)")

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc="parquet",
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)

    print(f"Saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def image_from_cell(cell) -> Image.Image | None:
    if cell is None:
        return None
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("buffer")
        if raw:
            return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    return None


def iter_products_from_parquet(parquet_path: Path, limit: int):
    pf = pq.ParquetFile(parquet_path)
    cols = pf.schema_arrow.names
    name_col = "productDisplayName" if "productDisplayName" in cols else "title"
    cat_col = "masterCategory" if "masterCategory" in cols else "subCategory"
    need = [c for c in (name_col, cat_col, "image") if c in cols]
    if "image" not in need:
        raise RuntimeError(f"Parquet missing image column. Columns: {cols}")

    count = 0
    for batch in pf.iter_batches(batch_size=64, columns=need):
        data = batch.to_pydict()
        names = data.get(name_col) or []
        categories = data.get(cat_col) or ["general"] * len(names)
        images = data.get("image") or []
        for name, category, img_cell in zip(names, categories, images):
            if count >= limit:
                return
            image = image_from_cell(img_cell)
            if image is None:
                continue
            yield count, str(name or f"product_{count}"), str(category or "general"), image
            count += 1


def build_from_parquet(num_products: int, force_download: bool, clean: bool) -> int:
    if clean:
        clean_image_directory(only_synthetic=False)

    parquet_path = download_parquet(CACHE_PARQUET, force=force_download)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    products = []

    for idx, title, category, image in tqdm(
        iter_products_from_parquet(parquet_path, num_products),
        total=num_products,
        desc="Saving products",
    ):
        image_path = IMAGE_DIR / f"product_{idx}.jpg"
        image.save(image_path, quality=90)
        products.append(
            {
                "id": idx + 1,
                "name": title,
                "description": f"{category} {title}",
                "image_path": str(image_path).replace("\\", "/"),
                "price": 999,
                "category": category,
            }
        )

    if not products:
        raise RuntimeError("No products extracted from parquet.")
    return save_csv(products)


def save_csv(products: list[dict]) -> int:
    df = pd.DataFrame(products)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nDone: {len(products)} products -> {CSV_PATH}")
    return len(products)


def build_offline(num_products: int) -> int:
    """离线：调用本地商品库生成脚本。"""
    print("Offline mode: generating synthetic product library ...")
    import subprocess

    per_cat = max(20, num_products // 5)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "scripts" / "build_product_library.py"),
        "--per-category",
        str(per_cat),
    ]
    subprocess.check_call(cmd, cwd=ROOT)
    df = pd.read_csv(CSV_PATH)
    if len(df) > num_products:
        df = df.head(num_products)
        df.to_csv(CSV_PATH, index=False)
    print(f"\nDone (offline): {len(df)} products -> {CSV_PATH}")
    return len(df)


def try_load_dataset_legacy(num_products: int) -> int | None:
    """部分环境仍可用旧 API，成功则返回数量，失败返回 None。"""
    try:
        from datasets import load_dataset

        ds = load_dataset(DATASET_REPO, split="train")
        ds = ds.select(range(min(num_products, len(ds))))
    except NotImplementedError:
        return None
    except Exception as exc:
        print(f"load_dataset skipped: {exc}")
        return None

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    products = []
    for idx, item in enumerate(tqdm(ds, desc="Saving products")):
        image = item.get("image")
        pil = image_from_cell(image)
        if pil is None:
            continue
        title = item.get("productDisplayName") or item.get("title") or f"product_{idx}"
        category = item.get("masterCategory") or item.get("subCategory") or "general"
        image_path = IMAGE_DIR / f"product_{idx}.jpg"
        pil.save(image_path, quality=90)
        products.append(
            {
                "id": idx + 1,
                "name": str(title),
                "description": f"{category} {title}",
                "image_path": str(image_path).replace("\\", "/"),
                "price": 999,
                "category": str(category),
            }
        )
    if not products:
        return None
    return save_csv(products)


def main():
    parser = argparse.ArgumentParser(description="Build product CSV from Hugging Face fashion dataset")
    parser.add_argument("--num", type=int, default=200, help="Number of products to export")
    parser.add_argument("--offline", action="store_true", help="Use local synthetic data (no download)")
    parser.add_argument("--force-download", action="store_true", help="Re-download parquet file")
    parser.add_argument("--legacy", action="store_true", help="Try old load_dataset() first")
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Delete all files in data/images before export (default: on)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="Keep existing images in data/images",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only remove old/synthetic images, do not rebuild",
    )
    parser.add_argument(
        "--clean-synthetic",
        action="store_true",
        help="Only remove laptop_/phone_/... placeholder images",
    )
    args = parser.parse_args()

    print(f"Project root: {ROOT}")
    print(f"HF mirror: {MIRROR}")

    if args.clean_only:
        clean_image_directory(only_synthetic=False)
        return
    if args.clean_synthetic:
        clean_image_directory(only_synthetic=True)
        return

    if args.offline:
        if args.clean:
            clean_image_directory(only_synthetic=False)
        build_offline(args.num)
        return

    if args.legacy:
        if args.clean:
            clean_image_directory(only_synthetic=False)
        n = try_load_dataset_legacy(args.num)
        if n:
            return

    try:
        build_from_parquet(args.num, force_download=args.force_download, clean=args.clean)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Use --offline for quick local data.")
        raise SystemExit(1) from None
    except Exception as exc:
        print(f"\nFailed: {exc}")
        print("Tip: run with --offline  or check network / HF mirror.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
