"""pytest 公共配置：让测试用临时目录跑，互不影响主数据。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_data(tmp_path, monkeypatch):
    """把所有模块的数据目录指向 tmp_path，并预置一份精简的 products.csv。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    images_dir = data_dir / "images"
    images_dir.mkdir()

    from PIL import Image

    sample_image = images_dir / "product_1.jpg"
    Image.new("RGB", (64, 64), (200, 100, 50)).save(sample_image, "JPEG")
    sample_image2 = images_dir / "product_2.jpg"
    Image.new("RGB", (64, 64), (50, 100, 200)).save(sample_image2, "JPEG")

    csv = data_dir / "products.csv"
    csv.write_text(
        "id,name,description,image_path,price,category,visible,featured,sort_order,stock\n"
        "1,Red Polo Shirt,Cotton red polo,data/images/product_1.jpg,199,Apparel,1,1,1,5\n"
        "2,Blue Sneakers,Casual blue sneakers,data/images/product_2.jpg,399,Footwear,1,0,2,2\n",
        encoding="utf-8",
    )

    from api import product_admin, store, user_auth, orders

    monkeypatch.setattr(product_admin, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(product_admin, "CSV_PATH", csv, raising=True)
    monkeypatch.setattr(product_admin, "IMAGE_DIR", images_dir, raising=True)

    monkeypatch.setattr(store, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(store, "CSV_PATH", csv, raising=True)
    store.reload_products()

    monkeypatch.setattr(user_auth, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(user_auth, "USERS_FILE", data_dir / "users.json", raising=True)
    monkeypatch.setattr(user_auth, "CART_DIR", data_dir / "carts", raising=True)

    monkeypatch.setattr(orders, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(orders, "ORDERS_FILE", data_dir / "orders.json", raising=True)

    from api import search_fallback

    monkeypatch.setattr(search_fallback, "BASE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(search_fallback, "CSV_PATH", csv, raising=True)
    search_fallback.reload_products()

    yield tmp_path

    shutil.rmtree(tmp_path, ignore_errors=True)
