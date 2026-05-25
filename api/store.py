"""商品商城：浏览、筛选、详情、购物袋。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from api.image_thumbs import display_image_path
from api.shopping_ui import _category_label, _price_label, gallery_label

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "products.csv"

PAGE_SIZE = 8

_df = pd.read_csv(CSV_PATH)
_all_products: list[dict] = []


def _row_to_product(row) -> dict:
    visible = row.get("visible", 1)
    featured = row.get("featured", 0)
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "price": float(row["price"]),
        "category": str(row["category"]),
        "image_path": str(row["image_path"]),
        "description": str(row.get("description", "") or ""),
        "visible": bool(int(visible)) if str(visible) not in ("", "nan") else True,
        "featured": bool(int(featured)) if str(featured) not in ("", "nan") else False,
        "sort_order": int(row.get("sort_order", row["id"])),
    }


def reload_products() -> None:
    global _df, _all_products
    from api.product_admin import ensure_schema

    ensure_schema()
    _df = pd.read_csv(CSV_PATH)
    _all_products = [_row_to_product(row) for _, row in _df.iterrows()]


reload_products()


def list_all_products() -> list[dict]:
    return list(_all_products)


def list_visible_products() -> list[dict]:
    return [p for p in _all_products if p.get("visible", True)]


def get_featured_products(limit: int = 4) -> list[dict]:
    items = [p for p in _all_products if p.get("visible", True) and p.get("featured")]
    items.sort(key=lambda p: (p.get("sort_order", p["id"]), p["id"]))
    return items[:limit]


def _customer_catalog() -> list[dict]:
    return list_visible_products()


def resolve_image_path(image_path: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / image_path)


def list_categories() -> list[str]:
    cats = sorted({p["category"] for p in _customer_catalog()})
    return ["全部"] + cats


def category_label(cat: str) -> str:
    if cat == "全部":
        return "全部"
    return _category_label(cat)


def get_product(product_id: int | None) -> dict | None:
    if product_id is None:
        return None
    for p in _customer_catalog():
        if p["id"] == int(product_id):
            return p
    return None


def filter_products(
    category: str = "全部",
    search: str = "",
    page: int = 1,
) -> tuple[list[dict], int, int]:
    items = _customer_catalog()
    items.sort(key=lambda p: (p.get("sort_order", p["id"]), p["id"]))
    if category and category != "全部":
        items = [p for p in items if p["category"] == category]
    q = (search or "").strip().lower()
    if q:
        items = [
            p
            for p in items
            if q in p["name"].lower()
            or q in p.get("description", "").lower()
            or q in p["category"].lower()
        ]
    total = len(items)
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(int(page), pages))
    start = (page - 1) * PAGE_SIZE
    page_items = items[start : start + PAGE_SIZE]
    return page_items, total, pages


def products_to_gallery(products: list[dict], *, use_thumb: bool = True) -> list[tuple[str, str]]:
    out = []
    for p in products:
        path = resolve_image_path(p["image_path"])
        if use_thumb:
            path = display_image_path(path)
        out.append((path, gallery_label(p)))
    return out


def product_choices(products: list[dict]) -> list[tuple[str, int]]:
    return [(f"{p['name']} · {_price_label(p['price'])}", p["id"]) for p in products]


def format_product_detail(product: dict | None) -> tuple[str | None, str]:
    if product is None:
        return None, "请从上方列表选择一件商品查看详情。"
    img = resolve_image_path(product["image_path"])
    cat = _category_label(product["category"])
    price = _price_label(product["price"])
    desc = product.get("description") or product["name"]
    text = f"""## {product['name']}

**售价** {price}  
**品类** {cat}  
**货号** #{product['id']:04d}

{desc}

---
✅ 正品保障 · 7 天无理由退换（演示）  
🚚 满 ¥99 包邮（演示）
"""
    return img, text


def add_to_cart(cart: list[dict], product_id: int | None, quantity: int) -> list[dict]:
    if product_id is None:
        return cart
    product = get_product(product_id)
    if product is None:
        return cart
    qty = max(1, int(quantity or 1))
    cart = [dict(item) for item in cart]
    for item in cart:
        if item["id"] == product["id"]:
            item["qty"] = item.get("qty", 1) + qty
            return cart
    cart.append({
        "id": product["id"],
        "name": product["name"],
        "price": product["price"],
        "image_path": product["image_path"],
        "qty": qty,
    })
    return cart


def remove_from_cart(cart: list[dict], product_id: int | None) -> list[dict]:
    if product_id is None:
        return cart
    return [item for item in cart if item["id"] != int(product_id)]


def clear_cart(_cart: list[dict]) -> list[dict]:
    return []


def format_cart_summary(cart: list[dict]) -> str:
    if not cart:
        return "🛒 购物袋是空的，去挑几件喜欢的吧～"
    lines = [f"🛒 **购物袋** · 共 {sum(i.get('qty', 1) for i in cart)} 件\n"]
    total = 0.0
    for i, item in enumerate(cart, start=1):
        qty = item.get("qty", 1)
        sub = item["price"] * qty
        total += sub
        lines.append(f"{i}. {item['name']}")
        lines.append(f"   {_price_label(item['price'])} × {qty} = {_price_label(sub)}")
        lines.append("")
    lines.append(f"**合计：{_price_label(total)}**")
    lines.append("\n*结算功能为演示，接入支付后可正式下单。*")
    return "\n".join(lines)


def cart_remove_choices(cart: list[dict]) -> list[tuple[str, int]]:
    return [(f"{item['name']} × {item.get('qty', 1)}", item["id"]) for item in cart]
