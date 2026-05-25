"""商家商品管理：增删改、展示控制、保存 CSV、刷新各模块缓存。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "products.csv"
IMAGE_DIR = BASE_DIR / "data" / "images"

DEFAULT_CATEGORIES = ["Apparel", "Accessories", "Footwear", "Personal Care"]

COLUMNS = [
    "id",
    "name",
    "description",
    "image_path",
    "price",
    "category",
    "visible",
    "featured",
    "sort_order",
    "stock",
]

DEFAULT_STOCK = 99


def _as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    return str(value).strip().lower() in ("1", "true", "yes", "是", "y")


def ensure_schema() -> None:
    """为旧版 CSV 补全展示字段。"""
    if not CSV_PATH.is_file():
        return
    df = pd.read_csv(CSV_PATH)
    changed = False
    if "visible" not in df.columns:
        df["visible"] = 1
        changed = True
    if "featured" not in df.columns:
        df["featured"] = 0
        changed = True
    if "sort_order" not in df.columns:
        df["sort_order"] = df["id"]
        changed = True
    if "stock" not in df.columns:
        df["stock"] = DEFAULT_STOCK
        changed = True
    if changed:
        _save_df(df)


def _read_df() -> pd.DataFrame:
    ensure_schema()
    df = pd.read_csv(CSV_PATH)
    for col in COLUMNS:
        if col not in df.columns:
            if col in ("visible", "featured"):
                df[col] = 1 if col == "visible" else 0
            elif col == "sort_order":
                df[col] = df["id"]
            else:
                df[col] = ""
    df["visible"] = df["visible"].apply(lambda x: 1 if _as_bool(x) else 0)
    df["featured"] = df["featured"].apply(lambda x: 1 if _as_bool(x) else 0)
    df["sort_order"] = pd.to_numeric(df["sort_order"], errors="coerce").fillna(df["id"]).astype(int)
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(DEFAULT_STOCK).astype(int).clip(lower=0)
    return df[COLUMNS]


def _save_df(df: pd.DataFrame) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for col in ("visible", "featured"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if _as_bool(x) else 0)
    if "sort_order" in df.columns:
        df["sort_order"] = pd.to_numeric(df["sort_order"], errors="coerce").fillna(df["id"]).astype(int)
    if "stock" in df.columns:
        df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(DEFAULT_STOCK).astype(int).clip(lower=0)
    df = df.sort_values("id").reset_index(drop=True)
    if CSV_PATH.is_file():
        try:
            shutil.copy2(CSV_PATH, CSV_PATH.with_suffix(".csv.bak"))
        except Exception:
            pass
    df.to_csv(CSV_PATH, index=False)


def _next_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return int(df["id"].max()) + 1


def _save_upload_image(image_path: str | None, product_id: int) -> str | None:
    if not image_path:
        return None
    src = Path(image_path)
    if not src.is_file():
        return None
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = IMAGE_DIR / f"product_{product_id}.jpg"
    try:
        img = Image.open(src).convert("RGB")
        img.save(dest, quality=90)
    except Exception:
        shutil.copy2(src, dest)
    return str(Path("data/images") / dest.name).replace("\\", "/")


def _row_dict(row) -> dict:
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "price": float(row["price"]),
        "category": str(row["category"]),
        "image_path": str(row["image_path"]),
        "description": str(row.get("description", "") or ""),
        "visible": bool(int(row.get("visible", 1))),
        "featured": bool(int(row.get("featured", 0))),
        "sort_order": int(row.get("sort_order", row["id"])),
        "stock": int(row.get("stock", DEFAULT_STOCK)),
    }


def adjust_stock(product_id: int, delta: int) -> bool:
    """库存增减；返回是否成功（库存不足返回 False）。"""
    df = _read_df()
    mask = df["id"] == int(product_id)
    if not mask.any():
        return False
    idx = df.index[mask][0]
    current = int(df.at[idx, "stock"])
    new_value = current + int(delta)
    if new_value < 0:
        return False
    df.at[idx, "stock"] = new_value
    _save_df(df)
    return True


def get_stock(product_id: int) -> int:
    df = _read_df()
    row = df[df["id"] == int(product_id)]
    if row.empty:
        return 0
    return int(row.iloc[0]["stock"])


def list_all_products() -> list[dict]:
    df = _read_df()
    return [_row_dict(row) for _, row in df.iterrows()]


def admin_choices() -> list[tuple[str, int]]:
    return [
        (
            f"#{p['id']:04d} · {p['name']} · ¥{p['price']:.0f}"
            + (" · 已隐藏" if not p["visible"] else "")
            + (" · 精选" if p["featured"] else ""),
            p["id"],
        )
        for p in list_all_products()
    ]


def get_product_fields(product_id: int | None) -> dict:
    if product_id is None:
        return {
            "name": "",
            "price": 299.0,
            "category": DEFAULT_CATEGORIES[0],
            "description": "",
            "image": None,
            "image_path": "",
            "visible": True,
            "featured": False,
            "sort_order": 0,
        }
    df = _read_df()
    row = df[df["id"] == int(product_id)]
    if row.empty:
        return get_product_fields(None)
    r = row.iloc[0]
    img_abs = BASE_DIR / str(r["image_path"])
    return {
        "name": str(r["name"]),
        "price": float(r["price"]),
        "category": str(r["category"]),
        "description": str(r.get("description", "") or ""),
        "image": str(img_abs) if img_abs.is_file() else None,
        "image_path": str(r["image_path"]),
        "visible": bool(int(r.get("visible", 1))),
        "featured": bool(int(r.get("featured", 0))),
        "sort_order": int(r.get("sort_order", r["id"])),
    }


def get_display_editor_df() -> pd.DataFrame:
    df = _read_df()
    out = pd.DataFrame({
        "货号": df["id"].astype(int),
        "商品名": df["name"],
        "价格": df["price"].astype(int),
        "品类": df["category"],
        "库存": df["stock"].astype(int),
        "商城展示": df["visible"].astype(bool),
        "精选推荐": df["featured"].astype(bool),
        "排序": df["sort_order"].astype(int),
    })
    return out


def save_display_editor_df(table) -> tuple[str, list[tuple[str, int]]]:
    if table is None or len(table) == 0:
        return "❌ 没有可保存的数据", admin_choices()

    if isinstance(table, pd.DataFrame):
        editor = table
    else:
        editor = pd.DataFrame(table)

    expected = ["货号", "商品名", "价格", "品类", "库存", "商城展示", "精选推荐", "排序"]
    if list(editor.columns)[: len(expected)] != expected:
        editor.columns = expected[: len(editor.columns)]

    df = _read_df()
    id_map = {}
    for _, row in editor.iterrows():
        try:
            pid = int(row["货号"])
        except (ValueError, TypeError, KeyError):
            continue
        id_map[pid] = row

    for idx, row in df.iterrows():
        pid = int(row["id"])
        if pid not in id_map:
            continue
        ed = id_map[pid]
        df.at[idx, "visible"] = 1 if _as_bool(ed["商城展示"]) else 0
        df.at[idx, "featured"] = 1 if _as_bool(ed["精选推荐"]) else 0
        df.at[idx, "sort_order"] = int(ed["排序"])
        try:
            df.at[idx, "stock"] = max(0, int(ed.get("库存", DEFAULT_STOCK)))
        except (ValueError, TypeError):
            pass

    _save_df(df)
    visible_n = int((df["visible"] == 1).sum())
    featured_n = int((df["featured"] == 1).sum())
    msg = refresh_runtime()
    return (
        f"✅ 展示设置已保存（商城展示 {visible_n} 件 · 精选 {featured_n} 件）\n\n{msg}",
        admin_choices(),
    )


def refresh_runtime() -> str:
    messages = []

    from api import store

    store.reload_products()
    messages.append(
        f"商城已刷新（展示 {len(store.list_visible_products())} 件 / 库内 {len(store.list_all_products())} 件）"
    )

    from api import search_fallback

    search_fallback.reload_products()
    messages.append("关键词检索已更新")

    try:
        from api import catalog

        if catalog.is_ready():
            catalog.rebuild_index()
            messages.append("AI 以图/文检索索引已重建")
        else:
            messages.append("AI 索引将在下次搜索或重启后使用新商品")
    except Exception as exc:
        messages.append(f"AI 索引未更新（可重启应用）: {exc}")

    return "✅ " + " · ".join(messages)


def create_product(
    name: str,
    price: float,
    category: str,
    description: str,
    image_path: str | None,
    *,
    visible: bool = True,
    featured: bool = False,
) -> tuple[str, list[tuple[str, int]]]:
    name = (name or "").strip()
    if not name:
        return "❌ 请填写商品名称", admin_choices()
    if not image_path:
        return "❌ 请上传商品图片", admin_choices()

    df = _read_df()
    new_id = _next_id(df)
    rel_image = _save_upload_image(image_path, new_id)
    if not rel_image:
        return "❌ 图片保存失败", admin_choices()

    desc = (description or "").strip() or f"{category} {name}"
    new_row = {
        "id": new_id,
        "name": name,
        "description": desc,
        "image_path": rel_image,
        "price": float(price or 0),
        "category": (category or DEFAULT_CATEGORIES[0]).strip(),
        "visible": 1 if visible else 0,
        "featured": 1 if featured else 0,
        "sort_order": new_id,
        "stock": DEFAULT_STOCK,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_df(df)
    msg = refresh_runtime()
    return f"✅ 已上架新商品 #{new_id:04d} · {name}\n\n{msg}", admin_choices()


def update_product(
    product_id: int | None,
    name: str,
    price: float,
    category: str,
    description: str,
    new_image_path: str | None,
    *,
    visible: bool = True,
    featured: bool = False,
    sort_order: int | None = None,
) -> tuple[str, list[tuple[str, int]]]:
    if product_id is None:
        return "❌ 请先选择要编辑的商品", admin_choices()
    name = (name or "").strip()
    if not name:
        return "❌ 商品名称不能为空", admin_choices()

    df = _read_df()
    mask = df["id"] == int(product_id)
    if not mask.any():
        return "❌ 商品不存在", admin_choices()

    idx = df.index[mask][0]
    df.at[idx, "name"] = name
    df.at[idx, "price"] = float(price or 0)
    df.at[idx, "category"] = (category or DEFAULT_CATEGORIES[0]).strip()
    df.at[idx, "description"] = (description or "").strip() or f"{df.at[idx, 'category']} {name}"
    df.at[idx, "visible"] = 1 if visible else 0
    df.at[idx, "featured"] = 1 if featured else 0
    if sort_order is not None:
        df.at[idx, "sort_order"] = int(sort_order)

    if new_image_path:
        rel = _save_upload_image(new_image_path, int(product_id))
        if rel:
            df.at[idx, "image_path"] = rel

    _save_df(df)
    msg = refresh_runtime()
    return f"✅ 已更新商品 #{int(product_id):04d}\n\n{msg}", admin_choices()


def delete_product(product_id: int | None) -> tuple[str, list[tuple[str, int]]]:
    if product_id is None:
        return "❌ 请选择要删除的商品", admin_choices()
    df = _read_df()
    mask = df["id"] == int(product_id)
    if not mask.any():
        return "❌ 商品不存在", admin_choices()
    name = str(df.loc[mask, "name"].iloc[0])
    df = df[~mask].reset_index(drop=True)
    _save_df(df)
    msg = refresh_runtime()
    return f"✅ 已删除「{name}」\n\n{msg}", admin_choices()


ensure_schema()
