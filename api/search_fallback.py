"""不依赖 PyTorch 的关键词检索（CLIP 不可用时的备用方案）。"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "products.csv"

_products_df = pd.read_csv(CSV_PATH)


# 中文 → 英文检索词（商品库标题为英文）
ZH_PHRASES = {
    "蓝色衬衫": "blue shirt",
    "红色连衣裙": "red dress",
    "黑色裤子": "black pants",
    "白色T恤": "white t-shirt",
    "白色t恤": "white t-shirt",
    "运动鞋": "sports shoes",
    "高跟鞋": "heels",
    "蓝色": "blue",
    "红色": "red",
    "黑色": "black",
    "白色": "white",
    "绿色": "green",
    "衬衫": "shirt",
    "连衣裙": "dress",
    "裙子": "dress",
    "裤子": "pants",
    "牛仔裤": "jeans",
    "手表": "watch",
    "鞋": "shoes",
    "包": "bag",
    "外套": "jacket",
    "T恤": "t-shirt",
    "t恤": "t-shirt",
}


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def expand_query_tokens(query: str) -> list[str]:
    """把中英文查询拆成可在英文商品名里匹配的 token。"""
    q = query.strip()
    tokens: list[str] = []

    remaining = q
    for zh, en in sorted(ZH_PHRASES.items(), key=lambda x: -len(x[0])):
        if zh in remaining:
            tokens.extend(en.lower().split())
            remaining = remaining.replace(zh, " ")

    for word in re.findall(r"[a-zA-Z0-9]+", remaining.lower()):
        if len(word) > 1:
            tokens.append(word)

    # 去重保序
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def reload_products() -> None:
    global _products_df
    from api.product_admin import ensure_schema

    ensure_schema()
    _products_df = pd.read_csv(CSV_PATH)


def _visible_rows():
    df = _products_df
    if "visible" in df.columns:
        return df[df["visible"].astype(int) == 1]
    return df


def search_by_text(query: str, top_k: int = 5) -> list[dict]:
    words = expand_query_tokens(query)
    if not words and _has_chinese(query):
        # 纯中文且未命中词典：用单字尝试常见品类（弱匹配）
        words = [c for c in query if "\u4e00" <= c <= "\u9fff"]

    if not words:
        words = [w for w in query.lower().replace(",", " ").split() if w]

    scored = []
    for _, row in _visible_rows().iterrows():
        text = f"{row['name']} {row.get('description', '')} {row['category']}".lower()
        score = sum(1 for w in words if w in text)
        if str(row["category"]).lower() in query.lower():
            score += 2
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, row in scored[:top_k]:
        results.append({
            "id": int(row["id"]),
            "name": str(row["name"]),
            "price": float(row["price"]),
            "category": str(row["category"]),
            "image_path": str(row["image_path"]),
            "description": str(row.get("description", "")),
            "score": float(score),
        })
    return results
