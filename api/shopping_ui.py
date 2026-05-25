"""面向用户的购物反馈文案（商业化展示，不含匹配度等技术字段）。"""

from __future__ import annotations

CATEGORY_ZH = {
    "Apparel": "服饰",
    "Accessories": "配饰",
    "Footwear": "鞋靴",
    "Personal Care": "个护",
    "general": "综合",
}


def _category_label(category: str) -> str:
    return CATEGORY_ZH.get(category, category)


def _price_label(price: float) -> str:
    if price <= 0:
        return "价格详询"
    return f"¥{price:,.0f}"


def gallery_label(item: dict) -> str:
    return f"{item['name']}\n{_price_label(item['price'])}"


def _format_product_block(item: dict, rank: int, *, highlight: bool = False) -> str:
    tag = "🏆 首选推荐" if highlight else f"{rank}. "
    name = item["name"]
    price = _price_label(item["price"])
    cat = _category_label(str(item.get("category", "")))
    desc = (item.get("description") or "").strip()
    lines = [f"{tag}{name}" if highlight else f"{rank}. {name}", f"   {price} · {cat}"]
    if desc and desc.lower() not in name.lower():
        lines.append(f"   {desc[:80]}")
    return "\n".join(lines)


def template_reply(
    user_request: str,
    results: list[dict],
    *,
    search_mode: str = "text",
) -> str:
    if not results:
        if search_mode == "image":
            lead = "抱歉，暂未找到与您上传图片足够相似的商品。"
        else:
            lead = f"抱歉，暂时没有找到与「{user_request or '您的需求'}」完全匹配的商品。"
        return (
            f"{lead}\n\n"
            "您可以试试：\n"
            "· 换一张更清晰、主体完整的参考图\n"
            "· 补充颜色、品类或预算，例如「蓝色衬衫 预算 500」\n"
            "· 使用英文关键词：shirt、dress、watch\n\n"
            "如需帮助，欢迎继续描述您的穿搭场景，我会为您重新挑选。"
        )

    if search_mode == "image":
        intro = "您好！根据您上传的参考图，我为您挑选了以下风格相近的商品："
    elif user_request:
        intro = f"您好！根据您的需求「{user_request}」，为您精选了 {len(results)} 款商品："
    else:
        intro = f"您好！为您精选了 {len(results)} 款商品："

    parts = [intro, ""]
    parts.append(_format_product_block(results[0], 1, highlight=True))
    parts.append("")

    if len(results) > 1:
        parts.append("📦 更多推荐")
        for i, item in enumerate(results[1:6], start=2):
            parts.append(_format_product_block(item, i))
        parts.append("")

    top = results[0]
    parts.append(
        "💡 购物建议\n"
        f"· 首推「{top['name']}」，与您{'上传的参考图' if search_mode == 'image' else '的描述'}最为接近\n"
        "· 下单前请确认尺码与颜色；如需搭配建议，可继续告诉我场合和预算\n"
        "· 支持在线客服咨询（演示）"
    )
    return "\n".join(parts)


def build_product_summary_for_llm(results: list[dict], limit: int = 6) -> str:
    lines = []
    for i, item in enumerate(results[:limit], start=1):
        lines.append(
            f"{i}. {item['name']} | {_price_label(item['price'])} | "
            f"{_category_label(str(item.get('category', '')))}"
        )
    return "\n".join(lines)


def generate_reply_with_llm(
    user_request: str,
    results: list[dict],
    *,
    search_mode: str = "text",
    llm_fn=None,
) -> str | None:
    if llm_fn is None or not results:
        return None

    if search_mode == "image":
        context = "用户上传了一张参考图，系统根据视觉相似度推荐了下列商品。"
        req_line = "用户未输入文字" if not user_request else f"用户补充说明：{user_request}"
    else:
        context = "用户通过文字描述提出购物需求。"
        req_line = f"用户需求：{user_request}"

    prompt = f"""你是专业、热情的电商购物顾问，用中文回复顾客。

{context}
{req_line}

候选商品（按推荐优先级）：
{build_product_summary_for_llm(results)}

请写一段给顾客的回复（150～280字），要求：
1. 语气自然、像真人导购，不要用「匹配度」「向量」「检索」等技术词
2. 明确推荐 1 款首选并简要说明理由（款式/场景/性价比）
3. 简要提及其他 2～3 款备选
4. 结尾给 1 条贴心购物建议（尺码、搭配或预算）
5. 不要使用 Markdown 标题符号 #，可用 emoji 点缀"""

    try:
        return llm_fn(prompt).strip()
    except Exception:
        return None
