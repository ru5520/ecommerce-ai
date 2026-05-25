"""关键词检索（无 CLIP 时的降级路径）。"""

from __future__ import annotations

from api import search_fallback


def test_english_keyword_hits(temp_data):
    search_fallback.reload_products()
    results = search_fallback.search_by_text("shirt", top_k=5)
    assert any("Shirt" in r["name"] for r in results)


def test_chinese_phrase_mapping(temp_data):
    search_fallback.reload_products()
    results = search_fallback.search_by_text("衬衫", top_k=5)
    assert results, "中文短语应能找到对应英文商品"


def test_empty_query(temp_data):
    search_fallback.reload_products()
    results = search_fallback.search_by_text("", top_k=5)
    assert results == []
