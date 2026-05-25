"""商城浏览 / 筛选 / 购物袋 业务逻辑。"""

from __future__ import annotations

from api import store


def test_filter_by_category(temp_data):
    items, total, _ = store.filter_products("Apparel", "", 1)
    assert total == 1
    assert items[0]["category"] == "Apparel"


def test_search_keyword(temp_data):
    items, total, _ = store.filter_products("全部", "sneaker", 1)
    assert total == 1
    assert "Sneakers" in items[0]["name"]


def test_invisible_products_hidden(temp_data, monkeypatch):
    from api import product_admin

    df = product_admin._read_df()
    df.loc[df["id"] == 1, "visible"] = 0
    product_admin._save_df(df)
    store.reload_products()
    items, total, _ = store.filter_products("全部", "", 1)
    assert total == 1
    assert all(p["id"] != 1 for p in items)


def test_add_and_remove_cart(temp_data):
    cart = []
    cart = store.add_to_cart(cart, 1, 2)
    assert len(cart) == 1 and cart[0]["qty"] == 2

    cart = store.add_to_cart(cart, 1, 1)
    assert cart[0]["qty"] == 3

    cart = store.remove_from_cart(cart, 1)
    assert cart == []


def test_cart_summary_total(temp_data):
    cart = [
        {"id": 1, "name": "A", "price": 100.0, "qty": 2},
        {"id": 2, "name": "B", "price": 50.0, "qty": 1},
    ]
    summary = store.format_cart_summary(cart)
    assert "250" in summary
