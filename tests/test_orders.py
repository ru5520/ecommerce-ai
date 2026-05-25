"""订单系统的核心流程测试。"""

from __future__ import annotations

from api import orders, product_admin


def test_create_order_deducts_stock(temp_data):
    cart = [
        {"id": 1, "name": "Red Polo Shirt", "price": 199.0, "qty": 2, "image_path": "data/images/product_1.jpg"},
    ]
    ok, msg, order = orders.create_order(
        "alice", cart, receiver="Alice", phone="13800000000", address="Test St"
    )
    assert ok, msg
    assert order["status"] == orders.STATUS_PENDING
    assert order["total"] == 199 * 2
    assert product_admin.get_stock(1) == 5 - 2


def test_order_requires_login_and_cart(temp_data):
    ok, msg, _ = orders.create_order("", [{"id": 1, "qty": 1}], receiver="A", phone="138", address="x")
    assert not ok and "登录" in msg

    ok, msg, _ = orders.create_order("alice", [], receiver="A", phone="138000000", address="x")
    assert not ok and "购物袋" in msg


def test_order_rejected_when_stock_insufficient(temp_data):
    cart = [{"id": 2, "name": "Blue Sneakers", "price": 399.0, "qty": 99, "image_path": "data/images/product_2.jpg"}]
    ok, msg, _ = orders.create_order("alice", cart, receiver="A", phone="13800000000", address="x")
    assert not ok and "库存" in msg
    assert product_admin.get_stock(2) == 2


def test_full_status_flow(temp_data):
    cart = [{"id": 1, "name": "Red Polo Shirt", "price": 199.0, "qty": 1}]
    _, _, order = orders.create_order("alice", cart, receiver="A", phone="13800000000", address="x")
    oid = order["id"]

    ok, _, o = orders.pay_order(oid)
    assert ok and o["status"] == orders.STATUS_PAID

    ok, _, o = orders.ship_order(oid)
    assert ok and o["status"] == orders.STATUS_SHIPPED

    ok, _, o = orders.complete_order(oid)
    assert ok and o["status"] == orders.STATUS_COMPLETED


def test_cancel_restores_stock(temp_data):
    cart = [{"id": 1, "name": "Red Polo Shirt", "price": 199.0, "qty": 3}]
    _, _, order = orders.create_order("alice", cart, receiver="A", phone="13800000000", address="x")
    assert product_admin.get_stock(1) == 5 - 3

    ok, _, o = orders.cancel_order(order["id"])
    assert ok and o["status"] == orders.STATUS_CANCELLED
    assert product_admin.get_stock(1) == 5


def test_cannot_cancel_shipped_order(temp_data):
    cart = [{"id": 1, "name": "Red Polo Shirt", "price": 199.0, "qty": 1}]
    _, _, order = orders.create_order("alice", cart, receiver="A", phone="13800000000", address="x")
    orders.pay_order(order["id"])
    orders.ship_order(order["id"])

    ok, msg, _ = orders.cancel_order(order["id"])
    assert not ok
    ok_admin, _, _ = orders.cancel_order(order["id"], by_admin=True)
    assert not ok_admin


def test_list_user_orders_isolated(temp_data):
    cart = [{"id": 1, "name": "Red Polo Shirt", "price": 199.0, "qty": 1}]
    orders.create_order("alice", cart, receiver="A", phone="13800000000", address="x")
    orders.create_order("bob", cart, receiver="B", phone="13900000000", address="y")
    assert len(orders.list_user_orders("alice")) == 1
    assert len(orders.list_user_orders("bob")) == 1
    assert len(orders.list_all_orders()) == 2
