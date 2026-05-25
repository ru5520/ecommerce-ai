"""顾客账号注册与登录的边界用例。"""

from __future__ import annotations

from api import user_auth


def test_register_and_login(temp_data):
    ok, _ = user_auth.register("alice", "secret123")
    assert ok
    ok, _, session = user_auth.login("alice", "secret123")
    assert ok and session and session["username"] == "alice"


def test_login_wrong_password(temp_data):
    user_auth.register("alice", "secret123")
    ok, msg, session = user_auth.login("alice", "wrong")
    assert not ok and session is None


def test_register_validation(temp_data):
    ok, msg = user_auth.register("a", "secret123")
    assert not ok and "用户名" in msg

    ok, msg = user_auth.register("alice", "123")
    assert not ok and "密码" in msg


def test_no_duplicate_username(temp_data):
    assert user_auth.register("alice", "secret123")[0]
    ok, msg = user_auth.register("alice", "anotherone")
    assert not ok and "已存在" in msg


def test_password_hash_uses_salt(temp_data):
    user_auth.register("alice", "secret123")
    user_auth.register("bob", "secret123")
    db = user_auth._load_db()
    assert db["users"]["alice"]["salt"] != db["users"]["bob"]["salt"]
    assert db["users"]["alice"]["password_hash"] != db["users"]["bob"]["password_hash"]


def test_cart_persistence(temp_data):
    user_auth.register("alice", "secret123")
    cart = [{"id": 1, "name": "Test", "price": 99.0, "qty": 1}]
    user_auth.save_user_cart("alice", cart)
    loaded = user_auth.load_user_cart("alice")
    assert loaded == cart
