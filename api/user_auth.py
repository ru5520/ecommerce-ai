"""顾客账号：注册、登录（本地 JSON 存储，演示用）。"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
USERS_FILE = BASE_DIR / "data" / "users.json"
CART_DIR = BASE_DIR / "data" / "carts"

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\u4e00-\u9fff]{2,32}$")


def _load_db() -> dict:
    if not USERS_FILE.is_file():
        return {"users": {}}
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_db(db: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return salt, digest.hex()


def register(username: str, password: str, email: str = "") -> tuple[bool, str]:
    username = (username or "").strip()
    password = password or ""
    if not _USERNAME_RE.match(username):
        return False, "用户名需 2～32 位，可用中文、字母、数字、下划线"
    if len(password) < 6:
        return False, "密码至少 6 位"

    db = _load_db()
    if username in db["users"]:
        return False, "用户名已存在，请直接登录或换一个"

    salt, pwd_hash = _hash_password(password)
    db["users"][username] = {
        "salt": salt,
        "password_hash": pwd_hash,
        "email": (email or "").strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_db(db)
    return True, f"注册成功，欢迎 {username}！"


def login(username: str, password: str) -> tuple[bool, str, dict | None]:
    username = (username or "").strip()
    db = _load_db()
    user = db["users"].get(username)
    if not user:
        return False, "用户名或密码错误", None

    salt, pwd_hash = _hash_password(password or "", user["salt"])
    if pwd_hash != user["password_hash"]:
        return False, "用户名或密码错误", None

    session = {"username": username, "email": user.get("email", "")}
    return True, f"登录成功，您好 {username}！", session


def user_cart_path(username: str) -> Path:
    CART_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", username)
    return CART_DIR / f"{safe}.json"


def load_user_cart(username: str | None) -> list[dict]:
    if not username:
        return []
    path = user_cart_path(username)
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_user_cart(username: str | None, cart: list[dict]) -> None:
    if not username:
        return
    path = user_cart_path(username)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cart, f, ensure_ascii=False, indent=2)


def get_store_stats() -> dict:
    from api.product_admin import list_all_products

    products = list_all_products()
    visible = [p for p in products if p.get("visible")]
    return {
        "total": len(products),
        "visible": len(visible),
        "hidden": len(products) - len(visible),
        "featured": sum(1 for p in products if p.get("featured") and p.get("visible")),
    }


def format_store_stats() -> str:
    s = get_store_stats()
    return (
        f"| 指标 | 数量 |\n|------|------|\n"
        f"| 商品总数 | {s['total']} |\n"
        f"| 商城展示中 | {s['visible']} |\n"
        f"| 已隐藏 | {s['hidden']} |\n"
        f"| 精选推荐 | {s['featured']} |"
    )
