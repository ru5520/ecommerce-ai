"""订单系统：下单 / 查询 / 发货 / 取消（JSON 持久化 + 文件锁保证并发安全）。

订单状态机：
    pending  → 待支付（下单后默认）
    paid     → 已付款（演示用：点击「模拟支付」立即跳到此状态）
    shipped  → 已发货（商家点「发货」）
    completed→ 已完成（顾客点「确认收货」）
    cancelled→ 已取消（顾客在 pending 状态下可取消，回滚库存）
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from api.product_admin import adjust_stock, get_product_fields

BASE_DIR = Path(__file__).resolve().parent.parent
ORDERS_FILE = BASE_DIR / "data" / "orders.json"

STATUS_PENDING = "pending"
STATUS_PAID = "paid"
STATUS_SHIPPED = "shipped"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"

STATUS_LABEL = {
    STATUS_PENDING: "待付款",
    STATUS_PAID: "已付款 · 待发货",
    STATUS_SHIPPED: "已发货",
    STATUS_COMPLETED: "已完成",
    STATUS_CANCELLED: "已取消",
}

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load() -> dict:
    if not ORDERS_FILE.is_file():
        return {"orders": [], "next_id": 1}
    with open(ORDERS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("orders", [])
    data.setdefault("next_id", len(data["orders"]) + 1)
    return data


def _save(data: dict) -> None:
    ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ORDERS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(ORDERS_FILE)


def _make_order_no(order_id: int) -> str:
    return f"E{datetime.now():%Y%m%d}{order_id:05d}"


def create_order(
    username: str,
    cart: list[dict],
    *,
    receiver: str,
    phone: str,
    address: str,
) -> tuple[bool, str, dict | None]:
    """从购物车创建订单：校验 → 扣库存 → 写订单。

    返回 (是否成功, 消息, 订单对象)。
    """
    if not username:
        return False, "请先登录后再结算（顾客订单会绑定到您的账号）。", None
    if not cart:
        return False, "购物袋是空的，先去挑几件商品吧。", None
    receiver = (receiver or "").strip()
    phone = (phone or "").strip()
    address = (address or "").strip()
    if not (receiver and phone and address):
        return False, "请填写完整的收货人、电话、地址。", None
    if len(phone) < 6:
        return False, "电话号码格式不正确。", None

    items: list[dict] = []
    total = 0.0
    deducted: list[tuple[int, int]] = []

    with _lock:
        # 校验库存并预扣
        for item in cart:
            pid = int(item["id"])
            qty = max(1, int(item.get("qty", 1)))
            info = get_product_fields(pid)
            if not info["name"]:
                _rollback(deducted)
                return False, f"商品 #{pid} 不存在或已下架。", None
            stock_ok = adjust_stock(pid, -qty)
            if not stock_ok:
                _rollback(deducted)
                return False, f"「{info['name']}」库存不足，请减少数量或移除。", None
            deducted.append((pid, qty))
            unit_price = float(item.get("price", info["price"]))
            items.append({
                "id": pid,
                "name": info["name"] or item.get("name", ""),
                "price": unit_price,
                "qty": qty,
                "subtotal": round(unit_price * qty, 2),
                "image_path": item.get("image_path", info.get("image_path", "")),
            })
            total += unit_price * qty

        data = _load()
        order_id = int(data["next_id"])
        order = {
            "id": order_id,
            "order_no": _make_order_no(order_id),
            "username": username,
            "items": items,
            "total": round(total, 2),
            "receiver": receiver,
            "phone": phone,
            "address": address,
            "status": STATUS_PENDING,
            "created_at": _now(),
            "paid_at": None,
            "shipped_at": None,
            "completed_at": None,
            "cancelled_at": None,
        }
        data["orders"].append(order)
        data["next_id"] = order_id + 1
        _save(data)
        return True, f"下单成功！订单号 {order['order_no']}，请在 24 小时内付款。", order


def _rollback(deducted: Iterable[tuple[int, int]]) -> None:
    for pid, qty in deducted:
        adjust_stock(pid, qty)


def list_user_orders(username: str) -> list[dict]:
    if not username:
        return []
    data = _load()
    return sorted(
        [o for o in data["orders"] if o.get("username") == username],
        key=lambda o: o["id"],
        reverse=True,
    )


def list_all_orders(status: str | None = None) -> list[dict]:
    data = _load()
    orders = list(data["orders"])
    if status and status != "all":
        orders = [o for o in orders if o.get("status") == status]
    return sorted(orders, key=lambda o: o["id"], reverse=True)


def get_order(order_id: int) -> dict | None:
    data = _load()
    for o in data["orders"]:
        if int(o["id"]) == int(order_id):
            return o
    return None


def _update_status(order_id: int, expect: set[str], new_status: str, time_key: str | None) -> tuple[bool, str, dict | None]:
    with _lock:
        data = _load()
        for o in data["orders"]:
            if int(o["id"]) != int(order_id):
                continue
            if o["status"] not in expect:
                return False, f"当前状态「{STATUS_LABEL.get(o['status'], o['status'])}」不可执行该操作。", o
            o["status"] = new_status
            if time_key:
                o[time_key] = _now()
            _save(data)
            return True, "操作成功。", o
        return False, "订单不存在。", None


def pay_order(order_id: int) -> tuple[bool, str, dict | None]:
    """模拟支付：直接把订单状态推到 paid。"""
    return _update_status(order_id, {STATUS_PENDING}, STATUS_PAID, "paid_at")


def ship_order(order_id: int) -> tuple[bool, str, dict | None]:
    return _update_status(order_id, {STATUS_PAID}, STATUS_SHIPPED, "shipped_at")


def complete_order(order_id: int) -> tuple[bool, str, dict | None]:
    return _update_status(order_id, {STATUS_SHIPPED}, STATUS_COMPLETED, "completed_at")


def cancel_order(order_id: int, *, by_admin: bool = False) -> tuple[bool, str, dict | None]:
    """取消订单：pending 状态下可取消，回滚库存。"""
    allowed = {STATUS_PENDING}
    if by_admin:
        allowed = {STATUS_PENDING, STATUS_PAID}
    with _lock:
        data = _load()
        for o in data["orders"]:
            if int(o["id"]) != int(order_id):
                continue
            if o["status"] not in allowed:
                return False, f"当前状态「{STATUS_LABEL.get(o['status'], o['status'])}」不可取消。", o
            for item in o["items"]:
                adjust_stock(int(item["id"]), int(item["qty"]))
            o["status"] = STATUS_CANCELLED
            o["cancelled_at"] = _now()
            _save(data)
            return True, "订单已取消，库存已回滚。", o
        return False, "订单不存在。", None


def order_stats() -> dict:
    data = _load()
    counts = {k: 0 for k in STATUS_LABEL}
    revenue = 0.0
    for o in data["orders"]:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
        if o["status"] in (STATUS_PAID, STATUS_SHIPPED, STATUS_COMPLETED):
            revenue += float(o.get("total", 0))
    return {
        "total_orders": len(data["orders"]),
        "by_status": counts,
        "revenue": round(revenue, 2),
    }


def format_order_brief(order: dict) -> str:
    if not order:
        return "（无订单）"
    lines = [
        f"**订单号** `{order['order_no']}` · {STATUS_LABEL.get(order['status'], order['status'])}",
        f"下单时间：{order['created_at']}",
        f"收货人：{order['receiver']} · {order['phone']}",
        f"地址：{order['address']}",
        "",
        "**商品清单**",
    ]
    for it in order["items"]:
        lines.append(f"- {it['name']} × {it['qty']}  ¥{it['price']:.0f}  小计 ¥{it['subtotal']:.0f}")
    lines.append("")
    lines.append(f"**合计：¥{order['total']:.0f}**")
    return "\n".join(lines)


def orders_table_for_admin(status: str | None = None) -> list[list]:
    rows = []
    for o in list_all_orders(status):
        rows.append([
            o["order_no"],
            o["username"],
            f"¥{o['total']:.0f}",
            len(o["items"]),
            STATUS_LABEL.get(o["status"], o["status"]),
            o["receiver"],
            o["phone"],
            o["created_at"][:19].replace("T", " "),
            o["id"],
        ])
    return rows


def orders_table_for_user(username: str) -> list[list]:
    rows = []
    for o in list_user_orders(username):
        rows.append([
            o["order_no"],
            f"¥{o['total']:.0f}",
            len(o["items"]),
            STATUS_LABEL.get(o["status"], o["status"]),
            o["created_at"][:19].replace("T", " "),
            o["id"],
        ])
    return rows


__all__ = [
    "STATUS_PENDING", "STATUS_PAID", "STATUS_SHIPPED", "STATUS_COMPLETED", "STATUS_CANCELLED",
    "STATUS_LABEL",
    "create_order", "list_user_orders", "list_all_orders", "get_order",
    "pay_order", "ship_order", "complete_order", "cancel_order",
    "order_stats", "format_order_brief",
    "orders_table_for_admin", "orders_table_for_user",
]
