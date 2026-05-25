"""AI 智能选购助手 — 商业化 Gradio 界面。

    python frontend/gradio_app.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FORCE_KEYWORD = os.environ.get("FORCE_KEYWORD", "").strip() in ("1", "true", "yes")
MODE = "keyword" if FORCE_KEYWORD else "clip"
INIT_ERROR = ""
CLIP_PRELOADED = False

if not FORCE_KEYWORD:
    try:
        import torch  # noqa: F401 — 主线程预加载，避免 Gradio 子线程 c10.dll 失败
    except Exception as exc:
        MODE = "keyword"
        INIT_ERROR = str(exc)

import re

import gradio as gr
import pandas as pd
from PIL import Image

from api.admin_auth import password_configured, verify_password
from api.user_auth import (
    format_store_stats,
    load_user_cart,
    login as user_login,
    register as user_register,
    save_user_cart,
)
from api.image_thumbs import display_image_path, warm_thumbs
from api.network_utils import print_access_urls
from api.shopping_ui import (
    gallery_label,
    generate_reply_with_llm,
    template_reply,
)
from api.product_admin import (
    DEFAULT_CATEGORIES,
    admin_choices,
    create_product,
    delete_product,
    get_display_editor_df,
    get_product_fields,
    save_display_editor_df,
    update_product,
)
from api.store import (
    add_to_cart,
    cart_remove_choices,
    category_label,
    clear_cart,
    filter_products,
    format_cart_summary,
    format_product_detail,
    get_featured_products,
    get_product,
    list_categories,
    product_choices,
    products_to_gallery,
    remove_from_cart,
)
from api.orders import (
    STATUS_LABEL,
    STATUS_PAID,
    STATUS_PENDING,
    STATUS_SHIPPED,
    cancel_order,
    complete_order,
    create_order,
    format_order_brief,
    get_order,
    list_user_orders,
    order_stats,
    orders_table_for_admin,
    orders_table_for_user,
    pay_order,
    ship_order,
)

clip_search_image = None
clip_search_text = None
keyword_search_text = None
rewrite_query = None
llm_chat = None

from api.search_fallback import search_by_text as keyword_search_text

if not FORCE_KEYWORD:
    try:
        from api.catalog import search_by_image as clip_search_image
        from api.catalog import search_by_text as clip_search_text
        from api.llm_agent import chat as llm_chat, rewrite_query
    except Exception as exc:
        MODE = "keyword"
        INIT_ERROR = str(exc)

if rewrite_query is None:
    try:
        from api.llm_agent import rewrite_query
    except Exception:
        rewrite_query = None

if llm_chat is None:
    try:
        from api.llm_agent import chat as llm_chat
    except Exception:
        llm_chat = None

try:
    from api.llm_agent import is_available as llm_available
except Exception:
    def llm_available() -> bool:
        return False


def _preload_clip() -> None:
    global MODE, INIT_ERROR, CLIP_PRELOADED
    if FORCE_KEYWORD or MODE != "clip":
        return
    try:
        from api.catalog import ensure_ready

        print("Preloading CLIP + product index (about 20–60s on first run)...")
        ensure_ready()
        from api.store import list_visible_products

        paths = [p["image_path"] for p in list_visible_products()]
        n = warm_thumbs(paths)
        print(f"Thumbnails ready ({n}/{len(paths)}).")
        CLIP_PRELOADED = True
        print("CLIP ready — image search enabled.")
    except Exception as exc:
        MODE = "keyword"
        INIT_ERROR = str(exc)
        CLIP_PRELOADED = False
        print(f"CLIP preload failed, keyword mode only: {exc}")


_preload_clip()


def resolve_image(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / path)


def build_advisor_reply(
    results: list[dict],
    user_request: str,
    *,
    search_mode: str,
    use_advisor: bool,
) -> str:
    if use_advisor and llm_available() and llm_chat and results:
        reply = generate_reply_with_llm(
            user_request,
            results,
            search_mode=search_mode,
            llm_fn=llm_chat,
        )
        if reply:
            return reply
    return template_reply(user_request, results, search_mode=search_mode)


def format_results(
    results: list[dict],
    *,
    user_request: str = "",
    search_mode: str = "text",
    use_advisor: bool = True,
) -> tuple[list, str]:
    gallery = [
        (display_image_path(resolve_image(item["image_path"])), gallery_label(item))
        for item in results
    ]
    message = build_advisor_reply(
        results,
        user_request,
        search_mode=search_mode,
        use_advisor=use_advisor,
    )
    return gallery, message


def search_by_image(image_path, use_advisor):
    if not image_path:
        return [], "请先上传一张参考图，我会为您在商品库中找相似款式。"
    if MODE != "clip" or clip_search_image is None:
        return [], (
            "图片找款功能暂时不可用，您可以在「文字选购」里描述想要的款式"
            "（例如：蓝色衬衫、休闲连衣裙）。\n\n"
            "如需恢复以图搜款，请重启应用并确保 PyTorch 正常加载。"
        )
    try:
        image = Image.open(image_path).convert("RGB")
        results = clip_search_image(image, top_k=6)
        return format_results(
            results,
            user_request="",
            search_mode="image",
            use_advisor=use_advisor,
        )
    except Exception as exc:
        return [], f"抱歉，找款时出了点问题，请换一张图片再试，或改用文字描述您的需求。\n（{exc}）"


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def recommend_by_text(query, use_llm, use_advisor):
    if not query.strip():
        return [], "请告诉我您想买什么，例如：蓝色衬衫、通勤连衣裙、男士手表。"

    original = query.strip()
    rewritten = original

    should_llm = use_llm or (_has_chinese(original) and llm_available())
    if should_llm and rewrite_query:
        try:
            rewritten = rewrite_query(original)
        except Exception:
            pass

    if MODE == "clip" and clip_search_text is not None:
        try:
            results = clip_search_text(rewritten, top_k=6)
            if not results:
                results = keyword_search_text(rewritten, top_k=6)
            return format_results(
                results,
                user_request=original,
                search_mode="text",
                use_advisor=use_advisor,
            )
        except Exception:
            results = keyword_search_text(rewritten, top_k=6)
            if not results:
                results = keyword_search_text(original, top_k=6)
            return format_results(
                results,
                user_request=original,
                search_mode="text",
                use_advisor=use_advisor,
            )

    results = keyword_search_text(rewritten, top_k=6)
    if not results and rewritten != original:
        results = keyword_search_text(original, top_k=6)
    return format_results(
        results,
        user_request=original,
        search_mode="text",
        use_advisor=use_advisor,
    )


# ---------- 商品商城 ----------

def _category_dropdown():
    return [("全部品类", "全部")] + [
        (category_label(c), c) for c in list_categories()[1:]
    ]


def browse_store(category, search, page):
    page = max(1, int(page or 1))
    items, total, pages = filter_products(category, search, page)
    first_id = items[0]["id"] if items else None
    img, detail = format_product_detail(get_product(first_id))
    page_info = f"共 **{total}** 件 · 第 **{page}** / **{pages}** 页"
    featured = products_to_gallery(get_featured_products())
    return (
        featured,
        products_to_gallery(items),
        gr.update(choices=product_choices(items), value=first_id),
        img,
        detail,
        page_info,
        page,
    )


def select_product(product_id):
    img, detail = format_product_detail(get_product(product_id))
    return img, detail


def store_prev_page(category, search, page):
    return browse_store(category, search, max(1, int(page or 1) - 1))


def store_next_page(category, search, page):
    _, _, pages = filter_products(category, search, int(page or 1))
    return browse_store(category, search, min(pages, int(page or 1) + 1))


def store_add_cart(cart, user_session, product_id, quantity):
    new_cart = add_to_cart(cart, product_id, quantity)
    username = (user_session or {}).get("username") if user_session else None
    save_user_cart(username, new_cart)
    return (
        new_cart,
        format_cart_summary(new_cart),
        gr.update(choices=cart_remove_choices(new_cart), value=None),
    )


def store_remove_cart(cart, user_session, product_id):
    new_cart = remove_from_cart(cart, product_id)
    username = (user_session or {}).get("username") if user_session else None
    save_user_cart(username, new_cart)
    return (
        new_cart,
        format_cart_summary(new_cart),
        gr.update(choices=cart_remove_choices(new_cart), value=None),
    )


def store_clear_cart(_cart, user_session):
    username = (user_session or {}).get("username") if user_session else None
    save_user_cart(username, [])
    return [], format_cart_summary([]), gr.update(choices=[], value=None)


# ---------- 结算 / 我的订单 ----------

ADMIN_ORDER_HEADERS = ["订单号", "顾客", "金额", "件数", "状态", "收件人", "电话", "下单时间", "ID"]
USER_ORDER_HEADERS = ["订单号", "金额", "件数", "状态", "下单时间", "ID"]


def _user_order_choices(username: str) -> list[tuple[str, int]]:
    return [
        (f"{o['order_no']} · ¥{o['total']:.0f} · {STATUS_LABEL.get(o['status'], o['status'])}", o["id"])
        for o in list_user_orders(username)
    ]


def _admin_order_choices(status: str | None) -> list[tuple[str, int]]:
    from api.orders import list_all_orders

    return [
        (f"{o['order_no']} · {o['username']} · ¥{o['total']:.0f} · {STATUS_LABEL.get(o['status'], o['status'])}", o["id"])
        for o in list_all_orders(status)
    ]


def checkout_submit(user_session, cart, receiver, phone, address):
    username = (user_session or {}).get("username") if user_session else None
    if not username:
        return (
            "❌ 请先在「登录 / 注册」Tab 登录后再结算。",
            cart,
            format_cart_summary(cart),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    ok, msg, order = create_order(username, cart, receiver=receiver, phone=phone, address=address)
    if not ok:
        return (
            f"❌ {msg}",
            cart,
            format_cart_summary(cart),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    save_user_cart(username, [])
    detail = format_order_brief(order)
    return (
        f"✅ {msg}\n\n{detail}\n\n*点击「模拟支付」即可推进到「待发货」。*",
        [],
        format_cart_summary([]),
        gr.update(choices=[], value=None),
        gr.update(choices=_user_order_choices(username), value=order["id"]),
        gr.update(value=orders_table_for_user(username)),
    )


def user_select_order(user_session, order_id):
    username = (user_session or {}).get("username") if user_session else None
    if not (username and order_id):
        return "选择一个订单查看详情。"
    order = get_order(int(order_id))
    if not order or order.get("username") != username:
        return "订单不存在或无权限查看。"
    return format_order_brief(order)


def user_pay(user_session, order_id):
    username = (user_session or {}).get("username") if user_session else None
    if not (username and order_id):
        return "请先选择订单。", gr.update(), gr.update()
    order = get_order(int(order_id))
    if not order or order.get("username") != username:
        return "订单不存在或无权限。", gr.update(), gr.update()
    ok, msg, order = pay_order(int(order_id))
    detail = format_order_brief(order) if order else msg
    return (
        ("✅ " if ok else "❌ ") + msg + ("\n\n" + detail if ok else ""),
        gr.update(choices=_user_order_choices(username), value=int(order_id)),
        gr.update(value=orders_table_for_user(username)),
    )


def user_cancel(user_session, order_id):
    username = (user_session or {}).get("username") if user_session else None
    if not (username and order_id):
        return "请先选择订单。", gr.update(), gr.update()
    order = get_order(int(order_id))
    if not order or order.get("username") != username:
        return "订单不存在或无权限。", gr.update(), gr.update()
    ok, msg, order = cancel_order(int(order_id))
    return (
        ("✅ " if ok else "❌ ") + msg,
        gr.update(choices=_user_order_choices(username), value=int(order_id) if order else None),
        gr.update(value=orders_table_for_user(username)),
    )


def user_complete(user_session, order_id):
    username = (user_session or {}).get("username") if user_session else None
    if not (username and order_id):
        return "请先选择订单。", gr.update(), gr.update()
    order = get_order(int(order_id))
    if not order or order.get("username") != username:
        return "订单不存在或无权限。", gr.update(), gr.update()
    ok, msg, order = complete_order(int(order_id))
    return (
        ("✅ " if ok else "❌ ") + msg,
        gr.update(choices=_user_order_choices(username), value=int(order_id)),
        gr.update(value=orders_table_for_user(username)),
    )


def user_refresh_orders(user_session):
    username = (user_session or {}).get("username") if user_session else None
    if not username:
        return (
            gr.update(value=[]),
            gr.update(choices=[], value=None),
            "请先登录后查看订单。",
        )
    return (
        gr.update(value=orders_table_for_user(username)),
        gr.update(choices=_user_order_choices(username), value=None),
        f"共 {len(list_user_orders(username))} 笔订单。",
    )


def admin_orders_refresh(authed, status_filter):
    if not authed:
        return (
            gr.update(value=[]),
            gr.update(choices=[], value=None),
            "请先登录商家后台。",
            "—",
        )
    rows = orders_table_for_admin(status_filter)
    stats = order_stats()
    summary_lines = [f"**订单总数** {stats['total_orders']} · **GMV ¥{stats['revenue']:.0f}**"]
    by = stats["by_status"]
    summary_lines.append(
        " · ".join(f"{STATUS_LABEL[k]} {v}" for k, v in by.items())
    )
    return (
        gr.update(value=rows),
        gr.update(choices=_admin_order_choices(status_filter), value=None),
        f"共 {len(rows)} 笔订单（已按状态筛选）。",
        "\n\n".join(summary_lines),
    )


def admin_order_detail(authed, order_id):
    if not authed or not order_id:
        return "选择一个订单查看详情。"
    order = get_order(int(order_id))
    return format_order_brief(order) if order else "订单不存在。"


def admin_ship(authed, order_id, status_filter):
    if not authed:
        return "请先登录商家后台。", *admin_orders_refresh(authed, status_filter)
    if not order_id:
        return "请选择订单。", *admin_orders_refresh(authed, status_filter)
    ok, msg, _ = ship_order(int(order_id))
    table, picker, hint, stats = admin_orders_refresh(authed, status_filter)
    return ("✅ " if ok else "❌ ") + msg, table, picker, hint, stats


def admin_cancel(authed, order_id, status_filter):
    if not authed:
        return "请先登录商家后台。", *admin_orders_refresh(authed, status_filter)
    if not order_id:
        return "请选择订单。", *admin_orders_refresh(authed, status_filter)
    ok, msg, _ = cancel_order(int(order_id), by_admin=True)
    table, picker, hint, stats = admin_orders_refresh(authed, status_filter)
    return ("✅ " if ok else "❌ ") + msg, table, picker, hint, stats


def do_user_register(username, password, email):
    ok, msg = user_register(username, password, email)
    return msg


def do_user_login(username, password, _session):
    ok, msg, session = user_login(username, password)
    if not ok:
        return None, msg, [], format_cart_summary([]), gr.update(choices=[], value=None)
    cart = load_user_cart(session["username"])
    return (
        session,
        msg,
        cart,
        format_cart_summary(cart),
        gr.update(choices=cart_remove_choices(cart), value=None),
    )


def do_user_logout(_session):
    return None, "已退出登录。您仍可游客浏览商城。", [], format_cart_summary([]), gr.update(choices=[], value=None)


def account_greeting(session):
    if session and session.get("username"):
        return f"👤 当前用户：**{session['username']}**（购物袋已云端保存到本机）"
    return "👤 当前为**游客**模式，登录后购物袋可在下次登录恢复"


def admin_load_fields(product_id):
    f = get_product_fields(product_id)
    return (
        f["name"],
        f["price"],
        f["category"],
        f["description"],
        f["image"],
        f["visible"],
        f["featured"],
        f["sort_order"],
    )


def admin_save_edit(authed, product_id, name, price, category, description, image, visible, featured, sort_order):
    if not authed:
        return "❌ 请先登录商家后台", gr.update()
    msg, choices = update_product(
        product_id, name, price, category, description, image,
        visible=visible, featured=featured, sort_order=int(sort_order or 0),
    )
    return msg, gr.update(choices=choices, value=product_id)


def admin_save_new(authed, name, price, category, description, image, visible, featured):
    if not authed:
        return "❌ 请先登录商家后台", gr.update()
    msg, choices = create_product(
        name, price, category, description, image,
        visible=visible, featured=featured,
    )
    new_id = choices[-1][1] if choices else None
    return msg, gr.update(choices=choices, value=new_id)


def admin_do_delete(authed, product_id):
    if not authed:
        return "❌ 请先登录商家后台", gr.update()
    msg, choices = delete_product(product_id)
    return msg, gr.update(choices=choices, value=None)


def admin_try_login(password, _authed):
    if not password_configured():
        return (
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            "❌ 未设置商家密码。请编辑 `start.ps1`，设置 `$env:ADMIN_PASSWORD = \"你的密码\"` 后重启。",
            get_display_editor_df(),
            gr.update(),
        )
    if verify_password(password):
        return (
            True,
            gr.update(visible=False),
            gr.update(visible=True),
            "✅ 登录成功，欢迎进入商家后台。",
            get_display_editor_df(),
            gr.update(choices=admin_choices(), value=None),
        )
    return (
        False,
        gr.update(visible=True),
        gr.update(visible=False),
        "❌ 密码错误，请重试。",
        get_display_editor_df(),
        gr.update(),
    )


def admin_logout(_authed):
    return (
        False,
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        get_display_editor_df(),
        gr.update(),
    )


def admin_save_display(authed, table):
    if not authed:
        return "❌ 请先登录商家后台", gr.update()
    msg, choices = save_display_editor_df(table)
    return msg, gr.update(choices=choices, value=None)


def admin_refresh_display(authed):
    if not authed:
        return get_display_editor_df(), "请先登录"
    df = get_display_editor_df()
    visible_n = int(df["商城展示"].sum()) if len(df) else 0
    featured_n = int(df["精选推荐"].sum()) if len(df) else 0
    return df, f"库内 {len(df)} 件 · 商城展示 {visible_n} 件 · 精选 {featured_n} 件"


# 用户可见的简洁状态；技术细节放进折叠区
if MODE == "clip" and CLIP_PRELOADED:
    user_status = "✅ 商品商城 · AI 文字选购 · 以图找款"
elif MODE == "clip":
    user_status = "✅ 商品商城 · AI 文字选购可用"
else:
    user_status = "✅ 商品商城 · AI 文字选购可用"

tech_status = []
if INIT_ERROR:
    tech_status.append(f"PyTorch/CLIP: {INIT_ERROR[:300]}")
tech_status.append(
    "DeepSeek: 已启用智能导购" if llm_available() else "DeepSeek: 未配置（使用模板导购文案）"
)
if MODE == "clip" and CLIP_PRELOADED:
    tech_status.append("CLIP 索引: 已加载 200 件商品")

with gr.Blocks(title="智能选购助手") as demo:
    cart_state = gr.State([])
    user_session = gr.State(None)

    account_banner = gr.Markdown("👤 当前为**游客**模式，登录后购物袋可在下次登录恢复")

    gr.Markdown(
        f"""# 🛍️ 智能选购助手

{user_status}

浏览全部商品，或使用 AI 帮您挑款、找相似。

*公网链接（gradio.live）会比本机慢一些，属正常现象；同 WiFi 用手机访问局域网地址会快很多。*
"""
    )

    with gr.Tab("商品商城", id="store"):
        gr.Markdown("### 挑选心仪商品 · 加入购物袋")

        store_featured = gr.Gallery(label="⭐ 精选推荐", columns=4, height=220, object_fit="contain")
        gr.Markdown("---")

        with gr.Row():
            store_category = gr.Dropdown(
                choices=_category_dropdown(),
                value="全部",
                label="品类",
                scale=1,
            )
            store_search = gr.Textbox(
                label="搜索",
                placeholder="输入商品名，如 shirt、watch、Puma…",
                scale=2,
            )
            store_page = gr.State(1)
            store_refresh = gr.Button("搜索 / 刷新", variant="secondary", scale=1)

        store_page_info = gr.Markdown("加载中…")
        store_gallery = gr.Gallery(label="商品列表", columns=4, height=320, object_fit="contain")

        with gr.Row():
            store_prev = gr.Button("← 上一页")
            store_next = gr.Button("下一页 →")

        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                store_pick = gr.Dropdown(label="查看详情（选择商品）", choices=[], value=None)
                store_detail_img = gr.Image(label="商品图", height=320, interactive=False)
            with gr.Column(scale=1):
                store_detail = gr.Markdown("选择一件商品查看详情")
                with gr.Row():
                    store_qty = gr.Number(label="数量", value=1, precision=0, minimum=1, maximum=99)
                    store_add_btn = gr.Button("🛒 加入购物袋", variant="primary")

        gr.Markdown("---")
        gr.Markdown("### 🛒 我的购物袋")
        with gr.Row():
            cart_remove_pick = gr.Dropdown(label="移除商品", choices=[], value=None)
            cart_remove_btn = gr.Button("移除所选")
            cart_clear_btn = gr.Button("清空购物袋")
        cart_summary = gr.Markdown("🛒 购物袋是空的，去挑几件喜欢的吧～")

        gr.Markdown("---")
        gr.Markdown("### 💳 结算下单")
        gr.Markdown("*请先登录后再下单；下单后到「我的订单」点「模拟支付」即可演示完整流程。*")
        with gr.Row():
            ck_receiver = gr.Textbox(label="收件人", placeholder="张三")
            ck_phone = gr.Textbox(label="电话", placeholder="13800138000")
        ck_address = gr.Textbox(label="收货地址", placeholder="北京市朝阳区 xxx 路 xx 号", lines=2)
        ck_submit = gr.Button("🚀 提交订单", variant="primary", size="lg")
        ck_result = gr.Markdown("")

        store_outputs = [
            store_featured,
            store_gallery,
            store_pick,
            store_detail_img,
            store_detail,
            store_page_info,
            store_page,
        ]
        store_inputs = [store_category, store_search, store_page]

        demo.load(browse_store, store_inputs, store_outputs)
        store_refresh.click(browse_store, store_inputs, store_outputs)
        store_category.change(lambda c, s: browse_store(c, s, 1), [store_category, store_search], store_outputs)
        store_search.submit(lambda c, s: browse_store(c, s, 1), [store_category, store_search], store_outputs)
        store_prev.click(store_prev_page, store_inputs, store_outputs)
        store_next.click(store_next_page, store_inputs, store_outputs)
        store_pick.change(select_product, store_pick, [store_detail_img, store_detail])

        store_add_btn.click(
            store_add_cart,
            [cart_state, user_session, store_pick, store_qty],
            [cart_state, cart_summary, cart_remove_pick],
        )
        cart_remove_btn.click(
            store_remove_cart,
            [cart_state, user_session, cart_remove_pick],
            [cart_state, cart_summary, cart_remove_pick],
        )
        cart_clear_btn.click(store_clear_cart, [cart_state, user_session], [cart_state, cart_summary, cart_remove_pick])
        # 结算按钮回调在「我的订单」Tab 之后挂，引用其控件

    with gr.Tab("登录 / 注册", id="account"):
        gr.Markdown("### 顾客账号（可选）")
        gr.Markdown("不登录也可以逛商城；登录后 **购物袋** 会保存到服务器，下次登录自动恢复。")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**已有账号**")
                login_user = gr.Textbox(label="用户名")
                login_pwd = gr.Textbox(label="密码", type="password")
                login_btn = gr.Button("登录", variant="primary")
            with gr.Column():
                gr.Markdown("**新用户注册**")
                reg_user = gr.Textbox(label="用户名")
                reg_pwd = gr.Textbox(label="密码", type="password")
                reg_email = gr.Textbox(label="邮箱（可选）")
                reg_btn = gr.Button("注册")
        account_msg = gr.Markdown("")
        logout_btn = gr.Button("退出登录")

        reg_btn.click(do_user_register, [reg_user, reg_pwd, reg_email], account_msg)
        login_btn.click(
            do_user_login,
            [login_user, login_pwd, user_session],
            [user_session, account_msg, cart_state, cart_summary, cart_remove_pick],
        ).then(account_greeting, user_session, account_banner)
        # 登录后顺手刷新「我的订单」会在 Tab 创建后挂
        logout_btn.click(
            do_user_logout,
            user_session,
            [user_session, account_msg, cart_state, cart_summary, cart_remove_pick],
        ).then(account_greeting, user_session, account_banner)

    with gr.Tab("我的订单", id="orders"):
        gr.Markdown("### 📦 我的订单")
        gr.Markdown("登录后查看您的全部订单。**状态流转**：待付款 → 已付款 → 已发货 → 已完成（演示）。")
        with gr.Row():
            user_order_refresh = gr.Button("刷新订单", variant="secondary")
        user_order_hint = gr.Markdown("请先登录后查看订单。")
        user_order_table = gr.Dataframe(
            headers=USER_ORDER_HEADERS,
            datatype=["str", "str", "number", "str", "str", "number"],
            interactive=False,
            label="订单列表",
            wrap=True,
        )
        with gr.Row():
            user_order_pick = gr.Dropdown(label="选择订单", choices=[], value=None)
        user_order_detail = gr.Markdown("选择一个订单查看详情。")
        with gr.Row():
            user_pay_btn = gr.Button("💳 模拟支付", variant="primary")
            user_complete_btn = gr.Button("✅ 确认收货")
            user_cancel_btn = gr.Button("✖ 取消订单", variant="stop")
        user_order_msg = gr.Markdown("")

        user_order_refresh.click(
            user_refresh_orders,
            user_session,
            [user_order_table, user_order_pick, user_order_hint],
        )
        user_order_pick.change(
            user_select_order,
            [user_session, user_order_pick],
            user_order_detail,
        )
        user_pay_btn.click(
            user_pay,
            [user_session, user_order_pick],
            [user_order_msg, user_order_pick, user_order_table],
        ).then(user_select_order, [user_session, user_order_pick], user_order_detail)
        user_complete_btn.click(
            user_complete,
            [user_session, user_order_pick],
            [user_order_msg, user_order_pick, user_order_table],
        ).then(user_select_order, [user_session, user_order_pick], user_order_detail)
        user_cancel_btn.click(
            user_cancel,
            [user_session, user_order_pick],
            [user_order_msg, user_order_pick, user_order_table],
        ).then(user_select_order, [user_session, user_order_pick], user_order_detail)

    # 现在挂结算回调（要等「我的订单」Tab 的控件先定义）
    ck_submit.click(
        checkout_submit,
        [user_session, cart_state, ck_receiver, ck_phone, ck_address],
        [ck_result, cart_state, cart_summary, cart_remove_pick, user_order_pick, user_order_table],
    )

    with gr.Tab("AI 文字选购", id="text"):
        txt_in = gr.Textbox(
            label="告诉我您想买什么",
            placeholder="例如：蓝色衬衫、约会穿的连衣裙、男士手表、通勤休闲风",
            lines=2,
        )
        with gr.Row():
            use_llm = gr.Checkbox(
                value=llm_available(),
                label="智能理解需求（推荐，支持中文）",
            )
            use_advisor = gr.Checkbox(
                value=True,
                label="AI 导购个性化回复",
            )
        txt_btn = gr.Button("为我推荐", variant="primary", size="lg")
        txt_gallery = gr.Gallery(label="为您挑选", columns=3, height=280, object_fit="contain")
        txt_text = gr.Textbox(label="购物顾问回复", lines=14)

        txt_btn.click(
            recommend_by_text,
            [txt_in, use_llm, use_advisor],
            [txt_gallery, txt_text],
        )

    with gr.Tab("AI 以图找款", id="image"):
        gr.Markdown(
            "上传您喜欢的款式图片，我会在商品库中找相似单品并给出搭配建议。"
            + ("" if MODE == "clip" and CLIP_PRELOADED else "\n\n*以图找款暂不可用，请先用「文字选购」。*")
        )
        img_in = gr.Image(type="filepath", label="上传参考图")
        img_advisor = gr.Checkbox(value=True, label="AI 导购个性化回复")
        img_btn = gr.Button("找相似款", variant="primary", size="lg")
        img_gallery = gr.Gallery(label="相似商品", columns=3, height=280, object_fit="contain")
        img_text = gr.Textbox(label="购物顾问回复", lines=14)
        img_btn.click(search_by_image, [img_in, img_advisor], [img_gallery, img_text])

    with gr.Tab("商家后台", id="admin"):
        admin_authed = gr.State(False)

        gr.Markdown(
            """### 🔐 商家后台

仅商家本人可进入。请先在 `start.ps1` 中设置 **`ADMIN_PASSWORD`**，再在此登录。
"""
        )

        with gr.Group(visible=True) as admin_login_box:
            admin_pwd = gr.Textbox(label="商家密码", type="password", placeholder="请输入后台密码")
            admin_login_btn = gr.Button("登录", variant="primary")

        with gr.Group(visible=False) as admin_panel:
            with gr.Row():
                admin_logout_btn = gr.Button("退出登录")
            admin_msg = gr.Markdown("已登录。")

            with gr.Tab("经营概览"):
                admin_stats = gr.Markdown(format_store_stats())
                stats_refresh = gr.Button("刷新数据")
                stats_refresh.click(lambda: format_store_stats(), None, admin_stats)
                gr.Markdown(
                    """
**快捷说明**
- **展示管理**：控制哪些商品给顾客看、哪些上精选
- **编辑商品**：改价格、换图、上下架
- **上架新商品**：添加新品

**访问方式**（见终端启动日志）
- 本机：`http://127.0.0.1:7860`
- 手机同 WiFi：用电脑局域网 IP
- 临时公网：`GRADIO_SHARE=1` 生成 72 小时链接
- 永久网站：部署到云服务器（见 README）
"""
                )

            with gr.Tab("展示管理"):
                gr.Markdown(
                    "勾选 **商城展示** 决定顾客能否看到；勾选 **精选推荐** 会出现在商城顶部。"
                    " **排序** 数字越小越靠前。"
                )
                display_hint = gr.Markdown("")
                display_table = gr.Dataframe(
                    headers=["货号", "商品名", "价格", "品类", "库存", "商城展示", "精选推荐", "排序"],
                    datatype=["number", "str", "number", "str", "number", "bool", "bool", "number"],
                    interactive=True,
                    label="全部商品展示设置",
                )
                with gr.Row():
                    display_refresh_btn = gr.Button("刷新列表")
                    display_save_btn = gr.Button("💾 保存展示设置", variant="primary")

            with gr.Tab("订单管理"):
                gr.Markdown("商家发货、取消订单与查看下单记录。**演示用**：模拟支付与发货均为一键推进状态。")
                with gr.Row():
                    admin_order_status = gr.Dropdown(
                        choices=[
                            ("全部", "all"),
                            ("待付款", STATUS_PENDING),
                            ("已付款 · 待发货", STATUS_PAID),
                            ("已发货", STATUS_SHIPPED),
                        ],
                        value="all",
                        label="按状态筛选",
                    )
                    admin_order_refresh_btn = gr.Button("刷新订单", variant="secondary")
                admin_order_stats_md = gr.Markdown("—")
                admin_order_hint = gr.Markdown("登录商家后台后点「刷新订单」。")
                admin_order_table = gr.Dataframe(
                    headers=ADMIN_ORDER_HEADERS,
                    datatype=["str", "str", "str", "number", "str", "str", "str", "str", "number"],
                    interactive=False,
                    label="订单列表",
                    wrap=True,
                )
                with gr.Row():
                    admin_order_pick = gr.Dropdown(label="选择订单", choices=[], value=None)
                admin_order_detail_md = gr.Markdown("选择一个订单查看详情。")
                with gr.Row():
                    admin_ship_btn = gr.Button("📦 标记发货", variant="primary")
                    admin_cancel_btn = gr.Button("✖ 取消订单", variant="stop")
                admin_order_msg = gr.Markdown("")

            with gr.Tab("编辑商品"):
                admin_pick = gr.Dropdown(label="选择商品", choices=admin_choices(), value=None)
                with gr.Row():
                    edit_name = gr.Textbox(label="商品名称", scale=2)
                    edit_price = gr.Number(label="价格（元）", value=999, precision=0, minimum=0)
                edit_category = gr.Dropdown(
                    label="品类",
                    choices=DEFAULT_CATEGORIES,
                    value=DEFAULT_CATEGORIES[0],
                    allow_custom_value=True,
                )
                edit_desc = gr.Textbox(label="商品描述", lines=2)
                edit_image = gr.Image(type="filepath", label="更换图片（不选则保留原图）")
                with gr.Row():
                    edit_visible = gr.Checkbox(value=True, label="在商城展示")
                    edit_featured = gr.Checkbox(value=False, label="精选推荐（置顶）")
                    edit_sort = gr.Number(label="排序", value=0, precision=0)
                with gr.Row():
                    edit_save = gr.Button("💾 保存修改", variant="primary")
                    edit_delete = gr.Button("🗑️ 删除商品", variant="stop")

            with gr.Tab("上架新商品"):
                new_name = gr.Textbox(label="商品名称", placeholder="例如：Puma Men Blue Polo Shirt")
                with gr.Row():
                    new_price = gr.Number(label="价格（元）", value=299, precision=0, minimum=0)
                    new_category = gr.Dropdown(
                        label="品类",
                        choices=DEFAULT_CATEGORIES,
                        value=DEFAULT_CATEGORIES[0],
                        allow_custom_value=True,
                    )
                new_desc = gr.Textbox(label="商品描述（可选）", lines=2)
                new_image = gr.Image(type="filepath", label="商品主图（必填）")
                with gr.Row():
                    new_visible = gr.Checkbox(value=True, label="在商城展示")
                    new_featured = gr.Checkbox(value=False, label="精选推荐")
                new_save = gr.Button("➕ 上架商品", variant="primary")

            gr.Markdown(
                f"""
---
**数据文件**：`{PROJECT_ROOT / "data" / "products.csv"}` · `{PROJECT_ROOT / "data" / "images"}`
"""
            )

        admin_login_btn.click(
            admin_try_login,
            [admin_pwd, admin_authed],
            [admin_authed, admin_login_box, admin_panel, admin_msg, display_table, admin_pick],
        )
        admin_logout_btn.click(
            admin_logout,
            admin_authed,
            [admin_authed, admin_login_box, admin_panel, admin_msg, display_table, admin_pick],
        )

        admin_pick.change(
            admin_load_fields,
            admin_pick,
            [edit_name, edit_price, edit_category, edit_desc, edit_image, edit_visible, edit_featured, edit_sort],
        )
        edit_save.click(
            admin_save_edit,
            [admin_authed, admin_pick, edit_name, edit_price, edit_category, edit_desc, edit_image,
             edit_visible, edit_featured, edit_sort],
            [admin_msg, admin_pick],
        )
        edit_delete.click(
            admin_do_delete,
            [admin_authed, admin_pick],
            [admin_msg, admin_pick],
        )
        new_save.click(
            admin_save_new,
            [admin_authed, new_name, new_price, new_category, new_desc, new_image, new_visible, new_featured],
            [admin_msg, admin_pick],
        )
        display_refresh_btn.click(
            admin_refresh_display,
            admin_authed,
            [display_table, display_hint],
        )
        display_save_btn.click(
            admin_save_display,
            [admin_authed, display_table],
            [admin_msg, admin_pick],
        )

        admin_order_refresh_btn.click(
            admin_orders_refresh,
            [admin_authed, admin_order_status],
            [admin_order_table, admin_order_pick, admin_order_hint, admin_order_stats_md],
        )
        admin_order_status.change(
            admin_orders_refresh,
            [admin_authed, admin_order_status],
            [admin_order_table, admin_order_pick, admin_order_hint, admin_order_stats_md],
        )
        admin_order_pick.change(
            admin_order_detail,
            [admin_authed, admin_order_pick],
            admin_order_detail_md,
        )
        admin_ship_btn.click(
            admin_ship,
            [admin_authed, admin_order_pick, admin_order_status],
            [admin_order_msg, admin_order_table, admin_order_pick, admin_order_hint, admin_order_stats_md],
        ).then(admin_order_detail, [admin_authed, admin_order_pick], admin_order_detail_md)
        admin_cancel_btn.click(
            admin_cancel,
            [admin_authed, admin_order_pick, admin_order_status],
            [admin_order_msg, admin_order_table, admin_order_pick, admin_order_hint, admin_order_stats_md],
        ).then(admin_order_detail, [admin_authed, admin_order_pick], admin_order_detail_md)

    with gr.Accordion("系统信息（开发者）", open=False):
        gr.Markdown("\n".join(f"- {line}" for line in tech_status))

IS_HF_SPACES = bool(os.environ.get("SPACE_ID") or os.environ.get("SYSTEM") == "spaces")

port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")

if IS_HF_SPACES:
    # HF Spaces 自带域名 + HTTPS，不需要 gradio.live 隧道
    share = False
    print(f"Running on Hugging Face Spaces: {os.environ.get('SPACE_ID', 'unknown')}")
else:
    # 本机/局域网/公网：默认开启 gradio.live；仅本机使用时在 start.ps1 设 GRADIO_SHARE=0
    _share_env = os.environ.get("GRADIO_SHARE", "1").strip().lower()
    share = _share_env not in ("0", "false", "no", "off")
    print_access_urls(port)
    if share:
        print("正在创建公网链接（gradio.live），请稍候…")
        print("创建成功后终端会出现: Running on public URL: https://....gradio.live")
        print("把该链接发给外地朋友即可（链接约 72 小时有效，电脑需保持运行）\n")

demo.queue(default_concurrency_limit=4)

_launch_kwargs = dict(
    server_name=host,
    share=share,
    show_error=True,
    theme=gr.themes.Soft(),
)
if not IS_HF_SPACES:
    _launch_kwargs["server_port"] = port

demo.launch(**_launch_kwargs)
