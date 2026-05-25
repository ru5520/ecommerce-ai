from playwright.sync_api import sync_playwright
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests
import time

# =========================
# 配置
# =========================

keyword = "游戏本"

# 可选：手动登录后保存的会话文件，见 scripts/scripts/jd_save_login.py
STORAGE_STATE = Path("data/jd_storage_state.json")

IMAGE_DIR = Path("data/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

products = []


def check_page_blocked(page) -> bool:
    """检测是否被京东反爬拦截（跳转到首页/登录页）。"""
    current_url = page.url

    if "passport.jd.com" in current_url or "plogin.m.jd.com" in current_url:
        print("\n[错误] 京东要求登录，当前页面是登录页，不是搜索结果页。")
        print("       原因：Playwright 被识别为自动化浏览器。")
        return True

    if current_url.rstrip("/") in ("https://www.jd.com", "https://www.jd.com/"):
        print("\n[错误] 京东把你重定向到了首页，搜索页没有加载。")
        print("       原因：触发了京东反爬/风控。")
        return True

    return False

# =========================
# 启动浏览器
# =========================

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled"],
    )

    context_kwargs = {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "locale": "zh-CN",
        "viewport": {"width": 1920, "height": 1080},
    }
    if STORAGE_STATE.exists():
        context_kwargs["storage_state"] = str(STORAGE_STATE)
        print(f"Using saved login state: {STORAGE_STATE}")

    context = browser.new_context(**context_kwargs)
    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    page = context.new_page()

    url = f"https://search.jd.com/Search?keyword={quote(keyword)}&enc=utf-8"

    print("Opening JD...")

    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    # 等待页面加载
    time.sleep(5)

    if check_page_blocked(page):
        print("\n建议：")
        print("  1. 运行 python scripts/scripts/jd_save_login.py 手动登录并保存会话")
        print("  2. 再重新运行本脚本")
        print("  3. 若仍失败，京东可能限制了当前 IP，需换网络/代理")
        browser.close()
        raise SystemExit(1)

    # =========================
    # 滚动页面
    # =========================

    print("Scrolling page...")

    for _ in range(8):

        page.mouse.wheel(0, 5000)

        time.sleep(2)

    if check_page_blocked(page):
        print("\n建议：")
        print("  1. 运行 python scripts/scripts/jd_save_login.py 手动登录并保存会话")
        print("  2. 再重新运行本脚本")
        print("  3. 若仍失败，京东可能限制了当前 IP，需换网络/代理")
        browser.close()
        raise SystemExit(1)

    # =========================
    # 获取商品
    # =========================

    items = page.locator(".gl-item")

    count = items.count()

    print(f"Found {count} items")

    if count == 0:
        print("\n[错误] 页面上没有找到商品元素 (.gl-item)。")
        print(f"       当前 URL: {page.url}")
        print("       通常不是选择器写错，而是搜索页根本没加载出来。")
        browser.close()
        raise SystemExit(1)

    # =========================
    # 解析商品
    # =========================

    for idx in range(min(count, 30)):

        try:

            item = items.nth(idx)

            # 商品标题
            title = item.locator(".p-name").inner_text()

            # 商品价格
            price = item.locator(".p-price").inner_text()

            # 图片
            img = item.locator("img").first

            img_url = (
                img.get_attribute("src")
                or img.get_attribute("data-lazy-img")
                or img.get_attribute("source-data-lazy-img")
            )

            if not img_url:
                print("No image")
                continue

            # 补全协议
            if img_url.startswith("//"):
                img_url = "https:" + img_url

            print(f"\n[{idx}]")
            print(title)
            print(price)
            print(img_url)

            # =========================
            # 下载图片
            # =========================

            image_name = f"jd_{idx}.jpg"

            image_path = IMAGE_DIR / image_name

            response = requests.get(img_url)

            with open(image_path, "wb") as f:

                f.write(response.content)

            # =========================
            # 保存商品
            # =========================

            products.append({
                "id": idx + 1,
                "name": title.strip(),
                "image_path": str(image_path).replace("\\", "/"),
                "price": price.strip(),
                "category": keyword
            })

            print("Saved.")

        except Exception as e:

            print("Error:", e)

    browser.close()

# =========================
# 保存 CSV
# =========================

df = pd.DataFrame(products)

df.to_csv("data/products.csv", index=False)

print("\nproducts.csv generated!")