"""手动登录京东并保存浏览器会话，供 jd_spider.py 使用。"""

from pathlib import Path
from playwright.sync_api import sync_playwright

STORAGE_STATE = Path("data/jd_storage_state.json")
STORAGE_STATE.parent.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        locale="zh-CN",
        viewport={"width": 1920, "height": 1080},
    )
    page = context.new_page()

    print("请在打开的浏览器里手动登录京东。")
    print("登录成功后，回到终端按 Enter 保存会话...")
    page.goto("https://passport.jd.com/new/login.aspx")

    input()

    context.storage_state(path=str(STORAGE_STATE))
    browser.close()

    print(f"会话已保存到: {STORAGE_STATE}")
    print("现在可以运行: python scripts/scripts/jd_spider.py")
