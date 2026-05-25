"""启动时显示本机 / 局域网访问地址。"""

import socket


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def print_access_urls(port: int) -> None:
    lan = get_lan_ip()
    print("\n========== 访问地址 ==========")
    print(f"  本机浏览器:     http://127.0.0.1:{port}")
    if lan != "127.0.0.1":
        print(f"  手机/同 WiFi:   http://{lan}:{port}")
        print("  （手机须与电脑连同一 WiFi；若打不开，见 Windows 防火墙提示）")
    share = __import__("os").environ.get("GRADIO_SHARE", "1").strip().lower() not in ("0", "false", "no", "off")
    if share:
        print("  公网分享:       已开启，启动后复制 Public URL 发给朋友")
    else:
        print("  公网分享:       已关闭（start.ps1 设 GRADIO_SHARE=0 则仅本机/LAN）")
    print("  永久网站:       需部署到云服务器，见 README 部署说明")
    print("==============================\n")
