"""Hugging Face Spaces 入口。

HF Spaces 默认运行根目录的 `app.py`。本文件只做一件事：
执行 `frontend/gradio_app.py` 把 Gradio 界面跑起来。

本机仍可使用 `start.ps1` 或直接 `python frontend/gradio_app.py`。
"""

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

runpy.run_path(str(PROJECT_ROOT / "frontend" / "gradio_app.py"), run_name="__main__")
