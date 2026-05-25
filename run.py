"""启动 API 服务。请在已激活 ai_env 的终端中运行:
    python run.py
Gradio 另开终端: python frontend/gradio_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
