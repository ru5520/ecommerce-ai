"""[已弃用] 请改用 build_product_library.py

Stable Diffusion 需要下载大模型且依赖 Hugging Face 网络。
在当前环境下容易失败，推荐用离线脚本一键生成商品库：

    python scripts/scripts/build_product_library.py --per-category 200
"""

import sys

print(__doc__)
sys.exit(1)
