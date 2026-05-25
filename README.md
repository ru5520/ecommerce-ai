---
title: Multimodal AI Ecommerce
emoji: 🛍️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
short_description: CLIP 多模态检索 + DeepSeek 导购 + 完整电商订单闭环
---

# 智能选购助手 · 多模态电商系统

> 一个面向**真实业务场景**的多模态 AI 电商演示项目：用 **CLIP + FAISS** 做图文检索，用 **DeepSeek LLM** 做导购对话和中文查询改写，覆盖 **顾客购物 / 商家运营 / 订单履约** 全闭环。

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)](https://pytorch.org/) [![Gradio](https://img.shields.io/badge/Gradio-6.14-orange)](https://gradio.app/) [![FAISS](https://img.shields.io/badge/FAISS-1.13-green)](https://github.com/facebookresearch/faiss) [![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)](./tests)

---

## ✨ 项目亮点

- **多模态语义检索**：用 **CLIP (ViT-B/32)** 把商品图与查询统一映射到同一向量空间，FAISS 余弦相似度近邻搜索；以图找款 / 文字找款 共用同一索引。
- **AI 导购双层流水线**：DeepSeek **重写中文查询**为英文（适配 CLIP 训练分布）→ CLIP 检索 → DeepSeek **生成个性化推荐文案**。无 LLM 时自动降级为本地关键词检索（含中文→英文映射词典）。
- **完整电商闭环**：顾客登录 → 浏览/精选/筛选/详情 → 购物袋 → 收货地址结算 → **订单状态机**（待付款 → 已付款 → 已发货 → 已完成 / 已取消）→ 库存原子扣减 / 回滚。
- **多角色 UI**：顾客（商城+订单）、商家（**经营概览 / 展示管理 / 编辑 / 上架 / 订单发货**）、未登录访客（仅浏览）。
- **工程化**：
  - CLIP 索引**磁盘缓存**：冷启动 30~60s → 二次启动 < 5s。
  - 商品图**自动缩略图**：公网链接传输体积 ↓ 80%。
  - Gradio Queue 并发限制 + 主线程预加载避免 Windows `c10.dll` 失败。
  - 21 个 `pytest` 单元测试覆盖订单 / 库存 / 认证 / 检索。
  - Docker 多阶段构建，一行 `docker compose up`。

---

## 🧩 系统架构

```
┌────────────────────────────────────────────────────────────────────┐
│                         Gradio 6 UI 层                             │
│  商品商城 │ 登录/注册 │ 我的订单 │ AI 文字选购 │ AI 以图找款 │ 商家后台  │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│                         业务服务层 (api/)                          │
│  store.py    商城浏览 / 购物袋                                       │
│  orders.py   订单状态机 / 库存扣减回滚 (线程锁 + 原子写)               │
│  user_auth.py 注册登录 (PBKDF2-SHA256 + 随机 salt)                  │
│  admin_auth.py 商家密码 (env + hmac.compare_digest)                 │
│  product_admin.py CRUD + CSV 迁移 + 自动备份                        │
│  shopping_ui.py 商业化文案模板                                       │
│  image_thumbs.py LRU 缩略图缓存                                     │
└──────────┬────────────────────────────────┬────────────────────────┘
           │                                │
┌──────────▼──────────────┐    ┌────────────▼────────────────────────┐
│  AI 检索层               │    │  LLM 智能层                          │
│  catalog.py             │    │  llm_agent.py                       │
│  ├─ CLIP 模型 (HF cache)│    │  ├─ rewrite_query()  中→英 查询改写  │
│  ├─ FAISS IndexFlatIP   │    │  └─ chat()           AI 导购回复     │
│  └─ embeddings.npy 缓存 │    │  (无 KEY 时模板降级)                  │
└─────────────────────────┘    └─────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│   数据层  data/                                                  │
│   products.csv (主数据) · users.json · orders.json · carts/     │
│   images/ (原图) · thumbs/ (缩略图) · clip_index/ (向量缓存)     │
└──────────────────────────────────────────────────────────────────┘
```

### 模块职责清单

| 模块 | 功能 | 关键设计 |
|------|------|----------|
| `api/catalog.py` | CLIP 加载、FAISS 索引、文/图检索 | 双锁单例 + 磁盘缓存（`meta.json + .npy`），CSV 指纹失效 |
| `api/orders.py` | 订单生命周期 | 文件锁 + tmp-rename 原子写；扣库存失败自动回滚 |
| `api/product_admin.py` | 商家 CRUD | CSV 自动迁移（新增字段 `visible/featured/sort_order/stock`）；保存前 `.bak` 备份 |
| `api/image_thumbs.py` | 缩略图缓存 | 480px / JPEG q=82；hash 文件名；mtime 比对失效 |
| `api/search_fallback.py` | 无 CLIP 时降级 | 中文短语词典 → 英文 token，按词频打分 |
| `frontend/gradio_app.py` | UI 集成 | 主线程 `import torch` 解决 Windows DLL；6 个 Tab；登录态 / 购物袋 / 订单跨 Tab 共享 |

---

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| 语言 / 运行时 | Python 3.10 |
| AI 模型 | OpenAI CLIP (ViT-B/32) · HuggingFace Transformers |
| 向量检索 | FAISS (`IndexFlatIP`) · NumPy |
| LLM | DeepSeek (OpenAI 兼容 SDK) |
| 前端 | Gradio 6 (Blocks API) |
| 后端框架 | FastAPI (`api/app.py` 提供 REST 接口可选) |
| 数据 | CSV / JSON (演示) · 设计上易迁 SQLite/PostgreSQL |
| 测试 | pytest |
| 打包 | Docker 多阶段构建 + docker-compose |

---

## 🚀 快速启动

### 方式 A：本地 Python (推荐开发)

```powershell
# Windows
python -m venv ai_env
.\ai_env\Scripts\Activate.ps1
pip install -r requirements.txt

# 设置密钥（可选；不设也能跑，仅缺 AI 文案）
$env:ADMIN_PASSWORD = "your_password"
$env:DEEPSEEK_API_KEY = "sk-xxxxx"

# 启动
.\start.ps1                       # 一键脚本
# 或：python frontend/gradio_app.py
```

### 方式 B：Docker (推荐部署)

```bash
docker compose up -d --build
# 访问 http://localhost:7860
```

### 访问地址

| 场景 | URL |
|------|------|
| 本机 | `http://127.0.0.1:7860` |
| 同 WiFi 手机 | `http://<电脑局域网 IP>:7860` |
| 临时公网 (72h) | 启动时设 `GRADIO_SHARE=1`，看终端 `*.gradio.live` |

---

## 🧪 测试

```bash
pytest tests/ -v
```

```
21 passed in 0.79s
```

覆盖：订单全状态流转 · 库存回滚 · 鉴权 · 购物袋 · 商城筛选 · 中文检索降级。

---

## 📊 性能数据 (实测，单机 CPU)

| 操作 | 首次 | 二次 (有缓存) | 说明 |
|------|------|----------------|------|
| CLIP 模型加载 | ~25 s | ~3 s | HF 本地 snapshot |
| 201 商品索引构建 | ~120 s | < 5 s | `embeddings.npy` 命中 CSV 指纹 |
| 文本查询 (CLIP) | < 80 ms | < 80 ms | FAISS IndexFlatIP, top-6 |
| 图片查询 (上传) | ~250 ms | ~250 ms | 224×224 resize + 1 次前向 |
| 关键词降级查询 | < 5 ms | < 5 ms | DataFrame 扫描 + 词频打分 |
| 商品列表首屏 (公网) | ~3 s | ~1.5 s | 缩略图后传输 ↓ 80% |

---

## 🎯 核心业务流程

### 1. AI 选购 (文本) — 中文体验

```
用户：「适合约会的蓝色衬衫」
  ↓  llm_agent.rewrite_query()      [DeepSeek]
"blue shirt for date / casual blue shirt"
  ↓  catalog.search_by_text()        [CLIP text encoder + FAISS]
top-6 商品 (id, name, score)
  ↓  shopping_ui.generate_reply_with_llm()
「为您挑选了这几件蓝色衬衫… 首推这件因为…」
```

### 2. 以图找款

上传参考图 → PIL 转 RGB → 224×224 → `clip.get_image_features` → FAISS 检索 → 返回 top-6。

### 3. 订单履约 (状态机)

```
下单 ─→ pending  ─pay→  paid  ─ship→  shipped  ─complete→  completed
        │ 24h 未付       │ 商家发货       │ 顾客确认收货
        ↓ cancel         ↓ cancel(admin)
     cancelled (库存回滚)
```

库存扣减用 `threading.Lock + 文件锁` 保证并发安全；任意一步异常自动回滚已扣减的商品。

---

## 📁 目录结构

```
ecommerce_ai_project/
├── api/                       # 业务 + AI 服务层
│   ├── catalog.py             # CLIP + FAISS 检索
│   ├── orders.py              # 订单状态机
│   ├── product_admin.py       # 商品 CRUD
│   ├── user_auth.py           # 顾客认证
│   ├── admin_auth.py          # 商家鉴权
│   ├── store.py               # 商城浏览
│   ├── shopping_ui.py         # 商业化文案
│   ├── image_thumbs.py        # 缩略图缓存
│   ├── search_fallback.py     # 关键词降级
│   ├── llm_agent.py           # DeepSeek 客户端
│   └── app.py                 # FastAPI 入口（可选）
├── frontend/
│   └── gradio_app.py          # 完整 6-Tab UI
├── tests/                     # pytest 单元测试
├── data/                      # 商品 / 用户 / 订单 / 图片
├── scripts/                   # HuggingFace 商品库构建脚本
├── Dockerfile                 # 多阶段构建
├── docker-compose.yml
├── requirements.txt           # 锁定版本
├── start.ps1                  # Windows 一键启动
├── INTERVIEW.md               # 面试讲解要点
└── README.md
```

---

## 🔐 安全说明 (演示项目)

- 顾客密码 PBKDF2-SHA256 + 16-byte salt + 120k 迭代。
- 商家密码 `hmac.compare_digest` 防时序攻击，存于环境变量。
- 订单 / 用户文件 tmp-rename 原子写。
- **演示用** SQLite/HTTPS/CSRF/rate-limit 未启用；生产部署见 [INTERVIEW.md](./INTERVIEW.md#可继续优化点)。

---

## 📜 License

仅供面试 / 学习使用。CLIP 与 DeepSeek 受其各自许可证约束。

商品图片来自 [Hugging Face Datasets](https://huggingface.co/datasets) 的公开服饰数据集，仅作演示用途。
