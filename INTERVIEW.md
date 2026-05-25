# 面试讲解要点 (INTERVIEW.md)

> 一份给作者自己看的「**项目讲解话术 + 设计决策 + 踩坑总结**」。  
> 面试时按以下顺序展开，每个点都准备了「我做了什么 / 为什么这么做 / 还能怎么优化」。

---

## 一、30 秒电梯陈述

> "这是一个**多模态 AI 电商系统**，覆盖顾客购物到商家发货的完整闭环。技术上用 **CLIP 把商品图和用户查询映射到同一向量空间** 做语义检索，**FAISS** 做近邻搜索，并接入 **DeepSeek LLM** 做中文查询改写和个性化导购文案。项目包含 21 个单元测试和 Docker 部署，重点是把 AI 检索做到工业级体验：冷启动从 3 分钟优化到 5 秒，公网传输体积降低 80%。"

---

## 二、关键技术决策

### 1. 为什么选 CLIP 而不是单独的图像分类器 / 词向量？

- **业务诉求是「以图搜款」+「文字搜款」共用一套索引**。CLIP 把图像和文本投到同一个 512 维空间，余弦相似度直接对齐——一份 embedding 同时服务两种检索。
- 替代方案：ResNet 图像特征 + word2vec 文本特征 → **两个空间不可比**，要再训一层映射；CLIP 已用 4 亿图文对预训练，零样本能力直接可用。

### 2. 为什么用 FAISS `IndexFlatIP` 而不是 IVF / HNSW？

- 商品规模 ~200 件，**精确暴搜 < 1 ms**，没必要近似。
- 留好接口：`embeddings.npy` 持久化后切到 `IndexHNSWFlat` 只需改两行（万级商品才有意义）。

### 3. 为什么 LLM 只做「改写 + 文案」，不直接 RAG？

- **检索精度由 CLIP 保证**，LLM 介入只在两端：
  - **入口**：中文 → 英文关键词（CLIP 训练语料以英文为主）
  - **出口**：把检索结果包装成"购物顾问"语气
- 这样**LLM 失败 / 没 KEY 时整体可降级**（关键词检索 + 模板文案），系统鲁棒性高。

### 4. 为什么用 CSV 而不是 SQLite？

- 演示项目，CSV 可读、可手工编辑、商家能直接打开 Excel 改。
- 工程上做了三层防护：
  - 写入前 `.csv.bak` 自动备份
  - 临时文件 `rename` 原子替换
  - `threading.Lock` 防并发
- **上生产前要换 SQLite/PostgreSQL**，已经在 `INTERVIEW.md` 的可优化点列出。

### 5. 为什么 Gradio 而不是 React + FastAPI？

- 一周内交付**可演示的多模态 AI 产品**，Gradio 把 Image / Gallery / Dataframe 控件做完了。
- 但保留了 `api/app.py`（FastAPI），表明**业务逻辑和 UI 解耦**：换 React 前端只需重新接 REST。

---

## 三、面试时一定要讲的「踩坑 + 解法」

### 坑 1：Windows 上 PyTorch 在 Gradio 子线程里 `c10.dll` 加载失败

```
OSError: [WinError 1114] Error loading c10.dll
```

**原因**：Gradio 把每个请求扔到 worker 线程，PyTorch DLL 必须先在主线程初始化。  
**解法**：`gradio_app.py` 顶部主线程 `import torch` + 启动前 `_preload_clip()` 把模型一次性装好。

### 坑 2：CLIP 索引每次重启都要 2 分钟

201 张商品图 → forward → 加入 FAISS，CPU 上耗时长。  
**解法**：把 `image_embeddings.npy + meta.json (csv_mtime, csv_size, count)` 存盘，下次启动先校验 CSV 指纹，命中就跳过重建。**3 min → 5 s**。

### 坑 3：公网 `gradio.live` 链接极慢

- 隧道走国外节点，每张原图 100-500KB。
- **解法**：`api/image_thumbs.py` 自动生成 480px JPEG q=82 缩略图（hash 文件名），商城列表 / 推荐画廊都用缩略图；只有商品详情用原图。**列表首屏传输体积下降 ~80%**。

### 坑 4：商家保存表格时空行导致 `int()` 崩溃

```
ValueError: invalid literal for int() with base 10: ''
```

**解法**：解析每行用 `try / except (ValueError, TypeError, KeyError): continue`，对脏数据宽容。

### 坑 5：中文查询 CLIP 检索效果差

CLIP 的文本编码器以英文为主。  
**解法**：DeepSeek 在查询前做一次 prompt rewrite（"蓝色衬衫" → "blue shirt"）；没 KEY 时走本地 `ZH_PHRASES` 词典 + 关键词检索。

### 坑 6：商家改商品后 AI 索引不同步

**解法**：`product_admin.refresh_runtime()` 在每次保存后：
- `store.reload_products()` 刷新商城
- `search_fallback.reload_products()` 刷新关键词检索
- `catalog.rebuild_index()` 重建 FAISS（已加载时才重建，未加载不强制启动）

---

## 四、订单系统的关键设计

### 状态机
```
pending → paid → shipped → completed
   ↓        ↓
cancelled (库存回滚)
```

- `pending` 顾客可取消；`paid` 仅商家可取消（演示）。
- 每次状态变更校验**期望前置状态集合**，防止跳状态。

### 库存原子扣减

```python
deducted = []
for item in cart:
    if not adjust_stock(pid, -qty):
        rollback(deducted)        # 全部回滚，避免半数下单
        return False
    deducted.append((pid, qty))
write_order(...)
```

- 进程级 `threading.Lock` + 临时文件原子 rename
- 取消订单时按订单项依次 `adjust_stock(+qty)` 回滚

### 测试覆盖

- 正常状态流转
- 库存不足拒单
- 取消订单库存回滚
- 已发货订单不可取消
- 多用户订单隔离

---

## 五、可继续优化点 (面试官常问 "下一步怎么做")

| 维度 | 现状 | 优化方向 |
|------|------|---------|
| **数据层** | CSV / JSON | SQLite (一行 `pandas.to_sql`) → PostgreSQL (SQLAlchemy ORM) |
| **AI 检索** | 英文 CLIP + DeepSeek 改写 | **Chinese-CLIP / Taiyi-CLIP** 直接吃中文，省一次 LLM 调用 |
| **AI 增强** | 单图 / 单文 | **图+文混合检索**（embedding 加权平均） |
| **AI 增强** | 单品推荐 | **搭配推荐**：CLIP 找相似 + LLM 推荐互补品（衬衫→裤子/包） |
| **支付** | 模拟支付按钮 | 接入微信支付 / 支付宝沙箱 + Webhook |
| **支付** | 单状态机 | 加 `payment_no` / `tracking_no` / 退款单 |
| **运维** | 单容器 | Nginx + Gunicorn + 多 worker，Redis 缓存热数据 |
| **可观测** | print 日志 | structlog + Sentry + Prometheus |
| **测试** | 21 单测 | 加 e2e (Playwright) + 负载 (locust) |
| **安全** | 演示级 | HTTPS / CSRF token / 限流 / 注册验证码 / 越权防护 |
| **CI/CD** | 无 | GitHub Actions: lint + test + 构建 + 推 Registry |

---

## 六、AI 部分的"加分题"答案

### Q: 为什么 CLIP 比 BERT 适合这个场景？
> BERT 是单模态文本模型，给商品做向量需要额外微调；CLIP 预训练就同时优化图文对齐，零样本上线即可用，且支持以图搜款。

### Q: FAISS 在大规模上怎么扩容？
> 当前 `IndexFlatIP` 是精确暴搜；商品到万级用 `IndexHNSWFlat`（图索引，毫秒级近似，召回 95%+）；到百万级上 `IndexIVFPQ` 量化。我把 embeddings 单独存盘正是为这个解耦做准备。

### Q: LLM 怎么避免幻觉？
> 我**没让 LLM 直接推荐商品**，只让它做：(1) 把中文查询改成英文 token；(2) 基于 **已经检索回来的真实商品列表** 写文案。商品 id / 名字 / 价格全部来自 CSV，LLM 只生成措辞。

### Q: 索引失效怎么处理？
> `meta.json` 存了 `csv_mtime`、`csv_size`、`count`、`model_path`。任意字段不匹配就触发重建。商家增删改后调用 `rebuild_index()` 主动重建。

### Q: 怎么衡量推荐效果？
> 演示项目没接埋点。生产中会加：
> - 点击率 (CTR) - top-N 推荐被点的比例
> - 加购率、转化率
> - A/B：纯 CLIP vs CLIP+LLM 改写 vs CLIP+LLM 文案
> - 离线评估：人工标注 100 条查询 → top-5 NDCG

---

## 七、Demo 演示脚本 (3 分钟版)

1. **首页商城** - 浏览精选 + 列表 + 分页 + 详情，加入购物袋
2. **AI 文字选购** - 输入 `适合约会的蓝色衬衫`，展示 (1) DeepSeek 重写 (2) CLIP 检索 (3) AI 文案
3. **AI 以图找款** - 上传一张参考图（淘宝随便截一张），展示返回相似单品
4. **登录注册 + 结算** - 注册账号 → 回到商城下单 → 跳到「我的订单」点「模拟支付」
5. **商家后台** - 输入密码 → 经营概览看 GMV / 订单数 → 订单管理标记发货 → 上架新商品
6. **回到「我的订单」** - 显示订单已发货 → 「确认收货」完成全流程

---

## 八、最容易被问到的 5 个问题（提前练）

1. **介绍一下这个项目** → 见第一部分电梯陈述。
2. **CLIP 是怎么工作的，为什么选它** → 见技术决策 1。
3. **遇到的最大挑战** → 选坑 2（CLIP 冷启动 3 min → 5 s 的缓存设计）。
4. **怎么保证并发安全** → 订单 `threading.Lock` + 原子 rename + 状态机校验。
5. **下一步会怎么做** → 见第五部分表格，重点讲 Chinese-CLIP + 真实支付 + SQLite 迁移。

---

**TL;DR**：这不是一个 demo，是**展示工程取舍能力的多模态 AI 项目**。AI 是亮点，工程闭环、性能优化、降级方案、测试与部署证明这个亮点能落地。
