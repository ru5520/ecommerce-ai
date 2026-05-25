# 王 硕

📞 13180126319 ｜ ✉ 491308100@qq.com ｜ 💬 QQ 491308100
🔗 [github.com/ru5520](https://github.com/ru5520) ｜ 🌐 [Live Demo](https://huggingface.co/spaces/ruandshuo/ai-product)

---

## 🎯 求职意向

**AI 应用 / 多模态算法 / 数据挖掘 / 大模型应用** 实习生（北京 / 上海 / 杭州 / 深圳 / 远程 均可）

---

## 🎓 教育背景

**东北大学（985）** ｜ 物联网工程 ｜ 2024.09 – 2028.06（在读）

**核心课程**：高等数学（**90+**）· 线性代数（**90+**）· 概率论与数理统计 · Python 程序设计 · 数据结构与算法 · 操作系统 · 计算机网络 · 机器学习基础

---

## 🔥 项目经历

### 🛍️ 智能选购助手 · 多模态 AI 电商系统 ｜ 个人独立开发 · 2026.04 – 2026.05

🔗 **源码**：[github.com/ru5520/ecommerce-ai](https://github.com/ru5520/ecommerce-ai) ｜ 🌐 **Demo**（24h 在线）：[huggingface.co/spaces/ruandshuo/ai-product](https://huggingface.co/spaces/ruandshuo/ai-product)

**技术栈**：Python · PyTorch · HuggingFace Transformers · **CLIP** · **FAISS** · **DeepSeek LLM** · Gradio · FastAPI · Docker · pytest

**项目背景**：从零设计、开发、测试、部署一个真实可用的多模态 AI 电商系统，覆盖顾客购物到商家发货全闭环，重点演示 AI 检索与电商业务的工程化整合。

**核心贡献**：

- **多模态语义检索引擎**：基于 OpenAI CLIP (ViT-B/32) 把商品图与查询文本统一映射到 512 维向量空间，FAISS `IndexFlatIP` 余弦相似度近邻搜索，"以图找款"与"文字找款"**共用同一索引**，单次检索 < 80ms

- **LLM 双层流水线导购**：接入 DeepSeek API：① 中文 → 英文查询改写（适配 CLIP 训练分布）② 基于真实检索结果生成"购物顾问"语气文案；LLM 不可用时**自动降级**为本地中文短语词典 + 关键词加权检索

- **完整电商业务闭环**：用户认证（PBKDF2-SHA256 + 随机 salt）· 购物袋 · 结算下单 · **订单状态机**（pending → paid → shipped → completed / cancelled）· **库存原子扣减 + 失败回滚** · 商家 CRUD 与发货后台

- **性能优化**：CLIP 索引磁盘缓存（CSV 指纹失效校验），**冷启动 3 min → 5 s**；商品图自动缩略图缓存，**公网传输体积 ↓ 80%**

- **工程化交付**：21 个 pytest 单元测试覆盖订单状态机 / 库存 / 鉴权 / 检索（**全部通过**）· 多阶段 Dockerfile + docker-compose · **一键部署至 Hugging Face Spaces** 永久公网访问

---

### 📊 Kaggle 数据挖掘三连击 ｜ 个人 · 2026.03 – 2026.04

| 比赛 | 任务 | 最终方案 | 成绩 |
|------|------|---------|------|
| **House Prices - Advanced Regression** | 房价回归（美国 Ames） | **XGBoost (60%) + LightGBM (40%)** 模型融合 · log1p 目标变换 · 特征工程（TotalSF / HouseAge / TotalBath / Remodeled）· One-Hot 合并编码 | **RMSE 0.12626** · **Top 15%** |
| **Store Sales Forecasting** | 多店多品类时序销量预测 | XGBoost 时序回归 · **递归预测**（每日预测追加到历史序列）· lag/rolling 特征（1/7/14/28 天）· 油价/节假日/门店多源融合 · 双方案融合提交 | 验证集 log-RMSE 已收敛 |
| **Titanic - ML from Disaster** | 二分类乘客生存预测 | RandomForest + 特征工程（Title / FamilySize / Pclass×Sex / AgeBand / Deck）· 误差分析 + 特征重要性可视化 | **0.78229** · 完整工程化 |

🔗 [house-prices](https://github.com/ru5520/house-prices) · [titanic-survival](https://github.com/ru5520/titanic-survival)

**核心技术沉淀**：
- **缺失值策略**：区分"语义缺失"（如 `PoolQC = None` 代表无泳池）vs "真实缺失"（用中位数/众数填补），避免简单丢行损失信息
- **目标变量变换**：log1p 缓解右偏分布，回归 RMSE 显著改善
- **类别特征处理**：训练集 + 测试集**合并编码**避免类别不一致
- **递归时序预测**：每日预测值追加到历史序列动态生成 lag/rolling 特征，相比固定 lag 更贴近实际推理场景
- **模型评估**：每个比赛都做 5 折交叉验证 · 特征重要性可视化 · 模型对比图 · 误差分析

---

### 🤖 AI 求职助手 · JD 分类与简历匹配 ｜ 个人 · 2026.03

🔗 [github.com/ru5520/ai-job-assistant](https://github.com/ru5520/ai-job-assistant) ｜ 同期姊妹项目：[resume-jd-matcher](https://github.com/ru5520) · [job-market-analyzer](https://github.com/ru5520)

**技术栈**：Python · matplotlib · 正则 · pandas

**核心实现**：
- 基于**可配置岗位词典**（Model_Algo / AI_Infra / Perception / Decision_Planning / Engineering_Tooling 五大类）+ 加权关键词，对 JD 文本做方向归类与置信度评分
- 中英文混合文本处理：正则清洗 + 中文短语精确匹配 + 英文单词边界识别（`\b` 防止 "ai" 被 "main" 误匹配）
- 简历 ↔ JD 关键词**交集 / 缺失分析**：高频未覆盖词标记为"高优先级缺失"，输出可视化雷达图
- 配套小项目：`resume-jd-matcher`（命令行匹配率计算）· `job-market-analyzer`（pandas 招聘数据分析、薪资排名、技能 Top10 词频）

---

## 🏆 竞赛与奖项

- **第 15 届蓝桥杯软件大赛** · Python 程序设计大学 B 组 · **省级二等奖**（2026）
- **Kaggle House Prices** · Top 15%（RMSE 0.12626）
- **Kaggle Titanic** · Public Score 0.78229
- 东北大学体能测试 · 优秀

---

## 🛠️ 技术栈

- **编程**：Python（熟练）· SQL · Git / GitHub · Linux 基础 · PowerShell
- **AI / ML**：PyTorch · HuggingFace Transformers · **CLIP** · **FAISS** · **XGBoost / LightGBM / scikit-learn**
- **大模型应用**：OpenAI / DeepSeek API · **Prompt Engineering** · **RAG** · 多轮对话
- **数据科学**：pandas · numpy · matplotlib · seaborn · 特征工程 · 模型融合 · 时序预测
- **工程框架**：FastAPI · Gradio · Docker · pytest · 多阶段构建
- **AI 辅助开发**：Cursor · Claude Code · 70%+ 代码 AI 辅助生成

---

## 🌟 个人特质

- **强自驱学习**：物联网专业背景，自学完成 PyTorch / 多模态 / LLM 应用栈，独立交付**上线级**项目
- **抗压能力**：项目全流程独立完成，遇到 PyTorch DLL / Gradio 隧道 / HF Spaces 版本兼容等多个陌生问题均独立排查解决
- **持续跟进前沿**：长期关注 InternVL / LLaVA / Qwen-VL / GLM-4 等多模态大模型进展
- **身体素质**：长期保持良好作息与体能，能承担高强度长期实习
