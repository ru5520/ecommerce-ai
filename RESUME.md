# 王 硕 · Wang Shuo

**东北大学 · 物联网工程 · 2028 届本科**

📞 13180126319 ｜ ✉ 491308100@qq.com ｜ 💬 QQ 491308100  
🔗 GitHub: **[github.com/ru5520](https://github.com/ru5520)**  
🌐 Live Demo: **[huggingface.co/spaces/ruandshuo/ai-product](https://huggingface.co/spaces/ruandshuo/ai-product)**

---

## 🎯 求职意向

**AI 应用 / 大模型 Agent / 多模态算法 / 数据挖掘** 实习生  
北京 / 上海 / 杭州 / 深圳 / 远程均可 · 可即时入职

---

## 🎓 教育背景

**东北大学（985）** · 物联网工程 · 2024.09 – 2028.06

**核心课程**：高等数学（**90+**）· 线性代数（**90+**）· 概率论与数理统计 · Python · 数据结构与算法 · 操作系统 · 计算机网络 · 机器学习基础

---

## 🔥 主要项目

### 🛍️ 智能选购助手 · 多模态 AI 电商系统（已上线 · 24h 可访问）

> [🔗 源码 github.com/ru5520/ecommerce-ai](https://github.com/ru5520/ecommerce-ai) · [🌐 Demo huggingface.co/spaces/ruandshuo/ai-product](https://huggingface.co/spaces/ruandshuo/ai-product)  
> **技术栈**：Python · PyTorch · CLIP · FAISS · DeepSeek LLM · Gradio · FastAPI · Docker · pytest

**从零设计并部署上线**的多模态 AI 电商系统，覆盖顾客购物到商家发货全闭环：

- **多模态检索**：CLIP (ViT-B/32) 将商品图与查询文本统一映射到 512 维向量空间，FAISS `IndexFlatIP` 余弦近邻搜索，"以图找款 / 文字找款"共用同一索引，**单次检索 < 80ms**
- **LLM 双层流水线**：DeepSeek ① 中文 → 英文查询改写（适配 CLIP 训练分布）② 基于检索结果生成导购文案；LLM 不可用时自动降级为本地词典 + 关键词检索
- **完整电商闭环**：用户认证（PBKDF2-SHA256）· 购物袋 · 订单状态机（待付款 → 已发货 → 已完成）· **库存原子扣减 + 失败回滚**
- **性能优化**：CLIP 索引磁盘缓存（CSV 指纹校验）**冷启动 3 min → 5 s**；缩略图缓存 **公网传输 ↓ 80%**
- **工程化**：21 个 pytest **全部通过** · 多阶段 Dockerfile · 一键部署 Hugging Face Spaces

---

### 🤖 Agent Lab · 智能体开发 + 领域模型微调（作品集 · 与电商项目联动）

> [🔗 源码 github.com/ru5520/agent-lab](https://github.com/ru5520/agent-lab) · 与上方电商项目**共用商品库与业务场景**  
> **技术栈**：OpenAI SDK · LangChain · LangGraph · FAISS · BGE · PEFT · TRL · QLoRA · Gradio

**分阶段实现「检索 → Agent 编排 → 小模型领域化」完整技术栈**，形成可演示、可讲解的 AI 应用工程师能力证明：

#### 智能购物 Agent（Phase 4 · 简历 Demo）

- 基于 **LangChain Tool Calling Agent**，封装搜索、详情、加购、购物车、模拟下单 **5 个电商工具**，支持自然语言全流程购物
- 编写导购 **System Prompt 与业务规则**（禁止编造商品、下单前信息采集、工具失败友好降级），Gradio 聊天界面 + 多轮对话记忆
- 工具层对接自研电商商品 CSV / 本地订单状态，与多模态 RAG 项目形成 **「RAG 底层 + Agent 中层」** 联动叙事

#### Agent 基础与框架实践（Phase 1–3）

- **手写 Function Calling 运行时**：工具 JSON Schema 注册、`tool_calls` 解析执行与结果回灌，理解 LLM 调工具的本质
- **手写 ReAct 循环**：Thought / Action / Observation 多步推理，对比 Function Calling 与显式推理链的取舍
- **LangChain LCEL + RAG**：FAISS + BGE 中文向量搭建商品库检索增强生成链，约束回答 grounded 于检索结果
- **LangGraph 状态机**：客服意图路由（退货 / 推荐 / 闲聊）；**Planner–Executor–Reviewer** 多 Agent 协作图，含评审打回与防死循环

#### Qwen2.5 导购领域微调（Phase 5 · QLoRA）

- 从电商商品库 **自动生成 Alpaca 格式 SFT 数据**（模板 + 可选 LLM 增强），80/20 训练验证划分
- **Qwen2.5-1.5B + 4bit QLoRA**（PEFT / TRL SFTTrainer，r=16, α=32），单卡可训，建立数据准备 → 训练 → 推理 → 评估可复现流水线
- 评估方案：**ROUGE-L + LLM-as-Judge**，为后续 DPO 对齐与替代 API 降本预留接口

---

### 📊 Kaggle 数据挖掘三连击

| 比赛 | 任务 | 核心方案 | 成绩 |
|------|------|---------|------|
| [**House Prices**](https://github.com/ru5520/house-prices) | 房价回归 | XGBoost + LightGBM 融合 · log1p · 特征工程 · 合并 One-Hot | **RMSE 0.12626 · Top 15%** |
| [**Store Sales**](https://github.com/ru5520/store-sales-forecasting) | 16 天时序销量预测 | lag/rolling 特征 · 递归预测 · 多源融合 · 双方案融合 | 验证集 log-RMSE 收敛 |
| [**Titanic**](https://github.com/ru5520/titanic-survival) | 生存二分类 | RandomForest + Title/FamilySize 等特征工程 | Public Score **0.78229** |

---

### 🔍 AI 求职助手 · JD 智能分析工具

> [🔗 ai-job-assistant](https://github.com/ru5520/ai-job-assistant)

基于可配置岗位词典 + 加权关键词，对 JD 做方向归类与置信度评分；支持简历 ↔ JD 关键词交集 / 缺失分析。

---

## 🏆 竞赛与奖项

- **第 15 届蓝桥杯软件大赛** · Python 程序设计大学 B 组 · **省级二等奖**（2026）
- **Kaggle House Prices** · Top 15%（RMSE 0.12626）
- **Kaggle Titanic** · Public Score 0.78229

---

## 🛠️ 技术栈

| 类别 | 技能 |
|------|------|
| **编程** | Python（熟练）· SQL · Git / GitHub · Linux 基础 |
| **AI / ML** | PyTorch · HuggingFace Transformers · CLIP · FAISS · XGBoost · LightGBM · scikit-learn |
| **大模型应用** | OpenAI / DeepSeek API · **Function Calling** · **ReAct Agent** · **LangChain / LangGraph** · **RAG** · Prompt Engineering |
| **模型微调** | LoRA / **QLoRA** · PEFT · TRL · SFT 数据工程 · LLM-as-Judge 评估 |
| **数据科学** | pandas · numpy · 特征工程 · 模型融合 · 时序预测 |
| **工程框架** | FastAPI · Gradio · Docker · pytest |

---

## 🌟 个人特质

- **强自驱学习**：物联网专业背景，自学 PyTorch / 多模态 / LLM Agent / 微调全栈，独立交付上线级项目
- **体系化思维**：电商 RAG + Agent + 微调三层联动，能讲清「为什么先手写再框架」
- **工程意识**：降级策略、单元测试、部署、性能数字可量化
- **抗压能力**：PyTorch DLL / Gradio 隧道 / HF Spaces 兼容等问题均独立排查解决

---

## 📋 作品集一句话（面试开场用）

> 我做了一条 **电商 AI 完整链路**：底层 CLIP + FAISS 多模态 RAG（已上线 Demo），中层 LangChain 智能购物 Agent（自然语言搜购），上层 Qwen QLoRA 导购微调（降 API 成本）——三个项目共用同一商品场景，不是孤立 Demo。

---

> **🌐 立即查看 Demo**：[huggingface.co/spaces/ruandshuo/ai-product](https://huggingface.co/spaces/ruandshuo/ai-product) · **💻 全部源码**：[github.com/ru5520](https://github.com/ru5520)
