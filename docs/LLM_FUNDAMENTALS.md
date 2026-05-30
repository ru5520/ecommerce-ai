# LLM 八股完整版（面试 / 笔试可直接背）

> 100 道高频题 + 标准答案，按主题分组。AI 应用岗 / 算法岗笔试 + 一面必备。  
> 学习方式：每天看 1-2 个主题，**用自己的话复述一遍**，不要死背。

---

## 目录

1. [Transformer 基础（必考 25 题）](#一transformer-基础)
2. [大语言模型（LLM）训练与推理（20 题）](#二llm-训练与推理)
3. [RAG 检索增强生成（15 题）](#三rag-检索增强生成)
4. [微调与对齐（10 题）](#四微调与对齐lora--sft--rlhf)
5. [Prompt Engineering（10 题）](#五prompt-engineering)
6. [多模态 / CLIP（10 题）](#六多模态--clip你的项目重点)
7. [向量数据库与检索（10 题）](#七向量数据库与检索)

---

## 一、Transformer 基础

### Q1. Transformer 整体结构？

**答**：Encoder + Decoder。
- **Encoder**：N 层（原论文 6 层），每层 = Multi-Head Self-Attention + FFN，每个子层有残差连接 + LayerNorm
- **Decoder**：N 层，每层 = Masked Self-Attention + Cross-Attention + FFN
- 输入：token embedding + 位置编码

```
Input → Embedding → +PosEmbedding → [Encoder Block × N] → 
       → [Decoder Block × N] → Linear → Softmax → Token
```

### Q2. Self-Attention 公式？

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

- Q, K, V 由输入 X 分别乘 W_Q, W_K, W_V 得到，shape: `[seq, d_k]`
- QKᵀ 得到 `[seq, seq]` 的相似度矩阵
- 除 √d_k 缩放，softmax 归一化成权重
- 加权求和 V 得到输出

### Q3. 为什么除 √d_k？

**答**：d_k 较大时，QKᵀ 的内积值会很大，softmax 进入饱和区，**梯度趋近 0**，训练不动。除 √d_k 让方差归一化到 1。

数学上：假设 q, k 是独立的 d_k 维零均值单位方差向量，q·k 的方差是 d_k，除 √d_k 后方差变 1。

### Q4. Multi-Head Attention 为什么有效？

**答**：
1. 并行多个低维子空间，捕捉**不同语义关系**（语法、指代、长距离依赖等）
2. 每个头的维度 d_model / h 较小，计算量与单头相近但表达力更强
3. 类似 CNN 多 filter，丰富特征

### Q5. 位置编码（PE）？

**Sinusoidal PE**（原版）：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
优点：可外推到训练时未见过的长度。

**Learnable PE**（BERT 用）：当成参数学习，无法外推。

**RoPE（旋转位置编码）**（LLaMA / Qwen 用）：把位置信息编码进 Q/K 的旋转角度，**相对位置友好**，长上下文外推性好。

**ALiBi**：在 attention score 上加位置偏置，无显式 PE。

### Q6. LayerNorm vs BatchNorm？

| | LayerNorm | BatchNorm |
|---|-----------|-----------|
| 归一化维度 | 单样本特征维度 | batch 内同一特征 |
| 依赖 batch | 否 | 是 |
| 推理 | 同训练 | 需要保存 running mean/var |
| 典型场景 | NLP / Transformer | CV / CNN |

NLP 不用 BN 是因为序列长度不定，batch 内样本不可比。

### Q7. Pre-Norm vs Post-Norm？

- **Post-Norm**（原 Transformer）：`x + Sublayer(LN(x))` 外面再做 norm；训练不稳定，需要 warmup
- **Pre-Norm**（GPT-2/LLaMA）：`x + Sublayer(LN(x))`；训练稳定，深层更容易收敛，但顶层退化效应

### Q8. FFN（前馈网络）结构？

```
FFN(x) = max(0, xW₁ + b₁) W₂ + b₂   # 原版 ReLU
```
- 两层全连接，中间维度通常 4 × d_model
- 现代变体：**SwiGLU**（LLaMA 用），效果更好
  ```
  SwiGLU(x) = (Swish(xW₁) ⊙ xW₃) W₂
  ```

### Q9. Encoder 和 Decoder 的 Attention 有什么区别？

- **Encoder Self-Attention**：双向，每个 token 看到全部
- **Decoder Masked Self-Attention**：单向，加上**下三角 mask**，第 i 个 token 只能看到 1~i
- **Decoder Cross-Attention**：Q 来自 decoder，K/V 来自 encoder 输出

### Q10. 为什么 Decoder 要 mask？

**答**：训练时 teacher forcing，整句一起喂；预测第 t 个 token 时不能看到 t 之后的（防作弊）。Mask 把上三角设为 -∞，softmax 后变 0。

### Q11. Transformer 复杂度？

- Self-Attention：**O(n² · d)**（n 是序列长，d 是维度）
- FFN：O(n · d²)
- 长序列瓶颈在 attention 的 n² → 各种 Efficient Transformer（Longformer、Performer、Flash Attention）

### Q12. Flash Attention 原理？

**答**：标准 attention 把 `[n, n]` 矩阵存到 HBM（显存），IO 慢。Flash Attention：
1. **分块计算**，在 SRAM 内做矩阵乘 + softmax
2. **重计算 backward**，省显存
3. 不改算法，**精度无损**，**速度 2-4 倍**

### Q13. Encoder-only / Decoder-only / Encoder-Decoder 各代表？

- **Encoder-only**：BERT / RoBERTa，理解任务（分类、NER）
- **Decoder-only**：GPT / LLaMA / Qwen / DeepSeek，生成任务，主流 LLM 都是这个
- **Encoder-Decoder**：T5 / BART，seq2seq（翻译、摘要）

### Q14. 为什么现在主流是 Decoder-only？

1. **统一架构**：所有任务都看作 next-token prediction
2. **In-context learning** 涌现
3. **KV-Cache** 加速推理
4. 大规模训练下，Decoder-only 比 Encoder-Decoder 更高效

### Q15. KV-Cache 是什么？为什么省时间？

**答**：自回归生成时，每生成一个新 token，前面所有 token 的 K, V 不变，**缓存起来**，避免重复计算。
- 不开 KV-Cache：每步 O(n²)，总 O(n³)
- 开 KV-Cache：每步 O(n)，总 O(n²)
- **代价是显存**：缓存大小 ≈ 2 × n × d × layer × batch

### Q16. 计算 LLaMA-7B 推理时 KV-Cache 大小？

参数：n=2048，d=4096，layer=32，dtype=fp16(2B)
```
KV-Cache = 2(K+V) × 2048 × 4096 × 32 × 2 = 1.07 GB  (单 batch)
```

### Q17. Softmax 数值稳定性问题？

**答**：直接算 `exp(x)` 大数会溢出。技巧：减最大值

```python
x_stable = x - x.max()
softmax = exp(x_stable) / sum(exp(x_stable))
```
不改变结果（分子分母同乘 `exp(-max)`）。

### Q18. Transformer 怎么处理变长序列？

**答**：
1. 同 batch padding 到相同长度
2. attention mask 把 padding 位置设为 -∞
3. loss 计算时忽略 padding

### Q19. Attention 的 mask 有哪几种？

1. **Padding mask**：忽略 padding token
2. **Causal mask**（look-ahead mask）：Decoder 防看未来
3. **Prefix mask**：U-PaLM / GLM 用，前缀双向，后缀单向

### Q20. Embedding 层和输出层是否共享权重？

**答**：可以。GPT-2 / LLaMA 共享输入 embedding 和输出 LM Head 的权重（**Tied Embedding**），省参数。

### Q21. Tokenization 方法？

- **Word-level**：词典大、OOV 严重
- **Char-level**：序列太长
- **Subword**：折中
  - **BPE**（GPT 用）：从字符开始合并最频繁对
  - **WordPiece**（BERT 用）：基于似然合并
  - **Unigram**（SentencePiece）：从大词表开始裁剪
  - **SentencePiece**：处理无空格语言（中日韩）

### Q22. BPE 怎么训练？

1. 初始词表 = 所有字符
2. 统计语料中所有相邻字符对的频率
3. 合并频率最高的对，加入词表
4. 重复 N 次（N = 目标词表大小 - 初始大小）

### Q23. 为什么 LLM 用 BPE 不用 word？

1. **OOV 处理**：词外的字仍可拆成 subword
2. **多语言友好**：跨语言共享 subword
3. **词表可控**：可调整词表大小

### Q24. 解码策略有哪几种？

| 方法 | 说明 |
|------|------|
| **Greedy** | 每步选概率最大，确定但缺多样性 |
| **Beam Search** | 维护 top-k 路径，翻译用 |
| **Sampling** | 按概率分布采样 |
| **Top-k** | 只在前 k 个候选中采样 |
| **Top-p / Nucleus** | 累积概率 p 以内采样 |
| **Temperature** | T<1 更确定，T>1 更随机 |

### Q25. Temperature / Top-k / Top-p 关系？

- **Temperature** 调整 softmax 分布陡峭程度：`softmax(logits / T)`
  - T → 0：等价 greedy
  - T → ∞：均匀分布
- **Top-k**：硬性截断到前 k 个
- **Top-p**：动态截断，累积概率 ≥ p
- 实际常**先 top-p 再 temperature**

---

## 二、LLM 训练与推理

### Q26. LLM 三阶段训练？

1. **Pretraining**：海量无标注语料，目标是 next-token prediction（CLM）
2. **SFT (Supervised Fine-Tuning)**：指令-回答对，让模型学会"听话"
3. **RLHF / DPO**：人类偏好对齐，让回答更有用、无害

### Q27. CLM vs MLM？

- **CLM (Causal LM)**：GPT 系，单向预测下一个 token
- **MLM (Masked LM)**：BERT 系，双向，预测被 mask 的 token
- 大模型主流是 CLM

### Q28. RLHF 流程？

1. **SFT 模型**：先 SFT 出基线
2. **奖励模型 (RM)**：人类对多个回答排序，训练 RM 打分
3. **PPO**：用 RM 当 reward，PPO 优化策略；加 KL 散度防止偏离 SFT 模型太远

### Q29. DPO vs PPO？

- **PPO** 需要奖励模型 + 复杂的强化学习训练，不稳定
- **DPO** 直接用偏好数据优化策略（数学等价），**单阶段、稳定、便宜**
- 现在 DPO 是主流

### Q30. LLM 涌现能力（Emergent Ability）？

**答**：小模型上几乎为 0，模型规模 / 训练数据 / 计算量过临界点后突然出现的能力，如：
- In-context learning
- 算术 / 多步推理
- 遵循复杂指令

### Q31. Scaling Law 是什么？

**答**：模型 loss 随**参数量、数据量、计算量**幂律下降（Kaplan & Hoffmann）。指导：
- 算力够 → 增大模型
- 数据是真瓶颈 → Chinchilla 法则：参数和 token 数 1:20 比例最优

### Q32. 上下文窗口怎么扩展？

1. **RoPE 内插 / 外推**：调整频率基数
2. **YaRN / NTK-aware**：在 RoPE 基础上更聪明地缩放
3. **稀疏 Attention**：Longformer / Sliding Window
4. **微调长上下文**：用长样本继续训

### Q33. 量化（Quantization）？

把 fp32/fp16 压到 int8/int4，省显存、加速推理。

- **PTQ**（训练后量化）：直接量化，简单但精度损失
- **QAT**（量化感知训练）：训练时模拟量化
- **GPTQ / AWQ**：LLM 专用，4bit 也能保留 99% 精度

### Q34. 蒸馏（Distillation）？

**答**：用大模型（teacher）的输出当 soft label，训练小模型（student）。比硬标签信息量大（"猫" vs 概率分布）。

### Q35. 推理加速手段汇总？

| 手段 | 收益 |
|------|------|
| KV-Cache | O(n³) → O(n²) |
| Flash Attention | 显存 ↓，速度 × 2-4 |
| 量化（INT8/INT4） | 显存 ↓ 4-8x |
| Continuous Batching | 吞吐 ↑ |
| 推测解码 (Speculative Decoding) | 小模型预测大模型验证 |
| MoE | 激活部分专家 |
| 模型并行 / 张量并行 | 大模型必须 |

### Q36. vLLM 是什么？

**答**：高吞吐 LLM 推理引擎，核心技术 **PagedAttention**（借鉴 OS 虚拟内存），高效管理 KV-Cache，吞吐比 HuggingFace 高 5-20x。

### Q37. MoE（Mixture of Experts）？

**答**：FFN 层换成多个专家，每个 token 只激活 top-k 个专家（如 Mixtral 8×7B 激活 2 个）。
- 优点：总参数大，激活少，**算力效率高**
- 缺点：训练复杂、负载均衡、显存仍占总参数

### Q38. LLM 幻觉（Hallucination）原因？

1. 训练数据噪声
2. 没有外部知识（RAG 解决）
3. 解码采样随机
4. 长尾事实记不住

**缓解**：RAG、引用来源、低 temperature、Chain-of-Thought 自验证。

### Q39. Chain-of-Thought (CoT)？

**答**：让模型在回答前生成推理过程。
- **Zero-shot CoT**：加一句 "Let's think step by step"
- **Few-shot CoT**：给带推理过程的例子
- **Self-Consistency**：采样多次 CoT 投票

### Q40. In-Context Learning (ICL)？

**答**：不更新参数，仅在 prompt 里给例子，模型即可学会任务。
- 解释：**注意力机制隐式实现梯度下降**（Garg et al.）
- 受限于上下文窗口

### Q41. ReAct / Agent？

**答**：Reason + Act 循环。
```
Thought: 我需要查询天气
Action: search_weather("北京")
Observation: 晴 25℃
Thought: 用户问穿什么，我可以建议短袖
Final Answer: ...
```

### Q42. Function Calling？

**答**：模型输出结构化函数调用，外部执行后把结果喂回。GPT-4 / Claude / Qwen 都支持。

### Q43. LLM 安全 / 越狱（Jailbreak）？

常见攻击：
- **Prompt Injection**：用户输入覆盖系统提示
- **DAN 类**：角色扮演绕过
- **多步诱导**

防御：
- **System Prompt 强化**
- **输入过滤**
- **输出审核**
- **Constitutional AI**

### Q44. 怎么减少 LLM 推理延迟？

1. KV-Cache + Continuous Batching
2. 量化（INT8 / INT4）
3. 推测解码
4. 模型蒸馏到小模型
5. Prefix Caching（系统提示缓存）
6. 流式输出（用户感知延迟 ↓）

### Q45. LLM 评测怎么做？

- **自动指标**：BLEU / ROUGE / BERTScore（生成质量）
- **基准测试**：MMLU / C-Eval（知识）、HumanEval（代码）、GSM8K（数学）
- **LLM-as-Judge**：用 GPT-4 当评委
- **人工评估**：相关性、流畅度、安全性

---

## 三、RAG 检索增强生成

### Q46. RAG 完整链路？

```
用户 Query 
  → 改写 / 扩展 (HyDE / Multi-Query)
  → Embedding 
  → 向量检索 (top-k)
  → BM25 召回 (混合检索)
  → Rerank (Cross-Encoder)
  → Prompt 拼接 (含引用)
  → LLM 生成
  → 后处理 (引用提取 / 安全审核)
```

### Q47. RAG 为什么比纯 LLM 好？

1. **知识可更新**：换文档库就行，不用重训
2. **减幻觉**：基于真实文档生成
3. **可溯源**：能给出引用
4. **领域专精**：私有数据不需要微调

### Q48. Chunk 怎么切？

| 策略 | 说明 |
|------|------|
| 固定长度 | 简单，500-1000 字 + 10-20% overlap |
| 按句子 | NLTK / spaCy 切句 |
| 按段落 / 标题 | Markdown 结构化文档 |
| 语义切分 | 用 embedding 相似度判边界 |
| 父子切分 | 大块给 LLM，小块用于检索 |

**经验**：太短上下文不足，太长检索精度差，**500 字 + 50 字 overlap** 是稳妥起点。

### Q49. Embedding 模型选什么？

| 模型 | 特点 |
|------|------|
| OpenAI text-embedding-3-small | 商用，1536 维，多语言强 |
| BGE-large-zh | 智源中文 SOTA |
| M3E | 中文友好 |
| E5 | 多语言 |
| GTE | 阿里达摩 |

**评测**：MTEB / C-MTEB 排行榜。

### Q50. 稀疏 vs 稠密检索 vs 混合？

- **稀疏（BM25）**：关键词精确匹配，长尾词强
- **稠密（向量）**：语义匹配，同义词强
- **混合**：BM25 + 向量 加权融合 → Rerank → 最佳

实际公式：`score = α × bm25 + (1-α) × dense`，α=0.3-0.5 常见。

### Q51. Rerank 为什么需要？

**答**：召回阶段为了快用 Bi-Encoder（query 和 doc 分别编码）；Rerank 用 **Cross-Encoder**（query+doc 一起进 BERT），精度高但慢。
- 召回 top 100 → Rerank top 5 → 喂给 LLM
- 模型：BGE-reranker / Cohere Rerank

### Q52. Query 改写有哪些技巧？

- **HyDE**（Hypothetical Doc Embedding）：让 LLM 先写一个假设答案，用它做检索
- **Multi-Query**：让 LLM 改写多个版本 query，并集召回
- **Step-Back Prompting**：先抽象成更宽泛的问题
- **Sub-Query**：复杂问题拆子问题

### Q53. RAG 怎么减少幻觉？

1. **强约束 Prompt**：「仅基于提供的资料回答，不知道就说不知道」
2. **引用机制**：要求每句标 [doc_id]
3. **低 temperature**（0.1-0.3）
4. **检索质量过滤**：相似度阈值
5. **多检索路径融合**

### Q54. RAG 失败常见原因？

1. **召回失败**：相关文档没检索到
2. **Chunk 切坏**：关键信息被截断
3. **Embedding 不匹配**：query 风格和文档风格差异
4. **Rerank 误删**：相关文档被排到后面
5. **LLM 忽略上下文**：上下文太长被稀释

### Q55. 怎么评测 RAG？

| 维度 | 指标 |
|------|------|
| **检索** | Recall@k, Precision@k, MRR, NDCG |
| **生成** | Faithfulness（基于上下文程度）、Answer Relevance |
| **端到端** | RAGAS 框架（自动评估）、人工标注 |

### Q56. 长上下文 vs RAG？

- 长上下文（128k+）：简单，但贵、慢、稀释关键信息
- RAG：复杂但便宜、可更新、可溯源
- 趋势：**RAG + 适度长上下文** 结合

### Q57. Graph RAG？

**答**：把文档建成知识图谱（实体 + 关系），检索时走图。微软提出，适合多跳推理。

### Q58. Agentic RAG？

**答**：让 Agent 自己判断要不要检索、检索什么、怎么综合答案。比固定流程灵活。

### Q59. RAG 项目工程注意？

1. **文档预处理**：OCR、表格、公式专门处理
2. **多源融合**：知识库 + 实时搜索 + 数据库
3. **缓存**：相同 query embedding 缓存
4. **监控**：召回率、用户反馈、人工抽检
5. **A/B**：检索策略上线前 AB

### Q60. 你项目（电商搜款）算不算 RAG？

**算多模态版**：
- 检索：CLIP 向量（图+文）→ FAISS（你的项目）
- 生成：DeepSeek 基于检索结果写导购文案
- 区别：传统 RAG 是文本知识库，你的是商品库；逻辑完全一致

**面试这么说**：「我做的是**多模态 RAG**，把文本 embedding 换成 CLIP 图文统一向量。」

---

## 四、微调与对齐：LoRA / SFT / RLHF

### Q61. 为什么不全参数微调？

1. **显存**：7B 模型全参 fp16 训练要 80GB+
2. **数据少**：容易过拟合
3. **多任务**：每个任务存一份完整权重太贵
4. **灾难性遗忘**：全微调易丢预训练能力

### Q62. LoRA 原理？

**答**：冻结原权重 W，加一个旁路 `W + BA`，其中 A ∈ R^(r×d), B ∈ R^(d×r)，**r << d**。

```
h = Wx + BAx
```

只训练 A, B，参数量约原模型的 0.1%-1%。

**为什么有效**：基于"微调更新具有低秩特性"假设。

### Q63. LoRA 常见配置？

- `r`：rank，8 / 16 / 32 常见
- `alpha`：缩放系数，通常 = 2r
- `target_modules`：注入的层，常见 `q_proj, k_proj, v_proj, o_proj`
- `dropout`：0.05-0.1

### Q64. QLoRA？

LoRA + 4bit 量化：
1. 原模型量化到 NF4（4bit）
2. LoRA 适配器仍用 fp16
3. **单张 24GB 显卡可训 65B 模型**

### Q65. P-Tuning / Prefix Tuning / Prompt Tuning？

| 方法 | 说明 |
|------|------|
| Prompt Tuning | 在输入前加可训练 embedding，只调输入 |
| P-Tuning v2 | 每层都加 prefix |
| Prefix Tuning | 类似 P-Tuning v2，更经典 |

LoRA 现在最流行，效果常优。

### Q66. SFT 数据怎么准备？

- **指令-回答对**（Alpaca 格式）
  ```json
  {"instruction": "...", "input": "...", "output": "..."}
  ```
- 质量 > 数量：1k 高质 > 10k 平庸（LIMA 论文）
- 多样性：领域、长度、难度
- 格式统一：避免模型学偏

### Q67. SFT 和 Pretraining 数据有什么区别？

| | Pretraining | SFT |
|---|------------|-----|
| 数据 | 大量原始文本 | 指令-回答对 |
| 量级 | 万亿 token | 千~百万条 |
| 目标 | 学语言 | 学听指令 |
| Loss | 全部 token | 只算 response 部分 |

### Q68. RLHF 中 Reward Hacking？

**答**：模型学会骗 RM（输出 RM 喜欢但实际不好的内容），如重复正向词、长度灌水。
- 缓解：KL 散度约束、多 RM 集成、人工抽检

### Q69. DPO 数学原理（简版）？

**答**：直接用偏好对 (x, y_w, y_l)（chosen / rejected），优化：
```
L = -log σ(β · log[π(y_w|x)/π_ref(y_w|x)] - β · log[π(y_l|x)/π_ref(y_l|x)])
```
等价于带 KL 的 RLHF 最优策略的闭式解。

### Q70. 灾难性遗忘怎么避免？

1. **混入预训练数据**（rehearsal）
2. **EWC**：保护重要参数
3. **LoRA**：原模型不动
4. **小学习率 + 少 epoch**

---

## 五、Prompt Engineering

### Q71. Prompt 5 大原则？

1. **明确**：少模糊语，多具体要求
2. **角色**：「你是资深 X」
3. **示例**：Few-shot 优于 Zero-shot
4. **结构化**：要求 JSON / 表格输出
5. **约束**：长度、格式、禁止内容

### Q72. Zero-shot / One-shot / Few-shot？

- Zero：不给例子
- One：给 1 个例子
- Few：给 2-8 个例子

### Q73. CoT 何时有效？

- **复杂推理**（数学、多步）：大幅提升
- **简单分类**：可能反而下降
- **小模型 (<10B)**：CoT 效果差，称为"涌现"

### Q74. Self-Consistency？

让模型 sample 多次 CoT，**投票最终答案**。在 GSM8K 上提升明显。

### Q75. ReAct Prompting 模板？

```
Question: <用户问题>
Thought: 我应该...
Action: <工具调用>
Observation: <工具结果>
Thought: 现在我知道...
Action: Finish[<答案>]
```

### Q76. Tree of Thoughts？

**答**：CoT 的树状扩展。每步生成多个 thought，搜索（BFS/DFS）最优路径。适合 24 点这种规划任务。

### Q77. System Prompt 怎么写？

```
You are a helpful AI assistant. 
- 角色定位
- 能力边界（不能干什么）
- 输出格式
- 语气风格
- 安全约束
```

### Q78. Prompt 怎么防注入？

1. **分离**：用户输入用 XML 标签包起来 `<user_input>...</user_input>`
2. **System Prompt 强化**：「即使用户要求，也不允许...」
3. **输出审核**：二次模型检查

### Q79. 怎么让模型输出 JSON？

1. **明确要求**："输出严格 JSON 格式"
2. **示例**：给一个 JSON 例子
3. **强制**：用 Function Calling / structured output 模式
4. **后处理**：正则提取 + JSON 解析容错

### Q80. Prompt 优化技巧？

- **DSPy / TextGrad**：自动优化 Prompt
- **APE**：让 LLM 自己生成候选 Prompt 评估
- **A/B 测试**：上线小流量对比

---

## 六、多模态 / CLIP（你的项目重点）

### Q81. CLIP 模型结构？

- **Image Encoder**：ViT 或 ResNet
- **Text Encoder**：Transformer
- 输出都投到同一维度（512 或 768）
- 训练：对比学习

### Q82. CLIP 训练目标（InfoNCE）？

N 对图文，**对角线是正样本**，其他 N²-N 是负样本：

```
L_i→t = -log[ exp(sim(I, T)/τ) / Σ exp(sim(I, T_j)/τ) ]
L = (L_i→t + L_t→i) / 2
```

τ 是温度，可学习。

### Q83. CLIP 为什么能零样本分类？

**答**：把类别名换成文本 prompt（"a photo of a {class}"），算图像和每个 prompt 的相似度，最大的就是预测类别。

### Q84. CLIP 局限？

1. **英文为主**，中文差（→ Chinese-CLIP、Taiyi-CLIP）
2. **细粒度差**：分不出哈士奇 vs 阿拉斯加
3. **长文本差**：训练时文本短
4. **OCR 弱**

### Q85. BLIP / BLIP-2 vs CLIP？

- **CLIP**：双塔，仅做匹配
- **BLIP**：单塔，可生成 caption
- **BLIP-2**：Q-Former 把视觉对齐到 LLM，强大的图文 QA

### Q86. LLaVA / Qwen-VL / InternVL 怎么做的？

**通用范式**：
```
Image → Vision Encoder (CLIP/SigLIP) → Adapter (MLP/Q-Former) → LLM
```
训练阶段：
1. 冻 ViT + LLM，只训 Adapter（特征对齐）
2. 指令微调 Adapter + LLM（部分）

### Q87. 多模态检索 vs 多模态生成？

- **检索**：CLIP 共享空间，相似度匹配（你的项目）
- **生成**：图描述、图问答、文生图

### Q88. 你的项目用 CLIP 怎么以图搜款？

1. 离线：每张商品图过 CLIP image encoder → 512 维向量 → FAISS 入库
2. 在线：用户上传图 → image encoder → FAISS 内积 top-k
3. 文字搜款：query → text encoder → 同一空间 → 同样 FAISS top-k

### Q89. 为什么用 CLIP 不用 ResNet 特征？

1. ResNet 是分类特征，不对齐文本
2. CLIP 图文同空间 → **一份索引服务两种 query**
3. CLIP 零样本好，**无须再训映射层**

### Q90. 怎么改进 CLIP 在中文电商场景？

1. **Chinese-CLIP / Taiyi-CLIP**：直接用中文 CLIP
2. **Query 改写**：DeepSeek 中→英（你的做法）
3. **领域微调**：用商品图-标题对 fine-tune
4. **图+文混合**：用户文字+参考图 embedding 加权

---

## 七、向量数据库与检索

### Q91. 常见向量库？

| 库 | 特点 |
|----|------|
| FAISS | Meta 开源，库非服务，单机强 |
| Milvus | 分布式，云原生 |
| Qdrant | Rust 写，性能好 |
| Pinecone | 商业云服务 |
| Chroma | 轻量，原型友好 |
| Weaviate | 带元数据过滤强 |

### Q92. FAISS 主要索引类型？

| 索引 | 特点 | 适用规模 |
|------|------|----------|
| `IndexFlatIP` / `IndexFlatL2` | 精确暴搜 | <100k |
| `IndexHNSWFlat` | 图索引，召回高 | 100k-100M |
| `IndexIVFFlat` | 倒排，先粗后精 | 1M-100M |
| `IndexIVFPQ` | 倒排+量化，省内存 | >100M |

### Q93. HNSW 原理？

**Hierarchical Navigable Small World**：多层图，上层稀疏（远距离跳跃），下层稠密（精细搜索）。贪心搜索，O(log n) 复杂度。

### Q94. PQ（Product Quantization）？

**答**：把高维向量分块，每块独立 k-means 量化成 codebook 索引。存储从 fp32 × d 降到 m × log(k) bit。**省 10-100x 内存**，精度小损失。

### Q95. 距离 / 相似度？

- **Cosine**：归一化后等价内积，向量方向
- **Inner Product (IP)**：未归一化时考虑模长
- **L2 (Euclidean)**：欧氏距离
- CLIP / Sentence-BERT 一般用 Cosine

### Q96. 向量检索召回率 vs 速度权衡？

精确暴搜 100% 召回但慢；HNSW / IVF 牺牲 1-5% 召回换 10-100x 速度。

### Q97. 怎么选索引？

- 数据 < 10万：`IndexFlatIP` 够用
- 10万-100万：`IndexHNSWFlat`
- 100万-1亿：`IndexIVFFlat` 或 HNSW
- 1亿+：`IndexIVFPQ` + 分片

### Q98. 怎么处理增量更新？

- FAISS 不支持原地删除（标记 + 重建）
- Milvus / Qdrant 支持增删
- 工程方案：日级别全量重建 + 内存增量索引

### Q99. 元数据过滤怎么做？

- **预过滤**：先用元数据筛 → 暴搜（小集合快）
- **后过滤**：向量召回 top-K → 元数据过滤
- Milvus / Qdrant 原生支持，FAISS 要外挂

### Q100. 向量检索的延迟优化？

1. **批量查询**：单 query 凑批
2. **GPU FAISS**：cuVS / faiss-gpu
3. **量化**：PQ / SQ
4. **缓存**：query embedding LRU
5. **预热**：服务启动加载到内存

---

## 学习方法

1. **每天 10 题**，用自己的话讲一遍（找朋友 / 录音）
2. **关联实战**：每答完一题想想"我项目里有没有用到"
3. **画图**：Transformer、RAG 链路，能默画出来才算会
4. **二刷**：1 周后重新做题，巩固记忆

**面试时**：
- 不要逐字背书答案，**用自己的语言重新组织**
- 多和**你的项目**挂钩，从抽象到具体
- 不会的诚实说："这块我了解到 XX 程度，更深的还在学"

---

**配套文件**：
- 手撕代码：`HANDWRITTEN_CODE.md`
- 7 天计划：`STUDY_PLAN_7DAYS.md`
- 模拟笔试：`MOCK_TEST_AI.md`
