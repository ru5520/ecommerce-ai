# AI 岗模拟笔试题（90 分钟）

> 模拟字节 / 阿里 / MiniMax / 智谱 等 AI 应用岗笔试。  
> **限时 90 分钟**，做完后再看答案。

总分：100 分
- 选择题（30 分） · 30 题 × 1 分
- 简答题（25 分） · 5 题 × 5 分
- SQL（15 分） · 2 题
- 算法（20 分） · 1 道 Medium
- 手撕（10 分） · 1 道

---

## 一、选择题（30 题，30 分）

> 单选，每题 1 分。

### Transformer / 模型基础（10 题）

**1.** Self-Attention 中除以 √d_k 的主要目的是？
A. 加快计算
B. 防止 softmax 进入饱和区导致梯度消失
C. 归一化输入
D. 减少参数量

**2.** Multi-Head Attention 中 8 个头，d_model=512，每个头的 d_k 是？
A. 8   B. 64   C. 256   D. 512

**3.** 以下哪个不是位置编码方法？
A. Sinusoidal   B. RoPE   C. ALiBi   D. ReLU

**4.** Transformer 的 self-attention 复杂度是？
A. O(n)   B. O(n log n)   C. O(n²)   D. O(n³)

**5.** LayerNorm 和 BatchNorm 的区别？
A. LN 跨样本，BN 跨特征
B. LN 跨特征，BN 跨样本
C. 两者一样
D. LN 只在 inference 用

**6.** 以下哪个模型是 Decoder-only？
A. BERT   B. T5   C. GPT   D. BART

**7.** KV-Cache 主要解决什么问题？
A. 训练速度   B. 推理速度   C. 显存占用   D. 精度

**8.** Flash Attention 的核心优化是？
A. 改变算法   B. SRAM 分块计算减少 IO   C. 量化   D. 蒸馏

**9.** Tokenization 中 BPE 是怎么生成词表的？
A. 从词典固定   B. 从字符开始合并最频繁对   C. 随机选择   D. 按字母顺序

**10.** Temperature = 0 时，等价于哪种解码？
A. Random Sampling   B. Beam Search   C. Greedy   D. Top-p

### 训练 / 微调（10 题）

**11.** LLM 通常三阶段训练顺序是？
A. SFT → Pretrain → RLHF
B. Pretrain → SFT → RLHF
C. RLHF → Pretrain → SFT
D. SFT → RLHF → Pretrain

**12.** LoRA 中加入的低秩矩阵是？
A. 替换原权重
B. 加在 W 旁边 W + BA
C. 加在输入前
D. 加在输出后

**13.** QLoRA 的 Q 是指？
A. Query
B. Quantization（量化）
C. Quick
D. Queue

**14.** DPO 相比 PPO 的优势？
A. 效果更好
B. 不需要奖励模型，训练更稳定
C. 速度更快
D. 以上都是

**15.** Chinchilla Scaling Law 建议参数和 token 的比例？
A. 1:1   B. 1:5   C. 1:20   D. 1:100

**16.** 过拟合的解决方法不包括？
A. 增加数据   B. Dropout   C. 增加模型容量   D. L2 正则

**17.** Cross Entropy + Softmax 合并求导的结果是？
A. p   B. y   C. p - y   D. y - p

**18.** 以下激活函数中，**LLaMA** 用的是？
A. ReLU   B. GeLU   C. SwiGLU   D. Sigmoid

**19.** MoE (Mixture of Experts) 模型的优势？
A. 参数少
B. 激活参数少 → 算力效率高
C. 训练简单
D. 显存占用低

**20.** Self-Attention 的 mask 主要用于？
A. 加速   B. 防 Decoder 看到未来 token   C. 减少参数   D. 量化

### RAG / 多模态 / 向量库（10 题）

**21.** RAG 中 Rerank 通常用哪种模型？
A. Bi-Encoder   B. Cross-Encoder   C. CNN   D. LSTM

**22.** 关于稀疏检索（BM25）和稠密检索（向量），错误的是？
A. BM25 关键词精确
B. 向量检索语义匹配
C. 混合检索通常效果最好
D. 向量检索 100% 召回

**23.** Chunk 切分的常见 overlap 比例？
A. 0%   B. 10-20%   C. 50%   D. 80%

**24.** FAISS 中适合 200 个向量精确检索的索引是？
A. IndexFlatIP   B. IndexIVFPQ   C. IndexHNSWFlat   D. IndexLSH

**25.** HNSW 索引的核心思想？
A. 倒排   B. 量化   C. 多层小世界图 + 贪心搜索   D. 哈希

**26.** CLIP 训练用的损失函数是？
A. CrossEntropy   B. InfoNCE   C. Triplet Loss   D. MSE

**27.** CLIP 怎么做零样本分类？
A. 微调输出层
B. 用 prompt（"a photo of a {class}"）算图文相似度
C. 用 KNN
D. 用决策树

**28.** 以下哪个向量库是 Meta 开源的？
A. Milvus   B. FAISS   C. Pinecone   D. Qdrant

**29.** PQ (Product Quantization) 的作用？
A. 加速训练
B. 压缩向量存储 + 加速检索
C. 提升精度
D. 减少 GPU 用量

**30.** RAG 怎么减少幻觉？
A. 强约束 Prompt + 引用 + 低 temperature
B. 提高 temperature
C. 不用 Prompt
D. 用更小模型

---

## 二、简答题（5 题 × 5 分 = 25 分）

**31.** 用 3-5 句话解释 Transformer 中 Self-Attention 的计算过程，包括关键公式。

**32.** 简述 RAG 的完整链路，并说明每一步可能失败的原因。

**33.** LoRA 为什么能用 0.1%-1% 参数达到接近全量微调的效果？

**34.** 解释 CLIP 是怎么训练的，为什么能同时支持"以图搜图"和"以文搜图"。

**35.** 列出 LLM 推理加速的至少 5 种手段，并简要说明原理。

---

## 三、SQL（2 题，15 分）

### 36（7 分）·  连续登录 N 天的用户

表 `login(user_id, login_date)`，找出**连续登录 3 天及以上**的用户和他们的最长连续天数。

期望输出：`user_id, max_consecutive_days`

### 37（8 分）·  漏斗转化率

表 `event(user_id, event_name, event_time)`，事件包括 `view, cart, pay`。  
计算总体 view→cart→pay 的两段转化率。

期望输出（一行）：
```
view_cnt | cart_cnt | pay_cnt | view_to_cart | cart_to_pay
```

---

## 四、算法（1 题，20 分）

### 38 · LRU Cache

设计一个 LRU（最近最少使用）缓存：
- `LRUCache(int capacity)` 初始化
- `int get(int key)` 不存在返回 -1
- `void put(int key, int value)` 存在则更新，否则插入；满了淘汰最久未用

**要求**：`get` 和 `put` 都是 **O(1)** 时间复杂度。

```python
class LRUCache:
    def __init__(self, capacity: int):
        # TODO
        pass
    def get(self, key: int) -> int:
        # TODO
        pass
    def put(self, key: int, value: int) -> None:
        # TODO
        pass
```

---

## 五、手撕代码（1 题，10 分）

### 39 · 实现 Scaled Dot-Product Attention（NumPy）

要求：
1. 支持 batch
2. 支持 attention mask
3. 数值稳定的 softmax
4. 测试代码 + 验证 weights 行和为 1

```python
import numpy as np

def softmax(x, axis=-1):
    # TODO
    pass

def attention(Q, K, V, mask=None):
    """
    Q: [B, n, d_k]
    K: [B, m, d_k]
    V: [B, m, d_v]
    mask: [B, n, m] or None
    返回: [B, n, d_v], [B, n, m]
    """
    # TODO
    pass
```

---

---
---

# 📝 参考答案与评分标准

> 做完后再看！

## 一、选择题答案

| 题号 | 答案 | 题号 | 答案 | 题号 | 答案 |
|------|------|------|------|------|------|
| 1 | B | 11 | B | 21 | B |
| 2 | B | 12 | B | 22 | D |
| 3 | D | 13 | B | 23 | B |
| 4 | C | 14 | D | 24 | A |
| 5 | B | 15 | C | 25 | C |
| 6 | C | 16 | C | 26 | B |
| 7 | B | 17 | C | 27 | B |
| 8 | B | 18 | C | 28 | B |
| 9 | B | 19 | B | 29 | B |
| 10 | C | 20 | B | 30 | A |

**评分**：每题 1 分，总 30 分。低于 20 分要回看 `LLM_FUNDAMENTALS.md`。

---

## 二、简答题答案

### 31. Self-Attention 计算过程

> 输入 X 分别乘 W_Q, W_K, W_V 得到 Q, K, V。然后算 `scores = QKᵀ/√d_k`，其中除以 √d_k 是为了防止 d_k 大时 softmax 进入饱和区导致梯度消失。对 scores 做 softmax 得到注意力权重，最后用权重对 V 加权求和：`Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`。

### 32. RAG 完整链路

```
Query → 改写/扩展 → Embedding → 向量召回 + BM25 → Rerank → Prompt 拼接 → LLM 生成
```

每步失败原因：
- **改写**：改写偏离原意
- **Embedding**：模型不匹配 query 风格
- **召回**：相关文档没召回（chunk 切坏 / 语义匹配失败）
- **Rerank**：误删相关文档
- **拼接**：上下文过长稀释关键信息
- **生成**：LLM 忽略上下文 / 幻觉

### 33. LoRA 为什么有效

> LoRA 假设微调过程中权重更新具有**低秩特性**，即 ΔW 可以用低秩矩阵分解 BA 近似（A: r×d, B: d×r, r<<d）。冻原始 W，只训 A、B，参数量从 d² 降到 2rd，约 0.1%-1%。实践证明在大多数下游任务上效果接近全参微调。优点：① 省显存、省存储；② 多任务共享 base model；③ 不影响推理速度（可合并回 W）。

### 34. CLIP 训练与跨模态检索

> CLIP 用**对比学习 InfoNCE** 损失训练：一个 batch 内 N 对图文，对角线是正样本，其他 N²-N 是负样本。让正样本图文 embedding 余弦相似度高，负样本低。  
> 损失对称：图→文 + 文→图 平均。  
> **同时支持以图搜图、以文搜图、以文搜图**：因为图和文都被投到同一个 512 维空间，余弦相似度直接对齐，一份 FAISS 索引服务所有跨模态查询。

### 35. LLM 推理加速 5 种以上

| 手段 | 原理 |
|------|------|
| **KV-Cache** | 缓存历史 K, V，避免重算，O(n³)→O(n²) |
| **Flash Attention** | SRAM 分块计算，减少 HBM IO，无精度损失 |
| **量化（INT8/INT4/GPTQ/AWQ）** | 降低数值精度，显存 ↓ 4-8x |
| **Continuous Batching** | 动态拼接 batch，吞吐 ↑ |
| **推测解码 (Speculative Decoding)** | 小模型预测大模型验证 |
| **MoE** | 激活部分专家，算力效率高 |
| **vLLM PagedAttention** | OS 虚拟内存式 KV-Cache 管理 |
| **Prefix Caching** | 系统提示缓存，避免重复 prefill |
| **流式输出** | 用户感知延迟 ↓ |

**评分**：每题 5 分，关键词 + 公式各 2 分，举例/对比 1 分。

---

## 三、SQL 答案

### 36. 连续登录 3 天

```sql
SELECT user_id, MAX(consec_days) AS max_consecutive_days
FROM (
  SELECT user_id, grp, COUNT(*) AS consec_days
  FROM (
    SELECT 
      user_id, 
      login_date,
      DATE_SUB(login_date, INTERVAL ROW_NUMBER() OVER (
        PARTITION BY user_id ORDER BY login_date
      ) DAY) AS grp
    FROM (SELECT DISTINCT user_id, login_date FROM login) t1
  ) t2
  GROUP BY user_id, grp
  HAVING COUNT(*) >= 3
) t3
GROUP BY user_id;
```

**思路**：
- 同一用户连续日期：`login_date - ROW_NUMBER()` 相同
- 按 `(user_id, grp)` 分组计数即为连续天数
- 过滤 ≥ 3 天

### 37. 漏斗转化率

```sql
SELECT
  SUM(CASE WHEN event_name='view' THEN 1 ELSE 0 END) AS view_cnt,
  SUM(CASE WHEN event_name='cart' THEN 1 ELSE 0 END) AS cart_cnt,
  SUM(CASE WHEN event_name='pay'  THEN 1 ELSE 0 END) AS pay_cnt,
  ROUND(SUM(CASE WHEN event_name='cart' THEN 1 ELSE 0 END) * 1.0 /
        NULLIF(SUM(CASE WHEN event_name='view' THEN 1 ELSE 0 END), 0), 4) AS view_to_cart,
  ROUND(SUM(CASE WHEN event_name='pay'  THEN 1 ELSE 0 END) * 1.0 /
        NULLIF(SUM(CASE WHEN event_name='cart' THEN 1 ELSE 0 END), 0), 4) AS cart_to_pay
FROM event;
```

**进阶版（按用户去重）**：
```sql
SELECT
  COUNT(DISTINCT CASE WHEN event_name='view' THEN user_id END) AS view_uv,
  COUNT(DISTINCT CASE WHEN event_name='cart' THEN user_id END) AS cart_uv,
  COUNT(DISTINCT CASE WHEN event_name='pay'  THEN user_id END) AS pay_uv
FROM event;
```

**评分**：基础正确 5 分；用窗口 / 不被 0 除 / 去重等加分。

---

## 四、算法答案 · LRU Cache

```python
class LRUCache:
    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

**进阶 · 手撕双向链表 + 哈希**（面试官常追问）：

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}
        self.head = Node()       # dummy
        self.tail = Node()       # dummy
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.value

    def put(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_head(node)
        else:
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            if len(self.cache) > self.cap:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
```

**评分**：
- OrderedDict 版本：15 分
- 手写双向链表 + 哈希：20 分
- 漏掉边界（cap=0, key 已存在）扣分

---

## 五、手撕答案 · Scaled Dot-Product Attention

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """
    Q: [B, n, d_k]
    K: [B, m, d_k]
    V: [B, m, d_v]
    mask: [B, n, m] or None；1 表示有效，0 表示 mask
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)  # [B, n, m]
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores, axis=-1)               # [B, n, m]
    out = weights @ V                                # [B, n, d_v]
    return out, weights

# 测试
np.random.seed(42)
Q = np.random.randn(2, 4, 8)
K = np.random.randn(2, 6, 8)
V = np.random.randn(2, 6, 8)
out, w = attention(Q, K, V)
assert out.shape == (2, 4, 8)
assert np.allclose(w.sum(axis=-1), 1.0)
print("OK")

# 测试 mask
mask = np.ones((2, 4, 6))
mask[:, :, 4:] = 0   # 最后 2 个 mask 掉
out2, w2 = attention(Q, K, V, mask)
assert np.allclose(w2[:, :, 4:], 0)
print("Mask OK")
```

**评分**：
- 公式正确：4 分
- 数值稳定 softmax：2 分
- mask 处理：2 分
- 测试代码 + 验证：2 分

---

## 总评分标准

| 分数段 | 评价 |
|--------|------|
| **85+** | 大厂 AI 岗笔试稳过 |
| **70-85** | 中小厂 AI 岗稳过，大厂可能要好运气 |
| **50-70** | 还要 3-7 天专项突击 |
| **<50** | 优先看 `LLM_FUNDAMENTALS.md` 全文 + 算法基础 50 题 |

---

## 复盘建议

做完后**必须**：
1. 错题按类型分类（八股 / 算法 / SQL / 手撕）
2. 每个错题写**为什么错 + 关联知识点**
3. 回到对应文档章节复习
4. 一周后**重做一次**

---

**记住**：笔试是门票，**项目和讲解能力**才是拿 offer 的核心。这份题做到 70+ 就可以全力准备面试了。
