# 手撕代码模板（AI 岗笔试必练）

> AI / 大模型公司笔试爱出"用 NumPy 实现 XXX"。  
> 下面 12 道都是高频题，**全部能默写 + 解释**就稳了。

学习方法：
1. 看一遍代码理解
2. 关掉文档自己默写
3. 跑通 + 打印中间结果验证
4. 准备口述思路（面试要边写边讲）

---

## 1. Softmax（数值稳定版）

### 代码

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x = x - np.max(x, axis=axis, keepdims=True)  # 减最大值防溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### 口述要点

- "Softmax 把任意实数映射到 [0,1] 且和为 1，常做分类概率"
- "直接算 exp(大数) 会溢出，**减最大值后等价但稳定**：分子分母同乘 exp(-max)"
- "axis=-1 表示在最后一维归一化（如 batch×n_class 的 class 维）"

### 测试

```python
x = np.array([1000, 1001, 1002])
print(softmax(x))  # [0.09 0.24 0.67]，不会 inf
```

---

## 2. Scaled Dot-Product Attention

### 代码

```python
import numpy as np

def attention(Q, K, V, mask=None):
    """
    Q: [..., n, d_k]
    K: [..., m, d_k]
    V: [..., m, d_v]
    mask: [..., n, m]，1 表示有效，0 表示 mask
    返回: [..., n, d_v]
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)  # [..., n, m]
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
```

### 口述要点

- "三步：算分 → 缩放 → 加权"
- "**除 √d_k**：防止 d_k 大时内积方差大，softmax 进饱和区梯度消失"
- "**mask 用 -1e9**：因为 exp(-1e9)→0，softmax 后那位变 0"
- "复杂度 O(n × m × d_k)，矩阵乘法"

### 测试

```python
Q = np.random.randn(2, 4, 8)   # batch=2, seq=4, d_k=8
K = np.random.randn(2, 6, 8)
V = np.random.randn(2, 6, 8)
out, w = attention(Q, K, V)
print(out.shape, w.shape)  # (2,4,8) (2,4,6)
print(w.sum(axis=-1))      # 每行和为 1
```

---

## 3. Multi-Head Attention

### 代码

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model=512, n_heads=8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        rng = np.random.RandomState(42)
        self.W_q = rng.randn(d_model, d_model) * 0.01
        self.W_k = rng.randn(d_model, d_model) * 0.01
        self.W_v = rng.randn(d_model, d_model) * 0.01
        self.W_o = rng.randn(d_model, d_model) * 0.01

    def split_heads(self, x):
        # [B, n, d_model] -> [B, h, n, d_k]
        B, n, _ = x.shape
        return x.reshape(B, n, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        # [B, h, n, d_k] -> [B, n, d_model]
        B, _, n, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, n, self.d_model)

    def __call__(self, x, mask=None):
        Q = self.split_heads(x @ self.W_q)
        K = self.split_heads(x @ self.W_k)
        V = self.split_heads(x @ self.W_v)
        out, _ = attention(Q, K, V, mask)
        out = self.combine_heads(out)
        return out @ self.W_o
```

### 口述要点

- "把 d_model 拆成 h 个 d_k=d_model/h 的子空间，**并行**做 attention"
- "每个头学不同语义子空间（语法、指代、长距离等）"
- "最后 concat → W_o 线性变回 d_model"

---

## 4. LayerNorm

### 代码

```python
import numpy as np

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def __call__(self, x):
        # x: [..., dim]
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### vs BatchNorm

```python
class BatchNorm:
    def __call__(self, x):
        # x: [B, dim]
        mean = x.mean(axis=0)        # 跨样本
        var = x.var(axis=0)
        return (x - mean) / np.sqrt(var + 1e-5)
```

### 口述要点

- "LN 单样本跨特征 / BN 跨样本同特征"
- "Transformer 用 LN 因为序列长度不定，batch 不可比"
- "γ, β 是可学习的缩放和平移"

---

## 5. 位置编码（Sinusoidal）

### 代码

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    pos = np.arange(seq_len)[:, None]            # [n, 1]
    i = np.arange(d_model)[None, :]              # [1, d]
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe
```

### 口述要点

- "偶数位 sin，奇数位 cos，频率指数衰减"
- "**相对位置可线性表示**：PE(pos+k) 可由 PE(pos) 线性变换得到"
- "可外推到训练时未见过的长度"

---

## 6. Cross Entropy Loss + 反向传播

### 代码

```python
import numpy as np

def cross_entropy(pred, label):
    """
    pred: [B, C] 已经 softmax 后的概率
    label: [B] 类别索引 or [B, C] one-hot
    """
    eps = 1e-12
    pred = np.clip(pred, eps, 1 - eps)
    if label.ndim == 1:
        # 类别索引
        B = pred.shape[0]
        return -np.mean(np.log(pred[np.arange(B), label]))
    else:
        # one-hot
        return -np.mean(np.sum(label * np.log(pred), axis=1))

def softmax_cross_entropy_grad(logits, label_onehot):
    """
    softmax + CE 合并求导，简化为 (p - y)
    """
    p = softmax(logits, axis=-1)
    return (p - label_onehot) / logits.shape[0]
```

### 口述要点

- "**clip 防 log(0)**"
- "softmax + CE 一起求导**等于 (预测 - 标签)**，非常优雅"
- "推导：∂L/∂z_i = p_i - y_i"

---

## 7. KV-Cache（自回归推理）

### 代码（伪代码版）

```python
def generate_with_kv_cache(model, prompt, max_new_tokens=100):
    # 初始化 KV cache
    K_cache = [None] * model.n_layers
    V_cache = [None] * model.n_layers

    # 1. Prefill: 整个 prompt 一次前向
    logits, K_cache, V_cache = model(prompt, K_cache, V_cache)
    next_token = logits[:, -1].argmax(-1)
    output = [next_token]

    # 2. Decode: 每次只喂新 token，复用 cache
    for _ in range(max_new_tokens - 1):
        logits, K_cache, V_cache = model(
            next_token[:, None],   # 只 1 个新 token
            K_cache, V_cache       # 拼接历史
        )
        next_token = logits[:, -1].argmax(-1)
        output.append(next_token)
        if next_token == eos_id:
            break
    return output
```

### 关键：每层 Attention 中

```python
# 没 KV-Cache:
K = X_all @ W_K   # 每次重算所有历史
V = X_all @ W_V

# 有 KV-Cache:
K_new = x_new @ W_K           # 只算新 token
V_new = x_new @ W_V
K_cache = np.concatenate([K_cache, K_new], axis=1)  # 拼接历史
V_cache = np.concatenate([V_cache, V_new], axis=1)
# 用 K_cache, V_cache 做 attention
```

### 口述要点

- "自回归生成每步前面的 K, V 不变，缓存避免重算"
- "复杂度从 O(n³) 降到 O(n²)"
- "代价是显存，大模型 KV-Cache 可占多 GB"

---

## 8. Top-k / Top-p Sampling

### 代码

```python
import numpy as np

def top_k_sample(logits, k=50, temperature=1.0):
    logits = logits / temperature
    idx = np.argsort(logits)[-k:]              # 前 k 大
    top_logits = logits[idx]
    probs = softmax(top_logits)
    choice = np.random.choice(k, p=probs)
    return idx[choice]

def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_idx = np.argsort(logits)[::-1]      # 降序
    sorted_logits = logits[sorted_idx]
    probs = softmax(sorted_logits)
    cumsum = np.cumsum(probs)
    cutoff = np.searchsorted(cumsum, p) + 1    # 累积 >= p 的位置
    top_idx = sorted_idx[:cutoff]
    top_probs = probs[:cutoff] / probs[:cutoff].sum()
    choice = np.random.choice(cutoff, p=top_probs)
    return top_idx[choice]
```

### 口述要点

- "**Top-k**：硬性截断到前 k 个"
- "**Top-p (nucleus)**：动态截断到累积概率 ≥ p"
- "**Temperature**：T<1 更确定，T>1 更随机；T=1 不变"

---

## 9. BPE 分词（核心逻辑）

### 代码（训练版简化）

```python
from collections import Counter

def get_pairs(word_freqs):
    """统计所有相邻字符对的频率"""
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    """合并最频繁的对"""
    new_freqs = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in word_freqs.items():
        new_word = word.replace(bigram, replacement)
        new_freqs[new_word] = freq
    return new_freqs

def train_bpe(corpus, n_merges=10):
    # 每个词用空格分开字符
    word_freqs = Counter(' '.join(list(w)) + ' </w>' for w in corpus)
    word_freqs = dict(word_freqs)
    merges = []
    for _ in range(n_merges):
        pairs = get_pairs(word_freqs)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        word_freqs = merge_pair(best, word_freqs)
        merges.append(best)
    return merges
```

### 口述要点

- "从字符级开始，每次合并最高频的相邻对，迭代 N 次"
- "OOV 友好（拆 subword）、多语言友好、词表可控"
- "`</w>` 标记词尾"

---

## 10. ReLU / Sigmoid / GeLU 激活

### 代码

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 防溢出

def tanh(x):
    return np.tanh(x)

def gelu(x):
    """近似版 GELU（GPT 用）"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    return x * sigmoid(x)
```

### 口述要点

| 激活 | 优点 | 缺点 |
|------|------|------|
| ReLU | 简单、快、缓解梯度消失 | 死神经元（负区永 0） |
| Sigmoid | 概率解释 | 梯度消失、非零中心 |
| GeLU | 平滑、效果好 | 略慢 |
| Swish | 平滑、自门控 | 略慢 |

---

## 11. Dropout

### 代码

```python
import numpy as np

class Dropout:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x, training=True):
        if not training:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * mask / (1 - self.p)  # inverted dropout
```

### 口述要点

- "训练时随机置零神经元，**inference 全开**"
- "**inverted dropout**：训练时除 (1-p)，inference 不用调整"
- "原理：类似集成多个子网络，减少协同适应"

---

## 12. K-Means（手撕聚类）

### 代码

```python
import numpy as np

def kmeans(X, k, max_iter=100):
    """
    X: [n, d]
    返回: centers [k, d], labels [n]
    """
    n, d = X.shape
    # 初始化：随机选 k 个点
    idx = np.random.choice(n, k, replace=False)
    centers = X[idx].copy()

    for _ in range(max_iter):
        # E 步：分配
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # [n, k]
        labels = dists.argmin(axis=1)
        # M 步：更新中心
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers, labels
```

### 口述要点

- "EM 迭代：E 步分配最近中心，M 步更新中心"
- "初始化敏感 → **K-Means++** 改进初始化"
- "k 的选择：肘部法、轮廓系数"

---

## 加分题（出现概率较低但拿分高）

### 13. Beam Search

```python
def beam_search(model, prompt, beam_width=4, max_len=20):
    beams = [(prompt, 0.0)]  # (序列, 累积 log prob)
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logits = model(seq)[-1]
            log_probs = np.log(softmax(logits) + 1e-12)
            top_k = np.argsort(log_probs)[-beam_width:]
            for tok in top_k:
                candidates.append((seq + [tok], score + log_probs[tok]))
        beams = sorted(candidates, key=lambda x: -x[1])[:beam_width]
    return beams[0][0]
```

### 14. RMSNorm（LLaMA 用）

```python
class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones(dim)
        self.eps = eps
    def __call__(self, x):
        rms = np.sqrt((x ** 2).mean(-1, keepdims=True) + self.eps)
        return self.gamma * x / rms
```

省了减均值，更快效果相当。

### 15. RoPE（旋转位置编码）

```python
def rope(x, base=10000):
    """x: [..., n, d]，d 必须偶数"""
    n, d = x.shape[-2:]
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    theta = pos / (base ** (2 * i / d))
    cos, sin = np.cos(theta), np.sin(theta)
    
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = np.empty_like(x)
    rotated[..., ::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos
    return rotated
```

---

## 笔试现场技巧

1. **先写正确解再优化**：能 work > 最优
2. **测试驱动**：写完立刻造小例子验证
3. **边界条件**：空输入、单元素、超长
4. **复杂度**：写完口述 "时间 O(?), 空间 O(?)"
5. **不会怎么办**：写思路注释 + 暴力解，**总比空着好**

---

## 三天速记计划

**Day 1（基础）**：Softmax + Attention + LayerNorm + 位置编码  
**Day 2（进阶）**：MultiHead + KV-Cache + CrossEntropy + Top-k/p  
**Day 3（加分）**：BPE + RoPE + RMSNorm + K-Means

**每个都要：看 → 默写 → 跑通 → 讲一遍**。

---

**配套**：八股理论看 `LLM_FUNDAMENTALS.md`，刷题计划看 `STUDY_PLAN_7DAYS.md`。
