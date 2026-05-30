# 7 天 AI 岗笔试冲刺计划（含 LeetCode 题号）

> 一天 4-5 小时，跟着做完笔试一面够用。  
> 配合：`LLM_FUNDAMENTALS.md` + `HANDWRITTEN_CODE.md` + `MOCK_TEST_AI.md`

---

## 总览

| Day | 上午 (2h) | 下午 (2h) | 晚上 (1h) |
|-----|-----------|-----------|-----------|
| 1 | 算法：哈希+双指针 | LLM：Transformer 25 题 | 手撕：Softmax+Attention |
| 2 | 算法：滑窗+二分 | LLM：训练推理 20 题 | 手撕：MultiHead |
| 3 | 算法：树+图 | LLM：RAG 15 题 | 手撕：LayerNorm+PE |
| 4 | 算法：DP 基础 | LLM：微调+Prompt 20 题 | 手撕：KV-Cache+CE |
| 5 | 算法：链表+栈 | LLM：CLIP+向量库 20 题 | 手撕：Top-k/p+BPE |
| 6 | 模拟笔试 90min | 复盘错题 | Python 八股 |
| 7 | 二刷错题 | 项目话术 + 简历 | 早睡 |

---

## Day 1 · 哈希、双指针 + Transformer 基础

### 上午 · 算法（2 小时）

**LeetCode 必刷（按顺序）**：

| 题号 | 题目 | 难度 | 类型 |
|------|------|------|------|
| 1 | 两数之和 | Easy | 哈希 |
| 49 | 字母异位词分组 | Medium | 哈希 |
| 128 | 最长连续序列 | Medium | 哈希 |
| 283 | 移动零 | Easy | 双指针 |
| 11 | 盛最多水的容器 | Medium | 双指针 |
| 15 | 三数之和 | Medium | 双指针 |
| 42 | 接雨水 | Hard | 双指针（看思路） |

**目标**：1-15 自己写出；42 看题解理解。

**通用模板**：
```python
# 哈希
def twoSum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i

# 双指针（对撞）
def container(height):
    l, r = 0, len(height) - 1
    res = 0
    while l < r:
        res = max(res, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]: l += 1
        else: r -= 1
    return res
```

### 下午 · LLM 八股（2 小时）

读 `LLM_FUNDAMENTALS.md` **Q1-Q25（Transformer 基础）**。

**重点掌握（必背）**：
- [ ] Q2 Self-Attention 公式 + 流程
- [ ] Q3 为什么除 √d_k
- [ ] Q4 Multi-Head 为什么有效
- [ ] Q5 位置编码三种（Sinusoidal / Learnable / RoPE）
- [ ] Q6 LN vs BN
- [ ] Q11 复杂度 O(n²d)
- [ ] Q13-14 Encoder-only / Decoder-only / Encoder-Decoder
- [ ] Q15-16 KV-Cache + 显存计算
- [ ] Q21-22 BPE 分词
- [ ] Q24-25 解码策略（Greedy / Beam / Top-k / Top-p）

**自测**：合上文档，用自己的话讲：
1. Transformer 整体结构 + 流程
2. Self-Attention 公式 + 为什么除 √d_k
3. KV-Cache 怎么省时间

### 晚上 · 手撕代码（1 小时）

`HANDWRITTEN_CODE.md` 第 1-2 题：
- [ ] **Softmax 数值稳定版**（默写 + 解释为什么减最大值）
- [ ] **Scaled Dot-Product Attention**（默写 + 测试 shape）

```python
# 自测
Q = np.random.randn(2, 4, 8)
K = np.random.randn(2, 6, 8)
V = np.random.randn(2, 6, 8)
out, w = attention(Q, K, V)
assert out.shape == (2, 4, 8)
assert np.allclose(w.sum(-1), 1.0)
```

---

## Day 2 · 滑窗、二分 + LLM 训练推理

### 上午 · 算法（2 小时）

| 题号 | 题目 | 难度 | 类型 |
|------|------|------|------|
| 3 | 无重复字符的最长子串 | Medium | 滑窗 |
| 76 | 最小覆盖子串 | Hard | 滑窗 |
| 209 | 长度最小的子数组 | Medium | 滑窗 |
| 35 | 搜索插入位置 | Easy | 二分 |
| 33 | 搜索旋转排序数组 | Medium | 二分 |
| 153 | 寻找旋转排序数组中的最小值 | Medium | 二分 |
| 4 | 寻找两个正序数组的中位数 | Hard | 二分（选做） |

**滑窗模板**：
```python
def slide_window(s):
    left = 0
    window = {}
    res = 0
    for right in range(len(s)):
        # 右扩
        window[s[right]] = window.get(s[right], 0) + 1
        # 左收
        while not_valid(window):
            window[s[left]] -= 1
            left += 1
        res = max(res, right - left + 1)
    return res
```

**二分模板**：
```python
def binary_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target: return m
        elif nums[m] < target: l = m + 1
        else: r = m - 1
    return -1
```

### 下午 · LLM 八股（2 小时）

`LLM_FUNDAMENTALS.md` **Q26-Q45（训练 + 推理）**。

**重点**：
- [ ] Q26 三阶段训练（Pretrain → SFT → RLHF/DPO）
- [ ] Q28-29 RLHF 流程 / DPO vs PPO
- [ ] Q31 Scaling Law / Chinchilla 1:20
- [ ] Q33 量化（PTQ/QAT/GPTQ/AWQ）
- [ ] Q35 推理加速汇总
- [ ] Q36 vLLM / PagedAttention
- [ ] Q37 MoE
- [ ] Q38-40 幻觉 / CoT / ICL

**作业**：合上文档画一张图 —— LLM 三阶段训练 + 推理加速链路。

### 晚上 · 手撕（1 小时）

`HANDWRITTEN_CODE.md` 第 3 题：
- [ ] **Multi-Head Attention 完整类**
- [ ] 测试 shape `[2, 10, 512]` → `[2, 10, 512]`

---

## Day 3 · 树、图 + RAG

### 上午 · 算法（2 小时）

| 题号 | 题目 | 难度 | 类型 |
|------|------|------|------|
| 104 | 二叉树最大深度 | Easy | 递归 |
| 226 | 翻转二叉树 | Easy | 递归 |
| 102 | 层序遍历 | Medium | BFS |
| 236 | 最近公共祖先 | Medium | 递归 |
| 105 | 前序+中序构造二叉树 | Medium | 递归 |
| 200 | 岛屿数量 | Medium | DFS/BFS |
| 207 | 课程表 | Medium | 拓扑排序 |

**树递归模板**：
```python
def helper(node):
    if not node: return 0
    left = helper(node.left)
    right = helper(node.right)
    # 处理逻辑
    return ...
```

**BFS 模板**：
```python
from collections import deque
def bfs(grid):
    q = deque([(0, 0)])
    visited = {(0, 0)}
    while q:
        x, y = q.popleft()
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited and valid(nx, ny):
                visited.add((nx, ny))
                q.append((nx, ny))
```

### 下午 · LLM 八股（2 小时）

`LLM_FUNDAMENTALS.md` **Q46-Q60（RAG）**。

**重点**：
- [ ] Q46 RAG 完整链路图（必默画）
- [ ] Q48 Chunk 切分策略
- [ ] Q49 Embedding 模型选型
- [ ] Q50 稀疏 / 稠密 / 混合
- [ ] Q51 Rerank（Bi vs Cross Encoder）
- [ ] Q52 Query 改写（HyDE / Multi-Query）
- [ ] Q53-54 减幻觉 / 失败原因
- [ ] Q55 RAG 评测（RAGAS）
- [ ] **Q60 你的项目是多模态 RAG**（必背！）

**作业**：用 5 句话讲清楚你的项目和 RAG 的对应关系。

### 晚上 · 手撕（1 小时）

`HANDWRITTEN_CODE.md` 第 4-5 题：
- [ ] **LayerNorm + BatchNorm 对比**
- [ ] **Sinusoidal Positional Encoding**

---

## Day 4 · 动态规划 + 微调/Prompt

### 上午 · 算法（2 小时）

| 题号 | 题目 | 难度 | 类型 |
|------|------|------|------|
| 70 | 爬楼梯 | Easy | DP 入门 |
| 198 | 打家劫舍 | Medium | DP |
| 53 | 最大子数组和 | Medium | DP |
| 322 | 零钱兑换 | Medium | 完全背包 |
| 300 | 最长上升子序列 | Medium | DP |
| 416 | 分割等和子集 | Medium | 0-1 背包 |
| 72 | 编辑距离 | Medium | DP 二维 |

**DP 思路**：
1. 定义状态 `dp[i]` 是什么
2. 写状态转移方程
3. 初始化
4. 遍历顺序

```python
# 打家劫舍
def rob(nums):
    if not nums: return 0
    dp = [0] * len(nums)
    dp[0] = nums[0]
    if len(nums) > 1:
        dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    return dp[-1]
```

### 下午 · LLM 八股（2 小时）

`LLM_FUNDAMENTALS.md` **Q61-Q80（微调 + Prompt）**。

**重点**：
- [ ] Q61-65 LoRA 原理 / QLoRA / P-Tuning
- [ ] Q66-68 SFT 数据 / RLHF Reward Hacking
- [ ] Q69 DPO 数学（知道思路即可）
- [ ] Q71-72 Prompt 5 原则 / Zero/Few-shot
- [ ] Q73-74 CoT / Self-Consistency
- [ ] Q75 ReAct 模板
- [ ] Q78 Prompt 防注入

**作业**：写一个 ReAct 风格的 Prompt 例子。

### 晚上 · 手撕（1 小时）

`HANDWRITTEN_CODE.md` 第 6-7 题：
- [ ] **Cross Entropy + softmax 求导**（推导 `(p-y)`）
- [ ] **KV-Cache 伪代码 + 解释**

---

## Day 5 · 链表、栈 + CLIP/向量库

### 上午 · 算法（2 小时）

| 题号 | 题目 | 难度 | 类型 |
|------|------|------|------|
| 206 | 反转链表 | Easy | 链表 |
| 21 | 合并两个有序链表 | Easy | 链表 |
| 141 | 环形链表 | Easy | 快慢指针 |
| 142 | 环形链表 II | Medium | 快慢指针 |
| 146 | LRU 缓存 | Medium | 链表+哈希 |
| 20 | 有效的括号 | Easy | 栈 |
| 155 | 最小栈 | Medium | 栈 |
| 215 | 数组中第 K 个最大元素 | Medium | 堆 |

**反转链表**（必会）：
```python
def reverse(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
```

**LRU**（高频）：
```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()
    def get(self, key):
        if key not in self.d: return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
```

### 下午 · LLM 八股（2 小时）

`LLM_FUNDAMENTALS.md` **Q81-Q100（CLIP + 向量库）**。

**重点（这是你的项目核心，必背）**：
- [ ] Q81-82 CLIP 结构 + InfoNCE
- [ ] Q83 零样本分类原理
- [ ] Q84 CLIP 局限（英文为主、细粒度差）
- [ ] Q86 LLaVA 范式（Vision Encoder + Adapter + LLM）
- [ ] Q88-89 你项目怎么用 CLIP（必背！）
- [ ] Q91-92 向量库 / FAISS 索引类型
- [ ] Q93-94 HNSW / PQ
- [ ] Q97 索引选择经验

### 晚上 · 手撕（1 小时）

`HANDWRITTEN_CODE.md` 第 8-9 题：
- [ ] **Top-k / Top-p Sampling**
- [ ] **BPE 训练简化版**

---

## Day 6 · 模拟笔试

### 上午 · 模拟 90 分钟（封闭环境）

打开 `MOCK_TEST_AI.md`，**计时 90 分钟**做完。  
做完前**不许查文档**。

### 下午 · 复盘

- 错题 / 不会的题，回到对应文档章节复习
- 在错题旁写下：「为什么错 + 知识点 + 类似题」
- 重做错题一次

### 晚上 · Python 八股（1 小时）

复习 `INTERVIEW_PREP.md` 第五节：
- [ ] GIL / 多线程多进程
- [ ] 装饰器（手写一个计时装饰器）
- [ ] 生成器 vs 列表
- [ ] 浅拷贝 vs 深拷贝
- [ ] `*args, **kwargs`
- [ ] async / await

```python
# 计时装饰器
import time
from functools import wraps
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = time.time()
        r = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time()-s:.4f}s")
        return r
    return wrapper
```

---

## Day 7 · 错题二刷 + 项目话术

### 上午 · 错题二刷（2 小时）

- 重写 Day 1-5 中所有错过 / 卡住的题
- 重新默写所有手撕代码
- 闭眼能讲：Transformer、Attention、RAG、CLIP、LoRA

### 下午 · 项目话术（2 小时）

打开 `INTERVIEW.md` 和 `INTERVIEW_PREP.md`：
- [ ] 30 秒 / 1 分钟 / 3 分钟自我介绍各练 3 遍
- [ ] 项目必答 10 题 闭眼复述
- [ ] 准备 3 个反问面试官的问题

**特别针对 AI 岗 — 把电商项目讲成 "多模态 RAG"**：

> 我做的是一个**面向真实业务的多模态 RAG 系统**。  
> - 检索：用 CLIP (ViT-B/32) 把商品图和查询文本统一映射到 512 维空间，**等价于多模态 embedding**  
> - 召回：FAISS IndexFlatIP 余弦相似度近邻检索  
> - LLM 层：DeepSeek 做 query 改写（中→英适配 CLIP 训练分布）和基于检索结果的导购文案生成  
> - 工程化：CLIP 索引磁盘缓存（CSV 指纹失效）让冷启动从 3min→5s；缩略图缓存让公网传输降 80%  
> - 完整业务闭环：用户认证 + 订单状态机 + 库存原子扣减 + 21 个 pytest 单测全过

### 晚上 · 早睡

明天笔试 / 一面**全力以赴**。

---

## 通用建议

### 算法刷题心法
- **不会就看题解，看完关掉自己写一遍**
- **同类型连刷**：滑窗就 5 道连着刷
- **写完口述复杂度**：时间 O(?)、空间 O(?)
- **每天保留 30 分钟错题二刷**

### 八股记忆心法
- **关联记忆**：每道八股关联到你项目里有没有用
- **画图记忆**：能默画的就别死背
- **讲给别人听**：能讲清楚才算真懂

### 笔试现场心法
1. **先扫题选难易**：先做有把握的
2. **SQL 不会就先暴力**：能 work 比优雅重要
3. **算法题先写思路 + 暴力解**
4. **手撕题边写边讲**：体现思路
5. **最后 10 分钟检查**：拼写、边界、shape

---

## 学习资源补充

### 免费 LLM 学习
- [📚 Stanford CS336 Language Modeling](https://stanford-cs336.github.io/)
- [📺 Andrej Karpathy YouTube](https://www.youtube.com/@AndrejKarpathy)（从零写 GPT）
- [📖 HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)
- [📖 The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### 八股聚合（中文）
- 🔥 [BAT 大佬笔记](https://github.com/datawhalechina/llms-from-scratch-cn)
- 🔥 [LLM 八股文](https://github.com/wdndev/llm_interview_note)
- 🔥 [深度学习面试宝典](https://github.com/amusi/Deep-Learning-Interview-Book)
- 🔥 [代码随想录](https://programmercarl.com/)（算法）

### 刷题平台
- LeetCode 中文站
- 牛客网（陌陌、字节等多用此平台）

---

**记住**：你的最大优势是**有 Live Demo 的多模态 AI 项目**。  
八股 + 算法只是为了**不在硬能力上被刷掉**，**项目讲好才是拿 offer 的关键**。
