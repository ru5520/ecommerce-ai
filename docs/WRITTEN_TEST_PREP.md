# 笔试 / 八股 速成手册（数据分析 + AI 应用 双线）

> 给王硕的硬能力补强清单。  
> **优先级**：⭐⭐⭐ 必背 / ⭐⭐ 高概率 / ⭐ 加分项。  
> 笔试一般 60–90 分钟，**SQL + 选择题 + 1 道编程**。AI 岗多一道 Prompt / 模型题。

---

## 一、岗位对照表（先看自己投了什么）

| 岗位 | 笔试重点（按权重） | 必背 |
|------|---------------------|------|
| **数据分析师 / 数据工程师** | SQL 40% · 统计概率 20% · Python/pandas 20% · 业务案例 10% · 算法 10% | SQL 七大子句、统计推断、AB 实验 |
| **AI 应用 / 大模型应用** | Python 30% · LLM / Prompt 25% · 算法 20% · 八股 15% · SQL 10% | Transformer、RAG、向量检索 |
| **算法 / 多模态** | 算法 35% · DL 八股 30% · 数学 15% · Python 10% · 论文题 10% | 反向传播、注意力、CLIP、对比学习 |
| **后端 / AI Infra** | 算法 40% · 操作系统 / 网络 / DB 30% · Python 20% · 系统设计 10% | 进程线程、TCP、Redis、索引 |

陌陌的数据分析岗 → **SQL + 统计 + Python + 1 道偏中等算法**，下面着重写这块。

---

## 二、SQL 高频题（⭐⭐⭐ 必背 8 类）

### 1. 七大子句执行顺序

```
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```
**笔试必出**：「WHERE 和 HAVING 区别？」  
→ WHERE 过滤行（GROUP BY 前），HAVING 过滤组（GROUP BY 后，可用聚合）。

### 2. JOIN 类型

```sql
-- INNER：两边都有
-- LEFT：保留左表
-- RIGHT：保留右表
-- FULL OUTER：两边都保留（MySQL 不支持，用 UNION）
-- 自连接（员工和经理同一张表）
SELECT a.name, b.name AS manager
FROM emp a LEFT JOIN emp b ON a.mgr_id = b.id;
```

### 3. 窗口函数（最高频，必练）

```sql
-- ROW_NUMBER / RANK / DENSE_RANK
SELECT user_id, score,
  ROW_NUMBER()  OVER (PARTITION BY class ORDER BY score DESC) AS rn,   -- 1,2,3,4
  RANK()        OVER (PARTITION BY class ORDER BY score DESC) AS rk,   -- 1,2,2,4
  DENSE_RANK()  OVER (PARTITION BY class ORDER BY score DESC) AS drk   -- 1,2,2,3
FROM scores;

-- 每个分组取 Top N
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) rn
  FROM sales_tbl
) t WHERE rn <= 3;

-- LAG/LEAD 取上一行/下一行（次日留存常用）
SELECT user_id, dt,
  LAG(dt) OVER (PARTITION BY user_id ORDER BY dt) AS prev_dt
FROM login_log;
```

### 4. 留存率 / 复购率（必出）

```sql
-- 次日留存
SELECT 
  a.dt,
  COUNT(DISTINCT b.user_id) * 1.0 / COUNT(DISTINCT a.user_id) AS d1_retention
FROM login a
LEFT JOIN login b 
  ON a.user_id = b.user_id 
  AND b.dt = DATE_ADD(a.dt, INTERVAL 1 DAY)
GROUP BY a.dt;
```

### 5. 日活 / 月活 / DAU/MAU

```sql
SELECT 
  DATE_FORMAT(dt, '%Y-%m') AS month,
  COUNT(DISTINCT user_id) AS mau
FROM login GROUP BY month;
```

### 6. 漏斗 / 转化率

```sql
SELECT 
  SUM(CASE WHEN step='view' THEN 1 ELSE 0 END) AS view_cnt,
  SUM(CASE WHEN step='cart' THEN 1 ELSE 0 END) AS cart_cnt,
  SUM(CASE WHEN step='pay'  THEN 1 ELSE 0 END) AS pay_cnt,
  SUM(CASE WHEN step='pay'  THEN 1 ELSE 0 END) * 1.0 /
  SUM(CASE WHEN step='view' THEN 1 ELSE 0 END) AS conv_rate
FROM event_log;
```

### 7. 连续登录 N 天（高频面试题）

```sql
-- 思路：dt - ROW_NUMBER 相同就是连续日
SELECT user_id, MIN(dt) start_dt, MAX(dt) end_dt, COUNT(*) days
FROM (
  SELECT user_id, dt,
    DATE_SUB(dt, INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY dt) DAY) grp
  FROM login
) t
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

### 8. 去重 / 行转列

```sql
-- 去重
SELECT DISTINCT col FROM t;
SELECT col FROM t GROUP BY col;

-- 行转列（透视）
SELECT 
  user_id,
  SUM(CASE WHEN month='2026-04' THEN amount ELSE 0 END) AS apr,
  SUM(CASE WHEN month='2026-05' THEN amount ELSE 0 END) AS may
FROM orders GROUP BY user_id;
```

**练习平台**：
- **LeetCode SQL 50 题**（中文站）→ 这 50 题刷完笔试 80% 能写
- **牛客 SQL 必知必会**
- **HackerRank SQL**（英文岗）

---

## 三、统计 / 概率 / AB 实验（⭐⭐⭐ 数据岗必背）

### 1. 概念速记

| 概念 | 一句话 |
|------|--------|
| 期望 E(X) | 加权平均 |
| 方差 Var(X) | E[(X-μ)²]，衡量离散程度 |
| 协方差 / 相关系数 | 线性相关度，相关系数 ∈ [-1, 1] |
| 大数定律 | 样本均值 → 总体均值 |
| 中心极限定理 | 独立同分布样本均值近似正态 |
| p 值 | 假设原假设成立时，看到当前或更极端结果的概率 |
| 显著性水平 α | 通常 0.05，p < α 拒绝原假设 |
| 第一类错误 | 弃真（α）｜第二类 | 取伪（β） |
| 置信区间 | 95% CI = 重复抽样 95% 包含真值 |

### 2. 常见分布

- **二项分布**：n 次独立伯努利，E=np
- **泊松分布**：单位时间事件次数，E=Var=λ
- **正态分布**：68-95-99.7 法则
- **指数分布**：事件间隔时间

### 3. AB 实验（数据分析必考！）

**问：怎么做 AB 实验？**

> 1. **明确指标**：北极星 + 护栏。新功能上线，主指标 = CTR，护栏 = 加载时长、崩溃率
> 2. **样本量估算**：根据 MDE（最小可检测效应）、α=0.05、Power=0.8 算样本
> 3. **分流**：用户 hash 取模，保证均匀；按设备/用户层级
> 4. **运行时间**：至少一个完整业务周期（含周末）
> 5. **显著性检验**：t 检验（均值类）/ 卡方（比率类）/ 秩和（非正态）
> 6. **检查 AA**：先开 AA 对照确认无偏
> 7. **下结论**：p < 0.05 且效应量有业务意义才上线

**辛普森悖论**：分组结论 ≠ 总体结论。AB 时按子人群再看一遍。

**问：AB 没显著怎么办？**
- 样本不够 → 延长时间
- 用户体感差 → 看分位数（中位数、p90）
- 漏斗下游影响 → 看护栏

### 4. 经典面试题

**Q：抛硬币 100 次出现 60 次正面，硬币是否公平？**  
→ H0: p=0.5；二项检验 / 近似正态 z = (0.6-0.5)/√(0.5*0.5/100) = 2，p≈0.046，**拒绝 H0**。

**Q：怎么判断异常值？**  
→ 3σ 原则 / IQR 法（Q1-1.5IQR ~ Q3+1.5IQR）/ Z-score。

**Q：偏态分布怎么处理？**  
→ log 变换、Box-Cox；分析时用中位数代替均值。

---

## 四、Python / pandas 速记（⭐⭐⭐）

### 1. pandas 必会 20 行

```python
import pandas as pd
df = pd.read_csv('x.csv')

df.head(); df.info(); df.describe()              # 看数据
df.isna().sum()                                  # 缺失
df['col'].value_counts(normalize=True)           # 频率
df.dropna(subset=['a']); df.fillna(0)            # 处理缺失
df[df['a'] > 0]                                  # 筛选
df.loc[df.a>0, 'b'] = 1                          # 条件赋值
df.groupby('cat').agg({'amt':['sum','mean']})    # 分组
df.pivot_table(index='a', columns='b', values='c', aggfunc='sum')
pd.merge(a, b, on='id', how='left')              # 表合并
df.sort_values('col', ascending=False).head(10)  # 排序取前 N
df['col'].apply(lambda x: x*2)                   # 应用函数
df['dt'] = pd.to_datetime(df['dt'])              # 时间
df.set_index('dt').resample('D').sum()           # 重采样
df['rank'] = df.groupby('cat')['amt'].rank(ascending=False)  # 组内排名
df.duplicated(); df.drop_duplicates()
```

### 2. 数据清洗 4 步

1. 缺失值：删 / 填均值/众数 / 模型预测 / 标记 "Unknown"
2. 异常值：截断（clip）/ 分箱 / 删除（少时）
3. 类型转换：`astype` / `to_datetime`
4. 标准化：`StandardScaler` (z-score) / `MinMaxScaler` (0-1)

### 3. Python 基础八股

- **可变 vs 不可变**：list/dict/set 可变；tuple/str/int 不可变
- **`is` vs `==`**：is 比较 id；== 比较值
- **浅 vs 深拷贝**：`copy()` vs `deepcopy()`
- **GIL**：CPython 全局锁；CPU 密集用 multiprocessing；IO 密集用 threading/asyncio
- **装饰器**：闭包 + 高阶函数
- **生成器 vs 列表**：`yield` 惰性求值省内存
- **`__init__` vs `__new__`**：new 创建实例；init 初始化
- **with 上下文**：`__enter__` / `__exit__`，自动释放资源
- **collections**：`Counter`、`defaultdict`、`OrderedDict`、`deque`

---

## 五、机器学习 / 深度学习八股（⭐⭐⭐ AI 岗必背）

### 1. 基础概念

| 问题 | 答案 |
|------|------|
| 过拟合怎么办？ | 增加数据、正则化（L1/L2/Dropout）、早停、降低模型复杂度、交叉验证 |
| L1 vs L2 | L1 产生稀疏解（特征选择）；L2 平滑系数 |
| 梯度消失 / 爆炸 | 残差连接、BN、合适激活函数（ReLU）、梯度裁剪 |
| 为什么 ReLU 不用 Sigmoid | Sigmoid 易饱和导致梯度消失；ReLU 简单且不饱和 |
| BN 作用 | 加速收敛、稳定训练、轻微正则 |
| Dropout 原理 | 训练随机置零神经元，inference 全开，类似集成 |
| Adam vs SGD | Adam 自适应学习率；SGD+momentum 泛化常更好 |
| 偏差 vs 方差 | 欠拟合高偏差；过拟合高方差 |

### 2. 经典模型一句话

- **LR**：线性 + sigmoid，二分类
- **SVM**：最大间隔，核技巧映射高维
- **决策树**：信息增益 / gini 分裂
- **随机森林**：bagging 多树投票，抗过拟合
- **GBDT/XGBoost/LightGBM**：boosting 残差拟合；XGB 二阶导 + 正则；LGB 直方图 + Leaf-wise 更快
- **K-Means**：距离最近的中心 → 更新中心
- **PCA**：协方差矩阵特征分解，最大方差方向降维

### 3. 评估指标

| 指标 | 公式 | 用法 |
|------|------|------|
| Accuracy | (TP+TN)/all | 类别均衡时 |
| Precision | TP/(TP+FP) | 不能误判（垃圾邮件） |
| Recall | TP/(TP+FN) | 不能漏判（癌症筛查） |
| F1 | 2PR/(P+R) | 平衡 |
| AUC | ROC 曲线下面积 | 排序能力 |
| RMSE / MAE | 回归 | RMSE 对大误差敏感 |

### 4. 损失函数

- 回归：MSE / MAE / Huber
- 二分类：BCE
- 多分类：CrossEntropy
- 排序：Pairwise / Triplet Loss
- 对比学习：InfoNCE（CLIP 用的！）

### 5. Transformer & Attention（必背）

**Self-Attention 三步**：
```
Q, K, V = X·W_Q, X·W_K, X·W_V
Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V
```

**为什么除 √d_k**：防止内积过大导致 softmax 梯度消失。

**Multi-Head**：并行多个头，捕捉不同语义子空间。

**位置编码**：Transformer 无序，需 sin/cos 或 RoPE 注入位置。

**Encoder vs Decoder**：Encoder 双向；Decoder 带 mask 防看到未来。

### 6. CLIP（你的项目，必精通）

- 对比学习：N 对图文，对角线是正样本，其他都是负样本
- 损失：InfoNCE，图→文 + 文→图 对称
- 推理：图和文共用一个空间，余弦相似度直接对齐
- 局限：英文为主、长文本表现差、细粒度识别弱

---

## 六、大模型 / LLM 八股（⭐⭐⭐ AI 应用岗）

### 1. 必背概念

| 概念 | 解释 |
|------|------|
| **Tokenization** | BPE / WordPiece / Unigram；中文常用 SentencePiece |
| **Pretraining** | 大语料无监督预训练（CLM / MLM） |
| **SFT** | 监督微调，喂指令-回答对 |
| **RLHF** | 人类反馈强化学习（PPO / DPO） |
| **In-Context Learning** | 不更新参数，靠 prompt 给例子 |
| **CoT** | Chain-of-Thought，让模型一步一步想 |
| **Few-shot / Zero-shot** | prompt 里有/无例子 |
| **Hallucination** | 模型编造事实 |
| **Temperature** | 0=确定 / 高=多样 |
| **Top-k / Top-p** | 采样策略 |

### 2. RAG 八股（高频）

**完整链路**：
```
Query → (改写/扩展) → Embedding → 向量检索 → Rerank → Prompt 拼接 → LLM 生成 → 后处理
```

**常见追问**：
- **怎么切 chunk**：固定长度（500–1000 字）+ 重叠（10–20%）；按语义/标题切更好
- **embedding 模型**：bge / m3e / OpenAI text-embedding-3-small
- **向量库**：FAISS / Milvus / Qdrant / Pinecone / Chroma
- **检索器**：BM25（稀疏）+ 向量（稠密）混合 + Rerank
- **怎么减少幻觉**：限制只回答检索结果中的内容、加引用、低 temperature

### 3. 微调方法

| 方法 | 一句话 |
|------|--------|
| Full Fine-tune | 全参数更新，效果好但贵 |
| **LoRA** | 低秩矩阵，冻原模型只学 A·B，参数 < 1% |
| QLoRA | LoRA + 4bit 量化，单卡可训 7B |
| P-Tuning v2 | 在每层加可训 prefix |
| Prompt Tuning | 只调输入 prompt embedding |

### 4. Prompt Engineering 5 招

1. **角色设定**：「你是资深数据分析师」
2. **Few-shot**：给 2-3 个例子
3. **CoT**：「请逐步推理」/「Let's think step by step」
4. **结构化输出**：要求 JSON / 表格
5. **约束**：「只用 50 字以内」「不要编造」

### 5. 评估 LLM

- 自动：BLEU / ROUGE / BERTScore（生成）；MMLU / C-Eval（知识）
- 人工：相关性、流畅度、安全性
- 业务：AB 实验、用户满意度

---

## 七、算法题刷题计划（⭐⭐⭐）

### 1. 笔试常考类型 + 模板

| 类型 | LeetCode 代表题 | 模板 |
|------|----------------|------|
| 哈希 | 1, 49, 128 | dict 一次遍历 |
| 双指针 | 11, 15, 167 | 左右收缩 |
| 滑窗 | 3, 76, 209 | 右扩 + 不满足时左收 |
| 二分 | 35, 33, 153 | l=0,r=n-1, while l<=r |
| 栈 | 20, 155, 739 | 单调栈 |
| BFS/DFS | 200, 102, 207 | queue / 递归 + visited |
| 回溯 | 46, 78, 39 | 选 / 不选 |
| DP | 53, 70, 322, 198 | dp[i] 状态定义 |
| 链表 | 206, 21, 141 | 虚拟头 + 快慢指针 |
| 树 | 104, 226, 235 | 递归三件套 |
| 排序 | 215, 912 | 快速 / 归并 / 堆 |
| 贪心 | 55, 122 | 局部最优 → 全局 |

### 2. 必刷 50 题（按重要性）

**第一周（基础）**：
1, 26, 27, 88, 121, 122, 136, 169, 217, 283  
20, 155, 234, 206, 21, 141, 160  
70, 198, 53, 121, 322

**第二周（中等）**：
3, 5, 11, 15, 33, 49, 56, 75, 102, 104  
146 (LRU), 200, 215, 236, 300, 322, 416

**第三周（巩固）**：
- 重新写一遍错过的题
- 每天 1 道 Hard 试试看（42 接雨水、76 最小覆盖子串）

### 3. 笔试编程题专项（牛客 / ACM 模式）

- **输入处理**：`input()` / `sys.stdin.readline()`；多行用 while  
- **大数据量**：`sys.stdin` 比 `input()` 快 10 倍

```python
import sys
input = sys.stdin.readline
n = int(input())
arr = list(map(int, input().split()))
```

- **常见小坑**：浮点精度、整型溢出（Python 无溢出但要注意）、负数取模

### 4. 推荐资源

- **代码随想录**（公众号 / 网站）：算法体系学习
- **LeetCode 热门 100 + Hot 75**：刷题首选
- **牛客网**：国内笔试模拟（陌陌也用牛客）
- **力扣面试经典 150 题**

---

## 八、操作系统 / 网络（⭐⭐ 后端/Infra 才考）

数据岗一般不深考，知道概念即可：

- **TCP 三次握手 / 四次挥手**
- **HTTP 状态码**：2xx 成功 / 3xx 重定向 / 4xx 客户端错 / 5xx 服务器错
- **HTTP vs HTTPS**：HTTPS 加 TLS
- **GET vs POST**：GET 幂等、参数在 URL；POST 不幂等、参数在 body
- **进程 vs 线程 vs 协程**
- **死锁四条件**：互斥、占有等待、不可抢占、循环等待
- **Cache 策略**：LRU / LFU / FIFO

---

## 九、业务分析 / Case Study（数据岗）

**经典题型**：

### Q：DAU 突然跌 20%，怎么排查？

> 1. **确认数据准确**：埋点是否有问题？日志是否丢失？
> 2. **拆维度**：新老用户 / 渠道 / 平台（iOS/Android）/ 版本 / 地区 / 时段
> 3. **看相关事件**：是否新版本上线？是否运营活动结束？是否竞品有大动作？
> 4. **看漏斗**：是登录前掉了还是登录后掉了？
> 5. **定位根因 + 制定方案**

### Q：怎么衡量一个新功能成功？

> 1. **定义目标**：吸引新用户 / 提留存 / 提 GMV？
> 2. **拆指标**：北极星 + 护栏 + 过程指标
> 3. **AB 实验**：随机分流 → 跑足时间 → 显著性检验
> 4. **看长期**：避免短期诱导（如推送轰炸提了点击但伤留存）

### Q：陌陌（社交）核心指标？

> - DAU/MAU、人均使用时长、消息数
> - 新增用户 LTV、付费率（VIP/礼物）
> - 关键社交动作完成率（注册→上传头像→首条动态→首次聊天）
> - 留存：次留、7留、30留

---

## 十、十二天冲刺时间表（建议）

| 天数 | 上午 | 下午 | 晚上 |
|------|------|------|------|
| Day 1 | SQL 七大子句 + 窗口函数 | 刷 LC SQL 1-10 | 复盘错题 |
| Day 2 | SQL 留存 / 漏斗 / 连续登录 | 刷 SQL 11-25 | 算法 LC 1, 20, 21 |
| Day 3 | 统计概念 + AB 实验 | SQL 26-40 | 算法 LC 53, 70, 121 |
| Day 4 | pandas 20 行 + 数据清洗 | LC SQL 41-50 | 算法 LC 3, 11, 15 |
| Day 5 | Python 八股 + collections | 业务案例 5 题 | 算法 LC 33, 49, 56 |
| Day 6 | ML 基础八股 | 评估指标 + 损失 | 算法 LC 102, 104, 200 |
| Day 7 | Transformer + Attention | CLIP / 对比学习 | 算法 LC 206, 215, 146 |
| Day 8 | LLM 八股 + RAG | LoRA / 微调 / Prompt | 算法 LC 300, 322, 416 |
| Day 9 | 模拟笔试 90min | 复盘 | 错题二刷 |
| Day 10 | 业务 case 5 题 | SQL 难题 5 题 | 项目话术 |
| Day 11 | 错题三刷 | 体力补充 + 简历再过 | 早睡 |
| Day 12 | 最后看一眼 cheatsheet | 笔试！ | - |

---

## 十一、面试 / 笔试当天 Checklist

- [ ] 电脑 + 充电器 + 网络稳定
- [ ] 准备好草稿纸和笔
- [ ] 提前 15 分钟登录牛客 / 在线编程平台
- [ ] **打字速度**：笔试 90min 写 1500 字以上才够答完
- [ ] **时间分配**：选择题 1min/题 → SQL 5min/题 → 编程留 30min
- [ ] **不会的题先跳**，最后回头看
- [ ] 编程题：先写**正确解**再优化，过样例最重要

---

## 十二、核心 Cheat Sheet（打印出来贴墙上）

```
=== SQL ===
执行: FROM→JOIN→WHERE→GROUP→HAVING→SELECT→ORDER→LIMIT
窗口: ROW_NUMBER/RANK/DENSE_RANK, LAG/LEAD, SUM() OVER (PARTITION BY)

=== 统计 ===
CLT: 均值近似正态; p<0.05 拒 H0
AB: 指标→样本量→分流→显著性→护栏

=== ML ===
过拟合: 数据+正则+早停+CV
评估: P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R)
Boosting: 残差拟合; XGB 二阶导, LGB 直方图

=== Transformer ===
Attention = softmax(QKᵀ/√d_k)·V
为什么 √d_k: 防 softmax 梯度消失
CLIP: 对比学习 InfoNCE

=== RAG ===
Query → Embedding → 向量检索 → Rerank → LLM
减幻觉: 限定来源 + 引用 + 低 temperature

=== Python ===
GIL: CPU 用进程, IO 用线程
浅深拷贝: copy / deepcopy
可变: list/dict/set; 不可变: tuple/str/int
```

---

**最终建议**：你已经有项目和 Demo（这是最大优势）。**八股 + 算法 = 让你不被一面刷掉**。  
**陌陌笔试**重点冲 **SQL 50 题 + LC 必刷 50 + AB 实验**，3-5 天可见效。

相关：
- 项目讲解：`../INTERVIEW.md`
- 面试流程：`INTERVIEW_PREP.md`
- 简历：`../RESUME.md`
